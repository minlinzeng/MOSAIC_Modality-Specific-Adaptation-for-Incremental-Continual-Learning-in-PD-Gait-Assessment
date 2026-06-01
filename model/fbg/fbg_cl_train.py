import os
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# 核心解耦资产引入
from data_loader import get_fbg_dataloaders
from encoder import MICL_CNN_PD_Model
from fbg_utility import (
    EarlyStopping, 
    CurriculumScheduler, 
    compute_kd_loss, 
    compute_repulsive_loss, 
    set_deterministic_seed,
    compute_fisher_information, 
    compute_ewc_loss,
    set_active_task_and_freeze_fbg  # 🌟 引入物理隔离锁
)

# =====================================================================
# 🌟 CL 终极版：适配 MixUp 的参数路由表 (彻底关闭双重平滑)
# =====================================================================
MODALITY_CONFIG = {
    # 彻底关闭 label_smoothing, 交由 MixUp 处理；延长耐心以穿越复杂流形
    "linear":  {"lr": 1e-4, "weight_decay": 0.05, "dropout": 0.3, "label_smoothing": 0.0, "patience": 15},
    "angular": {"lr": 1e-4, "weight_decay": 0.05, "dropout": 0.3, "label_smoothing": 0.0, "patience": 15}, 
    "grf":     {"lr": 2e-4, "weight_decay": 0.05, "dropout": 0.1, "label_smoothing": 0.0, "patience": 20}   
}

def train_cl_epoch(model, teacher_model, dataloader, criterion, optimizer, device, 
                   active_mod, prev_mod, task_idx, ewc_memories, alpha_rep, args):
    # 物理锁定机制启动
    model.train()
    if not args.disable_msbn:
        set_active_task_and_freeze_fbg(model, task_idx)
    
    if teacher_model: 
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
            
    total_loss, total_ce, total_ewc, total_kd, total_rep = 0, 0, 0, 0, 0
    all_preds, all_labels = [], []
    
    for batch in dataloader:
        lin = batch['linear'].to(device)
        ang = batch['angular'].to(device)
        grf = batch['grf'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # 🌟 修复 1：重新植入受试者流形 MixUp
        alpha_mix = 0.3
        lam = np.random.beta(alpha_mix, alpha_mix)
        batch_size = labels.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mix_lin = lam * lin + (1 - lam) * lin[index] if active_mod == 'linear' else None
        mix_ang = lam * ang + (1 - lam) * ang[index] if active_mod == 'angular' else None
        mix_grf = lam * grf + (1 - lam) * grf[index] if active_mod == 'grf' else None
        labels_a, labels_b = labels, labels[index]
        
        logits_s, feat_s = model(x_lin=mix_lin, x_ang=mix_ang, x_grf=mix_grf, current_task=task_idx)
        
        # 交叉熵损失 (MixUp 版本)
        loss_ce = lam * criterion(logits_s, labels_a) + (1 - lam) * criterion(logits_s, labels_b)
        loss = loss_ce
        total_ce += loss_ce.item()
        
        if task_idx > 0 and teacher_model is not None:
            with torch.no_grad():
                # 🌟 修复 2：教师模型必须接收 MixUp 后的同态数据，以约束插值流形
                prev_mix_lin = lam * lin + (1 - lam) * lin[index] if prev_mod == 'linear' else None
                prev_mix_ang = lam * ang + (1 - lam) * ang[index] if prev_mod == 'angular' else None
                prev_mix_grf = lam * grf + (1 - lam) * grf[index] if prev_mod == 'grf' else None
                
                logits_t, feat_t = teacher_model(x_lin=prev_mix_lin, x_ang=prev_mix_ang, x_grf=prev_mix_grf, current_task=task_idx-1)
            
            if args.lambda_kd > 0:
                raw_kd = compute_kd_loss(logits_s, logits_t, tau=args.kd_tau)
                weighted_kd = args.lambda_kd * raw_kd
                loss += weighted_kd
                total_kd += weighted_kd.item()  # 记录砸向梯度的真实力量
            
            if alpha_rep > 0:
                raw_rep = compute_repulsive_loss(feat_s, feat_t, margin=args.repulsive_margin)
                weighted_rep = alpha_rep * raw_rep
                loss += weighted_rep
                total_rep += weighted_rep.item() # 记录砸向梯度的真实力量
            
            if args.lambda_ewc > 0:
                weighted_ewc = compute_ewc_loss(model, ewc_memories, lambda_ewc=args.lambda_ewc)
                loss += weighted_ewc
                total_ewc += (weighted_ewc.item() if isinstance(weighted_ewc, torch.Tensor) else weighted_ewc)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits_s, dim=1)
        # 仅为指标监控映射回硬标签
        actual_labels = labels_a if lam > 0.5 else labels_b
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(actual_labels.cpu().numpy())
        
    metrics = {
        'loss': total_loss / len(dataloader), 'ce': total_ce / len(dataloader),
        'ewc': total_ewc / len(dataloader) if task_idx > 0 else 0,
        'kd': total_kd / len(dataloader) if task_idx > 0 else 0,
        'rep': total_rep / len(dataloader) if task_idx > 0 else 0,
        'f1': f1_score(all_labels, all_preds, average='macro') * 100
    }
    return metrics

@torch.no_grad()
def evaluate_cl(model, dataloader, device, eval_mod, eval_task_idx):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        lin = batch['linear'].to(device)
        ang = batch['angular'].to(device)
        grf = batch['grf'].to(device)
        labels = batch['label'].to(device)
        
        logits, _ = model(x_lin=lin if eval_mod=='linear' else None,
                          x_ang=ang if eval_mod=='angular' else None,
                          x_grf=grf if eval_mod=='grf' else None, 
                          current_task=eval_task_idx)
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return f1_score(all_labels, all_preds, average='macro') * 100

def get_all_subjects(data_root):
    import glob
    files = glob.glob(os.path.join(data_root, "*.pkl"))
    subjects = sorted(list(set([os.path.basename(f).split('_')[0] for f in files])))
    return subjects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--order', type=str, default="linear,angular,grf")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--lambda_ewc', type=float, default=50.0)
    parser.add_argument('--lambda_kd', type=float, default=1.0)
    parser.add_argument('--kd_tau', type=float, default=4.0)
    parser.add_argument('--alpha_max', type=float, default=0.5)
    parser.add_argument('--repulsive_margin', type=float, default=0.3)
    parser.add_argument('--p_degree', type=float, default=5.0)
    parser.add_argument('--disable_curriculum', action='store_true')
    parser.add_argument('--save_dir', type=str, default="./checkpoints")
    parser.add_argument('--window_size', type=int, default=256, help="修改窗口大小")
    parser.add_argument('--step_size', type=int, default=64, help="建议设置为 window_size 的 50% 或 25%")
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--disable_msbn', action='store_true', help="关闭MSBN，降级为Shared BN基线")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_deterministic_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tasks = [t.strip().lower() for t in args.order.split(",")]
    num_tasks = len(tasks)
    subjects = get_all_subjects(args.data_root)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    R_matrix = np.zeros((5, num_tasks, num_tasks))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        print(f"\n{'='*60}\n 🌟 INITIATING FOLD {fold+1}/5 WITH MSBN SECURITY \n{'='*60}")
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        train_loader, test_loader = get_fbg_dataloaders(
            args.data_root, train_subjects, test_subjects, 
            batch_size=args.batch_size,
            window_size=args.window_size, 
            step_size=args.step_size
        )
        
        model = None
        teacher_model = None
        ewc_memories = {}
        
        for task_idx, active_mod in enumerate(tasks):
            print(f"\n  >>> Task {task_idx}: Learning [{active_mod.upper()}] <<<")
            config = MODALITY_CONFIG[active_mod]
            prev_mod = tasks[task_idx-1] if task_idx > 0 else None
            
            # 1. 动态实例化模型与梯度继承管理
            if task_idx == 0:
                model = MICL_CNN_PD_Model(d_model=args.d_model, dropout=config["dropout"], 
                                          num_tasks=num_tasks, disable_msbn=args.disable_msbn).to(device)
                teacher_model = None
            else:
                model.dropout.p = config["dropout"]
                model.res1.drop1d.p = config["dropout"]
                model.res2.drop1d.p = config["dropout"]
                model.res3.drop1d.p = config["dropout"]
            
            # 2. 严格的物理隔离：首先绝对锁定不需要的前端编码器的梯度
            # 这一步是 WearGait 隐藏的必修课：进入当前任务，必须冻结其他模态的特异前端！
            if hasattr(model, 'enc_lin'): model.enc_lin.requires_grad_(active_mod == 'linear')
            if hasattr(model, 'enc_ang'): model.enc_ang.requires_grad_(active_mod == 'angular')
            if hasattr(model, 'enc_grf'): model.enc_grf.requires_grad_(active_mod == 'grf')
            
            # 3. 激活当前任务特异性的 MSBN 分支，并强制将历史 MSBN 锁死
            if not args.disable_msbn:
                set_active_task_and_freeze_fbg(model, task_idx)
                
            # 4. 🌟 终极对齐修复：严格重置优化器 (Strict Optimizer Reset)
            # 过滤出当前真正允许求梯度的参数，彻底斩断跨任务二阶矩爆炸的步长隐患
            active_params = list(filter(lambda p: p.requires_grad, model.parameters()))
            actual_wd = 0.0 if (task_idx > 0 and args.lambda_ewc > 0) else config["weight_decay"]
            optimizer = optim.AdamW(active_params, lr=config["lr"], weight_decay=actual_wd) # <--- 传入净化后的 active_params
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
            early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
            curriculum = CurriculumScheduler(
                alpha_max=args.alpha_max, kd_lambda_base=args.lambda_kd, 
                kd_lambda_min=0.1, p_degree=args.p_degree, total_epochs=args.epochs
            )
            
            if task_idx > 0:
                print(f"      [🛡️ Warm-up] Aligning Random Encoder & Warming MSBN for {active_mod.upper()}...")
                for param in model.parameters():
                    param.requires_grad = False
                    
                # 2. 仅解冻当前模态的前端卷积核
                current_encoder = getattr(model, f"enc_{active_mod[:3]}")
                for param in current_encoder.parameters():
                    param.requires_grad = True
                
                # 3. 核心修正：依靠你底层的 set_active_task_and_freeze_fbg 
                # 此时只有 task_idx 对应的 BN 的 requires_grad 被设为了 True
                if not args.disable_msbn:
                    set_active_task_and_freeze_fbg(model, task_idx)
                
                # 4. 使用单独的优化器，仅对前端进行轻量级预热
                warmup_opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
                warmup_criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
                
                for w_ep in range(args.warmup_epochs):
                    # 每次迭代前，强制唤醒当前任务的 MSBN 进入 train 模式以累积 running_mean/var
                    # 同时保证历史任务的 BN 绝对死锁在 eval 状态
                    model.train()
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm1d):
                            if list(m.parameters())[0].requires_grad:
                                m.train()
                            else:
                                m.eval()

                    for batch in train_loader:
                        lin = batch['linear'].to(device) if active_mod == 'linear' else None
                        ang = batch['angular'].to(device) if active_mod == 'angular' else None
                        grf = batch['grf'].to(device) if active_mod == 'grf' else None
                        labels = batch['label'].to(device)
                        
                        warmup_opt.zero_grad()
                        logits_w, _ = model(x_lin=lin, x_ang=ang, x_grf=grf, current_task=task_idx)
                        loss_w = warmup_criterion(logits_w, labels)
                        loss_w.backward()
                        warmup_opt.step()
                        
                print(f"      [🛡️ Warm-up] Alignment & MSBN Initialization Complete.")
                teacher_model = copy.deepcopy(model).eval()
                for param in teacher_model.parameters():
                    param.requires_grad = False
                # 5. 恢复 CL 阶段应当被训练的参数梯度
                for param in model.parameters():
                    param.requires_grad = True
                    
                # 重新冻结其他模态的前端
                if hasattr(model, 'enc_lin'): model.enc_lin.requires_grad_(active_mod == 'linear')
                if hasattr(model, 'enc_ang'): model.enc_ang.requires_grad_(active_mod == 'angular')
                if hasattr(model, 'enc_grf'): model.enc_grf.requires_grad_(active_mod == 'grf')

            # 5. 任务级训练循环
            for ep in range(1, args.epochs + 1):
                if args.disable_curriculum:
                    alpha_rep = args.alpha_max
                    current_kd_lambda = args.lambda_kd # 保持传入的 base 值绝对恒定
                else:
                    alpha_rep, current_kd_lambda = curriculum.get_weights(ep)
                args.lambda_kd = current_kd_lambda 
                
                metrics = train_cl_epoch(model, teacher_model, train_loader, criterion, optimizer, device, 
                                         active_mod, prev_mod, task_idx, ewc_memories, alpha_rep, args)
                
                val_f1 = evaluate_cl(model, test_loader, device, active_mod, task_idx)
                scheduler.step(val_f1)
                
                stop_signal = early_stopper(val_f1, model)
                
                # 动态课程早停锁
                lockout_horizon = int((0.5 ** (1.0 / args.p_degree)) * args.epochs)
                curriculum_active = (task_idx > 0) and (ep <= lockout_horizon) and (args.alpha_max > 0.0)
                
                if ep % 5 == 0 or ep == 1:
                    log_str = f"      [Ep {ep:02d}] Tr_Loss: {metrics['loss']:.3f} | Tr_F1: {metrics['f1']:.1f}% | Val_F1: {val_f1:.1f}%"
                    if task_idx > 0: 
                        log_str += f" || wEWC: {metrics['ewc']:.3f} | wRep: {metrics['rep']:.3f} (α={alpha_rep:.2f}) | wKD: {metrics['kd']:.3f} (λ={args.lambda_kd:.2f})"
                    print(log_str, flush=True)
                    
                if stop_signal:
                    if curriculum_active:
                        early_stopper.counter = 0
                        early_stopper.early_stop = False
                    else:
                        print(f"      🛑 Convergence: Early Stop at Ep {ep}.")
                        break
                    
            model.load_state_dict(early_stopper.best_model_state)
            
            # 保存各 Fold 阶段最优权重
            ckpt_path = os.path.join(args.save_dir, f"fbg_fold{fold+1}_task{task_idx}_{active_mod}.pth")
            torch.save(model.state_dict(), ckpt_path)
            
            # 计算全量 Fisher 并安全迁移到 CPU 内存
            fisher_matrix = compute_fisher_information(model, train_loader, device, active_mod, task_idx)
            opt_params = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
            fisher_matrix = {name: f.cpu() for name, f in fisher_matrix.items()}
            ewc_memories[task_idx] = {'fisher': fisher_matrix, 'opt_params': opt_params}
            
            # 严格计算增量评估状态矩阵 R
            for j in range(task_idx + 1):
                eval_mod = tasks[j]
                f1_score_j = evaluate_cl(model, test_loader, device, eval_mod, j)
                R_matrix[fold, task_idx, j] = f1_score_j
                print(f"      [EVAL] Post-{active_mod.upper()} -> Testing {eval_mod.upper()}: {f1_score_j:.2f}%")

    print("\n" + "="*60 + "\n 🏆 FINAL METRIC MATRIX (R_N,N) WITH MSBN INTEGRITY \n" + "="*60)
    mean_R = np.mean(R_matrix, axis=0)
    std_R = np.std(R_matrix, axis=0)
    
    print("      [" + "]\t[".join([t.upper()[:3] for t in tasks]) + "]")
    for i in range(num_tasks):
        row_str = f"T{i}:  "
        for j in range(num_tasks):
            if j <= i:
                row_str += f"{mean_R[i,j]:.1f}±{std_R[i,j]:.1f}\t"
            else:
                row_str += "-----\t"
        print(row_str)
        
    bwt = np.mean([mean_R[-1, j] - mean_R[j, j] for j in range(num_tasks - 1)])
    avg_acc = np.mean(mean_R[-1, :])
    print(f"\nFinal Average F1 (A_N): {avg_acc:.2f}%")
    print(f"Backward Transfer (BWT): {bwt:.2f}%")

if __name__ == "__main__":
    main()