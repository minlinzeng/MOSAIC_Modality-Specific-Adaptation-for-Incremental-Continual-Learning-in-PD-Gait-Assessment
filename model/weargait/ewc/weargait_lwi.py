import os
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path

# --- 导入 WearGait 核心组件 ---
from model.weargait.ewc.config import Config
import model.weargait.ewc.utility as U
from model.weargait.ewc.data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, build_subj2label
)
from model.weargait.ewc.encoder import WearGaitUniversal

# 导入 LwI 最优传输模块 (确保路径对齐你的项目结构)
from model.baselines.LwI import optimal_transport as ot

# ==========================================
# LwI 配置与工具函数
# ==========================================
class OTConfig:
    def __init__(self, args, device):
        self.args = args
        self.layers = args.layers                
        self.ensemble_step = args.step       
        self.ensemble_step_diff = args.step_diff
        
        self.ground_metric = 'euclidean' 
        self.ground_metric_normalize = 'log'
        self.reg = 0.01
        self.unbalanced = False
        self.gpu_id = 0 if device.type == 'cuda' else -1
        self.geom_ensemble_type = 'wts'
        self.clip_gm = False
        self.dist_normalize = True
        self.debug = False

        self.ground_metric_eff = False
        self.clip_min = 0.0
        self.clip_max = 1.0
        self.normalize_wts = False
        self.act_num_samples = 1.0
        self.not_squared = False

def recalibrate_bn(model, loader, device, mod, task_idx):
    """
    权重经过 OT 融合后，必须使用当前新模态数据重新校准 Shared BN 的统计量。
    """
    model.train()
    model.set_active_modality(mod)
    if hasattr(model, 'set_active_task'):
        model.set_active_task(task_idx)

    for p in model.parameters(): 
        p.requires_grad = False
        
    print(f"   🔄 [LwI] Recalibrating Shared Batch Norm statistics using '{mod.upper()}' data...")
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i > 50: break
            _ = model(x.to(device))
            
    for p in model.parameters(): 
        p.requires_grad = True
    print("   ✅ [LwI] Recalibration Complete.")

# ==========================================
# 核心训练逻辑 (包含 Chimera KD)
# ==========================================
def train_lwi_task(args, model, model_old, train_loader, val_loader, mod, task_id, device):
    print(f"\n   >>> [LwI] Training '{mod.upper()}' (Task {task_id}) | Feat KD $\lambda$: {args.kd_lambda}")
    
    model.train()
    model.set_active_modality(mod)
    if hasattr(model, 'set_active_task'):
        model.set_active_task(task_id)

    if model_old is not None:
        model_old.eval()
        model_old.set_active_modality(mod)
        if hasattr(model_old, 'set_active_task'):
            model_old.set_active_task(task_id - 1)
        for p in model_old.parameters(): p.requires_grad = False

    # 仅开启当前模态的前端编码器，冻结其他历史模态前端
    for k in model.encoders.keys():
        for p in model.encoders[k].parameters():
            p.requires_grad = (k == mod)

    active_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(active_params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopper = U.EarlyStopping(patience=args.patience, mode='max')
    
    criterion = nn.CrossEntropyLoss() 
    mse_loss = nn.MSELoss()
    best_eval = 0.0

    # LwI Warmup 设定 (前 5 个 Epoch 冻结共享层，只适应 Encoder)
    WARMUP_EPOCHS = 5

    for ep in range(1, args.epochs + 1):
        if model_old is not None and ep <= WARMUP_EPOCHS:
            phase = "WARMUP"
            current_lambda = 0.0
            for p in model.shared_backbone.parameters(): p.requires_grad = False
            for p in model.shared_head.parameters(): p.requires_grad = False
        else:
            phase = "TRAIN "
            current_lambda = args.kd_lambda
            for p in model.shared_backbone.parameters(): p.requires_grad = True
            for p in model.shared_head.parameters(): p.requires_grad = True

        model.train()
        accum = {"loss": 0, "ce": 0, "kd": 0, "correct": 0, "total": 0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 前向传播提取表征
            z_encoder = model.encoders[mod](x)
            z_new = model.shared_backbone(z_encoder)
            logits_new = model.shared_head(z_new)
            
            loss_ce = criterion(logits_new, y)
            loss = loss_ce
            loss_kd_val = torch.tensor(0.0)

            # LwI 特征级 KD (Chimera Distillation)
            if model_old is not None and current_lambda > 0:
                with torch.no_grad():
                    z_old_enc = model_old.encoders[mod](x)
                    z_old = model_old.shared_backbone(z_old_enc)
                
                # 严格 L2 归一化特征
                z_new_norm = F.normalize(z_new, p=2, dim=1)
                z_old_norm = F.normalize(z_old, p=2, dim=1)
                
                loss_kd = mse_loss(z_new_norm, z_old_norm)
                loss += current_lambda * loss_kd
                loss_kd_val = loss_kd

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            accum["loss"] += loss.item()
            accum["ce"] += loss_ce.item()
            accum["kd"] += loss_kd_val.item()
            accum["correct"] += (logits_new.argmax(dim=1) == y).sum().item()
            accum["total"] += y.size(0)

        # 验证评估
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                all_preds.extend(v_logits.argmax(1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        best_eval = max(val_f1, best_eval)
        scheduler.step(val_f1)

        if ep % 5 == 0 or ep == 1:
            n = len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   [{mod[:3].upper()}] Ep {ep:02d} [{phase}] | LR: {current_lr:.1e} | "
                  f"Loss:{accum['loss']/n:.3f} [CE:{accum['ce']/n:.3f} KD:{accum['kd']/n:.3f}] | "
                  f"TrAcc:{accum['correct']/accum['total']*100:.1f}% ValF1:{val_f1:.1f}%")

        if early_stopper(val_f1, model):
            print(f"      🛑 [Early Stop] Triggered at Epoch {ep}")
            model.load_state_dict(early_stopper.best_model_state)
            break

    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)

# ==========================================
# 工具函数: WearGait 数据划分
# ==========================================
def _scan_subjects(dir_path: Path):
    return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})

def init_subjects_and_folds(args):
    pd_ids, hc_ids = _scan_subjects(Config.PD_PATH), _scan_subjects(Config.HC_PATH)
    if not pd_ids or not hc_ids: raise ValueError("No subjects found.")
    
    # 借助 data_loader.py 暴露的方法创建 Folds
    from model.weargait.ewc.data_loader import make_fixed_balanced_folds_no_overlap
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    return subj2label, folds

# ==========================================
# 主控与交叉验证 (Cross Validation & OT Fusion)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="WearGait LwI Baseline Execution Script")
    parser.add_argument('--order', type=str, default="walkway,insole,imu")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--win_len', type=int, default=Config.WINDOW_SIZE)
    parser.add_argument('--hop_len', type=int, default=int(Config.WINDOW_SIZE * Config.STRIDE))
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g., cuda:0)")
    
    # 🚨 WearGait 核心降级开关 
    parser.add_argument("--disable_dbn", action='store_true', help="Force network to drop DBN and downgrade to Shared BN")

    # LwI (OT) Specific Arguments
    parser.add_argument('--step', type=float, default=0.3, help="Max similarity fusion step")
    parser.add_argument('--step_diff', type=float, default=0.5, help="Min similarity fusion step")
    parser.add_argument('--layers', type=int, default=2, help="Number of deep layers to apply min-sim to")
    parser.add_argument('--kd_lambda', type=float, default=300.0, help="Chimera Distillation weight")
    
    args = parser.parse_args()
    
    # 强制物理降级为传统的 Shared BN，防止提取私有化统计红利
    args.disable_dbn = True

    U.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ot_config = OTConfig(args, device)
    
    tasks = [t.strip().lower() for t in args.order.split(",")]
    num_tasks = len(tasks)
    
    subj2label, folds = init_subjects_and_folds(args)
    global_cache = preload_all_subjects(Config.OUTPUT_DIR)
    
    R_matrix = np.zeros((len(folds), num_tasks, num_tasks))
    eval_loader_cache = {fi: {} for fi in range(len(folds))}

    print("\n" + "="*65)
    print(f" 🚀 WearGait BASELINE ENGINE: Learning without Isolation (LwI)")
    print(f" Seed: {args.seed} | Modalities: {tasks} | DBN Protection: DISABLED")
    print("="*65)

    for fold, (train_subs, test_subs) in enumerate(folds):
        print(f"\n{'='*60}\n 🌟 INITIATING FOLD {fold+1}/{len(folds)} WITH OPTIMAL TRANSPORT \n{'='*60}")
        
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=True).to(device)
        model_old = None

        for task_idx, active_mod in enumerate(tasks):
            # 加载当前模态数据
            prep = prepare_split(train_subs, test_subs, data_cache=global_cache, win=args.win_len, hop=args.hop_len, modalities=(active_mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            
            tr_loader = DataLoader(U.SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(U.SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[fold][active_mod] = te_loader 
            
            # 1. 训练当前增量任务
            train_lwi_task(args, model, model_old, tr_loader, te_loader, active_mod, task_idx, device)

            # 2. 执行 OT 权重拓扑融合
            if model_old is not None:
                print("\n   🧬 [LwI] Performing Optimal Transport (OT) Weight Fusion...")
                # 🚨 WearGait 中私有前端保存在 ModuleDict 'encoders' 中
                fused_dict = ot.get_wassersteinized_layers_modularized(
                    ot_config, device, networks=[model_old, model], ignore_keyword='encoders'
                )
                
                current_state = model.state_dict()
                for layer_name, new_weight in fused_dict.items():
                    if layer_name in current_state:
                        current_state[layer_name].copy_(new_weight)
                model.load_state_dict(current_state)
                
                # 3. 重新校准 Shared BN
                recalibrate_bn(model, tr_loader, device, active_mod, task_idx)

            model_old = copy.deepcopy(model)

            # 4. 增量矩阵 R 计算
            print(f"   [EVAL] Sequential Backward Testing...")
            for j in range(task_idx + 1):
                eval_mod = tasks[j]
                model.set_active_modality(eval_mod)
                if hasattr(model, 'set_active_task'): model.set_active_task(j)
                
                f1_score_j = U.evaluate_classification(model, eval_loader_cache[fold][eval_mod], device, metric='f1_macro')
                R_matrix[fold, task_idx, j] = f1_score_j
                print(f"      Post-{active_mod.upper()} -> Testing {eval_mod.upper()}: {f1_score_j:.2f}%")

    print("\n" + "="*60 + "\n 🏆 FINAL LwI METRIC MATRIX (R_N,N) \n" + "="*60)
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
    print(f"\n[LwI BASELINE] Final Average F1 (A_N): {avg_acc:.2f}%")
    print(f"[LwI BASELINE] Backward Transfer (BWT): {bwt:.2f}%")

if __name__ == "__main__":
    main()