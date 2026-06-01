import os, argparse, copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

# --- 导入 FOG 核心组件 ---
import utility as U
from encoder import WearGaitUniversal
from data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    build_subj2label_fog, make_stratified_folds, SingleModalityDataset
)

# 导入 LwI 最优传输模块
from model.baselines.LwI import optimal_transport as ot
from model.paths import FOG_CACHE

# 全局路径配置 (对接 FOG 的预处理输出)
CACHE_DIR = FOG_CACHE

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

def recalibrate_bn(model, loader, device, mod):
    """
    权重经过 OT 融合后，由于底层分布改变，必须使用新数据重新校准 BN 层的运行统计量。
    """
    model.train()
    model.set_active_modality(mod)
    # 冻结所有权重参数，仅允许 BN 层更新 running_mean 和 running_var
    for p in model.parameters(): 
        p.requires_grad = False
        
    print(f"   🔄 [LwI] Recalibrating Shared Batch Norm statistics using '{mod}' data...")
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i > 50: break # 50 个 batch 足够校准
            x = x.to(device)
            _ = model(x) 
            
    # 恢复梯度计算
    for p in model.parameters(): 
        p.requires_grad = True
    print("   ✅ [LwI] Recalibration Complete.")

# ==========================================
# 核心训练逻辑 (包含 Chimera KD)
# ==========================================
def train_lwi_task(args, model, model_old, train_loader, val_loader, mod, task_id, device):
    print(f"\n   >>> [LwI] Training '{mod}' (Task {task_id+1}) | Feat KD $\lambda$: {args.kd_lambda}")
    
    model.train()
    model.set_active_modality(mod)
    if hasattr(model, 'set_active_task'):
        model.set_active_task(task_id)

    if model_old is not None:
        model_old.eval()
        model_old.set_active_modality(mod)

    # 仅更新当前模态的 encoder 和共享部件
    for k in model.encoders.keys():
        for p in model.encoders[k].parameters():
            p.requires_grad = (k == mod) 

    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(active_params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    early_stopper = U.EarlyStopping(patience=args.patience, mode='max')
    
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    best_eval = 0.0

    # LwI 的 Warmup 设定 (前 5 个 Epoch 冻结共享层，只适应 Encoder)
    WARMUP_EPOCHS = 5

    for ep in range(1, args.epochs + 1):
        if model_old is not None and ep <= WARMUP_EPOCHS:
            phase = "WARMUP"
            current_lambda = 0.0
            for p in model.shared_backbone.parameters(): p.requires_grad = False
            for p in model.shared_head.parameters(): p.requires_grad = False
        else:
            phase = "TRAIN"
            current_lambda = args.kd_lambda
            for p in model.shared_backbone.parameters(): p.requires_grad = True
            for p in model.shared_head.parameters(): p.requires_grad = True

        model.train()
        accum = {"loss": 0, "ce": 0, "kd": 0, "correct": 0, "total": 0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 前向传播 (适配 WearGaitUniversal)
            features = model.encoders[mod](x)
            z_new = model.shared_backbone(features)
            logits = model.shared_head(z_new)
            
            loss_ce = criterion(logits, y)
            loss = loss_ce
            loss_kd_val = torch.tensor(0.0)

            # LwI 特征级 KD (Chimera Distillation)
            if model_old is not None and current_lambda > 0:
                with torch.no_grad():
                    features_old = model_old.encoders[mod](x)
                    z_old = model_old.shared_backbone(features_old)
                
                # 必须对特征进行 L2 归一化再算 MSE
                z_new_norm = F.normalize(z_new, p=2, dim=1)
                z_old_norm = F.normalize(z_old, p=2, dim=1)
                
                loss_kd = mse_loss(z_new_norm, z_old_norm)
                loss += current_lambda * loss_kd
                loss_kd_val = loss_kd

            loss.backward()
            optimizer.step()

            accum["loss"] += loss.item()
            accum["ce"] += loss_ce.item()
            accum["kd"] += loss_kd_val.item()
            accum["correct"] += (logits.argmax(dim=1) == y).sum().item()
            accum["total"] += y.size(0)

        scheduler.step()

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

        if ep % 5 == 0 or ep == 1:
            n = len(train_loader)
            print(f"[{mod}] Ep {ep:02d} [{phase}] | LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Loss:{accum['loss']/n:.4f} [CE:{accum['ce']/n:.4f} KD:{accum['kd']/n:.4f}] | "
                  f"TrAcc:{accum['correct']/accum['total']*100:.1f}% ValF1:{val_f1:.2f}% (Best:{best_eval:.2f}%)")

        if early_stopper(val_f1, model):
            print(f"   🛑 Early Stopping at Ep {ep}")
            model.load_state_dict(early_stopper.best_model_state)
            break

    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)


# ==========================================
# 主控与交叉验证 (Cross Validation & OT Fusion)
# ==========================================
def run_cv_lwi(args, data_cache):
    # FOG 专用的 Json 和 subj2label
    json_path = CACHE_DIR / "subj2label.json"
    subj2label = build_subj2label_fog(str(json_path)) 
    folds = make_stratified_folds(subj2label, n_folds=args.n_folds, seed=args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ot_config = OTConfig(args, device)
    
    tasks = [t.strip() for t in args.order.split(",") if t.strip()]
    eval_loader_cache = {fi: {} for fi in range(len(folds))}
    step_history, fold_scores = {}, []

    for fi in range(len(folds)):
        print(f"\n{'='*20} Fold {fi+1}/{len(folds)} {'='*20}")
        
        # 🚨 LwI 基线必须关闭 DBN (Shared BN) 以保证参数量完全公平
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=True).to(device)
        model_old = None
        seen_mods = []

        for ti, mod in enumerate(tasks):
            print(f"\n=== LwI Task {ti+1}/{len(tasks)} : {mod} ===")
            
            # 数据加载
            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            
            # 使用包装器转换为单一模态 Dataset
            tr_loader = DataLoader(SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[fi][mod] = te_loader 

            # 1. 训练当前任务
            train_lwi_task(args, model, model_old, tr_loader, te_loader, mod, ti, device)

            # 2. 核心：执行 OT 权重融合 (仅在任务 > 1 时)
            if model_old is not None:
                print("\n   🧬 [LwI] Performing Optimal Transport (OT) Weight Fusion...")
                # ignore_keyword 'encoders' 确保只融合 shared_backbone 和 shared_head
                fused_dict = ot.get_wassersteinized_layers_modularized(
                    ot_config, device, networks=[model_old, model], ignore_keyword='encoders'
                )
                
                # 注入融合后的权重
                current_state = model.state_dict()
                for layer_name, new_weight in fused_dict.items():
                    if layer_name in current_state:
                        current_state[layer_name].copy_(new_weight)
                model.load_state_dict(current_state)
                
                # 3. 重新校准 BN
                recalibrate_bn(model, tr_loader, device, mod)

            # 更新旧模型
            model_old = copy.deepcopy(model)
            model_old.eval()

            # 4. 评估所有已学模态
            seen_mods.append(mod)
            print(f"\n--- Evaluation (Step {ti+1}) ---")
            scores = []
            for seen_task_idx, m in enumerate(seen_mods):
                model.eval()
                model.set_active_modality(m)
                if hasattr(model, 'set_active_task'):
                    model.set_active_task(seen_task_idx)
                
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for vx, vy in eval_loader_cache[fi][m]:
                        vx, vy = vx.to(device), vy.to(device)
                        v_logits = model(vx)
                        all_preds.extend(v_logits.argmax(1).cpu().numpy())
                        all_targets.extend(vy.cpu().numpy())
                
                score = f1_score(all_targets, all_preds, average='macro') * 100.0
                scores.append(score)
                print(f"  {m}: {score:.2f}")

            if ti not in step_history: step_history[ti] = {}
            for m_idx, m_score in enumerate(scores):
                if m_idx not in step_history[ti]: step_history[ti][m_idx] = []
                step_history[ti][m_idx].append(m_score) 

            avg_seen = sum(scores) / len(scores)
            print(f"  Avg Seen: {avg_seen:.2f}")

        fold_scores.append(avg_seen)

    print(f"\n🏆 Final Avg F1 across folds: {sum(fold_scores)/len(fold_scores):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    # FOG 三大模态
    ap.add_argument("--order", type=str, default="acc,gyr,skeleton")
    
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16) # 与 FOG 对齐的 batch size
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=80)     # 与 FOG 对齐的 epochs
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--win_len", type=int, default=120)
    ap.add_argument("--hop_len", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=3) # FOG (H&Y 3分类)
    ap.add_argument("--disable_dbn", action='store_true')

    # LwI (OT) Specific Arguments
    ap.add_argument('--step', type=float, default=0.3, help="Max similarity fusion step")
    ap.add_argument('--step_diff', type=float, default=0.5, help="Min similarity fusion step")
    ap.add_argument('--layers', type=int, default=14, help="Number of deep layers to apply min-sim to")
    ap.add_argument('--kd_lambda', type=float, default=300.0, help="Chimera Distillation weight")

    args = ap.parse_args()
    print(f"LwI Baseline Mode | Arguments: {', '.join(f'{k}={v}' for k, v in vars(args).items())}")
    
    U.set_seed(args.seed)
    global_cache = preload_all_subjects(CACHE_DIR)
    
    run_cv_lwi(args, global_cache)