import os, argparse, copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

# --- Core imports ---
import utility as U
from encoder import WearGaitUniversal
from data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    build_subj2label_fog, make_stratified_folds, SingleModalityDataset
)
from model.weargait.ewc.EWC import ElasticWeightConsolidation
from model.paths import FOG_CACHE

CACHE_DIR = FOG_CACHE
CHECKPOINT_DIR = Path("./checkpoints")

MODALITY_CONFIG = {
    "acc":      {"lr": 1e-4, "weight_decay": 2e-2}, # raised from 1e-2 to 2e-2
    "gyr":      {"lr": 1e-4, "weight_decay": 2e-2},
    "skeleton": {"lr": 1e-4, "weight_decay": 5e-2}  # heavy 5e-2 WD for skeleton overfitting
}

# ==========================================
# 1. Loss engine
# ==========================================
class LossEngine:
    def __init__(self, ewc, teacher_model, args, device):
        self.ewc = ewc
        self.teacher_model = teacher_model
        self.args = args
        self.device = device
        
        self.kd_lambda = args.kd_lambda
        self.repulsive_alpha = getattr(args, 'repulsive_alpha', 0.0)
        self.repulsive_margin = getattr(args, 'repulsive_margin', 0.1)

    def compute(self, logits, z, y, x, mod): 
        raw_ce = self.ewc.criterion(logits, y)
        weighted_ewc_tensor = self.ewc._ewc_penalty()
        weighted_ewc = weighted_ewc_tensor.item() if torch.is_tensor(weighted_ewc_tensor) else weighted_ewc_tensor
        
        ewc_multiplier = 0.5 * self.ewc.weight
        raw_ewc = (weighted_ewc / ewc_multiplier) if ewc_multiplier > 0 else 0.0

        raw_kd = torch.tensor(0.0, device=self.device)
        raw_repulsion = torch.tensor(0.0, device=self.device)
        weighted_kd, weighted_repulsion = 0.0, 0.0
        
        requires_teacher = (self.teacher_model is not None) and (self.kd_lambda > 0 or self.repulsive_alpha > 0)
                           
        if requires_teacher:
            with torch.no_grad():
                t_features = self.teacher_model.encoders[mod](x)
                t_z = self.teacher_model.shared_backbone(t_features)
                if self.kd_lambda > 0:
                    t_logits = self.teacher_model.shared_head(t_z)

            # A. Semantic knowledge distillation
            if self.kd_lambda > 0:
                T = 2.0
                p_s = F.log_softmax(logits / T, dim=1)
                p_t = F.softmax(t_logits / T, dim=1)
                raw_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)
                weighted_kd = self.kd_lambda * raw_kd

            # B. Repulsive manifold loss
            if self.repulsive_alpha > 0:
                cos_sim = F.cosine_similarity(z, t_z, dim=1)
                raw_repulsion = F.relu(cos_sim - self.repulsive_margin).mean()
                weighted_repulsion = self.repulsive_alpha * raw_repulsion

        total_loss = raw_ce + weighted_ewc_tensor + weighted_kd + weighted_repulsion
        metrics = {
            "loss": total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            "raw_ce": raw_ce.item() if torch.is_tensor(raw_ce) else raw_ce,
            "raw_ewc": raw_ewc,                 
            "raw_kd": raw_kd.item() if torch.is_tensor(raw_kd) else raw_kd,
            "raw_repul": raw_repulsion.item() if torch.is_tensor(raw_repulsion) else raw_repulsion,
            "w_ewc": weighted_ewc,              
            "w_kd": weighted_kd.item() if torch.is_tensor(weighted_kd) else weighted_kd,
            "w_repul": weighted_repulsion.item() if torch.is_tensor(weighted_repulsion) else weighted_repulsion
        }
        return total_loss, metrics

# ==========================================
# 2. State management
# ==========================================
def register_shared_ewc(model, ewc, dataloader, num_batches, task_id):
    set_active_task_and_freeze(model, task_id)
    ewc.register_ewc_params(dataloader, task_id=task_id, num_batches=num_batches)

def set_active_task_and_freeze(model, task_id):
    if hasattr(model, 'set_active_task'):
        model.set_active_task(task_id)
    
    for m in model.modules():
        if hasattr(m, 'bn1_list') and hasattr(m, 'bn2_list'):
            for bn in m.bn1_list:
                for p in bn.parameters(): p.requires_grad = False
            for bn in m.bn2_list:
                for p in bn.parameters(): p.requires_grad = False
            
            if task_id < len(m.bn1_list):
                for p in m.bn1_list[task_id].parameters(): p.requires_grad = True
                for p in m.bn2_list[task_id].parameters(): p.requires_grad = True

def unfreeze_shared_components(model, mod):
    for p in model.shared_backbone.parameters(): p.requires_grad = True
    for p in model.shared_head.parameters():     p.requires_grad = True
    for p in model.encoders[mod].parameters():   p.requires_grad = True

# ==========================================
# 3. Training loop
# ==========================================
def run_warmup_phase(args, model, train_loader, val_loader, device, mod, task_id, mod_cfg):
    warmup_ep = args.kd_we
    print(f"\n   >>> [Phase 1] PURE WARM-UP: Adapting '{mod}' Encoder ({warmup_ep} epochs)...")
    
    for p in model.parameters(): p.requires_grad = False
    for p in model.encoders[mod].parameters(): p.requires_grad = True
    set_active_task_and_freeze(model, task_id) 
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=mod_cfg["lr"], weight_decay=mod_cfg["weight_decay"])
    model.eval() 
    model.set_active_modality(mod)
    
    best_val_f1, best_epoch = 0.0, 0
    best_model_state = copy.deepcopy(model.state_dict()) 
    
    for ep in range(1, warmup_ep + 1): 
        model.encoders[mod].train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and list(m.parameters())[0].requires_grad:
                m.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            feats = model.encoders[mod](x)
            z = model.shared_backbone(feats)
            logits = model.shared_head(z)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                v_logits = model(vx.to(device))
                all_preds.extend(v_logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(vy.numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        if val_f1 >= best_val_f1:
            best_val_f1, best_epoch = val_f1, ep
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"   >>> [Snapshot] Restoring Best Warmup Model from Epoch {best_epoch}...")
    model.load_state_dict(best_model_state)
    return best_val_f1

def train_one_task(args, model, ewc, train_loader, val_loaders_dict, tasks_list, mod, device, task_id, fold_idx=0):
    current_val_loader = val_loaders_dict[mod]
    aligned_teacher = None
    requires_warmup = (getattr(args, 'kd_lambda', 0.0) > 0) or (getattr(args, 'repulsive_alpha', 0.0) > 0)
    mod_cfg = MODALITY_CONFIG.get(mod, {"lr": args.lr, "weight_decay": 1e-3})

    if task_id > 0 and requires_warmup:
        _ = run_warmup_phase(args, model, train_loader, current_val_loader, device, mod, task_id, mod_cfg)
        aligned_teacher = copy.deepcopy(model)
        aligned_teacher.eval()
        for p in aligned_teacher.parameters(): p.requires_grad = False

    print(f"\n   >>> [Phase 2] FINE-TUNE: Curriculum Consolidation...")
    unfreeze_shared_components(model, mod)
    set_active_task_and_freeze(model, task_id)
    print(f"   ⚙️  [Dynamic HP] Retaining optimized momentum. LR: {mod_cfg['lr']}, WD: {mod_cfg['weight_decay']}")

    loss_engine = LossEngine(ewc, aligned_teacher, args, device)
    base_rep_alpha = getattr(args, 'repulsive_alpha', 0.0)
    base_kd_lambda = getattr(args, 'kd_lambda', 0.0)
    min_kd_lambda = getattr(args, 'min_kd_lambda', 0.1)
    
    if len(train_loader.dataset.labels) > 0:
        counts = [train_loader.dataset.labels.count(i) for i in range(args.num_classes)]
        ewc.criterion = nn.CrossEntropyLoss(weight=U.class_weight_tensor(counts, device), label_smoothing=0.05)
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ewc.optimizer, mode='max', factor=0.5, patience=50)
    early_stopper = U.EarlyStopping(patience=args.patience, mode='max')

    best_eval = 0
    for ep in range(1, args.epochs+1):
        t = ep / args.epochs 
        
        # 🚨 Fix 3: unified curriculum schedule
        if task_id > 0:
            if base_rep_alpha > 0.0 and not getattr(args, 'disable_curriculum', False):
                loss_engine.repulsive_alpha = base_rep_alpha * (t ** args.p_degree)
                loss_engine.kd_lambda = min_kd_lambda + (base_kd_lambda - min_kd_lambda) * (1.0 - (t ** args.p_degree))
            elif base_rep_alpha > 0.0:
                loss_engine.repulsive_alpha, loss_engine.kd_lambda = base_rep_alpha, base_kd_lambda
            else:
                loss_engine.repulsive_alpha, loss_engine.kd_lambda = 0.0, base_kd_lambda

        model.train()
        model.set_active_modality(mod)
        set_active_task_and_freeze(model, task_id)
        accum = {"loss": 0, "raw_ce": 0, "w_ewc": 0, "w_kd": 0, "w_repul": 0, "correct": 0, "total": 0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            ewc.optimizer.zero_grad()
            features = model.encoders[mod](x) 
            z = model.shared_backbone(features)
            logits = model.shared_head(z)
            
            loss, metrics = loss_engine.compute(logits, z, y, x, mod)
            loss.backward()
            ewc.optimizer.step()
            
            for k in metrics:
                if k in accum: accum[k] += metrics[k]

            accum["correct"] += (logits.argmax(dim=1) == y).sum().item()
            accum["total"] += y.size(0)
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in current_val_loader:
                v_logits = model(vx.to(device))
                all_preds.extend(v_logits.argmax(1).cpu().numpy())
                all_targets.extend(vy.numpy())
        
        current_val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        best_eval = max(current_val_f1, best_eval)
        
        if task_id > 0: 
            avg_metrics = {k: v / len(train_loader) for k, v in accum.items() if k not in ["correct", "total"]}
            U.log_training_curves_to_csv(getattr(args, 'csv_log', ""), fold_idx, mod, ep, avg_metrics, loss_engine.repulsive_alpha, loss_engine.kd_lambda, current_val_f1)

        if ep % 10 == 0: 
            tr_acc = (accum['correct'] / accum['total']) * 100.0
            avg_tot = accum.get('loss', 0) / len(train_loader)
            avg_ce = accum.get('raw_ce', 0) / len(train_loader)
            avg_ewc = accum.get('w_ewc', 0) / len(train_loader)
            avg_kd = accum.get('w_kd', 0) / len(train_loader)
            avg_rep = accum.get('w_repul', 0) / len(train_loader)
            
            print(f"[{mod}] Ep {ep:02d} | TrAcc:{tr_acc:.2f}% | ValF1:{current_val_f1:.2f} (Best:{best_eval:.2f}) | "
                  f"Tot:{avg_tot:.4f} [CE:{avg_ce:.4f} | L_ewc:{avg_ewc:.4f} | "
                  f"L_kd:{avg_kd:.4f} (λ={loss_engine.kd_lambda:.2f}) | "
                  f"L_rep:{avg_rep:.4f} (α={loss_engine.repulsive_alpha:.2f})]")

        if ep > args.lr_we: scheduler.step(current_val_f1)
            
        stop_signal = early_stopper(current_val_f1, model)
        
        # 🚨 Fix 4: early stopping with 50% protection
        lockout_horizon = int(0.5 * args.epochs) if (getattr(args, 'disable_curriculum', False) or base_rep_alpha == 0.0) else int((0.5 ** (1.0 / args.p_degree)) * args.epochs)
        curriculum_active = (task_id > 0) and (ep <= lockout_horizon) and (getattr(args, 'kd_lambda', 0.0) > 0) and (base_rep_alpha > 0.0)
        
        if stop_signal:
            if curriculum_active:
                early_stopper.counter = 0 
                early_stopper.early_stop = False
            else:
                print(f"   🛑 Task Convergence Reached! Early Stopping at Ep {ep}")
                break
            
    if early_stopper.best_model_state: model.load_state_dict(early_stopper.best_model_state)

# ==========================================
# 4. Main CV pipeline
# ==========================================
def run_cv_with_cache(args, data_cache):
    json_path = CACHE_DIR / "subj2label.json"
    subj2label = build_subj2label_fog(str(json_path))
    folds = make_stratified_folds(subj2label, n_folds=args.n_folds, seed=args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.order.split(",") if t.strip()]
    eval_loader_cache = {fi: {} for fi in range(len(folds))}
    step_history, fold_scores = {}, []
    
    for fi in range(len(folds)):
        print(f"\n========== Fold {fi+1}/{len(folds)} ==========")
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=args.disable_dbn).to(device)
        ewc = ElasticWeightConsolidation(model, nn.CrossEntropyLoss(), lr=args.lr, weight=args.ewc_lambda)
        seen = []
        frozen_task1_model = None

        for ti, mod in enumerate(tasks, 1):
            print(f"\n=== Task {ti}/{len(tasks)} : {mod} ===")
            current_task_idx = ti - 1
            mod_cfg = MODALITY_CONFIG.get(mod, {"lr": args.lr, "weight_decay": 1e-2})

            if ti == 2 and getattr(args, 'analyze_bn_shift', False):
                frozen_task1_model = copy.deepcopy(model).cpu()
                frozen_task1_model.eval()

            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            
            tr_loader = DataLoader(SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            te_loader = DataLoader(SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            eval_loader_cache[fi][mod] = te_loader 

            seen_val_loaders = {m: eval_loader_cache[fi][m] for m in seen + [mod]}
            
            # 🚨 Fix 2: optimizer momentum injection
            # 🚨 Dynamic weight-decay guard (protect EWC)
            if args.mode == 'cl':
                # Read base config
                current_wd = mod_cfg["weight_decay"]
                
                # Zero/minimal WD on later tasks to avoid corrupting EWC-protected weights
                if current_task_idx > 0:
                    current_wd = 0.0   # Disable L2 regularization
                    print(f"   🛡️ [CL Defense] Task {current_task_idx} 触发! 强制关闭 Weight Decay (WD={current_wd}) 以保护历史旧知识。")

                if ti == 1:
                    active_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                    ewc.optimizer = torch.optim.Adam(active_params, lr=mod_cfg["lr"], weight_decay=current_wd)
                else:
                    current_active_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                    old_group = ewc.optimizer.param_groups[0]
                    
                    retained_params = [p for p in old_group['params'] if p.requires_grad]
                    old_group['params'] = retained_params
                    for p in list(ewc.optimizer.state.keys()):
                        if not p.requires_grad: del ewc.optimizer.state[p]
                        
                    existing_params = set(retained_params)
                    new_params = [p for p in current_active_params if p not in existing_params]
                    if new_params: 
                        ewc.optimizer.add_param_group({'params': new_params, 'lr': mod_cfg["lr"], 'weight_decay': current_wd})
                        
            train_one_task(args, model, ewc, tr_loader, seen_val_loaders, tasks, mod, device, current_task_idx, fi)

            is_last_task = (ti == len(tasks))
            should_register_fisher = (args.ewc_lambda > 0 and not is_last_task) or getattr(args, 'analyze_overlap', False)
            if args.mode == 'cl' and should_register_fisher:
                print(f">> [CL] Registering Fisher for {mod} (Required for Overlap Analysis)...")
                register_shared_ewc(model, ewc, tr_loader, args.fisher_batches, task_id=current_task_idx)
            
            # 🚨 Fix 1: dynamic matrix overlap
            if getattr(args, 'analyze_overlap', False) and ti >= 2:
                for past_task in range(current_task_idx):
                    U.analyze_fisher_cosine_similarity(ewc, task_A=past_task, task_B=current_task_idx)
                
            if ti == 2 and getattr(args, 'analyze_bn_shift', False) and frozen_task1_model:
                U.compute_bn_statistics_shift(frozen_task1_model, model)
                
            seen.append(mod)
            print(f"\n--- Evaluation ---")
            scores = []
            for m in seen:
                model.set_active_modality(m) 
                model.set_active_task(tasks.index(m))
                s = U.evaluate_classification(model, eval_loader_cache[fi][m], device, metric='f1_macro')
                scores.append(s)

            if current_task_idx not in step_history: step_history[current_task_idx] = {}
            for m, score in zip(seen, scores):
                real_task_idx = tasks.index(m)
                if real_task_idx not in step_history[current_task_idx]:
                    step_history[current_task_idx][real_task_idx] = []
                step_history[current_task_idx][real_task_idx].append(score)
                print(f"  {m}: {score:.2f}")

            final_score = sum(scores)/len(scores)
            print(f"  Avg Seen: {final_score:.2f}")

        fold_scores.append(final_score)

    avg_f1 = sum(fold_scores) / len(fold_scores)
    return avg_f1, step_history

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", type=str, default="cl", choices=["cl", "specialist"])
    ap.add_argument("--order", type=str, default="acc,gyr,skeleton")
    
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4) 
    ap.add_argument("--lr_we", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    
    ap.add_argument("--win_len", type=int, default=120) 
    ap.add_argument("--hop_len", type=int, default=15)
    
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--disable_dbn", action='store_true')
    
    ap.add_argument("--ewc_lambda", type=float, default=5000.0)
    ap.add_argument("--fisher_batches", type=int, default=64) 
    ap.add_argument("--kd_lambda", type=float, default=1.0)
    ap.add_argument("--kd_we", type=int, default=10)
    
    ap.add_argument("--repulsive_alpha", type=float, default=1.0)
    ap.add_argument("--repulsive_margin", type=float, default=0.1)

    ap.add_argument("--analyze_overlap", default=False, action='store_true')
    ap.add_argument("--analyze_bn_shift", default=False, action='store_true')
    ap.add_argument("--csv_log", type=str, default="")
    ap.add_argument("--disable_curriculum", action='store_true')
    ap.add_argument("--min_kd_lambda", type=float, default=0.1)
    ap.add_argument("--p_degree", type=float, default=5.0)
    
    args = ap.parse_args()
    
    print(f"Arguments: {vars(args)}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    U.set_seed(args.seed)
    global_cache = preload_all_subjects(CACHE_DIR)
    
    print(f"\n{'='*40}\nSTARTING SINGLE SEED: {args.seed}\n{'='*40}")
    avg_f1, step_history = run_cv_with_cache(args, global_cache)
    print(f"\nFINISHED SEED {args.seed} | Macro F1: {avg_f1:.2f}")
    U.print_experiment_summary(args, step_history)