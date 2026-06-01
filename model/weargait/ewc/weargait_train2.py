import os, argparse, sys
from pathlib import Path
import copy
import csv

# =====================================================================
# 1. ENVIRONMENT & IMPORTS
# =====================================================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use('Agg') # Crucial for headless server

# Local Project Imports
from model.weargait.ewc.config import Config
import model.weargait.ewc.utility as U
import model.weargait.ewc.joint_train as joint_train
from model.weargait.ewc.data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    make_fixed_balanced_folds_no_overlap, build_subj2label
)
from model.weargait.ewc.EWC import ElasticWeightConsolidation
from model.weargait.ewc.encoder import WearGaitUniversal


# =====================================================================
# 2. DATA INITIALIZATION HELPERS
# =====================================================================
def _scan_subjects(dir_path: Path):
    return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})

def init_subjects_and_folds(args):
    pd_ids, hc_ids = _scan_subjects(Config.PD_PATH), _scan_subjects(Config.HC_PATH)
    if not pd_ids or not hc_ids: raise ValueError("No subjects found.")
    
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    return subj2label, folds


# =====================================================================
# 3. PHASE 1: WARMUP (SPATIAL ALIGNMENT)
# =====================================================================
def run_warmup_phase(args, model, train_loader, val_loader, device, mod, task_id):
    warmup_ep = args.kd_we
    print(f"\n   >>> [Phase 1] PURE WARM-UP: Adapting '{mod}' Encoder via CE ({warmup_ep} epochs)...")
    
    # Freeze backbone, only train current encoder
    for p in model.parameters(): p.requires_grad = False
    for p in model.encoders[mod].parameters(): p.requires_grad = True
    U.set_active_task_and_freeze(model, task_id) 
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-3)
    model.eval() 
    model.set_active_modality(mod)
    
    best_val_f1, best_epoch = 0.0, 0
    best_model_state = copy.deepcopy(model.state_dict()) 
    
    for ep in range(1, warmup_ep + 1): 
        model.encoders[mod].train()
        for m in model.modules(): # Ensure BN is active for requires_grad params
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and list(m.parameters())[0].requires_grad:
                m.train()

        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            
            logits = model.shared_head(model.shared_backbone(model.encoders[mod](x)))
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            
        # Evaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                all_preds.extend(model(vx).argmax(dim=1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        if val_f1 >= best_val_f1:
            best_val_f1, best_epoch = val_f1, ep
            best_model_state = copy.deepcopy(model.state_dict())

        if ep % 5 == 0 or ep == warmup_ep:
            print(f"       Warmup Ep {ep:02d}/{warmup_ep} | Loss:{total_loss/len(train_loader):.4f} | "
                  f"Tr:{(correct/total)*100:.2f}% | ValF1:{val_f1:.2f}% (Best:{best_val_f1:.2f}%)")

    print(f"   >>> [Snapshot] Restoring Best Warmup Model from Epoch {best_epoch}...")
    model.load_state_dict(best_model_state)
    return best_val_f1


# =====================================================================
# 4. PHASE 2: CORE TRAINING LOOP
# =====================================================================
def train_one_task(args, model, ewc, train_loader, val_loaders_dict, tasks_list, mod, device, epochs, num_classes, patience, task_id, fold_idx=0):
    current_val_loader = val_loaders_dict[mod]
    aligned_teacher = None
    requires_warmup = (getattr(args, 'kd_lambda', 0.0) > 0) or (getattr(args, 'repulsive_alpha', 0.0) > 0)
    
    # --- [4.1] Semantic Anchor Generation ---
    if task_id > 0 and requires_warmup:
        _ = run_warmup_phase(args, model, train_loader, current_val_loader, device, mod, task_id)
        aligned_teacher = copy.deepcopy(model)
        aligned_teacher.eval()
        for p in aligned_teacher.parameters(): p.requires_grad = False

    # --- [4.2] Network Unfreezing & Config ---
    print(f"\n   >>> [Phase 2] FINE-TUNE: Curriculum Consolidation...")
    U.unfreeze_shared_components(model, mod)
    U.set_active_task_and_freeze(model, task_id)

    # -----------------------------------------------------------------
    # ⮑ 双重稳定锁 (Stabilizer Locks): 阻断向零点侵蚀与决策边界梯度噪声
    # -----------------------------------------------------------------
    if task_id > 0:
        for param_group in ewc.optimizer.param_groups: 
            param_group['weight_decay'] = 0.0
        for m in model.shared_backbone.modules():
            if isinstance(m, (nn.Dropout1d, nn.Dropout)): 
                m.p = 0.0  

    # 🚨 核心修正 1：LossEngine 内部严格接收 z 并在深层隐空间进行余弦正交隔离
    loss_engine = U.LossEngine(ewc, aligned_teacher, args, device)
    base_rep_alpha, base_kd_lambda, min_kd_lambda = getattr(args, 'repulsive_alpha', 0.0), getattr(args, 'kd_lambda', 0.0), getattr(args, 'min_kd_lambda', 0.1)
    
    if len(train_loader.dataset.labels) > 0:
        counts = [train_loader.dataset.labels.count(i) for i in range(num_classes)]
        ewc.criterion = nn.CrossEntropyLoss(weight=U.class_weight_tensor(counts, device))
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ewc.optimizer, mode='max', factor=0.5, patience=50)
    early_stopper = U.EarlyStopping(patience=patience, mode='max')
    best_eval = 0

    # --- [4.3] Epoch Loop ---
    for ep in range(1, epochs+1):
        t = ep / epochs 
        
        # -------------------------------------------------------------
        # ⮑ 课程调度器 (Curriculum Scheduler)
        # -------------------------------------------------------------
        if task_id > 0:
            if base_rep_alpha > 0.0 and not getattr(args, 'disable_curriculum', False):
                loss_engine.repulsive_alpha = base_rep_alpha * (t ** args.p_degree)
                loss_engine.kd_lambda = min_kd_lambda + (base_kd_lambda - min_kd_lambda) * (1.0 - (t ** args.p_degree))
            elif base_rep_alpha > 0.0:
                loss_engine.repulsive_alpha, loss_engine.kd_lambda = base_rep_alpha, base_kd_lambda
            else:
                loss_engine.repulsive_alpha, loss_engine.kd_lambda = 0.0, base_kd_lambda

        # -------------------------------------------------------------
        # ⮑ Forward & Backward
        # -------------------------------------------------------------
        model.train()
        model.set_active_modality(mod)
        U.set_active_task_and_freeze(model, task_id)
        accum = {k: 0 for k in ["loss", "raw_ce", "raw_ewc", "raw_kd", "raw_repul", "w_ewc", "w_kd", "w_repul", "correct", "total"]}

        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            ewc.optimizer.zero_grad()
            
            features = model.encoders[mod](x)
            z = model.shared_backbone(features)
            logits = model.shared_head(z)
            
            # 此处严格传入深层潜在表征 z 进行计算
            loss, metrics = loss_engine.compute(logits, z, y, x, mod)
            
            loss.backward()
            ewc.optimizer.step()
            
            for k in metrics: accum[k] += metrics[k]
            accum["correct"] += (logits.argmax(dim=1) == y).sum().item()
            accum["total"] += y.size(0)
        
        # -------------------------------------------------------------
        # ⮑ Strict Task-Specific Evaluation
        # -------------------------------------------------------------
        model.eval()
        model.set_active_modality(mod)
        if hasattr(model, 'set_active_task'): model.set_active_task(task_id)
            
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in current_val_loader:
                vx, vy = vx.to(device), vy.to(device)
                all_preds.extend(model(vx).argmax(1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        current_val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        best_eval = max(current_val_f1, best_eval)
        
        # -------------------------------------------------------------
        # ⮑ Logging & Early Stopping Lockout
        # -------------------------------------------------------------
        avg = {k: v / len(train_loader) for k, v in accum.items() if k not in ["correct", "total"]}
        if task_id > 0: 
            U.log_training_curves_to_csv(getattr(args, 'csv_log', ""), fold_idx, mod, ep, avg, loss_engine.repulsive_alpha, loss_engine.kd_lambda, current_val_f1)

        if ep % 5 == 0:
            print(f"[{mod}] Ep {ep:02d} | TrAcc:{(accum['correct']/accum['total'])*100:.2f} | ValF1:{current_val_f1:.2f} (Best:{best_eval:.2f}) | "
                  f"Tot:{avg['loss']:.4f} [CE:{avg['raw_ce']:.4f} | wEWC:{avg['w_ewc']:.4f} | "
                  f"wKD:{avg['w_kd']:.4f} (λ={loss_engine.kd_lambda:.1f}) | wRepul:{avg['w_repul']:.4f} (α={loss_engine.repulsive_alpha:.2f})]")

        if ep > args.lr_we: scheduler.step(current_val_f1)
            
        stop_signal = early_stopper(current_val_f1, model)
        
        # 🚨 核心修正 2：自适应的 50% 能量期曲线动态锁定
        lockout_horizon = int(0.5 * epochs) if (getattr(args, 'disable_curriculum', False) or base_rep_alpha == 0.0) else int((0.5 ** (1.0 / args.p_degree)) * epochs)
        curriculum_active = (task_id > 0) and (ep <= lockout_horizon) and (getattr(args, 'kd_lambda', 0.0) > 0) and (base_rep_alpha > 0.0)
        
        if stop_signal:
            if curriculum_active:
                early_stopper.counter, early_stopper.early_stop = 0, False
                print(f"   [Curriculum Lockout] Suppressing Early Stop at Ep {ep} (Curriculum Active)")
            else:
                print(f"   🛑 Task Convergence Reached! Early Stopping at Ep {ep}"); break
            
    if early_stopper.best_model_state: model.load_state_dict(early_stopper.best_model_state)


# =====================================================================
# 5. CROSS VALIDATION ENGINE
# =====================================================================
def run_cv_with_cache(args, data_cache):
    subj2label, folds = init_subjects_and_folds(args)
    if args.mode == 'joint': return joint_train.run_joint_experiment(args, data_cache, subj2label, folds)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.order.split(",") if t.strip()]
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    eval_loader_cache, step_history, fold_scores = {fi: {} for fi in range(len(folds))}, {}, []
    
    for fi in range(len(folds)):
        print(f"\n========== Fold {fi+1}/{len(folds)} ==========")
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=args.disable_dbn).to(device)
        ewc   = ElasticWeightConsolidation(model, nn.CrossEntropyLoss(), lr=args.lr, weight=args.ewc_lambda)
        seen  = []

        for ti, mod in enumerate(tasks, 1):
            print(f"\n=== Task {ti}/{len(tasks)} : {mod} ===")
            current_task_idx = ti - 1

            # --- Data Loading ---
            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            tr_loader = DataLoader(U.SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(U.SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[fi][mod] = te_loader 

            # --- Optimizer Setup ---
            seen_val_loaders = {m: eval_loader_cache[fi][m] for m in seen + [mod]}
            current_decay = 1e-3
            
            if args.mode == 'specialist':
                model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=args.disable_dbn).to(device)
                ewc = ElasticWeightConsolidation(model, nn.CrossEntropyLoss(), lr=args.lr, weight=0.0, weight_decay=current_decay)
            elif args.mode == 'cl':
                # -----------------------------------------------------------------
                # ⮑ 强制清空优化器动量 (Strict Optimizer Reset)
                # 彻底斩断跨任务二阶矩爆炸导致的步长灾难，强迫启动 Adam 第一步偏差校正
                # -----------------------------------------------------------------
                active_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                ewc.optimizer = torch.optim.Adam(active_params, lr=args.lr, weight_decay=current_decay)

            # --- Execution ---
            train_one_task(args, model, ewc, tr_loader, seen_val_loaders, tasks, mod, device, args.epochs, args.num_classes, args.patience, current_task_idx, fi)
            
            # EWC --- Fisher Registration ---
            is_last_task = (ti == len(tasks))
            should_register_fisher = (args.ewc_lambda > 0 and not is_last_task) or getattr(args, 'analyze_overlap', False)
            if args.mode == 'cl' and should_register_fisher:
                print(f">> [CL] Registering Fisher for {mod} (Required for Overlap Analysis)...")
                U.register_shared_ewc(model, ewc, tr_loader, args.fisher_batches, task_id=current_task_idx)

            # 🚨 核心修正 3：动态嵌套循环，拉满抽取所有历史任务与当前任务的 Fisher 相似度
            if getattr(args, 'analyze_overlap', False) and ti >= 2:
                for past_task in range(current_task_idx):
                    U.analyze_fisher_cosine_similarity(ewc, task_A=past_task, task_B=current_task_idx)
                
            # --- Evaluation Matrix ---
            seen.append(mod)
            print(f"\n--- Evaluation ---")
            eval_targets = [mod] if args.mode == 'specialist' else seen
            scores = []
            
            for m in eval_targets:
                model.set_active_modality(m) 
                if hasattr(model, 'set_active_task'): model.set_active_task(tasks.index(m))
                scores.append(U.evaluate_classification(model, eval_loader_cache[fi][m], device))

            if current_task_idx not in step_history: step_history[current_task_idx] = {}
            for m, score in zip(eval_targets, scores):
                real_task_idx = tasks.index(m)
                if real_task_idx not in step_history[current_task_idx]: step_history[current_task_idx][real_task_idx] = []
                step_history[current_task_idx][real_task_idx].append(score)
                print(f"  {m}: {score:.2f}")

            final_score = scores[-1] if args.mode == 'specialist' else sum(scores)/len(scores)
            if args.mode != 'specialist': print(f"  Avg Seen: {final_score:.2f}")

        fold_scores.append(final_score)

    return sum(fold_scores) / len(fold_scores), step_history


# =====================================================================
# 6. MAIN ENTRY POINT
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", type=str, default="specialist", choices=["cl", "specialist", "joint"])
    ap.add_argument("--order", type=str, default="walkway,insole,imu")
    
    ap.add_argument("--seed", type=int, default=Config.SEED)
    ap.add_argument("--n_folds", type=int, default=Config.N_FOLDS)
    ap.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=1e-3) 
    ap.add_argument("--lr_we", type=int, default=10)
    
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--win_len", type=int, default=Config.WINDOW_SIZE)
    ap.add_argument("--hop_len", type=int, default=int(Config.WINDOW_SIZE * Config.STRIDE))
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=2)
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
    ap.add_argument("--analyze_intrinsic_gap", action="store_true")
    
    args = ap.parse_args()
    print(f"Arguments: {', '.join(f'{k}={v}' for k, v in vars(args).items())}")
    
    global_cache = preload_all_subjects(Config.OUTPUT_DIR)
    U.set_seed(args.seed)

    print(f"\n{'='*40}\nSTARTING SINGLE SEED: {args.seed}\n{'='*40}")
    avg_f1, step_history = run_cv_with_cache(args, global_cache)
    print(f"\nFINISHED SEED {args.seed} | Macro F1: {avg_f1:.2f}")
    U.print_experiment_summary(args, step_history)

if __name__ == "__main__":
    main()