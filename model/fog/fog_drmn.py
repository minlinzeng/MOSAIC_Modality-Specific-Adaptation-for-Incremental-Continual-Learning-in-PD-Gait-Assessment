import os, argparse, sys
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ==========================================
# 🚨 Local FOG-only imports
# ==========================================
import utility as U
from encoder import WearGaitUniversal
from data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    build_subj2label_fog, make_stratified_folds, SingleModalityDataset
)

from model.paths import FOG_CACHE

CACHE_DIR = FOG_CACHE

def init_subjects_and_folds(args):
    json_path = CACHE_DIR / "subj2label.json"
    subj2label = build_subj2label_fog(str(json_path))
    folds = make_stratified_folds(subj2label, n_folds=args.n_folds, seed=args.seed)
    return subj2label, folds

# ==========================================
# 1. DRMN Manager (The "Freezing" Logic)
# ==========================================
class DRMN_Manager:
    def __init__(self, model, lock_ratio=0.4):
        self.model = model
        self.lock_ratio = lock_ratio 
        
        self.task_masks = {}       
        self.global_free_mask = {} 
        self.master_weights = {}   
        self.active_task_id = None
        
        for name, p in self._get_tracked_params():
            self.global_free_mask[name] = torch.ones_like(p, dtype=torch.bool, requires_grad=False).to(p.device)
            self.master_weights[name] = p.data.clone().detach()

    def _get_tracked_params(self):
        for name, p in list(self.model.shared_backbone.named_parameters()) + list(self.model.shared_head.named_parameters()):
            if "bn" not in name and "downsample.1" not in name:
                yield name, p

    def switch_task(self, task_id):
        if self.active_task_id is not None:
            old_mask = self.task_masks[self.active_task_id]
            for name, p in self._get_tracked_params():
                self.master_weights[name][old_mask[name]] = p.data[old_mask[name]].clone()

        if task_id not in self.task_masks:
            self.task_masks[task_id] = {k: v.clone() for k, v in self.global_free_mask.items()}
            
        self.active_task_id = task_id
        current_mask = self.task_masks[task_id]
        
        for name, p in self._get_tracked_params():
            p.data.copy_(self.master_weights[name])
            p.data[~current_mask[name]] = 0.0 

    def apply_gradient_mask(self):
        current_mask = self.task_masks[self.active_task_id]
        for name, p in self._get_tracked_params():
            if p.grad is not None:
                p.grad[~current_mask[name]] = 0.0

    def update_relevance_map(self):
        print(f"   🔒 [DRMN] Updating Relevance Maps (Locking top {self.lock_ratio*100}% of utilized weights)...")
        current_mask = self.task_masks[self.active_task_id]
        total_params, total_locked = 0, 0
        
        for name, p in self._get_tracked_params():
            active_weights = p.data[current_mask[name]]
            total_params += p.numel()
            
            if active_weights.numel() > 0:
                num_to_lock = int(len(active_weights) * self.lock_ratio)
                if num_to_lock > 0:
                    threshold = torch.kthvalue(torch.abs(active_weights), len(active_weights) - num_to_lock + 1).values
                    new_task_mask = (torch.abs(p.data) >= threshold) & current_mask[name]
                    self.task_masks[self.active_task_id][name] = new_task_mask
                    self.global_free_mask[name] &= ~new_task_mask
                    total_locked += new_task_mask.sum().item()
                    
        print(f"       -> Permanent Backbone Capacity Claimed by Task {self.active_task_id}: {total_locked / total_params * 100:.2f}%")

# ==========================================
# 2. Training Loop
# ==========================================
def train_drmn_task(args, model, drmn_manager, train_loader, val_loader, mod, task_id, device, epochs, patience):
    print(f"\n   >>> [DRMN] Training '{mod}' (Task {task_id}) with Hard Gradient Masking...")
    drmn_manager.switch_task(task_id)
    for k in model.encoders.keys():
        for p in model.encoders[k].parameters():
            p.requires_grad = (k == mod) 
            
    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(active_params, lr=args.lr, weight_decay=1e-2) # FOG weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    early_stopper = U.EarlyStopping(patience=patience, mode='max')
    criterion = nn.CrossEntropyLoss()
    best_eval = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        model.set_active_task(task_id)
        model.set_active_modality(mod)
        accum = {"loss": 0, "correct": 0, "total": 0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            features = model.encoders[mod](x)
            z = model.shared_backbone(features)
            logits = model.shared_head(z)
            
            loss = criterion(logits, y)
            loss.backward()
            drmn_manager.apply_gradient_mask()
            optimizer.step()

            accum["loss"] += loss.item()
            preds = logits.argmax(dim=1)
            accum["correct"] += (preds == y).sum().item()
            accum["total"] += y.size(0)

        model.eval()
        model.set_active_task(task_id)
        model.set_active_modality(mod)
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

        if ep % 10 == 0:
            n = len(train_loader)
            print(f"[{mod}] Ep {ep:02d} | Loss:{accum['loss']/n:.4f} | "
                  f"Acc:{accum['correct']/accum['total']*100:.1f}% ValF1:{val_f1:.2f}%")

        if early_stopper(val_f1, model):
            print(f"   🛑 Early Stopping at Ep {ep}")
            model.load_state_dict(early_stopper.best_model_state)
            break
    if early_stopper.best_model_state: model.load_state_dict(early_stopper.best_model_state)

# ==========================================
# 3. Main Experiment Loop
# ==========================================
def run_cv_drmn(args, data_cache):
    subj2label, folds = init_subjects_and_folds(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.order.split(",") if t.strip()]
    
    step_history = {}
    fold_scores = []

    for fi in range(len(folds)):
        print(f"\n{'='*20} Fold {fi+1}/{len(folds)} {'='*20}")
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=args.disable_dbn).to(device)
        drmn_manager = DRMN_Manager(model, lock_ratio=args.lock_ratio)
        seen_mods, eval_loader_cache = [], {}

        for ti, mod in enumerate(tasks):
            print(f"\n=== DRMN Task {ti+1}/{len(tasks)} : {mod} ===")
            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            tr_loader = DataLoader(SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[mod] = te_loader 

            train_drmn_task(args, model, drmn_manager, tr_loader, te_loader, mod, ti, device, args.epochs, args.patience)

            if ti < len(tasks) - 1:
                drmn_manager.update_relevance_map()

            seen_mods.append(mod)
            print(f"\n--- Evaluation (Step {ti+1}) ---")
            scores = []
            
            for seen_task_idx, m in enumerate(seen_mods):
                model.eval()
                model.set_active_task(seen_task_idx) 
                model.set_active_modality(m)
                drmn_manager.switch_task(seen_task_idx)
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for vx, vy in eval_loader_cache[m]:
                        vx, vy = vx.to(device), vy.to(device)
                        v_logits = model(vx)
                        all_preds.extend(v_logits.argmax(1).cpu().numpy())
                        all_targets.extend(vy.cpu().numpy())
                
                score = f1_score(all_targets, all_preds, average='macro') * 100.0
                scores.append(score)
                print(f"  {m}: {score:.2f}")

            if ti not in step_history: step_history[ti] = {}
            for m_idx, m_score in enumerate(scores):
                step_history[ti][m_idx] = [m_score] 

            avg_seen = sum(scores) / len(scores)
            print(f"  Avg Seen: {avg_seen:.2f}")

        fold_scores.append(avg_seen)
    print(f"\nFinal Avg F1 across folds: {sum(fold_scores)/len(fold_scores):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    # FOG Defaults
    ap.add_argument("--order", type=str, default="acc,gyr,skeleton")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--win_len", type=int, default=120)
    ap.add_argument("--hop_len", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lock_ratio", type=float, default=0.4)
    ap.add_argument("--disable_dbn", action='store_true')

    args = ap.parse_args()
    print(f"FOG DRMN Mode | Arguments: {', '.join(f'{k}={v}' for k, v in vars(args).items())}")
    
    global_cache = preload_all_subjects(CACHE_DIR)
    U.set_seed(args.seed)
    run_cv_drmn(args, global_cache)

if __name__ == "__main__":
    main()