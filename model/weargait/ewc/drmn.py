import os, argparse, sys
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# --- Path Setup ---
current_file = Path(__file__).resolve()
current_dir = current_file.parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from model.weargait.ewc.config import Config
import model.weargait.ewc.utility as U
from model.weargait.ewc.data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    make_fixed_balanced_folds_no_overlap, build_subj2label
)
from model.weargait.ewc.encoder import WearGaitUniversal
# from model.weargait.ewc.encoder_res18 import WearGaitResNet18

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Crucial for headless server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_activation_death(model, walkway_loader, device, method_name, save_path):
    """
    Plots the histogram of the final backbone activations (post-ReLU) for Walkway data
    AFTER the model has trained on all tasks. High spikes at 0.0 indicate Catastrophic Forgetting.
    """
    print(f"\n   📊 [Analysis] Generating Activation Death plot for {method_name}...")
    model.eval()

    # Route model if it has active task/modality setters (DRMN/Ours usually do)
    if hasattr(model, 'set_active_modality'): model.set_active_modality('walkway')
    if hasattr(model, 'set_active_task'): model.set_active_task(0)

    all_activations = []
    
    with torch.no_grad():
        for x, _ in walkway_loader:
            x = x.to(device)
            
            # 1. Base Encoder
            feats = model.encoders['walkway'](x)
            
            # 2. SOTA-Specific Routing (Harmony has ACFM)
            if hasattr(model, 'acfm') and 'walkway' in model.acfm:
                feats = model.acfm['walkway'](feats)
                
            # 3. Shared Backbone
            z = model.shared_backbone(feats)
            
            # Flatten to 1D array
            all_activations.extend(z.cpu().numpy().flatten())
            break # One batch is enough for a statistical distribution
            
    # Plotting
    plt.figure(figsize=(7, 5))
    
    # Use a solid color for SOTAs to contrast with your method later
    plot_color = '#d62728' if "Ours" not in method_name else '#1f77b4'
    sns.histplot(all_activations, bins=50, color=plot_color, kde=False, stat="density")
    
    # Calculate exactly how many neurons are dead (0.0)
    zero_pct = (np.array(all_activations) == 0).sum() / len(all_activations) * 100
    
    plt.title(f"Task 1 Activations Post-Training ({method_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Activation Value ($z$)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    # Highlight the "Death" percentage
    plt.text(0.5, 0.8, f"Dead Activations (0.0): {zero_pct:.1f}%", 
             transform=plt.gca().transAxes, fontsize=12, color='black', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Activation plot saved to {save_path}")

def plot_tsne_latent(model, eval_loaders, device, method_name, save_path):
    """
    Plots a t-SNE of the shared backbone's latent space for Walkway and Insole.
    A confused model (SOTA) will show overlapping classes. A good model (Ours) will separate them.
    """
    print(f"\n   📊 [Analysis] Generating t-SNE plot for {method_name}...")
    model.eval()
    
    all_z = []
    all_labels = []     
    all_modalities = [] 
    
    # Limit to 250 samples per modality so t-SNE runs in seconds, not hours
    MAX_SAMPLES_PER_MOD = 250 
    
    with torch.no_grad():
        for task_idx, mod_name in enumerate(['walkway', 'insole']):
            if mod_name not in eval_loaders: continue
            
            if hasattr(model, 'set_active_modality'): model.set_active_modality(mod_name)
            if hasattr(model, 'set_active_task'): model.set_active_task(task_idx)
                
            samples_collected = 0
            for x, y in eval_loaders[mod_name]:
                x = x.to(device)
                
                # SOTA-agnostic forward pass
                feats = model.encoders[mod_name](x)
                if hasattr(model, 'acfm') and mod_name in model.acfm:
                    feats = model.acfm[mod_name](feats)
                z = model.shared_backbone(feats)
                
                all_z.append(z.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_modalities.extend([mod_name] * len(y))
                
                samples_collected += len(y)
                if samples_collected >= MAX_SAMPLES_PER_MOD:
                    break

    Z_np = np.concatenate(all_z, axis=0)
    labels_np = np.array(all_labels)
    mod_np = np.array(all_modalities)
    
    print("       -> Running PCA initialization & t-SNE reduction...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    Z_2d = tsne.fit_transform(Z_np)
    
    plt.figure(figsize=(8, 6))
    
    # Visual styling: Circles for Walkway, Triangles for Insole
    markers = {'walkway': 'o', 'insole': '^'}
    colors = {0: '#1f77b4', 1: '#d62728'} # 0: Control (Blue), 1: Parkinson's (Red)
    class_names = {0: 'Control', 1: 'PD'}
    
    for mod in np.unique(mod_np):
        for cls in np.unique(labels_np):
            idx = (mod_np == mod) & (labels_np == cls)
            plt.scatter(Z_2d[idx, 0], Z_2d[idx, 1], 
                        c=colors[cls], marker=markers[mod], 
                        label=f"{mod.capitalize()} - {class_names[cls]}", 
                        alpha=0.7, edgecolors='white', s=70)
            
    plt.title(f"Latent Space Anchoring ({method_name})", fontsize=14, fontweight='bold')
    plt.xticks([]) 
    plt.yticks([])
    
    # Put legend outside the plot so it doesn't cover data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ t-SNE plot saved to {save_path}")

# ==========================================
# 1. DRMN Manager (The "Freezing" Logic)
# ==========================================

class DRMN_Manager:
    """
    Simulates the Disjoint Relevance Mapping Network.
    Enforces strict parameter isolation during BOTH Forward and Backward passes.
    """
    def __init__(self, model, lock_ratio=0.4):
        self.model = model
        self.lock_ratio = lock_ratio 
        
        self.task_masks = {}       # {task_id: {layer_name: boolean_tensor}}
        self.global_free_mask = {} # Tracks weights not yet claimed by ANY task
        self.master_weights = {}   # Holds the true weights safely in RAM
        self.active_task_id = None
        
        # 1. Initialize global masks and master weights (excluding BNs)
        for name, p in self._get_tracked_params():
            self.global_free_mask[name] = torch.ones_like(p, dtype=torch.bool, requires_grad=False).to(p.device)
            self.master_weights[name] = p.data.clone().detach()

    def _get_tracked_params(self):
        """Generator to cleanly yield maskable parameters."""
        for name, p in list(self.model.shared_backbone.named_parameters()) + list(self.model.shared_head.named_parameters()):
            if "bn" not in name and "downsample.1" not in name:
                yield name, p

    def switch_task(self, task_id):
        """
        Physically isolates the sub-network for the requested task.
        Must be called before training OR evaluating a specific task.
        """
        # 1. Save the active weights back to the master bank
        if self.active_task_id is not None:
            old_mask = self.task_masks[self.active_task_id]
            for name, p in self._get_tracked_params():
                # Only save the weights this task was mathematically allowed to touch
                self.master_weights[name][old_mask[name]] = p.data[old_mask[name]].clone()

        # 2. If this is a NEW task, allocate all currently free weights to it
        if task_id not in self.task_masks:
            self.task_masks[task_id] = {k: v.clone() for k, v in self.global_free_mask.items()}
            
        self.active_task_id = task_id
        current_mask = self.task_masks[task_id]
        
        # 3. Inject Master Weights and ENFORCE DISJOINT FORWARD PASS
        for name, p in self._get_tracked_params():
            p.data.copy_(self.master_weights[name])
            # The Magic: Zero out any weight not belonging to this task
            p.data[~current_mask[name]] = 0.0 

    def apply_gradient_mask(self):
        """Called during backward pass to freeze unauthorized weights."""
        current_mask = self.task_masks[self.active_task_id]
        for name, p in self._get_tracked_params():
            if p.grad is not None:
                p.grad[~current_mask[name]] = 0.0

    def update_relevance_map(self):
        """Called after a task finishes. Locks the top % of weights and frees the rest."""
        print(f"   🔒 [DRMN] Updating Relevance Maps (Locking top {self.lock_ratio*100}% of utilized weights)...")
        current_mask = self.task_masks[self.active_task_id]
        
        total_params = 0
        total_locked = 0
        
        for name, p in self._get_tracked_params():
            # Get the weights this task was using
            active_weights = p.data[current_mask[name]]
            total_params += p.numel()
            
            if active_weights.numel() > 0:
                num_to_lock = int(len(active_weights) * self.lock_ratio)
                
                if num_to_lock > 0:
                    # Find magnitude threshold
                    threshold = torch.kthvalue(torch.abs(active_weights), len(active_weights) - num_to_lock + 1).values
                    
                    # Create the final, permanent mask for this specific task
                    new_task_mask = (torch.abs(p.data) >= threshold) & current_mask[name]
                    self.task_masks[self.active_task_id][name] = new_task_mask
                    
                    # Remove these permanently locked weights from the global free pool
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
            p.requires_grad = (k == mod) # Only True for current modality
    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(active_params, lr=args.lr, weight_decay=1e-4)
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

            # 1. Forward
            features = model.encoders[mod](x)
            z = model.shared_backbone(features)
            logits = model.shared_head(z)
            
            loss = criterion(logits, y)
            
            # 2. Backward
            loss.backward()
            
            # 3. DRMN MAGIC: Zero out gradients of locked weights
            drmn_manager.apply_gradient_mask()
            
            # 4. Step (Only free weights will update!)
            optimizer.step()

            # Metrics
            accum["loss"] += loss.item()
            preds = logits.argmax(dim=1)
            accum["correct"] += (preds == y).sum().item()
            accum["total"] += y.size(0)

        # Validation
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

    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)

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
        
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=True).to(device)
        # model = WearGaitResNet18(num_classes=args.num_classes).to(device)
        # Initialize DRMN Manager with 40% lock ratio
        drmn_manager = DRMN_Manager(model, lock_ratio=args.lock_ratio)
        
        seen_mods = []
        eval_loader_cache = {}

        for ti, mod in enumerate(tasks):
            print(f"\n=== DRMN Task {ti+1}/{len(tasks)} : {mod} ===")
            
            # Prepare Data
            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            tr_loader = DataLoader(U.SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(U.SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[mod] = te_loader 

            # Train Task
            train_drmn_task(args, model, drmn_manager, tr_loader, te_loader, mod, ti, device, args.epochs, args.patience)

            # Update Relevance Map (Lock the weights for this task)
            if ti < len(tasks) - 1: # No need to lock after the final task
                drmn_manager.update_relevance_map()

            # Evaluation
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

        # if fi == 0: 
        #     # 1. Activation Death Plot
        #     act_path = os.path.join(Config.OUTPUT_DIR, "activation_death_drmn_dualbn.png")
        #     plot_activation_death(model, eval_loader_cache['walkway'], device, 
        #                           method_name="DRMN", save_path=act_path)
            
        #     # 2. t-SNE Plot
        #     tsne_path = os.path.join(Config.OUTPUT_DIR, "tsne_drmn_dualbn.png")
        #     plot_tsne_latent(model, eval_loader_cache, device, 
        #                      method_name="DRMN", save_path=tsne_path)

    print(f"\nFinal Avg F1 across folds: {sum(fold_scores)/len(fold_scores):.2f}")


def init_subjects_and_folds(args):
    def _scan_subjects(dir_path: Path):
        return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})
    pd_ids = _scan_subjects(Config.PD_PATH)
    hc_ids = _scan_subjects(Config.HC_PATH)
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    return subj2label, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--order", type=str, default="imu,walkway,insole")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n_folds", type=int, default=Config.N_FOLDS)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--win_len", type=int, default=120)
    ap.add_argument("--hop_len", type=int, default=60)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=2)
    
    # DRMN Specific Argument
    ap.add_argument("--lock_ratio", type=float, default=0.4, help="Percentage of free weights to lock per task")

    args = ap.parse_args()
    print(f"DRMN Mode | Arguments: {', '.join(f'{k}={v}' for k, v in vars(args).items())}")
    
    global_cache = preload_all_subjects(Config.OUTPUT_DIR)
    U.set_seed(args.seed)
    run_cv_drmn(args, global_cache)

if __name__ == "__main__":
    main()