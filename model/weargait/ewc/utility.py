#from __future__ import annotations
import os
import random
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set RNG seeds across libs. Optionally force deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Uncomment if you need hard determinism on some CUDA ops:
        # torch.use_deterministic_algorithms(True)
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def class_weight_tensor(counts: List[int], device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency weights normalized to sum to C. Safe for zeros via epsilon.
    """
    w = 1.0 / (torch.tensor(counts, dtype=torch.float32, device=device) + 1e-8)
    w = w / w.sum() * len(counts)
    return w

def count_params(m, trainable_only=True):
    ps = [p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only)]
    return sum(ps)

def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# ----------------------- Training Helpers -----------------------
class EarlyStopping:
    """Stops training if metric doesn't improve after a given patience."""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        else:
            if self.mode == 'max': 
                improvement = current_score - self.best_score
            else: 
                improvement = self.best_score - current_score

            if improvement > self.min_delta:
                self.best_score = current_score
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        self.best_model_state = copy.deepcopy(model.state_dict())

class SingleModalityDataset(torch.utils.data.Dataset):
    """
    Generic Wrapper: extracting a single tensor from a Dict-based dataset.
    Assumes base dataset returns {'xs': [t1, t2...], 'y': label}.
    """
    def __init__(self, base_ds, mod_index=0):
        super().__init__()
        self.base = base_ds
        self.mod_index = mod_index
        # Try to expose labels if available (for class weighting)
        if hasattr(self.base, 'labels'):
            self.labels = self.base.labels
        elif hasattr(self.base, 'y'):
             self.labels = self.base.y

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        b = self.base[i]
        # Extract the specific modality tensor and label
        x = b["xs"][self.mod_index]         
        y = b["y"]
        return x, y

@torch.no_grad()
def evaluate_classification(model, loader, device, metric='f1_macro'):
    """
    Standard eval loop for classification. 
    Returns the score (e.g. Macro F1 * 100).
    """
    from sklearn.metrics import f1_score, accuracy_score
    
    model.eval()
    all_preds, all_targets = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
    
    if metric == 'f1_macro':
        return f1_score(all_targets, all_preds, average='macro') * 100.0
    elif metric == 'accuracy':
        return accuracy_score(all_targets, all_preds) * 100.0
    else:
        return 0.0

def compute_modality_analysis(features_A, features_B, target_dim=None):
    """
    Modality gap and variance shift (literature-based)
    - Gap from: Liang et al., NeurIPS 2022
    - Variance from: Li et al., ICLR 2017
    """
    # No grad tracking
    features_A = features_A.detach()
    features_B = features_B.detach()
    
    # ==========================================
    # 1. Variance shift ratio
    # ==========================================
    # Channel variance over batch (BN stats)
    var_A = features_A.var(dim=0).mean().item()
    var_B = features_B.var(dim=0).mean().item()
    var_ratio = var_B / var_A if var_A != 0 else 0

    # ==========================================
    # 2. Modality gap
    # ==========================================
    
    # L2-normalize to unit sphere
    norm_A = F.normalize(features_A, p=2, dim=1)
    norm_B = F.normalize(features_B, p=2, dim=1)
    
    # Per-modality centroids
    centroid_A = norm_A.mean(dim=0)
    centroid_B = norm_B.mean(dim=0)
    
    # Centroid Euclidean distance
    delta_gap = torch.norm(centroid_A - centroid_B, p=2).item()
        
    return delta_gap, var_ratio

def print_experiment_summary(args, step_history):
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    tasks = [t.strip() for t in args.order.split(',')]
    t_order_str = "".join([t[0] for t in tasks]) 
    
    header = f"{'lambda':<8} {'t_order':<8} {'step':<6}"
    for i in range(len(tasks)):
        header += f" {f't{i+1}_mean':<8}"
    header += " avg_acc"
    print(header)
    
    sorted_steps = sorted(step_history.keys())
    for step_idx in sorted_steps:
        row_str = f"{args.ewc_lambda:<8} {t_order_str:<8} {step_idx+1:<6}"
        current_step_means = []
        for task_idx in range(len(tasks)):
            if task_idx in step_history[step_idx]:
                scores = step_history[step_idx][task_idx]
                mean_score = sum(scores) / len(scores)
                row_str += f" {mean_score:<8.2f}"
                current_step_means.append(mean_score)
            else:
                row_str += f" {'NaN':<8}"
        
        if current_step_means:
            avg_acc = sum(current_step_means) / len(current_step_means)
            row_str += f" {avg_acc:.2f}"
        else:
            row_str += " 0.00"
        print(row_str)
    print("="*70 + "\n")

def log_training_curves_to_csv(csv_path, fold_idx, mod, ep, avg_metrics, alpha, kd_lambda, val_f1=0.0):
    if not csv_path: return
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Fold', 'Task', 'Epoch', 'Total_Loss', 'CE_Loss', 'EWC_Loss', 'KD_Loss', 'Repul_Loss', 'Alpha', 'Lambda', 'Val_F1'])
        writer.writerow([
            fold_idx, mod, ep, 
            f"{avg_metrics['loss']:.4f}", 
            f"{avg_metrics['raw_ce']:.4f}",       # CE uses raw (no weighting)
            f"{avg_metrics['w_ewc']:.4f}",        # use w_ewc
            f"{avg_metrics['w_kd']:.4f}",         # use w_kd
            f"{avg_metrics['w_repul']:.4f}",      # use w_repul
            f"{alpha:.4f}", f"{kd_lambda:.4f}",
            f"{val_f1:.4f}"
        ])

# --- 2. Replace analyze_fisher_cosine_similarity (dim fix) ---
def analyze_fisher_cosine_similarity(ewc_instance, task_A=0, task_B=1):
    print(f"\n   🔍 [Analysis] Computing Fisher Cosine Similarity between Task {task_A} and Task {task_B}...")
    model = ewc_instance.model
    total_sim, count = 0.0, 0
    for name, param in model.named_parameters():
        if 'shared_backbone' in name and 'weight' in name and len(param.shape) in [3, 4]:
            buffer_name_A = name.replace('.', '__') + f'_fisher_diag_t{task_A}'
            buffer_name_B = name.replace('.', '__') + f'_fisher_diag_t{task_B}'
            if hasattr(model, buffer_name_A) and hasattr(model, buffer_name_B):
                fisher_A, fisher_B = getattr(model, buffer_name_A), getattr(model, buffer_name_B)
                
                # Flatten spatial/input dims; keep C_out Fisher fingerprint
                imp_A = fisher_A.view(fisher_A.size(0), -1).sum(dim=1)
                imp_B = fisher_B.view(fisher_B.size(0), -1).sum(dim=1)
                
                sim = F.cosine_similarity(imp_A.unsqueeze(0), imp_B.unsqueeze(0), eps=1e-8).item()
                total_sim += sim
                count += 1

    if count == 0: 
        print("   ⚠️ [Analysis] No valid convolutional layers found for Overlap.")
        return 0.0
    
    avg_sim = total_sim / count
    print(f"   📊 Latent Crowding {task_A} vs {task_B}: {avg_sim:.4f}")
    return avg_sim

def analyze_fisher_overlap(model_prev, model_curr, ewc_instance, task_id_to_analyze=0):
    print(f"\n   🔍 [Analysis] Computing Fisher-Update Overlap (vs Task {task_id_to_analyze} Constraints)...")
    all_fisher, all_delta = [], []
    params_prev, params_curr = dict(model_prev.named_parameters()), dict(model_curr.named_parameters())
    
    for name, p_curr in params_curr.items():
        if name in params_prev:
            p_prev = params_prev[name]
            buffer_name = name.replace('.', '__') + f'_fisher_diag_t{task_id_to_analyze}'
            if hasattr(ewc_instance.model, buffer_name):
                fisher_tensor = getattr(ewc_instance.model, buffer_name)
                delta = (p_curr.cpu() - p_prev).abs().detach().view(-1)
                fish = fisher_tensor.detach().cpu().view(-1)
                all_fisher.append(fish)
                all_delta.append(delta)

    if not all_fisher: return 0.0
    F_vec, D_vec = torch.cat(all_fisher).numpy(), torch.cat(all_delta).numpy()
    if np.linalg.norm(D_vec) < 1e-9: return 0.0
        
    D_norm = D_vec / np.linalg.norm(D_vec)
    F_norm = F_vec / np.linalg.norm(F_vec) 
    overlap_score = np.dot(F_norm, D_norm)
    print(f"   📊 Fisher-Update Overlap Score: {overlap_score:.4f}")
    return overlap_score

class LossEngine:
    def __init__(self, ewc, teacher_model, args, device):
        self.ewc = ewc
        self.teacher_model = teacher_model
        self.args = args
        self.device = device
        
        # Hyperparameters
        self.kd_lambda = args.kd_lambda
        self.repulsive_alpha = getattr(args, 'repulsive_alpha', 0.0)
        self.repulsive_margin = getattr(args, 'repulsive_margin', 0.0)

    def compute(self, logits, z, y, x, mod):
        raw_ce = self.ewc.criterion(logits, y)
        
        # --- EWC Magnitude Extraction ---
        # 1. Get the WEIGHTED loss from the EWC class
        weighted_ewc_tensor = self.ewc._ewc_penalty()
        weighted_ewc = weighted_ewc_tensor.item() if torch.is_tensor(weighted_ewc_tensor) else weighted_ewc_tensor
        
        # 2. Reverse engineer the RAW displacement
        ewc_multiplier = 0.5 * self.ewc.weight
        raw_ewc = (weighted_ewc / ewc_multiplier) if ewc_multiplier > 0 else 0.0

        raw_kd = torch.tensor(0.0, device=self.device)
        raw_repulsion = torch.tensor(0.0, device=self.device)
        weighted_kd = 0.0
        weighted_repulsion = 0.0
        
        # Check if we need teacher features
        requires_teacher = (self.teacher_model is not None) and \
                           (self.kd_lambda > 0 or self.repulsive_alpha > 0)
                           
        if requires_teacher:
            with torch.no_grad():
                # t_features and t_logits now exclusively come from the ALIGNED teacher!
                t_features = self.teacher_model.encoders[mod](x)
                t_z = self.teacher_model.shared_backbone(t_features)
                if self.kd_lambda > 0:
                    t_logits = self.teacher_model.shared_head(t_z)

            # --- A. KD ---
            if self.kd_lambda > 0:
                T = 2.0
                p_s = F.log_softmax(logits / T, dim=1)
                p_t = F.softmax(t_logits / T, dim=1)
                raw_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)
                weighted_kd = self.kd_lambda * raw_kd

            # --- B. Physics-Decoupled Distillation (-L_f) ---
            if self.repulsive_alpha > 0:
                # Calculate Cosine Similarity against the Tangled State
                cos_sim = F.cosine_similarity(z, t_z, dim=1)
                
                # Penalize similarity greater than the margin
                raw_repulsion = F.relu(cos_sim - self.repulsive_margin).mean()
                weighted_repulsion = self.repulsive_alpha * raw_repulsion

        # Total Loss Assembly
        total_loss = raw_ce + weighted_ewc_tensor + weighted_kd + weighted_repulsion
        
        # Tracking both RAW and WEIGHTED metrics for magnitude observation
        metrics = {
            "loss": total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            "raw_ce": raw_ce.item() if torch.is_tensor(raw_ce) else raw_ce,
            "raw_ewc": raw_ewc,                 # True microscopic drift
            "raw_kd": raw_kd.item() if torch.is_tensor(raw_kd) else raw_kd,
            "raw_repul": raw_repulsion.item() if torch.is_tensor(raw_repulsion) else raw_repulsion,
            "w_ewc": weighted_ewc,              # Actual penalty applied to graph
            "w_kd": weighted_kd.item() if torch.is_tensor(weighted_kd) else weighted_kd,
            "w_repul": weighted_repulsion.item() if torch.is_tensor(weighted_repulsion) else weighted_repulsion
        }
        return total_loss, metrics

# ----------------- Helpers -----------------
def register_shared_ewc(model, ewc, dataloader, num_batches, task_id):
    set_active_task_and_freeze(model, task_id)
    # Manually toggle grads as U.py doesn't know about 'shared_backbone' specifics
    ewc.register_ewc_params(dataloader, task_id=task_id, num_batches=num_batches)
    # # Restore
    # for m in model.encoders.values(): 
    #     for p in m.parameters(): p.requires_grad = True

def set_active_task_and_freeze(model, task_id):
    """
    1. Sets the active task index for routing.
    2. Unfreezes the Batch Norms for the current task.
    3. FREEZES the Batch Norms for all other tasks.
    """

    # task_id = 0

    # 1. Set Routing
    # model.set_active_task(0)
    if hasattr(model, 'set_active_task'):
        model.set_active_task(task_id)
    
    # 2. Iterate through all modules to find Dual-BN blocks
    # (Checking for 'bn1_list' ensures we target your ResBlock1D)
    for m in model.modules():
        if hasattr(m, 'bn1_list') and hasattr(m, 'bn2_list'):
            # Freeze ALL first
            for bn in m.bn1_list:
                for p in bn.parameters(): p.requires_grad = False
            for bn in m.bn2_list:
                for p in bn.parameters(): p.requires_grad = False
            
            # Unfreeze CURRENT only
            if task_id < len(m.bn1_list):
                for p in m.bn1_list[task_id].parameters(): p.requires_grad = True
                for p in m.bn2_list[task_id].parameters(): p.requires_grad = True

def unfreeze_shared_components(model, mod):
    """
    Restores requires_grad=True for:
    1. The Shared Backbone
    2. The Shared Head
    3. The Active Encoder
    """
    for p in model.shared_backbone.parameters(): p.requires_grad = True
    for p in model.shared_head.parameters():     p.requires_grad = True
    for p in model.encoders[mod].parameters():   p.requires_grad = True

