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
    严格基于经典文献的 Modality Gap & Variance Shift 计算工具
    - Gap 基于: Liang et al., NeurIPS 2022
    - Variance 基于: Li et al., ICLR 2017
    """
    # 确保没有梯度追踪干扰
    features_A = features_A.detach()
    features_B = features_B.detach()
    
    # ==========================================
    # 1. 方差比例计算 (Variance Shift)
    # ==========================================
    # dim=0 计算通道级别在 Batch 上的分布方差，对应 BN 的统计维度
    var_A = features_A.var(dim=0).mean().item()
    var_B = features_B.var(dim=0).mean().item()
    var_ratio = var_B / var_A if var_A != 0 else 0

    # ==========================================
    # 2. 模态鸿沟计算 (Modality Gap)
    # ==========================================
    
    # a. 特征 L2 归一化 (将其投射到半径为 1 的超球面上，消除尺度影响)
    norm_A = F.normalize(features_A, p=2, dim=1)
    norm_B = F.normalize(features_B, p=2, dim=1)
    
    # b. 计算两个模态流形的质心 (Centroid)
    centroid_A = norm_A.mean(dim=0)
    centroid_B = norm_B.mean(dim=0)
    
    # c. 计算质心之间的欧氏距离 (Euclidean Distance)
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
        
        # 🚨 极客级容错写法：优先获取加权值 (w_)，没有则回退原始值 (raw_)，再没有则为 0.0
        loss_val = avg_metrics.get('loss', 0.0)
        ce_val = avg_metrics.get('raw_ce', 0.0)
        ewc_val = avg_metrics.get('w_ewc', avg_metrics.get('raw_ewc', 0.0))
        kd_val = avg_metrics.get('w_kd', avg_metrics.get('raw_kd', 0.0))
        repul_val = avg_metrics.get('w_repul', avg_metrics.get('raw_repul', 0.0))

        writer.writerow([
            fold_idx, mod, ep, 
            f"{loss_val:.4f}", f"{ce_val:.4f}", 
            f"{ewc_val:.4f}", f"{kd_val:.4f}", 
            f"{repul_val:.4f}", f"{alpha:.4f}", f"{kd_lambda:.4f}", 
            f"{val_f1:.4f}"
        ])

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
                imp_A, imp_B = fisher_A.sum(dim=(1, 2)), fisher_B.sum(dim=(1, 2))
                sim = F.cosine_similarity(imp_A.unsqueeze(0), imp_B.unsqueeze(0), eps=1e-8).item()
                total_sim += sim
                count += 1

    if count == 0: return 0.0
    avg_sim = total_sim / count
    # 🚨 核心修复：修改打印格式以匹配 extraction.py 的正则捕获组
    print(f"   📊 Latent Crowding {task_A} vs {task_B}: {avg_sim:.4f}")
    return avg_sim

def compute_bn_statistics_shift(model_task1, model_task2):
    shift_metrics = {}
    total_mu_shift, total_var_shift, count = 0.0, 0.0, 0
    print("\n   [Layer-by-Layer BN Shift]")
    for (name1, module1), (name2, module2) in zip(model_task1.named_modules(), model_task2.named_modules()):
        if isinstance(module1, (nn.BatchNorm1d, nn.BatchNorm2d)) and 'shared' in name1:
            mu_1, var_1 = module1.running_mean.detach().cpu(), module1.running_var.detach().cpu()
            mu_2, var_2 = module2.running_mean.detach().cpu(), module2.running_var.detach().cpu()
            delta_mu = torch.norm(mu_1 - mu_2, p=2).item()
            delta_var = torch.norm(var_1 - var_2, p=2).item()
            print(f"      {name1} | Δμ: {delta_mu:.4f} | Δσ²: {delta_var:.4f}")
            shift_metrics[name1] = {'delta_mu': delta_mu, 'delta_var': delta_var}
            total_mu_shift += delta_mu; total_var_shift += delta_var; count += 1
            
    avg_mu = total_mu_shift / count if count > 0 else 0.0
    avg_var = total_var_shift / count if count > 0 else 0.0
    if count > 0:
        print(f"   📊 [BN Covariate Shift] Fold Averages: Δμ={avg_mu:.4f} | Δσ²={avg_var:.4f}\n")
    return shift_metrics, avg_mu, avg_var