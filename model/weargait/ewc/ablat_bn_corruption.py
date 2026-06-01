import os, argparse, sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
project_root = current_dir.parent.parent.parent

sys.path.append(str(project_root))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import model.weargait.ewc.joint_train as joint_train
from model.weargait.ewc.config import Config
import model.weargait.ewc.utility as U
import copy

from model.weargait.ewc.data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    make_fixed_balanced_folds_no_overlap, build_subj2label
)
from model.weargait.ewc.EWC import ElasticWeightConsolidation
from model.weargait.ewc.encoder import WearGaitUniversal

import matplotlib
matplotlib.use('Agg') # Crucial for headless server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_activation_death(model, walkway_loader, device, method_name, save_path):
    print(f"\n   📊 [Analysis] Generating Activation Death plot for {method_name}...")
    model.eval()

    if hasattr(model, 'set_active_modality'): model.set_active_modality('walkway')
    if hasattr(model, 'set_active_task'): model.set_active_task(0)

    all_activations = []
    
    with torch.no_grad():
        for x, _ in walkway_loader:
            x = x.to(device)
            feats = model.encoders['walkway'](x)
            
            # (Will be skipped for your method since it has no ACFM)
            if hasattr(model, 'acfm') and 'walkway' in model.acfm:
                feats = model.acfm['walkway'](feats)
                
            z = model.shared_backbone(feats)
            all_activations.extend(z.cpu().numpy().flatten())
            break 
            
    plt.figure(figsize=(7, 5))
    
    # Use Blue for your method
    plot_color = '#1f77b4' 
    sns.histplot(all_activations, bins=50, color=plot_color, kde=False, stat="density")
    
    zero_pct = (np.array(all_activations) == 0).sum() / len(all_activations) * 100
    
    plt.title(f"Task 1 Activations Post-Training ({method_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Activation Value ($z$)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
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
    print(f"\n   📊 [Analysis] Generating t-SNE plot for {method_name}...")
    model.eval()
    
    all_z = []
    all_labels = []     
    all_modalities = [] 
    
    MAX_SAMPLES_PER_MOD = 250 
    
    with torch.no_grad():
        for task_idx, mod_name in enumerate(['walkway', 'insole']):
            if mod_name not in eval_loaders: continue
            
            if hasattr(model, 'set_active_modality'): model.set_active_modality(mod_name)
            if hasattr(model, 'set_active_task'): model.set_active_task(task_idx)
                
            samples_collected = 0
            for x, y in eval_loaders[mod_name]:
                x = x.to(device)
                
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
    
    markers = {'walkway': 'o', 'insole': '^'}
    colors = {0: '#1f77b4', 1: '#d62728'} 
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
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ t-SNE plot saved to {save_path}")

def _scan_subjects(dir_path: Path):
    return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})

def init_subjects_and_folds(args):
    pd_ids = _scan_subjects(Config.PD_PATH)
    hc_ids = _scan_subjects(Config.HC_PATH)
    if not pd_ids or not hc_ids: raise ValueError("No subjects found.")
    
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(
        pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed
    )
    return subj2label, folds

def print_experiment_summary(args, step_history):
    """
    Prints a summary table of task performance evolution averaged across folds.
    
    Args:
        args: Argument parser object (needs .ewc_lambda and .order)
        step_history: Dict structure { step_idx: { task_idx: [score_fold1, score_fold2...] } }
    """
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    
    # 1. Prepare Headers
    tasks = [t.strip() for t in args.order.split(',')]
    t_order_str = "".join([t[0] for t in tasks]) # "walkway,insole" -> "wi"
    
    # Header: lambda  t_order  step  t1_mean  t2_mean ... avg_acc
    header = f"{'lambda':<8} {'t_order':<8} {'step':<6}"
    for i in range(len(tasks)):
        header += f" {f't{i+1}_mean':<8}"
    header += " avg_acc"
    print(header)
    
    # 2. Iterate Steps (Rows)
    sorted_steps = sorted(step_history.keys())
    
    for step_idx in sorted_steps:
        row_str = f"{args.ewc_lambda:<8} {t_order_str:<8} {step_idx+1:<6}"
        
        current_step_means = []
        
        # Iterate Tasks (Columns)
        for task_idx in range(len(tasks)):
            # Check if we have data for this task at this step
            if task_idx in step_history[step_idx]:
                scores = step_history[step_idx][task_idx]
                mean_score = sum(scores) / len(scores)
                row_str += f" {mean_score:<8.2f}"
                current_step_means.append(mean_score)
            else:
                # Task not yet seen or not evaluated
                row_str += f" {'NaN':<8}"
        
        # Calculate Row Average
        if current_step_means:
            avg_acc = sum(current_step_means) / len(current_step_means)
            row_str += f" {avg_acc:.2f}"
        else:
            row_str += " 0.00"
            
        print(row_str)
    print("="*70 + "\n")

# ----------------- EWC Helpers -----------------
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

def analyze_fisher_overlap(model_prev, model_curr, ewc_instance, task_id_to_analyze=0):
    """
    Computes the alignment between weight updates (Delta W) and Fisher Information.
    
    Args:
        model_prev: Snapshot of model BEFORE Task 2.
        model_curr: Current model AFTER Task 2.
        ewc_instance: The EWC object (used to access the model buffers).
        task_id_to_analyze: The task ID of the constraints we are violating. 
                            For the Task 1 -> Task 2 crash, this is 0.
    """
    print(f"\n   🔍 [Analysis] Computing Fisher-Update Overlap (vs Task {task_id_to_analyze} Constraints)...")
    
    all_fisher = []
    all_delta = []
    
    params_prev = dict(model_prev.named_parameters())
    params_curr = dict(model_curr.named_parameters())
    
    # Iterate through current parameters
    for name, p_curr in params_curr.items():
        if name in params_prev:
            p_prev = params_prev[name]
            
            # --- CRITICAL FIX ---
            # Construct the buffer name used by your EWC class
            # Format: name.replace('.', '__') + '_fisher_diag_t{id}'
            buffer_name = name.replace('.', '__') + f'_fisher_diag_t{task_id_to_analyze}'
            
            # Retrieve the Fisher buffer from the EWC model
            if hasattr(ewc_instance.model, buffer_name):
                fisher_tensor = getattr(ewc_instance.model, buffer_name)
                
                # 1. Calculate Absolute Change: |w_new - w_old|
                delta = (p_curr.cpu() - p_prev).abs().detach().view(-1)
                
                # 2. Get Fisher Importance
                fish = fisher_tensor.detach().cpu().view(-1)
                
                all_fisher.append(fish)
                all_delta.append(delta)

    if not all_fisher:
        print("   ⚠️ [Warning] No Fisher buffers found. Did you run register_ewc_params for Task 0?")
        return 0.0

    # Concatenate into single global vectors
    F_vec = torch.cat(all_fisher).numpy()
    D_vec = torch.cat(all_delta).numpy()
    
    if np.linalg.norm(D_vec) < 1e-9:
        print("   ⚠️ [Analysis] Zero weight update detected. Overlap is 0.")
        return 0.0
        
    # 3. Normalize vectors to compare Direction Only
    D_norm = D_vec / np.linalg.norm(D_vec)
    F_norm = F_vec / np.linalg.norm(F_vec) 
    
    # 4. Compute Cosine Similarity
    overlap_score = np.dot(F_norm, D_norm)
    
    print(f"   📊 Fisher-Update Overlap Score: {overlap_score:.4f}")
    
    return overlap_score

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

# ----------------- Training Loop -----------------
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

    def compute(self, logits, features, y, x, mod):
        raw_ce = self.ewc.criterion(logits, y)
        
        # --- EWC Magnitude Extraction ---
        # 1. Get the WEIGHTED loss from the EWC class
        weighted_ewc_tensor = self.ewc._ewc_penalty()
        weighted_ewc = weighted_ewc_tensor.item() if torch.is_tensor(weighted_ewc_tensor) else weighted_ewc_tensor
        
        # 2. Reverse engineer the RAW displacement for the supervisor's logs
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
                if self.kd_lambda > 0:
                    t_z = self.teacher_model.shared_backbone(t_features)
                    t_logits = self.teacher_model.shared_head(t_z)

            # --- A. Semantic Logit Distillation (L_o) ---
            if self.kd_lambda > 0:
                T = 2.0
                p_s = F.log_softmax(logits / T, dim=1)
                p_t = F.softmax(t_logits / T, dim=1)
                raw_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)
                weighted_kd = self.kd_lambda * raw_kd

            # --- B. Physics-Decoupled Distillation (-L_f) ---
            if self.repulsive_alpha > 0:
                # Calculate Cosine Similarity against the Tangled State
                cos_sim = F.cosine_similarity(features, t_features, dim=1)
                
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


def run_warmup_phase(args, model, train_loader, val_loader, device, mod, task_id):
    warmup_ep = args.kd_we
    print(f"\n   >>> [Phase 1] PURE WARM-UP: Adapting '{mod}' Encoder & PAMN via CE ({warmup_ep} epochs)...")
    
    for p in model.parameters(): p.requires_grad = False
    for p in model.encoders[mod].parameters(): p.requires_grad = True
    set_active_task_and_freeze(model, task_id) 
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-3)
    
    model.eval() 
    model.set_active_modality(mod)
    
    best_val_f1 = 0.0
    best_model_state = copy.deepcopy(model.state_dict()) 
    best_epoch = 0
    
    for ep in range(1, warmup_ep + 1): 
        total_loss = 0
        correct, total = 0, 0
        
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
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        train_acc = (correct / total) * 100.0 if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                all_preds.extend(v_logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_epoch = ep
            best_model_state = copy.deepcopy(model.state_dict())

        if ep % 5 == 0 or ep == warmup_ep:
            print(f"       Warmup Ep {ep:02d}/{warmup_ep} | Loss:{avg_loss:.4f} | "
                  f"Tr:{train_acc:.2f}% | ValF1:{val_f1:.2f}% (Best:{best_val_f1:.2f}%)")

    print(f"   >>> [Snapshot] Restoring Best Warmup Model from Epoch {best_epoch}...")
    model.load_state_dict(best_model_state)
    return best_val_f1


def run_bn_corruption_ablation(model, task1_val_loader, task2_train_loader, device, mod1='imu', mod2='walkway'):
    """
    Ablation to prove Statistical Catastrophic Forgetting (The EWC Blind Spot).
    We expose the network to Task 2 data purely to shift the BN running statistics,
    with ZERO weight updates (no optimizer.step()), then re-evaluate Task 1.
    """
    print("\n" + "="*60)
    print("🚀 RUNNING BN CORRUPTION ABLATION (EWC BLIND SPOT)")
    print("="*60)

    # ---------------------------------------------------------
    # STEP 1: Baseline Evaluation on Task 1 (IMU)
    # ---------------------------------------------------------
    model.eval()
    model.set_active_modality(mod1)
    if hasattr(model, 'set_active_task'):
        model.set_active_task(0)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for vx, vy in task1_val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_logits = model(vx)
            all_preds.extend(v_logits.argmax(1).cpu().numpy())
            all_targets.extend(vy.cpu().numpy())

    baseline_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
    print(f"   [Step 1] Baseline {mod1.upper()} F1 (Before Corruption): {baseline_f1:.2f}%")

    # ---------------------------------------------------------
    # STEP 2: Corrupt the BN Statistics using Task 2 (Walkway) Data
    # ---------------------------------------------------------
    print(f"\n   [Step 2] Corrupting BN stats with {mod2.upper()} data...")
    
    # CRITICAL: model.train() turns ON the tracking of BN running mean/variance
    model.train() 
    model.set_active_modality(mod2)
    if hasattr(model, 'set_active_task'):
        model.set_active_task(1)

    corruption_epochs = 3 # 3 epochs is plenty to entirely overwrite the moving average
    
    # CRITICAL: torch.no_grad() ensures ZERO gradients are computed. 
    # The convolutional weights remain 100% frozen. Only BN stats shift.
    with torch.no_grad(): 
        for ep in range(corruption_epochs):
            for x, _ in task2_train_loader:
                x = x.to(device)
                _ = model(x) # The forward pass alone triggers the BN stat update

    print(f"   [Step 2] BN statistics successfully shifted to {mod2.upper()} distribution.")

    # ---------------------------------------------------------
    # STEP 3: Re-Evaluate on Task 1 (IMU) with Corrupted Stats
    # ---------------------------------------------------------
    print(f"\n   [Step 3] Re-evaluating {mod1.upper()} with corrupted BN stats...")
    
    # CRITICAL: model.eval() locks the new, corrupted Walkway stats in place for inference
    model.eval() 
    model.set_active_modality(mod1)
    if hasattr(model, 'set_active_task'):
        model.set_active_task(0)

    all_preds_c, all_targets_c = [], []
    with torch.no_grad():
        for vx, vy in task1_val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_logits = model(vx)
            all_preds_c.extend(v_logits.argmax(1).cpu().numpy())
            all_targets_c.extend(vy.cpu().numpy())

    corrupt_f1 = f1_score(all_targets_c, all_preds_c, average='macro') * 100.0
    drop = baseline_f1 - corrupt_f1

    print(f"   [Step 3] Corrupted {mod1.upper()} F1: {corrupt_f1:.2f}%")
    print(f"   📉 ABSOLUTE DROP (Statistical Forgetting): -{drop:.2f}%")
    print("="*60 + "\n")

    return baseline_f1, corrupt_f1


def train_one_task(args, model, ewc, train_loader, val_loaders_dict, tasks_list, mod, device, epochs, num_classes, patience, task_id):
    
    current_val_loader = val_loaders_dict[mod]

    # --- A & B. Phase 1: Pure CE Teacher Warmup ---
    # The Historical Teacher has been strictly removed to prevent Garbage Distillation.
    aligned_teacher = None
    requires_warmup = (getattr(args, 'kd_lambda', 0.0) > 0) or (getattr(args, 'repulsive_alpha', 0.0) > 0)
    if task_id > 0 and requires_warmup:
        _ = run_warmup_phase(args, model, train_loader, current_val_loader, device, mod, task_id)
        
        # Capture the ALIGNED teacher AFTER warmup to serve as both Semantic & Spatial Anchor
        aligned_teacher = copy.deepcopy(model)
        aligned_teacher.eval()
        for p in aligned_teacher.parameters(): p.requires_grad = False

    # --- C. Phase 2: Fine-Tuning Setup ---
    print(f"\n   >>> [Phase 2] FINE-TUNE: Curriculum Consolidation...")
    unfreeze_shared_components(model, mod)
    set_active_task_and_freeze(model, task_id)

    if task_id > 0:
        # permanently remove overfitting regularizers for subsequent tasks
        for param_group in ewc.optimizer.param_groups:
            param_group['weight_decay'] = 0.0
        for m in model.shared_backbone.modules():
            if isinstance(m, nn.Dropout1d) or isinstance(m, nn.Dropout):
                m.p = 0.0  
    model.train()

    # Initialize LossEngine with the ALIGNED Teacher and the dynamically updated args
    loss_engine = LossEngine(ewc, aligned_teacher, args, device)
    base_repulsive_alpha = getattr(args, 'repulsive_alpha', 0.0)
    base_kd_lambda = getattr(args, 'kd_lambda', 0.0)
    min_kd_lambda = getattr(args, 'min_kd_lambda', 0.1)  # The floor value 

    if len(train_loader.dataset.labels) > 0:
        counts = [train_loader.dataset.labels.count(i) for i in range(num_classes)]
        ewc.criterion = nn.CrossEntropyLoss(weight=U.class_weight_tensor(counts, device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ewc.optimizer, mode='max', factor=0.5, patience=50)
    early_stopper = U.EarlyStopping(patience=patience, mode='max')

    # Curriculum Learning Parameters
    # repul_ramp_start = getattr(args, 'repul_ramp_start', 10)
    # repul_ramp_end   = getattr(args, 'repul_ramp_end', 25)
    # kd_decay_start   = getattr(args, 'kd_decay_start', 25)
    # kd_decay_end     = getattr(args, 'kd_decay_end', 40)

    best_eval = 0
    p = 3.0
    for ep in range(1, epochs+1):
        # # 1A. REPULSIVE SCHEDULER: Linearly ramp UP alpha
        # if task_id > 0 and base_repulsive_alpha > 0:
        #     if ep <= repul_ramp_start:
        #         loss_engine.repulsive_alpha = 0.0
        #     elif ep >= repul_ramp_end:
        #         loss_engine.repulsive_alpha = base_repulsive_alpha
        #     else:
        #         ratio = (ep - repul_ramp_start) / (repul_ramp_end - repul_ramp_start)
        #         loss_engine.repulsive_alpha = base_repulsive_alpha * ratio

        # # 1B. KD SCHEDULER: Linearly ramp DOWN lambda
        # if task_id > 0 and base_kd_lambda > 0:
        #     if ep <= kd_decay_start:
        #         loss_engine.kd_lambda = base_kd_lambda
        #     elif ep >= kd_decay_end:
        #         loss_engine.kd_lambda = min_kd_lambda
        #     else: # Calculate decay ratio (1.0 down to 0.0)
        #         ratio = 1.0 - ((ep - kd_decay_start) / (kd_decay_end - kd_decay_start))
        #         # Scale between base_kd and min_kd
        #         loss_engine.kd_lambda = min_kd_lambda + (base_kd_lambda - min_kd_lambda) * ratio

        # Calculate normalized training progress t in [0, 1]
        t = ep / epochs 

        # 1. CONTINUOUS ASYMMETRIC POLYNOMIAL SCHEDULER
        if task_id > 0:
            if base_repulsive_alpha > 0.0:
                # A. Repul: Root function (t^(1/p)) ensures rapid early spatial separation
                loss_engine.repulsive_alpha = base_repulsive_alpha * (t ** (1.0 / p))
                
                # B. KD: Cubic decay (1 - t^p) ensures late semantic release
                loss_engine.kd_lambda = min_kd_lambda + (base_kd_lambda - min_kd_lambda) * (1.0 - (t ** p))
            else: # EWC + KD
                loss_engine.repulsive_alpha = 0.0
                loss_engine.kd_lambda = base_kd_lambda  # Anchor remains strictly static
        # ==============================================================

        # 2. TRAIN STEP
        model.train()
        model.set_active_modality(mod)
        set_active_task_and_freeze(model, task_id)
        accum = {"loss": 0, "raw_ce": 0, "raw_ewc": 0, "raw_kd": 0, "raw_repul": 0, 
                 "w_ewc": 0, "w_kd": 0, "w_repul": 0, "correct": 0, "total": 0}

        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            ewc.optimizer.zero_grad()
            features = model.encoders[mod](x) 
            z = model.shared_backbone(features)
            logits = model.shared_head(z)
            
            loss, metrics = loss_engine.compute(logits, features, y, x, mod)
            loss.backward()
            ewc.optimizer.step()
            
            for k in metrics: 
                if k in accum: accum[k] += metrics[k]
            preds = logits.argmax(dim=1)
            accum["correct"] += (preds == y).sum().item()
            accum["total"]   += y.size(0)
        
        # 3. STRICT TASK-SPECIFIC EVALUATION
        model.eval()
        model.set_active_modality(mod)
        if hasattr(model, 'set_active_task'):
            model.set_active_task(task_id)
            
        all_preds, all_targets = [], []
        with torch.no_grad():
            # ONLY evaluate on the current training task's validation loader
            for vx, vy in current_val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                all_preds.extend(v_logits.argmax(1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        # Calculate F1 for the CURRENT task only
        current_val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        best_eval = max(current_val_f1, best_eval)
        
        # 4. LOGGING & SCHEDULING
        if ep % 5 == 0:
            n = len(train_loader)
            avg = {k: v / n for k, v in accum.items() if k not in ["correct", "total"]}
            train_acc = (accum["correct"] / accum["total"]) * 100.0
            
            print(f"[{mod}] Ep {ep:02d} | TrAcc:{train_acc:.2f} | ValF1:{current_val_f1:.2f} (Best:{best_eval:.2f}) | "
                  f"Tot:{avg['loss']:.4f} [CE:{avg['raw_ce']:.4f} | "
                  f"wEWC:{avg['w_ewc']:.4f} | "
                  f"wKD:{avg['w_kd']:.4f} (λ={loss_engine.kd_lambda:.1f}) | "
                  f"wRepul:{avg['w_repul']:.4f} (α={loss_engine.repulsive_alpha:.2f})]")

        if ep > args.lr_we:
            scheduler.step(current_val_f1)
            
        # 5. CURRICULUM-LOCKED EARLY STOPPING
        stop_signal = early_stopper(current_val_f1, model)
        lockout_horizon = int(0.5 * epochs)
        using_curriculum = (getattr(args, 'kd_lambda', 0.0) > 0) and (getattr(args, 'repulsive_alpha', 0.0) > 0)
        curriculum_is_active = (task_id > 0) and (ep <= lockout_horizon) and using_curriculum
        
        if stop_signal:
            if curriculum_is_active:
                # The CRITICAL fix is here: completely un-trip the stopper mechanism
                early_stopper.counter = 0 
                early_stopper.early_stop = False
                print(f"   [Curriculum Lockout] Suppressing Early Stop at Ep {ep} (Curriculum Active)")
            else:
                print(f"   🛑 Task Convergence Reached! Early Stopping at Ep {ep}")
                break
            
    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)


def run_cv_with_cache(args, data_cache):
    subj2label, folds = init_subjects_and_folds(args)
    if args.mode == 'joint':
        return joint_train.run_joint_experiment(args, data_cache, subj2label, folds)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.order.split(",") if t.strip()]
    log_dir = Config.CHECKPOINT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_loader_cache = {fi: {} for fi in range(len(folds))}
    step_history = {}
    fold_scores = []
    fold_ablation_results = []
    
    for fi in range(len(folds)):
        print(f"\n========== Fold {fi+1}/{len(folds)} ==========")
        model   = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=args.disable_dbn).to(device)
        ewc     = ElasticWeightConsolidation(model, nn.CrossEntropyLoss(), lr=args.lr, weight=args.ewc_lambda)
        seen    = []
        model_snapshot_before_task = None

        for ti, mod in enumerate(tasks, 1):
            print(f"\n=== Task {ti}/{len(tasks)} : {mod} ===")
            current_task_idx = ti - 1

            # --- Compute Overlap capture snapshot before Task 2 ---
            if ti == 2 and getattr(args, 'analyze_overlap', False): 
                 print("   📸 [Analysis] Taking snapshot before Task 2...")
                 model_snapshot_before_task = copy.deepcopy(model)
                 model_snapshot_before_task.cpu() # Save GPU memory

            # Data
            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            tr_loader = DataLoader(U.SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(U.SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[fi][mod] = te_loader 

            # =====================================================================
            # INJECT: BN CORRUPTION ABLATION (Runs right before Task 2 training)
            # =====================================================================
            # Set to True manually here if you don't want to pass it via bash args
            run_ablation = getattr(args, 'run_bn_ablation', True) 
            
            if ti == 2 and run_ablation:
                task1_mod = tasks[0] # E.g., 'imu'
                task2_mod = mod      # E.g., 'walkway'
                
                # We pass the cached IMU eval loader and the freshly generated Walkway train loader
                base_f1, corr_f1 = run_bn_corruption_ablation(
                    model=model, 
                    task1_val_loader=eval_loader_cache[fi][task1_mod], 
                    task2_train_loader=tr_loader, 
                    device=device, 
                    mod1=task1_mod, 
                    mod2=task2_mod
                )
                fold_ablation_results.append((base_f1, corr_f1))
                
                print(f"🛑 Ablation recorded for Fold {fi+1}. Skipping remaining tasks to speed up evaluation.")
                break
            # =====================================================================

            # Optimizer Setup
            current_seen_tasks = seen + [mod]
            seen_val_loaders = {m: eval_loader_cache[fi][m] for m in current_seen_tasks}
            # current_decay = 0.05 if mod == 'insole' else 1e-3
            current_decay=1e-3
            if args.mode == 'specialist':
                model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=args.disable_dbn).to(device)
                ewc = ElasticWeightConsolidation(model, nn.CrossEntropyLoss(), lr=args.lr, weight=0.0, weight_decay=current_decay)
            elif args.mode == 'cl':
                ewc.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=current_decay)

            # Train
            # train_one_task(args, model, ewc, tr_loader, te_loader, mod, device, args.epochs, args.num_classes, patience=args.patience, task_id=current_task_idx)
            train_one_task(args, model, ewc, tr_loader, seen_val_loaders, tasks, mod, device, args.epochs, args.num_classes, patience=args.patience, task_id=current_task_idx)

            # --- Compute Overlap After Task 2 ---
            if ti == 2 and getattr(args, 'analyze_overlap', False) and model_snapshot_before_task:
                overlap = analyze_fisher_overlap(model_snapshot_before_task, model, ewc, task_id_to_analyze=0)
                del model_snapshot_before_task
                model_snapshot_before_task = None

            # EWC
            is_last_task = (ti == len(tasks))
            should_register_fisher = (args.ewc_lambda > 0) or getattr(args, 'analyze_overlap', False)
            if args.mode == 'cl' and should_register_fisher and not is_last_task:
                print(f">> [CL] Registering Fisher for {mod} (Required for Overlap Analysis)...")
                register_shared_ewc(model, ewc, tr_loader, args.fisher_batches, task_id=current_task_idx)
            
            # Eval
            seen.append(mod)
            print(f"\n--- Evaluation ---")
            eval_targets = [mod] if args.mode == 'specialist' else seen
            scores = []
            for m in eval_targets:
                model.set_active_modality(m) 
                target_task_idx = tasks.index(m)
                if hasattr(model, 'set_active_task'):
                    model.set_active_task(target_task_idx)
                s = U.evaluate_classification(model, eval_loader_cache[fi][m], device)
                scores.append(s)

            current_step_idx = ti - 1 # 0-based step index
            if current_step_idx not in step_history:
                step_history[current_step_idx] = {}

            for m, score in zip(eval_targets, scores):
                real_task_idx = tasks.index(m)
                if real_task_idx not in step_history[current_step_idx]:
                    step_history[current_step_idx][real_task_idx] = []
                step_history[current_step_idx][real_task_idx].append(score)
                print(f"  {m}: {score:.2f}")

            if args.mode == 'specialist': 
                final_score = scores[-1]
            else: 
                final_score = sum(scores)/len(scores)
                print(f"  Avg Seen: {final_score:.2f}")

        fold_scores.append(final_score)

    # =====================================================================
    # Batch Norm Corruption ABLATION SUMMARY
    # =====================================================================
    if getattr(args, 'run_bn_ablation', False) and len(fold_ablation_results) > 0:
        print("\n" + "="*60)
        print(f"🎯 FINAL ABLATION RESULTS (Averaged over {len(folds)} Folds)")
        print("="*60)
        avg_base = sum(b for b, c in fold_ablation_results) / len(fold_ablation_results)
        avg_corr = sum(c for b, c in fold_ablation_results) / len(fold_ablation_results)
        
        print(f"   Average Baseline F1  : {avg_base:.2f}%")
        print(f"   Average Corrupted F1 : {avg_corr:.2f}%")
        print(f"   📉 AVERAGE ABSOLUTE DROP: -{avg_base - avg_corr:.2f}%")
        print("="*60 + "\n")
        
        # Return dummy values to satisfy the main script, since we skipped standard CV
        return (avg_base - avg_corr), {}
    #=====================================================================

        # if fi == 0 and args.mode == 'cl': 
        #     print("\n" + "="*40)
        #     print("   Generating 'Ours' Comparison Plots...")
        #     print("="*40)
            
        #     # Note: Your eval loaders are stored in a nested dict `eval_loader_cache[fi][mod]`
        #     fold_0_loaders = eval_loader_cache[0]
            
        #     # 1. Activation Death Plot
        #     act_path = os.path.join(Config.OUTPUT_DIR, "activation_death_ours.png")
        #     plot_activation_death(model, fold_0_loaders['walkway'], device, 
        #                           method_name="Dual-BN (Ours)", save_path=act_path)
            
        #     # 2. t-SNE Plot
        #     tsne_path = os.path.join(Config.OUTPUT_DIR, "tsne_ours.png")
        #     plot_tsne_latent(model, fold_0_loaders, device, 
        #                      method_name="Dual-BN (Ours)", save_path=tsne_path)

    avg_f1 = sum(fold_scores) / len(fold_scores)
    return avg_f1, step_history


def run_single_seed_experiment(args):
    global_cache = preload_all_subjects(Config.OUTPUT_DIR)
    print(f"\n{'='*40}\nSTARTING SINGLE SEED: {args.seed}\n{'='*40}")
    U.set_seed(args.seed)

    avg_f1, step_history = run_cv_with_cache(args, global_cache)
    print(f"\nFINISHED SEED {args.seed} | Macro F1: {avg_f1:.2f}")
    print_experiment_summary(args, step_history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", type=str, default="specialist", choices=["cl", "specialist", "joint"])
    ap.add_argument("--order", type=str, default="walkway,insole,imu")
    
    # --- Config Overrides ---
    ap.add_argument("--seed", type=int, default=Config.SEED)
    ap.add_argument("--n_folds", type=int, default=Config.N_FOLDS)
    ap.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=1e-3) 
    ap.add_argument("--lr_we", type=int, default=10, help="LR warmup epochs")
    
    # --- The "Runway" Parameters ---
    ap.add_argument("--epochs", type=int, default=100, help="Increased to give network time post-curriculum")
    ap.add_argument("--patience", type=int, default=25, help="Increased to survive the spatial wedge phase")
    
    ap.add_argument("--win_len", type=int, default=Config.WINDOW_SIZE)
    default_hop = int(Config.WINDOW_SIZE * Config.STRIDE)
    ap.add_argument("--hop_len", type=int, default=default_hop)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--disable_dbn", action='store_true', help="Forces shared BN for pure baselines")
    
    # --- The Optimization Anchors ---
    ap.add_argument("--ewc_lambda", type=float, default=5000.0)
    ap.add_argument("--fisher_batches", type=int, default=64) 
    ap.add_argument("--kd_lambda", type=float, default=1.0)
    ap.add_argument("--kd_we", type=int, default=10, help="warmup epochs for KD")
    
    # --- The Repulsive Forces ---
    ap.add_argument("--repulsive_alpha", type=float, default=1.0, help="Weight for Physics-Decoupled Distillation (-Lf)")
    ap.add_argument("--repulsive_margin", type=float, default=0.1, help="Target max cosine similarity (0.0 = orthogonal)")
    ap.add_argument("--analyze_overlap", default=False, action='store_true')
    
    # --- Phase-Shifted Curriculum Schedulers ---
    ap.add_argument("--repul_ramp_start", type=int, default=10, help="Epoch to start ramping up repulsive loss")
    ap.add_argument("--repul_ramp_end", type=int, default=25, help="Epoch to reach max repulsive alpha")
    ap.add_argument("--kd_decay_start", type=int, default=25, help="Epoch to start decaying KD lambda")
    ap.add_argument("--kd_decay_end", type=int, default=40, help="Epoch to reach min KD lambda")
    ap.add_argument("--min_kd_lambda", type=float, default=0.1, help="Floor value for KD decay")
    ap.add_argument("--run_bn_ablation", type=bool, default=False, help="Run BN corruption ablation")
    
    args = ap.parse_args()
    print(f"Arguments: {', '.join(f'{k}={v}' for k, v in vars(args).items())}")
    run_single_seed_experiment(args)


if __name__ == "__main__":
    main()