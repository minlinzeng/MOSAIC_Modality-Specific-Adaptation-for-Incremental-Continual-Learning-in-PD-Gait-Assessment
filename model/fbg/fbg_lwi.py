import os
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# --- FBG core imports ---
from data_loader import get_fbg_dataloaders
from encoder import MICL_CNN_PD_Model
from fbg_utility import EarlyStopping, set_deterministic_seed

# LwI optimal transport
from model.baselines.LwI import optimal_transport as ot

# ==========================================
# LwI config and helpers
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

def route_modality(batch, device, active_mod):
    lin = batch['linear'].to(device) if active_mod == 'linear' else None
    ang = batch['angular'].to(device) if active_mod == 'angular' else None
    grf = batch['grf'].to(device) if active_mod == 'grf' else None
    labels = batch['label'].to(device)
    return lin, ang, grf, labels

def recalibrate_bn(model, loader, device, active_mod, task_idx):
    """
    After OT fusion, recalibrate shared BN running stats on new data.
    """
    model.train()
    for p in model.parameters(): 
        p.requires_grad = False
        
    print(f"   🔄 [LwI] Recalibrating Shared Batch Norm statistics using '{active_mod.upper()}' data...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 50: break
            lin, ang, grf, _ = route_modality(batch, device, active_mod)
            _ = model(x_lin=lin, x_ang=ang, x_grf=grf, current_task=task_idx) 
            
    for p in model.parameters(): 
        p.requires_grad = True
    print("   ✅ [LwI] Recalibration Complete.")

# ==========================================
# Training loop (Chimera KD)
# ==========================================
def train_lwi_task(args, model, model_old, train_loader, val_loader, mod, task_id, device):
    print(f"\n   >>> [LwI] Training '{mod.upper()}' (Task {task_id}) | Feat KD $\lambda$: {args.kd_lambda}")
    
    model.train()
    if model_old is not None:
        model_old.eval()
        for p in model_old.parameters(): p.requires_grad = False

    # Train current modality encoder only
    if hasattr(model, 'enc_lin'): model.enc_lin.requires_grad_(mod == 'linear')
    if hasattr(model, 'enc_ang'): model.enc_ang.requires_grad_(mod == 'angular')
    if hasattr(model, 'enc_grf'): model.enc_grf.requires_grad_(mod == 'grf')

    # LR from args
    lr = args.lr if args.lr is not None else (2e-4 if mod == 'grf' else 1e-4)
    wd = 0.05
    dropout_rate = 0.1 if mod == 'grf' else 0.3
    
    model.dropout.p = dropout_rate
    model.res1.drop1d.p = dropout_rate
    model.res2.drop1d.p = dropout_rate
    model.res3.drop1d.p = dropout_rate

    active_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.AdamW(active_params, lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)  # plain CE
    mse_loss = nn.MSELoss()
    best_eval = 0.0

    # LwI warmup: freeze shared layers for first 5 epochs
    WARMUP_EPOCHS = 5

    for ep in range(1, args.epochs + 1):
        if model_old is not None and ep <= WARMUP_EPOCHS:
            phase = "WARMUP"
            current_lambda = 0.0
            model.res1.requires_grad_(False)
            model.res2.requires_grad_(False)
            model.res3.requires_grad_(False)
            model.head.requires_grad_(False)
        else:
            phase = "TRAIN "
            current_lambda = args.kd_lambda
            model.res1.requires_grad_(True)
            model.res2.requires_grad_(True)
            model.res3.requires_grad_(True)
            model.head.requires_grad_(True)

        model.train()
        accum = {"loss": 0, "ce": 0, "kd": 0, "correct": 0, "total": 0}

        for batch in train_loader:
            lin, ang, grf, y = route_modality(batch, device, mod)
            optimizer.zero_grad()
            
            # Forward; pooled features for Chimera KD
            logits_new, z_new = model(x_lin=lin, x_ang=ang, x_grf=grf, current_task=task_id)
            loss_ce = criterion(logits_new, y)
            loss = loss_ce
            loss_kd_val = torch.tensor(0.0)

            # Feature-level Chimera KD
            if model_old is not None and current_lambda > 0:
                with torch.no_grad():
                    _, z_old = model_old(x_lin=lin, x_ang=ang, x_grf=grf, current_task=task_id-1)
                
                # L2-normalize features before MSE
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

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                v_lin, v_ang, v_grf, vy = route_modality(batch, device, mod)
                v_logits, _ = model(x_lin=v_lin, x_ang=v_ang, x_grf=v_grf, current_task=task_id)
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


@torch.no_grad()
def evaluate_cl(model, dataloader, device, eval_mod, eval_task_idx):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        lin, ang, grf, labels = route_modality(batch, device, eval_mod)
        logits, _ = model(x_lin=lin, x_ang=ang, x_grf=grf, current_task=eval_task_idx)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro') * 100

def get_all_subjects(data_root):
    import glob
    # Scan root and subdirs for .pkl
    search_path = os.path.join(data_root, "**", "*.pkl")
    files = glob.glob(search_path, recursive=True)
    
    # Fallback glob if empty
    if not files:
        files = glob.glob(os.path.join(data_root, "*.pkl"))
        
    if len(files) == 0:
        raise FileNotFoundError(
            f"\n[Fatal Error] No .pkl files under --data_root!\n"
            f"Scanned path: {os.path.abspath(data_root)}\n"
            f"Verify the dataset directory is correct."
        )
        
    # Robust subject id parsing for varied naming
    subjects = set()
    for f in files:
        base = os.path.basename(f)
        # Split on '_' or '-'
        sub_part = base.split('_')[0] if '_' in base else base.split('-')[0]
        subjects.add(sub_part)
        
    return sorted(list(subjects))

# ==========================================
# CV and OT fusion driver
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="FBG LwI Baseline Execution Script")
    # CLI args for automated runner compatibility
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--order', type=str, default="linear,angular,grf")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--step_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=64)
    
    # disable_msbn flag (FBG encoder)
    parser.add_argument("--disable_msbn", action='store_true', help="Force network to drop MSBN and downgrade to Shared BN")

    # LwI (OT) Specific Arguments
    parser.add_argument('--step', type=float, default=0.3, help="Max similarity fusion step")
    parser.add_argument('--step_diff', type=float, default=0.5, help="Min similarity fusion step")
    parser.add_argument('--layers', type=int, default=14, help="Number of deep layers to apply min-sim to")
    parser.add_argument('--kd_lambda', type=float, default=300.0, help="Chimera Distillation weight")
    
    args = parser.parse_args()
    
    # Shared BN baseline for fair comparison
    disable_msbn_flag = True

    set_deterministic_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ot_config = OTConfig(args, device)
    
    tasks = [t.strip().lower() for t in args.order.split(",")]
    num_tasks = len(tasks)
    subjects = get_all_subjects(args.data_root)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    R_matrix = np.zeros((5, num_tasks, num_tasks))

    print("\n" + "="*65)
    print(f" 🚀 FBG BASELINE ENGINE: Learning without Isolation (LwI)")
    print(f" Seed: {args.seed} | Modalities: {tasks} | MSBN Protection: DISABLED")
    print("="*65)

    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        print(f"\n{'='*60}\n 🌟 INITIATING FOLD {fold+1}/5 WITH OPTIMAL TRANSPORT \n{'='*60}")
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        train_loader, test_loader = get_fbg_dataloaders(
            args.data_root, train_subjects, test_subjects, 
            batch_size=args.batch_size, window_size=args.window_size, step_size=args.step_size
        )
        
        # Model with disable_msbn=True (shared BN baseline)
        model = MICL_CNN_PD_Model(d_model=args.d_model, dropout=0.3, num_tasks=num_tasks, disable_msbn=disable_msbn_flag).to(device)
        model_old = None

        for task_idx, active_mod in enumerate(tasks):
            
            # 1. Train current task
            train_lwi_task(args, model, model_old, train_loader, test_loader, active_mod, task_idx, device)

            # 2. OT weight fusion
            if model_old is not None:
                print("\n   🧬 [LwI] Performing Optimal Transport (OT) Weight Fusion...")
                fused_dict = ot.get_wassersteinized_layers_modularized(
                    ot_config, device, networks=[model_old, model], ignore_keyword='enc_'
                )
                
                current_state = model.state_dict()
                for layer_name, new_weight in fused_dict.items():
                    if layer_name in current_state:
                        current_state[layer_name].copy_(new_weight)
                model.load_state_dict(current_state)
                
                # 3. Recalibrate shared BN
                recalibrate_bn(model, train_loader, device, active_mod, task_idx)

            model_old = copy.deepcopy(model)

            # 4. R matrix eval
            print(f"   [EVAL] Sequential Backward Testing...")
            for j in range(task_idx + 1):
                eval_mod = tasks[j]
                f1_score_j = evaluate_cl(model, test_loader, device, eval_mod, j)
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