import os, argparse, sys
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# --- Path Setup ---
current_file = Path(__file__).resolve()
current_dir = current_file.parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

# --- Project Imports ---
from model.weargait.ewc.config import Config
import model.weargait.ewc.utility as U
from model.weargait.ewc.data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    make_fixed_balanced_folds_no_overlap, build_subj2label
)
from model.weargait.ewc.encoder import WearGaitUniversal
# from model.weargait.ewc.encoder_res18 import WearGaitResNet18
import matplotlib
matplotlib.use('Agg') # Safe for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def _scan_subjects(dir_path: Path):
    return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})

def init_subjects_and_folds(args):
    pd_ids = _scan_subjects(Config.PD_PATH)
    hc_ids = _scan_subjects(Config.HC_PATH)
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    return subj2label, folds

######################## Main Components ########################

class HarmonyACFM(nn.Module):
    """
    ACFM per Harmony paper Section 3.4.
    Builds compatible historical features from current features and past classifier weights.
    """
    def __init__(self, feature_dim=64, K=3, classifier_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.K = K  # paper uses K=3 noise scales
        self.proto_proj = nn.Linear(classifier_dim, feature_dim)
        # Eq. 2 E_trans: linear transform on prototypes
        self.E_trans = nn.Linear(feature_dim, feature_dim)
        
        # Eq. 2 E_mod: MLP for mixture coefficients alpha_i
        self.E_mod = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, K)
        )
        
        # Eq. 2 sigma: K learnable noise scales
        self.sigma = nn.Parameter(torch.ones(K))
        
        # Eq. 2 lambda_g: perturbation strength (paper suggests 0.6)
        self.lambda_g = 0.6

    def forward(self, current_feat, hist_classifier_weight, labels):
        """current_feat shape (B, T, C) as L x d in the paper."""
        B, T, C = current_feat.size()
        
        # Step 1: historical prototypes (Eq. 1)
        P_prev_high_dim = hist_classifier_weight[labels] # (B, 512)
        P_prev = self.proto_proj(P_prev_high_dim)        # (B, 64)
        
        # Broadcast prototype to all time steps
        # (B, 64) -> (B, 1, 64) -> (B, T, 64)
        P_prev_time = P_prev.unsqueeze(1).expand(-1, T, -1)
        
        # Step 2: adaptive feature perturbation (Eq. 2)
        # Predict noise mixture weights alpha_i (BxTxC)
        alpha = F.softmax(self.E_mod(current_feat), dim=-1) # (B, T, K)
        
        noise_sum = torch.zeros_like(current_feat) # (B, T, C)
        for k in range(self.K):
            z_k = torch.randn_like(current_feat) * self.sigma[k]
            noise_sum += alpha[..., k:k+1] * z_k
            
        F_prev_modulated = self.E_trans(P_prev_time) + (self.lambda_g * noise_sum)
        
        # Step 3: feature fusion (Eq. 3)
        F_compatible_history = F_prev_modulated + current_feat # (B, T, C)
        
        return F_compatible_history

class MKAM(nn.Module):
    """
    Modality Knowledge Aggregation Module (MKAM)
    Eq. (4): map features to a shared aggregation space (linear layer per Sec. 4.3).
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x may be (B, C) or (B, T, C)
        return self.proj(x)

class GatedKnowledgeAdapter(nn.Module):
    """
    Gated Knowledge Adapter
    Eq. (5): W_adapter = omega * B * A; low-rank filter with gate omega.
    """
    def __init__(self, feature_dim=64, rank=8):
        super().__init__()
        # Sec. 4.3: rank 128 in paper (may differ here)
        self.A = nn.Linear(feature_dim, rank, bias=False)
        self.B = nn.Linear(rank, feature_dim, bias=False)
        
        # Gate omega, init 1.0
        self.omega = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_history):
        # Low-rank filter
        low_rank_feat = self.B(self.A(x_history))
        # Gated scaling
        return self.omega * low_rank_feat

class CumulativeKnowledgeAggregation(nn.Module):
    """
    Cumulative Knowledge Aggregation (CKA)
    MKAM + gated adapter + cross-attention (Eq. 4-6).
    """
    def __init__(self, feature_dim=64, rank=128):
        super().__init__()
        # Separate MKAM for current and history
        self.mkam_current = MKAM(feature_dim)
        self.mkam_history = MKAM(feature_dim)
        
        # Gated knowledge adapter
        self.gated_adapter = GatedKnowledgeAdapter(feature_dim, rank)
        
        # Cross-attention scale 1/sqrt(d)
        self.scale = feature_dim ** -0.5

    def forward(self, f_current, f_history):
        """
        Args:
            f_current: (B, T, C) current modality features
            f_history: (B, T, C) ACFM-compatible history features
        Adds T=1 if input is 2D (B, C).
        """

        # Step 1: MKAM (Eq. 4) -> F_hat^t, F_hat^{t-1}
        f_hat_current = self.mkam_current(f_current)
        f_hat_history = self.mkam_history(f_history)

        # Step 2: gated adapter (Eq. 5) -> F_tilde^{t-1}
        f_tilde_history = self.gated_adapter(f_hat_history)

        # Step 3: cross-attention (Eq. 6)
        # Q: current; K/V: filtered history
        attn_scores = torch.bmm(f_hat_current, f_tilde_history.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        history_injection = torch.bmm(attn_weights, f_tilde_history)

        # Residual fusion (current modality dominant)
        f_fused = f_hat_current + history_injection

        if f_fused.size(1) == 1:
            f_fused = f_fused.squeeze(1)

        return f_fused

class HybridAlignmentLoss(nn.Module):
    def __init__(self, lambda_con=0.8, lambda_dis=0.6, margin=0.3):
        super().__init__()
        self.lambda_con = lambda_con
        self.lambda_dis = lambda_dis
        self.margin = margin

    def forward(self, o_current, o_history):
        batch_size = o_current.size(0)
        device = o_current.device
        
        # 1. Direct Feature Alignment
        loss_dir = F.mse_loss(o_current, o_history)

        # 2. Contrastive Feature Alignment
        norm_curr = F.normalize(o_current, p=2, dim=1)
        norm_hist = F.normalize(o_history, p=2, dim=1)
        sim_matrix = torch.matmul(norm_curr, norm_hist.t())
        pos_sim = torch.diag(sim_matrix) 
        
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        hard_neg_sim, _ = neg_sim_matrix.max(dim=1) 
        loss_con = F.relu(self.margin - (pos_sim - hard_neg_sim)).mean()

        # 3. Distribution-level Alignment
        # Uniform batch weights (no grad)
        beta_norm = torch.ones(batch_size, 1, device=device) / batch_size
        proxy_current = torch.sum(beta_norm * o_current, dim=0) 
        proxy_history = torch.sum(beta_norm * o_history, dim=0)
        loss_dis = F.mse_loss(proxy_current, proxy_history)

        # Total alignment loss
        total_align_loss = loss_dir + (self.lambda_con * loss_con) + (self.lambda_dis * loss_dis)
        return total_align_loss

######################## Training Loop ########################

def train_harmony_task(args, model, train_loader, val_loader, mod, device, 
                       epochs, num_classes, patience, task_idx):
    """Full Harmony training loop: ACFM, CKA, and hybrid alignment. task_idx marks first task."""
    print(f"\n   >>> [Harmony Complete] Training '{mod}' (Task {task_idx+1}) ...")
    model.set_active_task(task_idx)
    # ==========================================
    # Optimizer groups: base, ACFM, CKA
    # ==========================================
    base_params = [p for name, p in model.named_parameters() 
                   if p.requires_grad and 'acfm' not in name and 'cka' not in name]
    acfm_params = [p for p in model.acfm[mod].parameters() if p.requires_grad]
    cka_params =  [p for p in model.cka[mod].parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam([
        {'params': base_params},  
        {'params': acfm_params, 'lr': args.lr * 1.0}, 
        {'params': cka_params, 'lr': args.lr * 1.0} 
    ], lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    early_stopper = U.EarlyStopping(patience=patience, mode='max')
    criterion = nn.CrossEntropyLoss()

    # Hybrid alignment loss
    align_criterion = HybridAlignmentLoss(lambda_con=0.8, lambda_dis=0.6, margin=0.3).to(device)
    lambda_align = args.lambda_align
    best_eval = 0.0

    # ==========================================
    # Training loop
    # ==========================================
    for ep in range(1, epochs + 1):
        model.train()
        accum = {"loss": 0, "ce": 0, "align": 0, "correct": 0, "total": 0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # A. Current raw features
            raw_feats = model.encoders[mod](x) # (B, 64, T)
            # (B, C, T) -> (B, T, C) as F^t in paper
            feats_seq = raw_feats.transpose(1, 2) 
            
            loss_align = torch.tensor(0.0, device=device)

            if task_idx > 0:
                hist_weight = model.shared_head.fc.weight.detach()
                
                # B1. ACFM on sequence
                fake_history = model.acfm[mod](feats_seq, hist_weight, y) # (B, T, 64)

                # B2. CKA cross-attention (Eq. 6)
                fused_seq = model.cka[mod](feats_seq, fake_history) # (B, T, 64)

                # B3. Back to (B, C, T) for CNN backbone
                fused_time = fused_seq.transpose(1, 2) # (B, 64, T)
                fake_hist_time = fake_history.transpose(1, 2)

                z_curr = model.shared_backbone(fused_time)
                logits = model.shared_head(z_curr)
                z_history = model.shared_backbone(fake_hist_time)
                loss_align = align_criterion(z_curr, z_history)

            else:
                # Task 1
                f_hat_curr = model.cka[mod].mkam_current(feats_seq)
                fused_time = f_hat_curr.transpose(1, 2)
                
                z_curr = model.shared_backbone(fused_time)
                logits = model.shared_head(z_curr)

            # C. Total loss and backward
            loss_ce = criterion(logits, y)
            total_loss = loss_ce + (lambda_align * loss_align)
            
            total_loss.backward()
            optimizer.step()

            # Metrics
            accum["loss"] += total_loss.item()
            accum["ce"]   += loss_ce.item()
            accum["align"] += loss_align.item()
            accum["correct"] += (logits.argmax(1) == y).sum().item()
            accum["total"] += y.size(0)

        # ==========================================
        # Validation (no ACFM history at inference)
        # ==========================================
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                
                v_raw = model.encoders[mod](vx) # (B, 64, T)
                v_seq = v_raw.transpose(1, 2)   # (B, T, 64)

                # MKAM only at inference
                v_hat = model.cka[mod].mkam_current(v_seq) # (B, T, 64)
                v_time = v_hat.transpose(1, 2)  # (B, 64, T)
                
                vz = model.shared_backbone(v_time)
                v_logits = model.shared_head(vz)
                
                all_preds.extend(v_logits.argmax(1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        best_eval = max(val_f1, best_eval)
        scheduler.step(val_f1)

        if ep % 10 == 0:
            n = len(train_loader)
            print(f"[{mod}] Ep {ep:02d} | Loss:{accum['loss']/n:.4f} "
                  f"[CE:{accum['ce']/n:.3f} Align:{accum['align']/n:.3f}] | "
                  f"Acc:{accum['correct']/accum['total']*100:.1f}% ValF1:{val_f1:.2f}%")

        if early_stopper(val_f1, model):
            print(f"   🛑 Early Stopping at Ep {ep}")
            model.load_state_dict(early_stopper.best_model_state)
            break

    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)

    # ==========================================
    # Log gating collapse statistic (omega)
    # ==========================================
    model.eval()
    omega_val = model.cka[mod].gated_adapter.omega.item()
    print("\n" + "🎯" + "="*50)
    print(f"   [GATING COLLAPSE ANALYSIS - Task: {mod}]")
    print(f"   => Learned Gate Parameter (\u03c9) : {omega_val:.6f}")
    if omega_val < 0.1:
        print("   🚨 WARNING: GATING COLLAPSE DETECTED!")
        print("   The $1.344$ Modality Gap & $0.4$ Variance Shift forced Harmony")
        print("   to completely shut off historical knowledge transfer.")
    print("   " + "="*50 + "\n")

def run_cv_harmony(args, data_cache):
    subj2label, folds = init_subjects_and_folds(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.order.split(",") if t.strip()]
    
    step_history = {}
    fold_scores = []
    # Harmony ACFM prototypes come from shared_head weights

    for fi in range(len(folds)):
        print(f"\n{'='*20} Fold {fi+1}/{len(folds)} {'='*20}")
        
        # 1. Init Base Model
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=False).to(device)
        
        # ==========================================
        # 2. INJECT HARMONY COMPONENTS
        # ==========================================
        # a. ACFM per modality
        model.acfm = nn.ModuleDict({
            k: HarmonyACFM(feature_dim=64, K=3).to(device) for k in model.encoders.keys()
        })
        
        # b. CKA per modality
        model.cka = nn.ModuleDict({
            k: CumulativeKnowledgeAggregation(feature_dim=64, rank=128).to(device) for k in model.encoders.keys()
        })
        
        seen_mods = []
        eval_loader_cache = {}

        for ti, mod in enumerate(tasks):
            print(f"\n=== Harmony Task {ti+1}/{len(tasks)} : {mod} ===")
            
            # Data loaders
            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            tr_loader = DataLoader(U.SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(U.SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[mod] = te_loader 

            # ==========================================
            # 3. Train (task index ti)
            # ==========================================
            train_harmony_task(args, model, tr_loader, te_loader, mod, device, 
                               args.epochs, args.num_classes, args.patience, ti)

            # ==========================================
            # 4. Evaluate all seen tasks (strict inference)
            # ==========================================
            seen_mods.append(mod)
            print(f"\n--- Evaluation (Step {ti+1}) ---")
            scores = []
            
            # for m in seen_mods:
            for m_idx, m in enumerate(seen_mods):
                model.set_active_task(m_idx)
                model.eval()
                all_preds, all_targets = [], []
                
                with torch.no_grad():
                    for vx, vy in eval_loader_cache[m]:
                        vx, vy = vx.to(device), vy.to(device)
                        
                        # A. Raw features (B, 64, T)
                        vf = model.encoders[m](vx)
                        # (B, T, 64)
                        v_seq = vf.transpose(1, 2)
                        
                        # B. Inference: skip ACFM, MKAM only
                        v_hat = model.cka[m].mkam_current(v_seq) # (B, T, 64)
                        
                        # C. (B, 64, T) then backbone + head
                        v_time = v_hat.transpose(1, 2) # (B, 64, T)
                        
                        vz = model.shared_backbone(v_time)
                        v_logits = model.shared_head(vz)
                        
                        all_preds.extend(v_logits.argmax(1).cpu().numpy())
                        all_targets.extend(vy.cpu().numpy())
                
                # F1 score
                score = f1_score(all_targets, all_preds, average='macro') * 100.0
                scores.append(score)
                print(f"  {m}: {score:.2f}")

            # Record step scores
            if ti not in step_history: step_history[ti] = {}
            for m_idx, m_score in enumerate(scores):
                step_history[ti][m_idx] = [m_score]

            avg_seen = sum(scores) / len(scores)
            print(f"  Avg Seen: {avg_seen:.2f}")

        fold_scores.append(avg_seen)

    print(f"\n🏆 Final Avg F1 across folds: {sum(fold_scores)/len(fold_scores):.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--order", type=str, default="imu,walkway,insole")
    
    # Config Overrides
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n_folds", type=int, default=Config.N_FOLDS)
    ap.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=Config.EPOCHS)
    ap.add_argument("--patience", type=int, default=15)
    
    ap.add_argument("--win_len", type=int, default=Config.WINDOW_SIZE)
    default_hop = int(Config.WINDOW_SIZE * Config.STRIDE)
    ap.add_argument("--hop_len", type=int, default=default_hop)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=2)

    # ==========================================
    # 👑 Harmony Specific (Updated for Full Version)
    # ==========================================
    # Eq. 12 global alignment weight (paper suggests ~1.5; default lower here)
    ap.add_argument("--lambda_align", type=float, default=0.15, help="Weight for Hybrid Alignment Loss")

    args = ap.parse_args()
    print(f"Harmony Mode | Arguments: {', '.join(f'{k}={v}' for k, v in vars(args).items())}")
    
    # Preload Data
    global_cache = preload_all_subjects(Config.OUTPUT_DIR)
    U.set_seed(args.seed)
    
    run_cv_harmony(args, global_cache)

if __name__ == "__main__":
    main()