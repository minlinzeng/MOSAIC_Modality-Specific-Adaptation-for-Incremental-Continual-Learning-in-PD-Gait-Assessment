import os, argparse, sys
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

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

######################## Harmony Components ########################
class HarmonyACFM(nn.Module):
    def __init__(self, feature_dim=64, K=3, classifier_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.K = K 
        self.proto_proj = nn.Linear(classifier_dim, feature_dim)
        self.E_trans = nn.Linear(feature_dim, feature_dim)
        self.E_mod = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, K)
        )
        self.sigma = nn.Parameter(torch.ones(K))
        self.lambda_g = 0.6

    def forward(self, current_feat, hist_classifier_weight, labels):
        B, T, C = current_feat.size()
        P_prev_high_dim = hist_classifier_weight[labels] 
        P_prev = self.proto_proj(P_prev_high_dim)        
        P_prev_time = P_prev.unsqueeze(1).expand(-1, T, -1)
        
        alpha = F.softmax(self.E_mod(current_feat), dim=-1) 
        
        noise_sum = torch.zeros_like(current_feat) 
        for k in range(self.K):
            z_k = torch.randn_like(current_feat) * self.sigma[k]
            noise_sum += alpha[..., k:k+1] * z_k
            
        F_prev_modulated = self.E_trans(P_prev_time) + (self.lambda_g * noise_sum)
        F_compatible_history = F_prev_modulated + current_feat 
        
        return F_compatible_history

class MKAM(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.proj = nn.Linear(feature_dim, feature_dim)
    def forward(self, x):
        return self.proj(x)

class GatedKnowledgeAdapter(nn.Module):
    def __init__(self, feature_dim=64, rank=8):
        super().__init__()
        self.A = nn.Linear(feature_dim, rank, bias=False)
        self.B = nn.Linear(rank, feature_dim, bias=False)
        self.omega = nn.Parameter(torch.tensor(1.0))
    def forward(self, x_history):
        low_rank_feat = self.B(self.A(x_history))
        return self.omega * low_rank_feat

class CumulativeKnowledgeAggregation(nn.Module):
    def __init__(self, feature_dim=64, rank=128):
        super().__init__()
        self.mkam_current = MKAM(feature_dim)
        self.mkam_history = MKAM(feature_dim)
        self.gated_adapter = GatedKnowledgeAdapter(feature_dim, rank)
        self.scale = feature_dim ** -0.5

    def forward(self, f_current, f_history):
        f_hat_current = self.mkam_current(f_current)
        f_hat_history = self.mkam_history(f_history)
        f_tilde_history = self.gated_adapter(f_hat_history)
        
        attn_scores = torch.bmm(f_hat_current, f_tilde_history.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        history_injection = torch.bmm(attn_weights, f_tilde_history)
        
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
        loss_dir = F.mse_loss(o_current, o_history)
        
        norm_curr = F.normalize(o_current, p=2, dim=1)
        norm_hist = F.normalize(o_history, p=2, dim=1)
        sim_matrix = torch.matmul(norm_curr, norm_hist.t())
        pos_sim = torch.diag(sim_matrix) 
        
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        hard_neg_sim, _ = neg_sim_matrix.max(dim=1) 
        loss_con = F.relu(self.margin - (pos_sim - hard_neg_sim)).mean()

        beta_norm = torch.ones(batch_size, 1, device=device) / batch_size
        proxy_current = torch.sum(beta_norm * o_current, dim=0) 
        proxy_history = torch.sum(beta_norm * o_history, dim=0)
        loss_dis = F.mse_loss(proxy_current, proxy_history)

        return loss_dir + (self.lambda_con * loss_con) + (self.lambda_dis * loss_dis)

######################## Training Loop ########################
def train_harmony_task(args, model, train_loader, val_loader, mod, device, epochs, num_classes, patience, task_idx):
    print(f"\n   >>> [Harmony Complete] Training '{mod}' (Task {task_idx+1}) ...")
    model.set_active_task(task_idx)

    base_params = [p for name, p in model.named_parameters() if p.requires_grad and 'acfm' not in name and 'cka' not in name]
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
    align_criterion = HybridAlignmentLoss(lambda_con=0.8, lambda_dis=0.6, margin=0.3).to(device)
    lambda_align = args.lambda_align
    best_eval = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        accum = {"loss": 0, "ce": 0, "align": 0, "correct": 0, "total": 0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            raw_feats = model.encoders[mod](x) 
            feats_seq = raw_feats.transpose(1, 2) 
            loss_align = torch.tensor(0.0, device=device)

            if task_idx > 0:
                hist_weight = model.shared_head.fc.weight.detach()
                if hasattr(model.shared_head, 'fc'):
                    hist_weight = model.shared_head.fc.weight.detach()
                else:
                    hist_weight = model.shared_head.weight.detach()
                
                fake_history = model.acfm[mod](feats_seq, hist_weight, y)
                fused_seq = model.cka[mod](feats_seq, fake_history) 
                
                fused_time = fused_seq.transpose(1, 2) 
                fake_hist_time = fake_history.transpose(1, 2)

                z_curr = model.shared_backbone(fused_time)
                logits = model.shared_head(z_curr)
                z_history = model.shared_backbone(fake_hist_time)
                loss_align = align_criterion(z_curr, z_history)
            else:
                f_hat_curr = model.cka[mod].mkam_current(feats_seq)
                fused_time = f_hat_curr.transpose(1, 2)
                z_curr = model.shared_backbone(fused_time)
                logits = model.shared_head(z_curr)

            loss_ce = criterion(logits, y)
            total_loss = loss_ce + (lambda_align * loss_align)
            total_loss.backward()
            optimizer.step()

            accum["loss"] += total_loss.item()
            accum["ce"]   += loss_ce.item()
            accum["align"] += loss_align.item()
            accum["correct"] += (logits.argmax(1) == y).sum().item()
            accum["total"] += y.size(0)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_raw = model.encoders[mod](vx) 
                v_seq = v_raw.transpose(1, 2)   
                v_hat = model.cka[mod].mkam_current(v_seq) 
                v_time = v_hat.transpose(1, 2)  
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

    model.eval()
    omega_val = model.cka[mod].gated_adapter.omega.item()
    print("\n" + "🎯" + "="*50)
    print(f"   [GATING COLLAPSE ANALYSIS - Task: {mod}]")
    print(f"   => Learned Gate Parameter (\u03c9) : {omega_val:.6f}")
    if omega_val < 0.1:
        print("   🚨 WARNING: GATING COLLAPSE DETECTED!")
    print("   " + "="*50 + "\n")

def run_cv_harmony(args, data_cache):
    subj2label, folds = init_subjects_and_folds(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.order.split(",") if t.strip()]
    
    step_history = {}
    fold_scores = []

    for fi in range(len(folds)):
        print(f"\n{'='*20} Fold {fi+1}/{len(folds)} {'='*20}")
        
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=args.disable_dbn).to(device)
        
        if hasattr(model.shared_head, 'fc'):
            actual_cls_dim = model.shared_head.fc.weight.shape[1]
        else:
            actual_cls_dim = model.shared_head.weight.shape[1]
            
        model.acfm = nn.ModuleDict({
            k: HarmonyACFM(feature_dim=128, K=3, classifier_dim=actual_cls_dim).to(device) for k in model.encoders.keys()
        })
        model.cka = nn.ModuleDict({
            k: CumulativeKnowledgeAggregation(feature_dim=128, rank=128).to(device) for k in model.encoders.keys()
        })
        
        seen_mods = []
        eval_loader_cache = {}

        for ti, mod in enumerate(tasks):
            print(f"\n=== Harmony Task {ti+1}/{len(tasks)} : {mod} ===")
            
            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            tr_loader = DataLoader(SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[mod] = te_loader 

            train_harmony_task(args, model, tr_loader, te_loader, mod, device, 
                               args.epochs, args.num_classes, args.patience, ti)

            seen_mods.append(mod)
            print(f"\n--- Evaluation (Step {ti+1}) ---")
            scores = []
            
            for m_idx, m in enumerate(seen_mods):
                model.set_active_task(m_idx)
                model.eval()
                all_preds, all_targets = [], []
                
                with torch.no_grad():
                    for vx, vy in eval_loader_cache[m]:
                        vx, vy = vx.to(device), vy.to(device)
                        vf = model.encoders[m](vx)
                        v_seq = vf.transpose(1, 2)
                        v_hat = model.cka[m].mkam_current(v_seq) 
                        v_time = v_hat.transpose(1, 2) 
                        vz = model.shared_backbone(v_time)
                        v_logits = model.shared_head(vz)
                        
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
    print(f"\n🏆 Final Avg F1 across folds: {sum(fold_scores)/len(fold_scores):.2f}")

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
    ap.add_argument("--lambda_align", type=float, default=0.15)
    ap.add_argument("--disable_dbn", action='store_true')

    args = ap.parse_args()
    print(f"FOG Harmony Mode | Arguments: {', '.join(f'{k}={v}' for k, v in vars(args).items())}")
    
    global_cache = preload_all_subjects(CACHE_DIR)
    U.set_seed(args.seed)
    run_cv_harmony(args, global_cache)

if __name__ == "__main__":
    main()