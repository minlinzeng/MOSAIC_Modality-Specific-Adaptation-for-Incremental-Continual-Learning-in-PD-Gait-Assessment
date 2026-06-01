import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model.weargait.ewc.config import Config
import model.weargait.ewc.utility as U
from model.weargait.ewc.data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    make_fixed_balanced_folds_no_overlap, build_subj2label
)
from model.weargait.ewc.encoder import WearGaitUniversal

def get_conv_grads(model):
    """
    🚨 Extract conv/linear grads from shared backbone only
    Filter BN/bias (dim==1); measure representation conflict.
    """
    grads = []
    for p in model.shared_backbone.parameters():
        if p.grad is not None and p.dim() > 1: 
            grads.append(p.grad.clone().detach().flatten())
    if len(grads) == 0:
        raise ValueError("No valid gradients found! Check your routing.")
    return torch.cat(grads)

def run_gradient_conflict_analysis(args, data_cache):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Mixed-modality DataLoader
    def _scan_subjects(dir_path: Path):
        return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})
    pd_ids, hc_ids = _scan_subjects(Config.PD_PATH), _scan_subjects(Config.HC_PATH)
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    
    train_subs, test_subs = folds[0] 
    
    modalities = ['walkway', 'insole', 'imu']
    prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=modalities)
    tr_sync, _ = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(tr_sync.dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Initialize dual models
    # Model 1: shared BN (variance collapse)
    model_shared = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=True).to(device)
    opt_shared = torch.optim.Adam(model_shared.parameters(), lr=args.lr)
    
    # Model 2: MSBN (isolated stats)
    model_msbn = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=False).to(device)
    opt_msbn = torch.optim.Adam(model_msbn.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    history = {'shared': {'w_i': [], 'w_m': [], 'i_m': []}, 'msbn': {'w_i': [], 'w_m': [], 'i_m': []}}

    print("\n" + "="*60)
    print("🚀 [Motivation Analysis] Commencing REAL Gradient Tug-of-War...")
    print("="*60)

    # 3. Main tracking loop
    for ep in range(1, args.epochs + 1):
        model_shared.train()
        model_msbn.train()
        
        ep_sim_shared = {'w_i': [], 'w_m': [], 'i_m': []}
        ep_sim_msbn   = {'w_i': [], 'w_m': [], 'i_m': []}

        for batch in train_loader:
            x_walk = batch["xs"][0].to(device)
            x_inso = batch["xs"][1].to(device)
            x_imu  = batch["xs"][2].to(device)
            y      = batch["y"].to(device)
            
            # =================================================================
            # 🔥 [A] Shared BN: stat collision and gradient conflict
            # =================================================================
            # --- Step A1: contaminated per-modality grads ---
            model_shared.zero_grad()
            f_w_s = model_shared.encoders['walkway'](x_walk)
            f_i_s = model_shared.encoders['insole'](x_inso)
            f_m_s = model_shared.encoders['imu'](x_imu)
            
            # 🚨 Concat batches -> mixed BN variance
            f_mixed = torch.cat([f_w_s, f_i_s, f_m_s], dim=0)
            z_mixed = model_shared.shared_backbone(f_mixed)
            
            # Split modalities after mixed BN contamination
            z_w_s, z_i_s, z_m_s = torch.split(z_mixed, [x_walk.size(0), x_inso.size(0), x_imu.size(0)], dim=0)

            # Per-modality backward (retain_graph as needed)
            loss_w_s = criterion(model_shared.shared_head(z_w_s), y)
            loss_w_s.backward(retain_graph=True)
            g_shared_w = get_conv_grads(model_shared)
            model_shared.zero_grad()
            
            loss_i_s = criterion(model_shared.shared_head(z_i_s), y)
            loss_i_s.backward(retain_graph=True)
            g_shared_i = get_conv_grads(model_shared)
            model_shared.zero_grad()
            
            loss_m_s = criterion(model_shared.shared_head(z_m_s), y)
            loss_m_s.backward() # Free contaminated graph
            g_shared_m = get_conv_grads(model_shared)
            model_shared.zero_grad()

            # Shared-group cosine conflict
            ep_sim_shared['w_i'].append(F.cosine_similarity(g_shared_w.unsqueeze(0), g_shared_i.unsqueeze(0)).item())
            ep_sim_shared['w_m'].append(F.cosine_similarity(g_shared_w.unsqueeze(0), g_shared_m.unsqueeze(0)).item())
            ep_sim_shared['i_m'].append(F.cosine_similarity(g_shared_i.unsqueeze(0), g_shared_m.unsqueeze(0)).item())
            
            # --- Step A2: single mixed loss step ---
            opt_shared.zero_grad()
            f_w_clean = model_shared.encoders['walkway'](x_walk)
            f_i_clean = model_shared.encoders['insole'](x_inso)
            f_m_clean = model_shared.encoders['imu'](x_imu)
            f_mixed_clean = torch.cat([f_w_clean, f_i_clean, f_m_clean], dim=0)
            
            z_mixed_clean = model_shared.shared_backbone(f_mixed_clean)
            logits_mixed = model_shared.shared_head(z_mixed_clean) # shape [3B, num_classes]
            
            y_mixed = y.repeat(3) # Repeat labels for [3B]
            loss_total_shared = criterion(logits_mixed, y_mixed)
            loss_total_shared.backward() 
            opt_shared.step()

            # =================================================================
            # 🛡️ [B] MSBN: isolated BN routing
            # =================================================================
            # MSBN uses set_active_task per modality
            
            # --- Step B1: clean per-modality grads ---
            # Walkway
            model_msbn.zero_grad()
            model_msbn.set_active_task(0)
            loss_w_m = criterion(model_msbn.shared_head(model_msbn.shared_backbone(model_msbn.encoders['walkway'](x_walk))), y)
            loss_w_m.backward()
            g_msbn_w = get_conv_grads(model_msbn)
            model_msbn.zero_grad()

            # Insole
            model_msbn.set_active_task(1)
            loss_i_m = criterion(model_msbn.shared_head(model_msbn.shared_backbone(model_msbn.encoders['insole'](x_inso))), y)
            loss_i_m.backward()
            g_msbn_i = get_conv_grads(model_msbn)
            model_msbn.zero_grad()

            # IMU
            model_msbn.set_active_task(2)
            loss_m_m = criterion(model_msbn.shared_head(model_msbn.shared_backbone(model_msbn.encoders['imu'](x_imu))), y)
            loss_m_m.backward()
            g_msbn_m = get_conv_grads(model_msbn)
            model_msbn.zero_grad()

            # MSBN cosine similarity
            ep_sim_msbn['w_i'].append(F.cosine_similarity(g_msbn_w.unsqueeze(0), g_msbn_i.unsqueeze(0)).item())
            ep_sim_msbn['w_m'].append(F.cosine_similarity(g_msbn_w.unsqueeze(0), g_msbn_m.unsqueeze(0)).item())
            ep_sim_msbn['i_m'].append(F.cosine_similarity(g_msbn_i.unsqueeze(0), g_msbn_m.unsqueeze(0)).item())
            
            # --- Step B2: optimization step ---
            # MSBN: accumulate grads per modality
            opt_msbn.zero_grad()
            model_msbn.set_active_task(0)
            criterion(model_msbn.shared_head(model_msbn.shared_backbone(model_msbn.encoders['walkway'](x_walk))), y).backward()
            model_msbn.set_active_task(1)
            criterion(model_msbn.shared_head(model_msbn.shared_backbone(model_msbn.encoders['insole'](x_inso))), y).backward()
            model_msbn.set_active_task(2)
            criterion(model_msbn.shared_head(model_msbn.shared_backbone(model_msbn.encoders['imu'](x_imu))), y).backward()
            opt_msbn.step()

        # Log epoch averages
        for k in ['w_i', 'w_m', 'i_m']:
            history['shared'][k].append(np.mean(ep_sim_shared[k]))
            history['msbn'][k].append(np.mean(ep_sim_msbn[k]))

        if ep % 2 == 0 or ep == 1:
            s_wi, s_wm, s_im = history['shared']['w_i'][-1], history['shared']['w_m'][-1], history['shared']['i_m'][-1]
            m_wi, m_wm, m_im = history['msbn']['w_i'][-1], history['msbn']['w_m'][-1], history['msbn']['i_m'][-1]
            print(f"Ep {ep:02d} | Shared [W-I:{s_wi:+.3f} W-M:{s_wm:+.3f} I-M:{s_im:+.3f}] | MSBN [W-I:{m_wi:+.3f} W-M:{m_wm:+.3f} I-M:{m_im:+.3f}]")

    print("\n   🎨 Generating Motivation Figure...")
    plot_gradient_dynamics(history, args.epochs, save_path=Config.OUTPUT_DIR / "motivation_gradient_conflict.png")
    print(f"   ✅ Figure saved to: {Config.OUTPUT_DIR / 'motivation_gradient_conflict.png'}")


def plot_gradient_dynamics(history, epochs, save_path):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    x = np.arange(1, epochs + 1)
    labels = {'w_i': 'Walkway vs Insole', 'w_m': 'Walkway vs IMU', 'i_m': 'Insole vs IMU'}
    colors = {'w_i': '#1f77b4', 'w_m': '#ff7f0e', 'i_m': '#2ca02c'}
    
    # Tight y-axis limits
    all_vals = []
    for k in ['w_i', 'w_m', 'i_m']:
        all_vals.extend(history['shared'][k])
        all_vals.extend(history['msbn'][k])
    
    # Max abs value * margin
    y_bound = max(abs(min(all_vals)), abs(max(all_vals))) * 1.2
    if y_bound < 0.05: y_bound = 0.05 # floor on y bound
    if y_bound > 0.15: y_bound = 0.15 # cap y bound for outliers

    titles = ["(a) Conventional Shared BN: Gradient Tug-of-War", 
              "(b) Proposed Modality-Specific BN: Orthogonal Routing"]
    
    for i, (ax, model_type) in enumerate(zip(axes, ['shared', 'msbn'])):
        # 🎨 Red/green shaded regions
        # Red: negative transfer
        ax.axhspan(-y_bound, 0, color='#ffe6e6', alpha=0.8, zorder=0)
        # Green: orthogonal / positive transfer
        ax.axhspan(0, y_bound, color='#e6ffe6', alpha=0.5, zorder=0)
        
        # Zero reference line
        ax.axhline(0, color='red', linestyle='--', linewidth=2.5, zorder=1)

        # Plot with markers
        for k in ['w_i', 'w_m', 'i_m']:
            ax.plot(x, history[model_type][k], label=labels[k], color=colors[k], 
                    linewidth=2.5, marker='o', markersize=4, zorder=2)
            
        ax.set_title(titles[i], fontsize=15, fontweight='bold')
        ax.set_xlabel("Training Epochs", fontsize=14)
        ax.set_ylim(-y_bound, y_bound)
        
        # Three-decimal axis format
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        
        if i == 0:
            ax.set_ylabel("Gradient Cosine Similarity", fontsize=14)
            ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30) 
    parser.add_argument("--win_len", type=int, default=Config.WINDOW_SIZE)
    parser.add_argument("--hop_len", type=int, default=int(Config.WINDOW_SIZE * Config.STRIDE))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2)
    
    args = parser.parse_args()
    U.set_seed(args.seed)
    global_cache = preload_all_subjects(Config.OUTPUT_DIR)
    run_gradient_conflict_analysis(args, global_cache)