import argparse
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for KD
import random
import numpy as np
from IJCAI_26.model.weargait.weargait_windows import (
    prepare_split,
    make_sync_loaders,
    make_fixed_balanced_folds_no_overlap,
    build_subj2label,
)
from IJCAI_26.model.weargait.weargait_encoder import WearGaitUniversal
from IJCAI_26.model.weargait.weargait_train import discover_pd_hc
from IJCAI_26.model.weargait.LwI import optimal_transport as ot


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN (conv algos)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Configuration for LwI (Learning without Isolation) ---
class OTConfig:
    def __init__(self, args, device):
        self.args=args
        self.layers = self.args.layers                
        self.ensemble_step = self.args.step       
        self.ensemble_step_diff = self.args.step_diff
        
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

class WearGaitLwITrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.data_root = Path(self.args.data_root)
        self.pd_dir = self.data_root / "PD"
        self.hc_dir = self.data_root / "HC"
        self.pkl_dir = self.data_root / "WearGait_preproc_SPmT_30Hz"
        
        self.model = None
        self.model_old = None
        self.ot_config = OTConfig(args, device)
        self.criterion = nn.CrossEntropyLoss()

    def recalibrate_bn(self, loader):
        """Update BN stats to match fused weights."""
        self.model.train()
        # Freeze weights to update only stats
        for p in self.model.parameters(): p.requires_grad = False
        print(">>> Recalibrating Batch Norm stats...")
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i > 50: break
                x = x.to(self.device)
                _ = self.model(x) 
        for p in self.model.parameters(): p.requires_grad = True
        print(">>> Recalibration Complete.")

    # --- KD Loss Helper ---
    def distillation_loss(self, new_logits, old_logits, T=2.0):
        """
        Knowledge Distillation Loss.
        Forces new model probabilities to match old model probabilities.
        """
        log_probs = F.log_softmax(new_logits / T, dim=1)
        targets = F.softmax(old_logits / T, dim=1)
        # kl_div expects input as log-probabilities and target as probabilities
        return F.kl_div(log_probs, targets, reduction='batchmean') * (T ** 2)

    def get_modality_loader(self, mod, split, subj2label, folds, fold_idx):
        modalities = ("walkway", "insole", "imu") 
        train_subs, test_subs = folds[fold_idx]
        
        prep = prepare_split(
            train_subs,
            test_subs,
            data_dir=self.pkl_dir,
            win=self.args.win_len,
            hop=self.args.hop_len,
            modalities=modalities,
        )
        tr, te = make_sync_loaders(
            prep,
            subj2label,
            batch_size=self.args.batch_size,
            modalities=modalities,
        )
        
        base_ds = tr.dataset if split == "train" else te.dataset
        mod_idx = modalities.index(mod)
        
        class SingleModDS(torch.utils.data.Dataset):
            def __init__(self, base, idx): self.base, self.idx = base, idx
            def __len__(self): return len(self.base)
            def __getitem__(self, i): 
                b = self.base[i]
                return b["xs"][self.idx], b["y"]
                
        return torch.utils.data.DataLoader(
            SingleModDS(base_ds, mod_idx), 
            batch_size=self.args.batch_size, 
            shuffle=(split == "train"),
            num_workers=4
        )

    def train_task(self, mod, loader):
        print(f"\n>>> Training Task: {mod} | Feat KD: {self.args.kd_lambda}")
        self.model.set_active_modality(mod)
        
        if self.model_old:
            self.model_old.eval()

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        ### NEW: Add Scheduler to decay LR from 1e-3 -> 0 smoothly
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0.0001) 
        
        mse_loss = nn.MSELoss()
        
        # Warmup configuration
        WARMUP_EPOCHS = 5
        
        for ep in range(self.args.epochs):
            
            # --- WARMUP LOGIC ---
            if self.model_old is not None and ep < WARMUP_EPOCHS:
                phase = "WARMUP"
                current_lambda = 0.0
                # Freeze Backbone & Head
                for p in self.model.shared_backbone.parameters(): p.requires_grad = False
                for p in self.model.shared_head.parameters(): p.requires_grad = False
            else:
                phase = "TRAIN"
                current_lambda = self.args.kd_lambda
                # Unfreeze everything
                for p in self.model.shared_backbone.parameters(): p.requires_grad = True
                for p in self.model.shared_head.parameters(): p.requires_grad = True

            total_loss = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                # 1. Get Embedding
                z = self.model.get_embedding(x)
                
                # 2. Forward New Backbone
                feat_new = self.model.forward_backbone(z)
                logits = self.model.forward_head(feat_new)
                
                loss = self.criterion(logits, y)

                # 3. CHIMERA DISTILLATION
                if self.model_old is not None and current_lambda > 0:
                    with torch.no_grad():
                        feat_old = self.model_old.forward_backbone(z)
                    
                    feat_new_norm = F.normalize(feat_new, p=2, dim=1)
                    feat_old_norm = F.normalize(feat_old, p=2, dim=1)
                    
                    loss_feat = mse_loss(feat_new_norm, feat_old_norm)
                    loss += current_lambda * loss_feat

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            ### NEW: Step the scheduler at the end of every epoch
            scheduler.step()

            if (ep+1) % 10 == 0:
                # Optional: Print current LR to verify it's dropping
                current_lr = scheduler.get_last_lr()[0] 
                print(f"   Epoch {ep+1}/{self.args.epochs} [{phase}] LR: {current_lr:.6f} Loss: {total_loss/len(loader):.4f}")
    
    def perform_lwi_fusion(self):
        if self.model_old is None:
            return

        print("\n>>> Performing LwI Fusion (Graph Matching)...")
        
        fused_dict = ot.get_wassersteinized_layers_modularized(
            self.ot_config,
            self.device,
            networks=[self.model_old, self.model], 
            ignore_keyword='encoders' 
        )
        
        current_state = self.model.state_dict()
        for layer_name, new_weight in fused_dict.items():
            if layer_name in current_state:
                current_state[layer_name].copy_(new_weight)
                
        self.model.load_state_dict(current_state)
        print(">>> Fusion Complete. Shared pathways updated.")

    def run_experiment(self):
        pd_ids, hc_ids = discover_pd_hc(self.pd_dir, self.hc_dir)
        subj2label = build_subj2label(pd_ids, hc_ids)
        folds = make_fixed_balanced_folds_no_overlap(
            pd_ids,
            hc_ids,
            n_folds=self.args.n_folds,
            per_class=self.args.test_per_class,
            seed=self.args.seed
        )

        task_order = ["walkway", "insole", "imu"] 
        # task_order = ["insole", "imu", "walkway"]

        for fold_idx in range(len(folds)):
            print(f"=== Starting Fold {fold_idx+1}/{len(folds)} ===")
            # self.model = WearGaitCL(enc_out_ch=12, backbone_dim=8, shared_out_ch=16, num_classes=2).to(self.device)
            self.model = WearGaitUniversal(
                enc_out_ch=12, 
                backbone_dim=8, 
                shared_out_ch=64, 
                num_classes=2
            ).to(self.device)
            self.model_old = None

            for task_idx, mod in enumerate(task_order):
                train_loader = self.get_modality_loader(mod, "train", subj2label, folds, fold_idx)
                self.train_task(mod, train_loader)

                if task_idx > 0:
                    self.perform_lwi_fusion()
                    self.recalibrate_bn(train_loader)

                print(f"--- Evaluation after Task {task_idx} ({mod}) ---")
                for seen_mod in task_order[:task_idx + 1]:
                    test_loader = self.get_modality_loader(seen_mod, "test", subj2label, folds, fold_idx)
                    acc = self.evaluate(seen_mod, test_loader)
                    print(f"   Task {seen_mod}: {acc:.2f}%")

                self.model_old = copy.deepcopy(self.model)
                self.model_old.eval()

    def evaluate(self, mod, loader):
        self.model.eval()
        self.model.set_active_modality(mod)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / max(1, total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/home/yongjie/Minlin/CIKM-2025-Minlin/PD_3D_motion-capture_data/WearGait")
    parser.add_argument('--win_len', type=int, default=64)
    parser.add_argument('--hop_len', type=int, default=64)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--test_per_class', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--step', type=float, default=0.3)
    parser.add_argument('--step_diff', type=float, default=0.6)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--kd_lambda', type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed) 
    print(f"Global seed set to {args.seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = WearGaitLwITrainer(args, device)
    trainer.run_experiment()

    """
    s42: seed 42, mf6: max fusion step 0.6
    nohup python -u -m IJCAI_26.model.weargait.LwI.ot_train > IJCAI_26/model/weargait/LwI/log/ot_s42_mf6.log 2>&1 &

    SEEDS=(2 3 4 42 43 44)
    STEPS_D=(0.3 0.6) # max fusion 0.6 0.7 0.8 0.9 1.0
    LAMBDAS=(0.5 0.9) # ld 0.5 0.6 0.7 0.9 1.0

    # Configuration
    SEEDS=(2 3 4 42 43 44)
    STEPS=(0.3)
    STEPS_D=(0.5) 
    LAMBDAS=(300)
    LAYERS=(14)
    MAX_JOBS=60 

    for s in "${SEEDS[@]}"; do
        for st in "${STEPS[@]}"; do
            for sd in "${STEPS_D[@]}"; do
                for ld in "${LAMBDAS[@]}"; do
                    for ly in "${LAYERS[@]}"; do
                        LOG_FILE="/home/yongjie/Minlin/IJCAI_26/model/weargait/LwI/log/wim_deep/warmup_mseNorm/ot_ld${ld}_sd${sd}_st${st}_ly${ly}_s${s}_mseNorm_45epochs.out"
                        
                        while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                            sleep 0.5
                        done

                        nohup python -u -m IJCAI_26.model.weargait.LwI.ot_train \
                            --seed "$s" \
                            --step "$st" \
                            --step_diff "$sd" \
                            --kd_lambda "$ld" \
                            --layers "$ly" \
                            > "$LOG_FILE" 2>&1 &
                        sleep 0.2 
                    done
                done
            done
        done
    done
    """