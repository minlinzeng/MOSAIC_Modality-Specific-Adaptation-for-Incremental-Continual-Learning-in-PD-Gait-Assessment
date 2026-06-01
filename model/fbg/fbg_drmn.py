import os
import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- FBG project imports ---
from data_loader import get_fbg_dataloaders
from encoder import MICL_CNN_PD_Model
from fbg_utility import set_deterministic_seed, EarlyStopping
from model.paths import FBG_PROCESSED, as_str

# =====================================================================
# 1. FBG architecture adapter (DRMN wrapper)
# =====================================================================
class FBGEncoderWrapper(nn.Module):
    def __init__(self, encoder, input_drop):
        super().__init__()
        self.encoder = encoder
        self.input_drop = input_drop
    def forward(self, x):
        return self.encoder(self.input_drop(x.permute(0, 2, 1)))

class FBGSharedBackbone(nn.Module):
    def __init__(self, res1, res2, res3):
        super().__init__()
        self.res1 = res1
        self.res2 = res2
        self.res3 = res3
    def forward(self, x):
        x = self.res1(x, task_id=0)
        x = self.res2(x, task_id=0)
        x = self.res3(x, task_id=0)
        return torch.mean(x, dim=2) 

class FBGSharedHead(nn.Module):
    def __init__(self, head, dropout, noise_std):
        super().__init__()
        self.fc = head 
        self.dropout = dropout
        self.noise_std = noise_std
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return self.fc(self.dropout(x))

class DRMNFBGModel(nn.Module):
    """FBG wrapper for DRMN; disable_msbn=True for a plain shared-BN network."""
    def __init__(self, d_model=64, num_tasks=3, dropout=0.3):
        super().__init__()
        self.base_model = MICL_CNN_PD_Model(d_model=d_model, num_tasks=num_tasks, dropout=dropout, disable_msbn=True)
        
        self.encoders = nn.ModuleDict({
            'linear': FBGEncoderWrapper(self.base_model.enc_lin, self.base_model.input_drop),
            'angular': FBGEncoderWrapper(self.base_model.enc_ang, self.base_model.input_drop),
            'grf': FBGEncoderWrapper(self.base_model.enc_grf, self.base_model.input_drop)
        })
        
        self.shared_backbone = FBGSharedBackbone(self.base_model.res1, self.base_model.res2, self.base_model.res3)
        self.shared_head = FBGSharedHead(self.base_model.head, self.base_model.dropout, self.base_model.noise_std)
        self.active_task = 0
        self.active_modality = None

    def set_active_task(self, task_idx):
        self.active_task = task_idx

    def set_active_modality(self, mod):
        self.active_modality = mod

    def forward(self, x):
        features = self.encoders[self.active_modality](x)
        z = self.shared_backbone(features)
        return self.shared_head(z)

# =====================================================================
# 2. DRMN manager (original weight-locking logic)
# =====================================================================
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
        
        total_params = 0
        total_locked = 0
        
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

# =====================================================================
# 3. FBG training loop and data routing
# =====================================================================
def train_drmn_task(args, model, drmn_manager, train_loader, val_loader, mod, task_id, device, epochs, patience):
    print(f"\n   >>> [DRMN] Training '{mod}' (Task {task_id}) with Hard Gradient Masking...")
    
    # Modality-specific learning rate
    lr = 2e-4 if mod == 'grf' else 1e-4

    drmn_manager.switch_task(task_id)
    for k in model.encoders.keys():
        for p in model.encoders[k].parameters():
            p.requires_grad = (k == mod) 
            
    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(active_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopper = EarlyStopping(patience=patience, min_delta=1e-4)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        model.set_active_task(task_id)
        model.set_active_modality(mod)
        
        accum = {"loss": 0, "correct": 0, "total": 0}

        # Unpack FBG dict batch format
        for batch in train_loader:
            x = batch[mod].to(device)
            y = batch['label'].to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # DRMN MAGIC
            drmn_manager.apply_gradient_mask()
            optimizer.step()

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
            for batch in val_loader:
                vx = batch[mod].to(device)
                vy = batch['label'].to(device)
                v_logits = model(vx)
                all_preds.extend(v_logits.argmax(1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        scheduler.step(val_f1)

        if ep % 5 == 0 or ep == 1:
            n = len(train_loader)
            print(f"      [Ep {ep:02d}] Tr_Loss: {accum['loss']/n:.3f} | Tr_Acc: {accum['correct']/accum['total']*100:.1f}% | Val_F1: {val_f1:.2f}%")

        if early_stopper(val_f1, model):
            print(f"      🛑 Convergence: Early Stop at Ep {ep}.")
            break

    model.load_state_dict(early_stopper.best_model_state)


def get_all_subjects(data_root):
    import glob
    files = glob.glob(os.path.join(data_root, "*.pkl"))
    subjects = sorted(list(set([os.path.basename(f).split('_')[0] for f in files])))
    return subjects

def run_cv_drmn(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tasks = [t.strip().lower() for t in args.order.split(",")]
    num_tasks = len(tasks)
    
    subjects = get_all_subjects(args.data_root)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    R_matrix = np.zeros((5, num_tasks, num_tasks))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        print(f"\n{'='*60}\n 🌟 INITIATING FOLD {fold+1}/5 \n{'='*60}")
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        # FBG multi-modality dict dataloader
        train_loader, test_loader = get_fbg_dataloaders(
            args.data_root, train_subjects, test_subjects, 
            batch_size=args.batch_size, window_size=args.window_size, step_size=args.step_size
        )
        
        model = DRMNFBGModel(d_model=64, num_tasks=num_tasks).to(device)
        drmn_manager = DRMN_Manager(model, lock_ratio=args.lock_ratio)
        
        seen_mods = []
        for task_idx, active_mod in enumerate(tasks):
            patience = 20 if active_mod == 'grf' else 15
            train_drmn_task(args, model, drmn_manager, train_loader, test_loader, 
                            active_mod, task_idx, device, args.epochs, patience)

            if task_idx < len(tasks) - 1:
                drmn_manager.update_relevance_map()

            seen_mods.append(active_mod)
            print(f"\n      --- Evaluation (Post-{active_mod.upper()}) ---")
            
            for j, eval_mod in enumerate(seen_mods):
                model.eval()
                model.set_active_task(j) 
                model.set_active_modality(eval_mod)
                drmn_manager.switch_task(j)
                
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        vx = batch[eval_mod].to(device)
                        vy = batch['label'].to(device)
                        v_logits = model(vx)
                        all_preds.extend(v_logits.argmax(1).cpu().numpy())
                        all_targets.extend(vy.cpu().numpy())
                
                f1_score_j = f1_score(all_targets, all_preds, average='macro') * 100.0
                R_matrix[fold, task_idx, j] = f1_score_j
                print(f"      [EVAL] Testing {eval_mod.upper()}: {f1_score_j:.2f}%")

    print("\n" + "="*60 + "\n 🏆 FINAL METRIC MATRIX (R_N,N) \n" + "="*60)
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
    print(f"\nFinal Average F1 (A_N): {avg_acc:.2f}%")
    print(f"Backward Transfer (BWT): {bwt:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default=as_str(FBG_PROCESSED))
    ap.add_argument('--order', type=str, default="linear,angular,grf")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    
    # Sliding-window alignment (physical time)
    ap.add_argument('--window_size', type=int, default=256)
    ap.add_argument('--step_size', type=int, default=64)
    
    # DRMN-specific
    ap.add_argument("--lock_ratio", type=float, default=0.4, help="Percentage of free weights to lock per task")
    
    args = ap.parse_args()
    set_deterministic_seed(args.seed)
    
    run_cv_drmn(args)