import os
import argparse
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import f1_score
from pathlib import Path
import sys

# Ensure paths are correct
current_file = Path(__file__).resolve()
project_root = current_file.parents[5]
sys.path.append(str(project_root))

from medcoss_model.Unimodel import Unified_Model
import model.weargait.ewc.utility as U
from model.weargait.ewc.data_loader import preload_all_subjects, prepare_split, make_sync_loaders, make_fixed_balanced_folds_no_overlap, build_subj2label
from model.weargait.ewc.config import Config

class LinearProbe_Model(nn.Module):
    def __init__(self, pretrained_model_path, num_classes=2, **kwargs):
        super().__init__()
        
        self.encoder = Unified_Model(
            now_1D_input_size=(112, 1), 
            now_2D_input_size=(512, 512), 
            now_3D_input_size=(16, 192, 192)
        )
        
        checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        
        # Smart Filter to handle dimension upgrades across CL steps
        encoder_dict = self.encoder.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in encoder_dict and v.shape == encoder_dict[k].shape:
                filtered_dict[k] = v

        self.encoder.load_state_dict(filtered_dict, strict=False)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.head = nn.Linear(768, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, data):
        with torch.no_grad():
            latent_features, _ = self.encoder(data, feature=True)
        cls_feature = latent_features[:, 0, :]
        logits = self.head(cls_feature)
        return logits

def _scan_subjects(dir_path: Path):
    return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})

def train_linear_probe(args):
    print(f"\n🚀 Starting 5-Fold Linear Probe Evaluation - Step {args.eval_step} (Seed {args.seed})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset Initialization
    data_cache = preload_all_subjects()
    pd_ids = _scan_subjects(Config.PD_PATH)
    hc_ids = _scan_subjects(Config.HC_PATH)
    subj2label = build_subj2label(pd_ids, hc_ids)
    
    n_folds = 5
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=n_folds, seed=args.seed)
    
    # 2. Setup Tasks
    mod_map = {'1D_text': 'imu', '2D_xray': 'walkway', '2D_path': 'insole'}
    seen_medcoss_tasks = args.seen_tasks.split(',')
    seen_native_tasks = [mod_map[t] for t in seen_medcoss_tasks]
    
    # Dictionary to hold the F1 scores across all folds for each task
    cross_fold_scores = {mod: [] for mod in seen_medcoss_tasks}
    
    # 3. 5-Fold Cross Validation Loop
    for fi in range(n_folds):
        print(f"\n--- Fold {fi+1}/{n_folds} ---")
        train_subs, test_subs = folds[fi]
        
        prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=120, hop=60, modalities=tuple(seen_native_tasks))
        tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=64, num_workers=4)
        
        # Instantiate a BRAND NEW head for every fold
        model = LinearProbe_Model(args.load_pretrained_weight, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=0.01)
        
        # Train the Probe  
        epochs = 50
        model.train()
        for ep in range(epochs):
            for step, batch in enumerate(tr_sync):
                # We know exactly what the collate function returns now!
                y = batch['y'].to(device)
                
                for i, native_mod in enumerate(seen_native_tasks):
                    # 'xs' contains a list of tensors matching the order of seen_native_tasks
                    x = batch['xs'][i]
                    
                    if x.dim() == 3: x = x.unsqueeze(1) 
                    x = x.to(device)
                    
                    data_dict = {'data': x, 'modality': seen_medcoss_tasks[i]}
                    
                    optimizer.zero_grad()
                    logits = model(data_dict)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

        # Evaluate the Fold
        model.eval()
        with torch.no_grad():
            for i, native_mod in enumerate(seen_native_tasks):
                medcoss_mod = seen_medcoss_tasks[i]
                all_preds, all_targets = [], []
                
                for batch in te_sync:
                    # Clean extraction directly from the collate keys
                    y = batch['y'].to(device)
                    x = batch['xs'][i]
                        
                    if x.dim() == 3: x = x.unsqueeze(1)
                    x = x.to(device)
                    
                    data_dict = {'data': x, 'modality': medcoss_mod}
                    
                    logits = model(data_dict)
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                    
                f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
                cross_fold_scores[medcoss_mod].append(f1)
                print(f"   {medcoss_mod} F1: {f1:.2f}")

    # 4. Calculate Final Averages for the Matrix
    print(f"\n{'='*50}")
    print(f"📊 FINAL 5-FOLD AVERAGES (Step {args.eval_step})")
    print(f"{'='*50}")
    
    avg_acc_total = 0
    for mod in seen_medcoss_tasks:
        mod_avg = sum(cross_fold_scores[mod]) / n_folds
        print(f"   -> {mod}: {mod_avg:.2f}")
        avg_acc_total += mod_avg
        
    print(f"   -> Avg Seen Tasks: {avg_acc_total / len(seen_medcoss_tasks):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_step", type=int, required=True)
    parser.add_argument("--seen_tasks", type=str, required=True)
    parser.add_argument("--load_pretrained_weight", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./log/linprobe")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    train_linear_probe(args)