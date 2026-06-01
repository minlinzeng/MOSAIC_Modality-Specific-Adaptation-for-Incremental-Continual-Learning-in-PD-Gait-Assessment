import os
import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

# FBG MedCoSS core imports
from fbg_medcoss_core import FBG_Unified_Model, FBG_Buffer_Dataset
from data_loader import get_fbg_dataloaders
from fbg_utility import set_deterministic_seed
from model.paths import FBG_PROCESSED, as_str
import random
from torch.utils.data import Sampler

class ModalityBatchSampler(Sampler):
    """
    Continual-learning sampler for heterogeneous modalities.
    Groups indices by modality so each batch has consistent feature dims;
    shuffles batches for smooth experience replay.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Partition indices by modality
        self.mod_indices = {}
        for idx in range(len(dataset)):
            is_buffer = idx >= len(dataset.native_dataset)
            if is_buffer:
                buffer_idx = idx - len(dataset.native_dataset)
                mod = dataset.buffer_modalities[buffer_idx]
            else:
                mod = dataset.target_modality
                
            if mod not in self.mod_indices:
                self.mod_indices[mod] = []
            self.mod_indices[mod].append(idx)
            
    def __iter__(self):
        batches = []
        for mod, indices in self.mod_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            # Form full batches (implicit drop_last)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        
        # Shuffle batches to mix current task and replay buffer
        if self.shuffle:
            random.shuffle(batches)
            
        return iter(batches)
        
    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.mod_indices.values())

# =====================================================================
# Stage 1: MAE pretrain + teacher distillation
# =====================================================================
def train_medcoss_epoch(model, teacher_model, dataloader, optimizer, scaler, device, args, current_mod):
    model.train()
    total_loss, total_mae, total_distill = 0, 0, 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        batch_modality = batch['modality'] # 'linear', 'angular', or 'grf'
        if isinstance(batch_modality, (list, tuple)):
            batch_modality = batch_modality[0]
        is_past_task = (batch_modality != current_mod)
        
        # Move data to GPU before any math
        x_data = batch['data'].to(device)
        
        with torch.amp.autocast('cuda'):
            # [A] Past task: feature distillation + IMM
            if is_past_task and teacher_model is not None:
                # IMM: mix with permuted batch
                N = x_data.size(0)
                # Keep perm on GPU to avoid cross-device indexing
                perm = torch.randperm(N, device=device) 
                lambda_val = torch.rand(N, 1, 1, device=device)
                
                # All tensors on same device
                mixed_data = lambda_val * x_data + (1 - lambda_val) * x_data[perm]
                
                input_dict = {'data': mixed_data, 'modality': batch_modality}
                
                # Student/teacher hidden features
                feat_s, noise = model(input_dict, mask_ratio=args.mask_ratio, feature=True)
                with torch.no_grad():
                    feat_t, _ = teacher_model(input_dict, mask_ratio=args.mask_ratio, feature=True, noise=noise)
                
                loss = ((feat_t.detach() - feat_s) ** 2).mean() * args.lambda_distill
                total_distill += loss.item()
                
            # [B] Current task: MAE masked reconstruction
            else:
                input_dict = {'data': x_data, 'modality': current_mod}
                (loss, _), _, _, _ = model(input_dict, mask_ratio=args.mask_ratio)
                total_mae += loss.item()
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    n = len(dataloader)
    return total_loss / n, total_mae / max(1, n), total_distill / max(1, n)

# =====================================================================
# Stage 2: K-Means buffer extraction
# =====================================================================

def extract_kmeans_buffer(model, native_dataset, device, current_mod, args, fold):
    print(f"      [Buffer] Extracting K-Means core set for {current_mod.upper()}...")
    model.eval()
    all_features = []
    sample_indices = []
    
    # Sequential loader, no shuffle; track absolute dataset indices
    temp_loader = DataLoader(native_dataset, batch_size=args.batch_size, shuffle=False)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for idx_batch, raw_batch in enumerate(temp_loader):
            # Read modality from native dict batch
            x = raw_batch[current_mod].to(device)
            input_dict = {'data': x, 'modality': current_mod}
            
            features, _ = model(input_dict, mask_ratio=0.0, feature=True)
            all_features.append(features.mean(1).cpu().numpy())
            
            # Absolute indices in native dataset
            start_idx = idx_batch * args.batch_size
            batch_indices = list(range(start_idx, start_idx + x.size(0)))
            sample_indices.extend(batch_indices)
            
    all_features = np.concatenate(all_features, axis=0)
    sample_indices = np.array(sample_indices)
    
    total_samples = all_features.shape[0]
    buffer_size = max(1, int(total_samples * args.buffer_ratio))
    num_centers = min(args.num_centers, buffer_size, total_samples)
    samples_per_center = max(1, buffer_size // num_centers)
    
    kmeans = KMeans(n_clusters=num_centers, n_init='auto', random_state=args.seed)
    kmeans.fit(all_features)
    distances = np.linalg.norm(all_features - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    
    buffer_indices = []
    for i in range(kmeans.n_clusters):
        cluster_distances = distances[kmeans.labels_ == i]
        cluster_dataset_indices = sample_indices[kmeans.labels_ == i]
        
        top_k_local = cluster_distances.argsort()[:samples_per_center]
        buffer_indices.extend(cluster_dataset_indices[top_k_local].tolist())
        
    # Buffer JSON with fold/seed suffix
    buffer_path = os.path.join(args.save_dir, f"{current_mod}_buffer_seed{args.seed}_fold{fold}.json")
    with open(buffer_path, 'w') as f:
        json.dump({"buffer_indices": buffer_indices}, f)
    print(f"      [Buffer] Saved {len(buffer_indices)} anchor indices to {os.path.basename(buffer_path)}.")


# =====================================================================
# Stage 3: linear probe evaluation
# =====================================================================
def evaluate_linear_probe(encoder, eval_mod, eval_dataset, device, args):
    """Freeze backbone; train linear head only."""
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
        
    head = nn.Linear(768, 2).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Minimal probe on eval set to check representation quality
    # Standard practice uses a small subset; here we train 10 epochs on full eval set
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    
    for ep in range(10):
        for batch in loader:
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            input_dict = {'data': x, 'modality': eval_mod}
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                feats, _ = encoder(input_dict, mask_ratio=0.0, feature=True)
                cls_feat = feats[:, 0, :]  # CLS token
                
            logits = head(cls_feat)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Inference
    head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch['data'].to(device), batch['label'].cpu().numpy()
            input_dict = {'data': x, 'modality': eval_mod}
            feats, _ = encoder(input_dict, mask_ratio=0.0, feature=True)
            logits = head(feats[:, 0, :])
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y)
            
    # Restore requires_grad for later MAE training
    for p in encoder.parameters(): p.requires_grad = True
    return f1_score(all_labels, all_preds, average='macro') * 100.0


# =====================================================================
# Main MedCoSS pipeline
# =====================================================================
def get_all_subjects(data_root):
    import glob
    files = glob.glob(os.path.join(data_root, "*.pkl"))
    return sorted(list(set([os.path.basename(f).split('_')[0] for f in files])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=as_str(FBG_PROCESSED))
    parser.add_argument('--order', type=str, default="linear,angular,grf")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=80)  # MAE needs more epochs
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    
    # Buffer and distillation
    parser.add_argument('--buffer_ratio', type=float, default=0.1)
    parser.add_argument('--num_centers', type=int, default=10)
    parser.add_argument('--lambda_distill', type=float, default=2.0)
    
    # Windowing
    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--step_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default="./logs_fbg_ablations/medcoss/medcoss_logs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_deterministic_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.cuda.amp.GradScaler()
    
    tasks = [t.strip().lower() for t in args.order.split(",")]
    subjects = get_all_subjects(args.data_root)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    R_matrix = np.zeros((5, len(tasks), len(tasks)))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        print(f"\n{'='*70}\n 🌟 INITIATING MEDCOSS FOLD {fold+1}/5 \n{'='*70}")
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        native_train_loader, native_test_loader = get_fbg_dataloaders(
            args.data_root, train_subjects, test_subjects, 
            batch_size=args.batch_size, window_size=args.window_size, step_size=args.step_size
        )
        
        # Native datasets for Buffer_Dataset
        native_train_dataset = native_train_loader.dataset
        native_test_dataset = native_test_loader.dataset

        # Student (+ teacher when needed)
        student = FBG_Unified_Model(is_teacher=False).to(device)
        teacher = None
        
        seen_tasks = []
        for task_idx, current_mod in enumerate(tasks):
            print(f"\n  >>> Task {task_idx}: MedCoSS Learning [{current_mod.upper()}] <<<")
            
            # 1. Dataset with replay buffer (T0: empty past_tasks)
            dataset = FBG_Buffer_Dataset(
                target_modality=current_mod, 
                native_dataset=native_train_dataset,
                buffer_json_dir=args.save_dir, 
                past_tasks=seen_tasks,
                seed=args.seed,   
                fold=fold
            )
            
            # Modality-isolated batch sampler
            sampler = ModalityBatchSampler(dataset, args.batch_size, shuffle=True)
            
            # With batch_sampler, omit batch_size/shuffle/drop_last
            dataloader = DataLoader(
                dataset, 
                batch_sampler=sampler, 
                num_workers=4,        
                pin_memory=True       
            )
            # 2. Teacher model
            if task_idx > 0:
                teacher = FBG_Unified_Model(is_teacher=True).to(device)
                teacher.load_state_dict(student.state_dict(), strict=False)
                teacher.eval()
                for p in teacher.parameters(): p.requires_grad = False
                
            # 3. MAE + distillation loop
            optimizer = optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.05)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            for ep in range(1, args.epochs + 1):
                t_loss, t_mae, t_dis = train_medcoss_epoch(student, teacher, dataloader, optimizer, scaler, device, args, current_mod)
                scheduler.step()
                if ep <= 10 or ep % 5 == 0:
                    print(f"      [Ep {ep:03d}] Loss: {t_loss:.4f} | MAE: {t_mae:.4f} | Distill: {t_dis:.4f}", flush=True)
            
            # 4. K-Means buffer for current task
            extract_kmeans_buffer(student, native_train_dataset, device, current_mod, args, fold)
            
            seen_tasks.append(current_mod)
            
            # 5. Linear probe eval
            print(f"\n      --- Linear Probe Evaluation (Post-{current_mod.upper()}) ---")
            for j, eval_mod in enumerate(seen_tasks):
                eval_dataset = FBG_Buffer_Dataset(
                    target_modality=eval_mod, 
                    native_dataset=native_test_dataset, 
                    buffer_json_dir=None, past_tasks=None,
                    seed=args.seed, fold=fold
                )
                f1_score_j = evaluate_linear_probe(student, eval_mod, eval_dataset, device, args)
                R_matrix[fold, task_idx, j] = f1_score_j
                print(f"      [EVAL] Testing {eval_mod.upper()}: {f1_score_j:.2f}%")

    print("\n" + "="*70 + "\n 🏆 MEDCOSS FINAL METRIC MATRIX (R_N,N) \n" + "="*70)
    mean_R = np.mean(R_matrix, axis=0)
    std_R = np.std(R_matrix, axis=0)
    
    print("      [" + "]\t[".join([t.upper()[:3] for t in tasks]) + "]")
    for i in range(len(tasks)):
        row_str = f"T{i}:  "
        for j in range(len(tasks)):
            if j <= i: row_str += f"{mean_R[i,j]:.1f}±{std_R[i,j]:.1f}\t"
            else: row_str += "-----\t"
        print(row_str)
        
    bwt = np.mean([mean_R[-1, j] - mean_R[j, j] for j in range(len(tasks) - 1)])
    avg_acc = np.mean(mean_R[-1, :])
    print(f"\nFinal Average F1 (A_N): {avg_acc:.2f}%")
    print(f"Backward Transfer (BWT): {bwt:.2f}%")

if __name__ == "__main__":
    main()