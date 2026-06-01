import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Sampler

# CNN backbone and data pipeline imports
from encoder import WearGaitUniversal
from data_loader import preload_all_subjects, prepare_split, make_sync_loaders, build_subj2label_fog, make_stratified_folds
from utility import set_seed
from medcoss_core import FOG_Buffer_Dataset
from model.paths import FOG_CACHE, as_str

class ModalityBatchSampler(Sampler):
    """CL heterogeneous batch sampler; isolates modalities for safe stacking"""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.mod_indices = {}
        for idx in range(len(dataset)):
            is_buffer = idx >= len(dataset.native_dataset)
            if is_buffer:
                mod = dataset.buffer_modalities[idx - len(dataset.native_dataset)]
            else:
                mod = dataset.target_modality
            if mod not in self.mod_indices: self.mod_indices[mod] = []
            self.mod_indices[mod].append(idx)
            
    def __iter__(self):
        batches = []
        for mod, indices in self.mod_indices.items():
            if self.shuffle: random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size: batches.append(batch)
        if self.shuffle: random.shuffle(batches)
        return iter(batches)
        
    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.mod_indices.values())


def extract_kmeans_buffer(model, native_dataset, mod_order, device, current_mod, args, fold):
    print(f"      [Buffer] 正在为 {current_mod.upper()} 提取 K-Means 核心集...")
    model.eval()
    all_features = []
    sample_indices = []
    
    temp_dataset = FOG_Buffer_Dataset(
        target_modality=current_mod, native_dataset=native_dataset, 
        mod_order=mod_order, buffer_json_dir=None, past_tasks=None
    )
    temp_loader = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=False)
    model.set_active_modality(current_mod)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for idx_batch, batch in enumerate(temp_loader):
            # Buffer [B,T,C] -> CNN [B,C,T]
            x = batch['data'].transpose(1, 2).to(device)
            # K-means on intermediate features
            features = model.shared_backbone(model.encoders[current_mod](x))
            all_features.append(features.cpu().numpy())
            
            start_idx = idx_batch * args.batch_size
            sample_indices.extend(list(range(start_idx, start_idx + x.size(0))))
            
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
        
    buffer_path = os.path.join(args.save_dir, f"{current_mod}_buffer_seed{args.seed}_fold{fold}.json")
    with open(buffer_path, 'w') as f: json.dump({"buffer_indices": buffer_indices}, f)
    print(f"      [Buffer] 已保存 {len(buffer_indices)} 个特征锚点至 {os.path.basename(buffer_path)}。")


def evaluate_direct(model, eval_dataset, eval_mod, device, args):
    """End-to-end eval (no linear probe)"""
    model.eval()
    model.set_active_modality(eval_mod)
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['data'].transpose(1, 2).to(device)
            y = batch['label'].cpu().numpy()
            logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(y)
            
    return f1_score(all_targets, all_preds, average='macro') * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=as_str(FOG_CACHE))
    parser.add_argument('--order', type=str, default="skeleton,gyr,acc") # default modality order
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80) 
    
    parser.add_argument('--buffer_ratio', type=float, default=0.1)
    parser.add_argument('--num_centers', type=int, default=10)
    parser.add_argument('--lambda_distill', type=float, default=1.0)
    
    parser.add_argument('--win_len', type=int, default=120)
    parser.add_argument('--hop_len', type=int, default=15)
    parser.add_argument('--save_dir', type=str, default="./log/fog_baselines/cnn_medcoss/")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda')
    
    tasks = [t.strip().lower() for t in args.order.split(",")]
    print(">> 加载 FOG 全局缓存...")
    data_cache = preload_all_subjects(args.data_root)
    subj2label = build_subj2label_fog(os.path.join(args.data_root, "subj2label.json"))
    folds = make_stratified_folds(subj2label, n_folds=args.n_folds, seed=args.seed)
    
    R_matrix = np.zeros((args.n_folds, len(tasks), len(tasks)))
    criterion = nn.CrossEntropyLoss()
    
    for fold, (train_subs, test_subs) in enumerate(folds):
        print(f"\n{'='*70}\n 🌟 CNN-MEDCOSS FOLD {fold+1}/{args.n_folds} \n{'='*70}")
        
        # 🚨 Fairness: disable DBN (MedCoSS has no domain routing)
        student = WearGaitUniversal(num_classes=3, disable_dbn=True).to(device)
        teacher = None
        seen_tasks = []
        
        for task_idx, current_mod in enumerate(tasks):
            print(f"\n  >>> Task {task_idx}: CNN-MedCoSS Learning [{current_mod.upper()}] <<<")
            needed_mods = tuple(seen_tasks + [current_mod])
            
            prep = prepare_split(train_subs, test_subs, data_cache, win=args.win_len, hop=args.hop_len, modalities=needed_mods)
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=1, num_workers=0)

            dataset = FOG_Buffer_Dataset(
                target_modality=current_mod, native_dataset=tr_sync.dataset, 
                mod_order=list(needed_mods), buffer_json_dir=args.save_dir, 
                past_tasks=seen_tasks, seed=args.seed, fold=fold
            )
            
            sampler = ModalityBatchSampler(dataset, args.batch_size, shuffle=True)
            dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)
            
            if task_idx > 0:
                teacher = WearGaitUniversal(num_classes=3, disable_dbn=True).to(device)
                teacher.load_state_dict(student.state_dict(), strict=False)
                teacher.eval()
                for p in teacher.parameters(): p.requires_grad = False
                
            optimizer = optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            for ep in range(1, args.epochs + 1):
                student.train()
                t_ce, t_kd = 0, 0
                
                for batch in dataloader:
                    optimizer.zero_grad()
                    mod_str = batch['modality'][0] if isinstance(batch['modality'], list) else batch['modality']
                    is_past = (mod_str != current_mod)
                    
                    x = batch['data'].transpose(1, 2).to(device)
                    y = batch['label'].to(device)
                    student.set_active_modality(mod_str)
                    
                    with torch.amp.autocast('cuda'):
                        logits = student(x)
                        loss_ce = criterion(logits, y)
                        loss_kd = 0.0
                        
                        # 🌟 MedCoSS: historical feature distillation
                        if is_past and teacher is not None:
                            teacher.set_active_modality(mod_str)
                            with torch.no_grad():
                                feat_t = teacher.shared_backbone(teacher.encoders[mod_str](x))
                            feat_s = student.shared_backbone(student.encoders[mod_str](x))
                            loss_kd = F.mse_loss(feat_s, feat_t) * args.lambda_distill
                            
                        loss = loss_ce + loss_kd
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    t_ce += loss_ce.item()
                    if is_past: t_kd += loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd
                        
                scheduler.step()
                if ep <= 10 or ep % 10 == 0:
                    print(f"      [Ep {ep:03d}] CE: {t_ce/len(dataloader):.4f} | Feature KD: {t_kd/len(dataloader):.4f}")
            
            extract_kmeans_buffer(student, tr_sync.dataset, list(needed_mods), device, current_mod, args, fold)
            seen_tasks.append(current_mod)
            
            print(f"\n      --- End-to-End Evaluation (Post-{current_mod.upper()}) ---")
            for j, eval_mod in enumerate(seen_tasks):
                eval_dataset = FOG_Buffer_Dataset(
                    target_modality=eval_mod, native_dataset=te_sync.dataset, 
                    mod_order=list(needed_mods), buffer_json_dir=None, past_tasks=None,
                    seed=args.seed, fold=fold
                )
                f1_score_j = evaluate_direct(student, eval_dataset, eval_mod, device, args)
                R_matrix[fold, task_idx, j] = f1_score_j
                print(f"      [EVAL] Testing {eval_mod.upper()}: {f1_score_j:.2f}%")

    print("\n" + "="*70 + "\n 🏆 CNN-MEDCOSS FINAL METRIC MATRIX (R_N,N) \n" + "="*70)
    mean_R = np.mean(R_matrix, axis=0)
    std_R = np.std(R_matrix, axis=0)
    
    print("      [" + "]\t[".join([t.upper()[:4] for t in tasks]) + "]")
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


# 1. Ensure log directory exists
# mkdir -p ./log/fog_baselines/medcoss/

# # 2. Launch 5 concurrent jobs on GPU 0
# for s in 42 43 44 2 3; do
#     echo "Launch CNN-MedCoSS | Seed $s | GPU 0"
#     CUDA_VISIBLE_DEVICES=0 nohup python -u fog_medcoss.py --seed $s > ./log/fog_baselines/medcoss/seed_${s}.log 2>&1 &
    
#     # 5s stagger to avoid PKL I/O thrashing
#     sleep 0.5
# done