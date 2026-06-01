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

# 引入我们重构的 FBG 核心
from fbg_medcoss_core import FBG_Unified_Model, FBG_Buffer_Dataset
from data_loader import get_fbg_dataloaders
from fbg_utility import set_deterministic_seed
from model.paths import FBG_PROCESSED, as_str
import random
from torch.utils.data import Sampler

class ModalityBatchSampler(Sampler):
    """
    连续学习专属的异构数据采样器。
    它会自动将数据按模态分组，确保每个 Batch 内部的物理维度绝对一致，
    同时在 Batch 级别打乱顺序，实现平滑的经验回放 (Experience Replay)。
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 预先将索引按模态进行物理隔离
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
            # 划分 Batch (自动实现 drop_last=True)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        
        # 打乱所有 Batch 的顺序，使得模型交替学习当前任务和历史缓冲
        if self.shuffle:
            random.shuffle(batches)
            
        return iter(batches)
        
    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.mod_indices.values())

# =====================================================================
# 🌟 阶段 1: MAE 预训练与教师蒸馏 (MedCoSS 训练引擎)
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
        
        # 🌟 核心修复：在进行任何数学运算前，统一将数据推入显存
        x_data = batch['data'].to(device)
        
        with torch.amp.autocast('cuda'):
            # [A] 历史任务：特征蒸馏 + IMM (Intra-Modal MixUp)
            if is_past_task and teacher_model is not None:
                # IMM: 与打乱的数据进行插值
                N = x_data.size(0)
                # 顺手把随机索引也分配到 GPU，避免索引时的跨设备拷贝
                perm = torch.randperm(N, device=device) 
                lambda_val = torch.rand(N, 1, 1, device=device)
                
                # 此时所有的张量都在 cuda:0 上，计算极其安全且迅速
                mixed_data = lambda_val * x_data + (1 - lambda_val) * x_data[perm]
                
                input_dict = {'data': mixed_data, 'modality': batch_modality}
                
                # 获取学生和老师的隐层特征
                feat_s, noise = model(input_dict, mask_ratio=args.mask_ratio, feature=True)
                with torch.no_grad():
                    feat_t, _ = teacher_model(input_dict, mask_ratio=args.mask_ratio, feature=True, noise=noise)
                
                loss = ((feat_t.detach() - feat_s) ** 2).mean() * args.lambda_distill
                total_distill += loss.item()
                
            # [B] 当前任务：MAE 掩码重建预训练
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
# 🌟 阶段 2: K-Means 核心集提取引擎
# =====================================================================

def extract_kmeans_buffer(model, native_dataset, device, current_mod, args, fold):
    print(f"      [Buffer] 正在为 {current_mod.upper()} 提取 K-Means 核心集...")
    model.eval()
    all_features = []
    sample_indices = []
    
    # 🌟 修复 2：使用临时顺序加载器，绝对禁止 Shuffle，追踪真实的绝对索引
    temp_loader = DataLoader(native_dataset, batch_size=args.batch_size, shuffle=False)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for idx_batch, raw_batch in enumerate(temp_loader):
            # 直接从底层字典提取所需模态，不经过 Buffer_Dataset
            x = raw_batch[current_mod].to(device)
            input_dict = {'data': x, 'modality': current_mod}
            
            features, _ = model(input_dict, mask_ratio=0.0, feature=True)
            all_features.append(features.mean(1).cpu().numpy())
            
            # 计算这批特征在原生数据集中的绝对索引 [0, 1, 2... N]
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
        
    # 🌟 修复 3：写入带隔离后缀的文件名
    buffer_path = os.path.join(args.save_dir, f"{current_mod}_buffer_seed{args.seed}_fold{fold}.json")
    with open(buffer_path, 'w') as f:
        json.dump({"buffer_indices": buffer_indices}, f)
    print(f"      [Buffer] 已保存 {len(buffer_indices)} 个特征锚点至 {os.path.basename(buffer_path)}。")


# =====================================================================
# 🌟 阶段 3: 线性探测评估引擎 (Linear Probe)
# =====================================================================
def evaluate_linear_probe(encoder, eval_mod, eval_dataset, device, args):
    """冻结 Backbone，仅训练一个分类头进行评估"""
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
        
    head = nn.Linear(768, 2).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 我们用 Eval Dataset 做极简的自监督 Probe（仅为验证表征质量）
    # 在标准的 Linear Probe 中，通常会划分一个极小的子集，这里我们直接用它本身跑 10 轮
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    
    for ep in range(10):
        for batch in loader:
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            input_dict = {'data': x, 'modality': eval_mod}
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                feats, _ = encoder(input_dict, mask_ratio=0.0, feature=True)
                cls_feat = feats[:, 0, :] # 取 CLS Token
                
            logits = head(cls_feat)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 正式推理
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
            
    # 清理现场，恢复需要梯度的状态以便后续 MAE
    for p in encoder.parameters(): p.requires_grad = True
    return f1_score(all_labels, all_preds, average='macro') * 100.0


# =====================================================================
# 🌟 终极总控流水线
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
    parser.add_argument('--epochs', type=int, default=80) # MAE 通常需要更长的轮数
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    
    # Buffer 与 蒸馏参数
    parser.add_argument('--buffer_ratio', type=float, default=0.1)
    parser.add_argument('--num_centers', type=int, default=10)
    parser.add_argument('--lambda_distill', type=float, default=2.0)
    
    # 物理参数
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
        
        # 剥离出底层的 Dataset 供 Buffer_Dataset 使用
        native_train_dataset = native_train_loader.dataset
        native_test_dataset = native_test_loader.dataset

        # 初始化双模型
        student = FBG_Unified_Model(is_teacher=False).to(device)
        teacher = None
        
        seen_tasks = []
        for task_idx, current_mod in enumerate(tasks):
            print(f"\n  >>> Task {task_idx}: MedCoSS Learning [{current_mod.upper()}] <<<")
            
            # 1. 构建带有 Buffer 机制的数据集 (如果是 T0，则 past_tasks 为空，不加载 JSON)
            dataset = FBG_Buffer_Dataset(
                target_modality=current_mod, 
                native_dataset=native_train_dataset,
                buffer_json_dir=args.save_dir, 
                past_tasks=seen_tasks,
                seed=args.seed,   
                fold=fold
            )
            
            # 🌟 修复：加载异构物理隔离采样器
            sampler = ModalityBatchSampler(dataset, args.batch_size, shuffle=True)
            
            # 注意：使用了 batch_sampler 后，必须移除 batch_size, shuffle, drop_last 参数
            dataloader = DataLoader(
                dataset, 
                batch_sampler=sampler, 
                num_workers=4,        
                pin_memory=True       
            )
            # 2. 准备 Teacher
            if task_idx > 0:
                teacher = FBG_Unified_Model(is_teacher=True).to(device)
                teacher.load_state_dict(student.state_dict(), strict=False)
                teacher.eval()
                for p in teacher.parameters(): p.requires_grad = False
                
            # 3. 训练循环 (MAE + Distillation)
            optimizer = optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.05)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            for ep in range(1, args.epochs + 1):
                t_loss, t_mae, t_dis = train_medcoss_epoch(student, teacher, dataloader, optimizer, scaler, device, args, current_mod)
                scheduler.step()
                if ep <= 10 or ep % 5 == 0:
                    print(f"      [Ep {ep:03d}] Loss: {t_loss:.4f} | MAE: {t_mae:.4f} | Distill: {t_dis:.4f}", flush=True)
            
            # 4. 提取并保存当前任务的 K-Means Buffer
            extract_kmeans_buffer(student, native_train_dataset, device, current_mod, args, fold)
            
            seen_tasks.append(current_mod)
            
            # 5. 线性探测评估 (Linear Probe)
            print(f"\n      --- Linear Probe Evaluation (Post-{current_mod.upper()}) ---")
            for j, eval_mod in enumerate(seen_tasks):
                eval_dataset = FBG_Buffer_Dataset(
                    target_modality=eval_mod, 
                    native_dataset=native_test_dataset, 
                    buffer_json_dir=None, past_tasks=None,
                    seed=args.seed, fold=fold  # 👈 新增
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