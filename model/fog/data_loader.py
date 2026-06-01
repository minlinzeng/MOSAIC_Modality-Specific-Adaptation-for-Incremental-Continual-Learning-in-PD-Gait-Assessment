import numpy as np
import pandas as pd
import torch
import random
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# ==================== 0. Global feature map (matches preprocessing_fog output) ====================
ACC_COLS = ['Acc_X', 'Acc_Y', 'Acc_Z']
GYR_COLS = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
SKEL_COLS = [f'Skel_{i}' for i in range(21)]

FEAT_MAP = {
    "acc": ACC_COLS,
    "gyr": GYR_COLS,
    "skeleton": SKEL_COLS
}

class GaitAugmenter:
    def __init__(self, p=0.5, jitter_sigma=0.05, scale_min=0.8, scale_max=1.2, mask_ratio=0.1):
        self.p = p                   
        self.sigma = jitter_sigma    
        self.scale_min = scale_min   
        self.scale_max = scale_max   
        self.mask_ratio = mask_ratio # temporal mask ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        aug_x = x.clone()
        # 1. Scale
        scale = torch.empty(1, 1).uniform_(self.scale_min, self.scale_max)
        aug_x = aug_x * scale
        # 2. Jitter
        noise = torch.randn_like(aug_x) * self.sigma
        aug_x = aug_x + noise
        
        # 3. 🚨 Strong regularization: temporal cutout
        # Zero out a contiguous time span
        seq_len = aug_x.size(1) # x is [Channels, Time]
        mask_len = int(seq_len * self.mask_ratio)
        if mask_len > 0:
            start = torch.randint(0, seq_len - mask_len, (1,)).item()
            aug_x[:, start:start+mask_len] = 0.0
            
        return aug_x

        
# ==================== 1. Global in-memory cache (I/O speedup) ====================
def preload_all_subjects(data_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all aligned 30Hz PKL files into RAM.
    Expected: data_dir / "sub01-1_acc_raw.pkl"
    """
    print(f">> 🚀 [I/O] 预加载 FOG 流形数据至内存: {data_dir}...")
    cache = {}
    files = list(Path(data_dir).glob("*.pkl"))
    
    if not files:
        raise FileNotFoundError(f"❌ 严重错误: {data_dir} 下找不到 .pkl 文件，请检查预处理步骤。")

    count = 0
    for f in files:
        parts = f.name.replace("_raw.pkl", "").split("_")
        if len(parts) != 2: continue
        
        sid = parts[0] # sub01-1
        mod = parts[1] # acc, gyr, skeleton
        
        if sid not in cache: cache[sid] = {}
        try:
            cache[sid][mod] = pd.read_pickle(f)
        except Exception as e:
            print(f"⚠️ [警告] 无法读取 {f.name}: {e}")
            
    # Drop sessions missing any of the three modalities
    valid_cache = {sid: mods for sid, mods in cache.items() if len(mods) == 3}
    print(f">> ✅ 成功缓存 {len(valid_cache)} 个完整多模态 Sessions。")
    return valid_cache

# ==================== 2. Global fold statistics and normalization ====================
def calc_fold_stats(train_subs: List[str], global_cache: Dict, modalities: Tuple[str]) -> Dict[str, Tuple[float, float]]:
    """
    Compute train-fold mean/std only to avoid leakage.
    """
    print(f"  > 🧮 [Math] 严格计算当前 Fold 训练分布参数...")
    sums, sumsqs, counts = defaultdict(float), defaultdict(float), defaultdict(int)

    for sid in train_subs:
        if sid not in global_cache: continue
        for mod in modalities:
            if mod not in global_cache[sid]: continue
            
            df = global_cache[sid][mod]
            cols = FEAT_MAP[mod]
            arr = df[cols].fillna(0).to_numpy(dtype=np.float32)
            
            if arr.shape[0] == 0: continue
            
            for i, c in enumerate(cols):
                vals = arr[:, i]
                sums[c] += float(vals.sum())
                sumsqs[c] += float(np.dot(vals, vals))
                counts[c] += vals.size

    stats = {}
    for c, n in counts.items():
        if n == 0: continue
        mean = sums[c] / n
        var = max((sumsqs[c] / n) - mean**2, 0.0)
        stats[c] = (mean, max(np.sqrt(var), 1e-6))
    return stats

# ==================== 3. Sliding-window dataset ====================
class FOGLazyDataset(Dataset):
    """
    FOG continual-learning dataset.
    At 30Hz, win_len=120 is a 4s gait cycle.
    """
    def __init__(self, subject_ids: List[str], global_cache: Dict, stats: Dict, 
                 modalities: Tuple[str], win_len: int = 120, hop_len: int = 15, 
                 subj2label: Dict = None, mode: str = 'train'):
        self.cache = global_cache
        self.stats = stats
        self.modalities = modalities
        self.win = win_len
        self.hop = hop_len
        self.subj2label = subj2label or {}
        self.mode = mode
        self.augmenter = GaitAugmenter(p=0.5) if mode == 'train' else None
        
        self.indices = []
        
        # Strict sliding windows
        for sid in subject_ids:
            if sid not in self.cache: continue
            
            # Aligned preprocessing: any modality row count works
            seq_len = len(self.cache[sid][modalities[0]])
            if seq_len < self.win:
                continue # Skip windows shorter than 4s
                
            n_windows = int((seq_len - self.win) // self.hop + 1)
            for i in range(n_windows):
                start_idx = i * self.hop
                self.indices.append((sid, start_idx))

        self.labels = [self.subj2label.get(sid, -1) for sid, _ in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sid, start = self.indices[idx]
        y = self.subj2label.get(sid, -1)
        xs = []
        
        for mod in self.modalities:
            df = self.cache[sid][mod]
            window_df = df.iloc[start : start + self.win]
            cols = FEAT_MAP[mod]
            
            arr = window_df[cols].to_numpy(dtype=np.float32)
            
            # Z-score normalization
            means = np.array([self.stats.get(c, (0.0, 1.0))[0] for c in cols], dtype=np.float32)
            stds  = np.array([self.stats.get(c, (0.0, 1.0))[1] for c in cols], dtype=np.float32)
            stds[stds == 0] = 1.0 
            
            arr = (arr - means) / stds
            
            # Tensor (C, T) for ResBlock1D
            tensor = torch.tensor(arr, dtype=torch.float32).transpose(0, 1) 
            if self.augmenter is not None:
                tensor = self.augmenter(tensor)
            xs.append(tensor)
            
        return {"xs": xs, "y": torch.tensor(y, dtype=torch.long), "sid": sid}

# ==================== 4. CV splits and labels ====================
def build_subj2label_fog(json_path: str) -> Dict[str, int]:
    """Load preprocessed subj2label JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)

def make_stratified_folds(subj2label: Dict[str, int], n_folds: int = 5, seed: int = 42):
    """
    Stratified subject-level K-fold CV for balanced severity per fold.
    No subject appears in both train and test.
    """
    rng = random.Random(seed)
    
    # Group by subject base (-1/-2 suffix) to prevent trial leakage
    subject_bases = defaultdict(list)
    base_labels = {}
    
    for sid, label in subj2label.items():
        base_id = sid.split("-")[0] # sub01
        subject_bases[base_id].append(sid)
        base_labels[base_id] = label # Assume consistent label per subject base

    # Stratify by class
    class_groups = defaultdict(list)
    for base_id, label in base_labels.items():
        class_groups[label].append(base_id)
        
    for label in class_groups:
        rng.shuffle(class_groups[label])

    folds = []
    for f in range(n_folds):
        test_bases = []
        for label, bases in class_groups.items():
            # Distribute each class across folds
            chunk_size = max(1, len(bases) // n_folds)
            start_idx = f * chunk_size
            end_idx = start_idx + chunk_size if f < n_folds - 1 else len(bases)
            test_bases.extend(bases[start_idx:end_idx])
            
        train_bases = [b for b in base_labels.keys() if b not in test_bases]
        
        # Expand to session IDs
        train_subs = [sid for base in train_bases for sid in subject_bases[base]]
        test_subs = [sid for base in test_bases for sid in subject_bases[base]]
        
        folds.append((train_subs, test_subs))
        
    return folds

# ==================== 5. Training API wrappers ====================
def prepare_split(train_subs, test_subs, data_cache, win: int = 120, hop: int = 15, modalities=("acc", "gyr", "skeleton")):
    stats = calc_fold_stats(train_subs, data_cache, modalities)
    return {
        "train_subs": train_subs,
        "test_subs": test_subs,
        "stats": stats,
        "cache": data_cache,
        "win": win,
        "hop": hop,
        "modalities": modalities
    }

def make_sync_loaders(prep_data, subj2label, batch_size=64, num_workers=4, **kwargs):
    train_ds = FOGLazyDataset(prep_data["train_subs"], prep_data["cache"], prep_data["stats"],
                              modalities=prep_data["modalities"], win_len=prep_data["win"], 
                              hop_len=prep_data["hop"], subj2label=subj2label, mode='train')
    
    test_ds = FOGLazyDataset(prep_data["test_subs"], prep_data["cache"], prep_data["stats"],
                             modalities=prep_data["modalities"], win_len=prep_data["win"], 
                             hop_len=prep_data["hop"], subj2label=subj2label, mode='test')
    
    def collate(batch):
        ys = torch.stack([b['y'] for b in batch])
        sids = [b['sid'] for b in batch]
        num_mods = len(batch[0]['xs'])
        xs_batched = []
        for i in range(num_mods):
            mod_stack = torch.stack([b['xs'][i] for b in batch])
            xs_batched.append(mod_stack)
        return {"xs": xs_batched, "y": ys, "sid": sids}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, collate_fn=collate, pin_memory=True)
    
    return train_loader, test_loader

# WearGaitTrain-compatible dataset wrapper
class SingleModalityDataset(Dataset):
    def __init__(self, full_dataset, mod_index=0):
        self.ds = full_dataset
        self.mod_idx = mod_index
        self.labels = self.ds.labels # Expose labels for WeightedSampler
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        item = self.ds[idx]
        return item["xs"][self.mod_idx], item["y"]