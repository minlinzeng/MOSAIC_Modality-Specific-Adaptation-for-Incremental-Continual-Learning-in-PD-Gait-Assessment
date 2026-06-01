import numpy as np
import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
from model.weargait.ewc.config import Config as C
import warnings
warnings.filterwarnings('ignore', r'All-NaN slice encountered') # Suppress expected dead-sensor warnings

# ==================== 1. GLOBAL RAW CACHE (Singleton) ====================
def preload_all_subjects(data_dir: Path = C.OUTPUT_DIR) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Loads ALL .pkl files into RAM once. 
    Returns: dict { 'subject_id': { 'walkway': df, 'imu': df, 'insole': df } }
    """
    print(f">> 🚀 Pre-loading ALL raw data from {data_dir}...")
    cache = {}
    files = list(data_dir.glob("*.pkl"))
    
    if not files:
        print(f"❌ CRITICAL: No .pkl files found in {data_dir}")
        return {}

    # Group files by Subject ID
    # Filename format: sid_modality_raw.pkl (e.g., nls036_walkway_raw.pkl)
    subject_map = {} 
    for f in files:
        parts = f.name.split("_")
        sid = parts[0].lower()
        mod = parts[1] # walkway, insole, imu
        
        if sid not in subject_map: subject_map[sid] = {}
        subject_map[sid][mod] = f

    # Load content
    count = 0
    for sid, mods in subject_map.items():
        cache[sid] = {}
        for mod, fpath in mods.items():
            try:
                df = pd.read_pickle(fpath)
                if not df.empty:
                    cache[sid][mod] = df
            except Exception as e:
                print(f"⚠️ Failed to load {fpath}: {e}")
        if cache[sid]: count += 1
                
    print(f">> ✅ Cached {count} subjects in RAM.")
    return cache

# ==================== 2. COLUMN DEFINITIONS & EXPANDERS ====================
# We define these globally so both Stats Calculator and Dataset use the exact same logic.

WALKWAY_COLS = C.WALKWAY_COLS

# Insole: 16L + 16R Pressure + 2 Force + 4 CoP (Total 38)
# Note: We EXCLUDE Acc/Gyr here because they are in IMU now.
INSOLE_SCALARS = ["LTotalForce_BW", "RTotalForce_BW", "LCoP_X", "LCoP_Y", "LCoP_Vel", "RCoP_X", "RCoP_Y", "RCoP_Vel"]
INSOLE_KINEMATICS = [f"{side}{sns}_{ax}" for side in ["Linsole", "Rinsole"] for sns in ["Acc", "Gyr"] for ax in ["X", "Y", "Z"]]

# --- Map exactly to the 8 pooled zones output by InsolePreprocessor ---
INSOLE_ZONES = ["L_Heel", "L_Arch", "L_Meta", "L_Toes", "R_Heel", "R_Arch", "R_Meta", "R_Toes"]

# Total is now exactly 28 mathematically aligned features
INSOLE_TARGET_COLS = INSOLE_SCALARS + INSOLE_KINEMATICS + INSOLE_ZONES

# IMU: 15 Sites * 6 Axes (Total 90)
IMU_TARGET_COLS = []
for s in C.BODY_IMU_SITES:

    if "insole" in s.lower():
        continue

    for m in ["Acc", "Gyr"]:
        for ax in ["0", "1", "2"]:
            IMU_TARGET_COLS.append(f"{s}_{m}_{ax}")


class InsolePreprocessor:
    def __init__(self):
        pass

    def _pool_zones(self, pressure_matrix: np.ndarray) -> np.ndarray:
        if pressure_matrix.shape[1] != 16:
            return np.zeros((pressure_matrix.shape[0], 4), dtype=np.float32)

        # Ignore warnings for slices that are entirely NaN (dead sensor zones)
        with np.errstate(all='ignore'):
            # Use nanmax: Captures the peak pressure applied, ignoring dead (NaN) sensors
            heel = np.nanmax(pressure_matrix[:, 0:4], axis=1, keepdims=True)
            arch = np.nanmax(pressure_matrix[:, 4:8], axis=1, keepdims=True)
            meta = np.nanmax(pressure_matrix[:, 8:13], axis=1, keepdims=True)
            toes = np.nanmax(pressure_matrix[:, 13:16], axis=1, keepdims=True)

        zones = np.hstack([heel, arch, meta, toes])
        
        # If an entire zone was dead, nanmax returns -inf. Convert these back to 0.
        zones[np.isinf(zones)] = 0.0
        return np.nan_to_num(zones, nan=0.0)

    def process(self, df: pd.DataFrame) -> np.ndarray:
        flat_cols = INSOLE_SCALARS + INSOLE_KINEMATICS
        
        if df.empty:
            return np.zeros((0, len(flat_cols) + 8), dtype=np.float32)

        # Extract available flat columns, fill missing with 0 to keep tensor shape strict
        flat_data = np.zeros((len(df), len(flat_cols)), dtype=np.float32)
        for i, col in enumerate(flat_cols):
            if col in df.columns:
                flat_data[:, i] = df[col].fillna(0).to_numpy(dtype=np.float32)

        def stack_raw(col):
            if col not in df.columns: return np.zeros((len(df), 16), dtype=np.float32)
            arr = [x if isinstance(x, (list, tuple, np.ndarray)) else (0.0,)*16 for x in df[col]]
            return np.array(arr, dtype=np.float32)

        l_raw = stack_raw("LPressure_Block")
        r_raw = stack_raw("RPressure_Block")

        l_zones = self._pool_zones(l_raw)
        r_zones = self._pool_zones(r_raw)
        
        # Output Shape: 20 flat features + 8 zone features = 28 columns
        return np.hstack([flat_data, l_zones, r_zones])

def _expand_imu_data(df: pd.DataFrame) -> np.ndarray:
    """Unpacks all IMU tuple columns into a flat (N, 78) array."""
    if df.empty: return np.zeros((0, 78), dtype=np.float32)
    
    # 1. STRICT CHECK: Missing Critical Sensors
    expected_tuple_cols = []
    for s in C.BODY_IMU_SITES:
        # --- THE FIX: Tell the strict checker to ignore the Insole IMUs ---
        if "insole" in s.lower():
            continue 
            
        for mod in ["Acc", "Gyr"]:
            expected_tuple_cols.append(f"{s}_{mod}")
            
    if not set(expected_tuple_cols).issubset(df.columns):
        # Found a missing sensor file? Skip this subject.
        return np.zeros((0, 78), dtype=np.float32)

    arrays = []
    for col in expected_tuple_cols:
        # Stack tuples (N, 3)
        valid_data = df[col].tolist()
        
        # Verify integrity of the first row to ensure it's a tuple/list
        if len(valid_data) > 0 and not isinstance(valid_data[0], (list, tuple, np.ndarray)):
             # Corrupt data (scalar found where vector expected)
             return np.zeros((0, 78), dtype=np.float32)
             
        arr = np.array(valid_data, dtype=np.float32)
        arrays.append(arr)
            
    return np.hstack(arrays) # (N, 78)

    
def _get_expanded_data(df: pd.DataFrame, mod: str) -> np.ndarray:
    """Router for expansion logic with STRICT checks."""
    if mod == "walkway":
        if not set(WALKWAY_COLS).issubset(df.columns):
            return np.zeros((0, 8), dtype=np.float32)
        return df[WALKWAY_COLS].fillna(0).to_numpy(dtype=np.float32)
    elif mod == "insole":
        INSOLE_PREP = InsolePreprocessor()
        return INSOLE_PREP.process(df)
    elif mod == "imu":
        return _expand_imu_data(df)
        
    return np.array([])

# ==================== 3. STATS CALCULATOR ====================
def calc_fold_stats(train_subs, global_cache, modalities) -> Dict[str, Tuple[float, float]]:
    """
    Calculates Mean/Std on the fly using the RAM Cache.
    """
    sums = {}
    sumsqs = {}
    counts = {}
    
    # Map modality to its list of feature names for tracking
    feat_map = {
        "walkway": WALKWAY_COLS,
        "insole": INSOLE_TARGET_COLS,
        "imu": IMU_TARGET_COLS
    }

    print("  > 🧮 Calculating Stats on RAM Cache...")
    
    for sid in train_subs:
        if sid not in global_cache: continue
        
        for mod in modalities:
            if mod not in global_cache[sid]: continue
            
            # Expand data (N_samples, N_features)
            arr = _get_expanded_data(global_cache[sid][mod], mod)
            if arr.shape[0] == 0: continue
            
            # Feature Names
            cols = feat_map[mod]
            if arr.shape[1] != len(cols):
                # Shape mismatch safety (e.g. if config changed but data didn't)
                continue

            # Accumulate
            # We assume no NaNs here because extractors filled them with 0
            for i, c in enumerate(cols):
                vals = arr[:, i]
                sums[c] = sums.get(c, 0.0) + float(vals.sum())
                sumsqs[c] = sumsqs.get(c, 0.0) + float(np.dot(vals, vals))
                counts[c] = counts.get(c, 0) + vals.size

    # Finalize
    stats = {}
    for c, n in counts.items():
        if n == 0: continue
        mean = sums[c] / n
        var = max((sumsqs[c]/n) - mean**2, 0.0)
        stats[c] = (mean, max(np.sqrt(var), 1e-6))
        
    return stats


class GaitAugmenter:
    """
    Applies physical time-series augmentations to combat memorization in 1D CNNs.
    """
    def __init__(self, p=0.5, jitter_sigma=0.05, scale_min=0.8, scale_max=1.2):
        self.p = p                   # Probability of applying augmentation
        self.sigma = jitter_sigma    # Strength of the Gaussian noise
        self.scale_min = scale_min   # Lower bound for amplitude scaling
        self.scale_max = scale_max   # Upper bound for amplitude scaling

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects a tensor of shape (Channels, Sequence_Length)
        """
        # Only augment a subset of the batches to maintain stable gradient baselines
        if torch.rand(1).item() > self.p:
            return x
            
        aug_x = x.clone()
        
        # 1. Magnitude Scaling
        # Generate a single scalar (e.g., 0.95) to uniformly stretch/squash the amplitude
        scale = torch.empty(1, 1).uniform_(self.scale_min, self.scale_max)
        aug_x = aug_x * scale
        
        # 2. Gaussian Jitter
        # Generate random noise of the exact same shape as the signal
        noise = torch.randn_like(aug_x) * self.sigma
        aug_x = aug_x + noise
        
        return aug_x


# ==================== 4. LAZY DATASET ====================
class WearGaitLazyDataset(Dataset):
    def __init__(self, subject_ids, global_cache, stats, 
                 modalities=("walkway",), win_len=120, hop_len=60, subj2label=None, mode='train'):
        self.cache = global_cache
        self.stats = stats
        self.modalities = modalities
        self.win = win_len
        self.hop = hop_len
        self.subj2label = subj2label or {}
        self.mode = mode
        self.feat_map = {
            "walkway": WALKWAY_COLS,
            "insole": INSOLE_TARGET_COLS,
            "imu": IMU_TARGET_COLS
        }
        self.augmenter = GaitAugmenter(p=0.5) if mode == 'train' else None
        self.indices = []
        
        # --- TRACKING METRICS ---
        kept_subjects = 0
        dropped_subject_reasons = [] 
        
        total_possible_windows = 0
        rejected_windows = 0
        
        for sid in subject_ids:
            # 0. Basic Cache Check
            if sid not in self.cache: 
                dropped_subject_reasons.append(f"{sid}: Not found in RAM cache")
                continue
            
            valid_subject = True
            min_len = float('inf')
            
            for m in modalities:
                # 1. Modality Existence Check
                if m not in self.cache[sid]: 
                    valid_subject = False
                    dropped_subject_reasons.append(f"{sid}: Missing modality '{m}'")
                    break
                
                df = self.cache[sid][m]
                
                # 2. Strict Column/Data Integrity Check
                if len(df) > 0:
                    sample = _get_expanded_data(df.iloc[:1], m)
                    if sample.shape[0] == 0:
                        valid_subject = False
                        dropped_subject_reasons.append(f"{sid}: Corrupt/Missing Cols in '{m}'")
                        break
                else:
                    valid_subject = False
                    dropped_subject_reasons.append(f"{sid}: Empty DataFrame for '{m}'")
                    break
                
                # 3. Length Check
                l = len(df)
                if l < win_len: 
                    valid_subject = False
                    dropped_subject_reasons.append(f"{sid}: Too short ({l} < {win_len}) for '{m}'")
                    break
                if l < min_len: min_len = l
            
            if not valid_subject: continue
            
            kept_subjects += 1
            
            # --- WINDOW GENERATION & REJECTION TRACKING ---
            n_windows = int((min_len - win_len) // self.hop + 1)
            if n_windows > 0:
                total_possible_windows += n_windows
                
                for i in range(n_windows):
                    start_idx = i * self.hop
                    is_window_valid = True
                    
                    # Check for macro-gaps across all required modalities
                    for m in modalities:
                        window_slice = self.cache[sid][m].iloc[start_idx : start_idx + self.win]
                        raw_check_cols = [c for c in self.feat_map[m] if c in window_slice.columns]
                        # If any NaN exists in the target feature columns, reject the window
                        if len(raw_check_cols) > 0 and window_slice[raw_check_cols].isna().any().any():
                            is_window_valid = False
                            break
                    
                    if is_window_valid:
                        self.indices.append((sid, start_idx))
                    else:
                        rejected_windows += 1

        self.labels = [self.subj2label.get(sid, -1) for sid, _ in self.indices]

        # --- PRINT COMPREHENSIVE SUMMARY ---
        # print(f"   [Loader] {modalities} (Mode: {mode}): Subjects Kept [{kept_subjects}/{len(subject_ids)}]")
        # if len(dropped_subject_reasons) > 0:
            # print(f"      -> Dropped {len(dropped_subject_reasons)} subjects due to missing/short data.")
            
        # if total_possible_windows > 0:
            # reject_rate = (rejected_windows / total_possible_windows) * 100.0
            # print(f"   [Loader] {modalities}: Windows Generated: {total_possible_windows} | Rejected: {rejected_windows} ({reject_rate:.1f}%) | Valid: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sid, start = self.indices[idx]
        y = self.subj2label.get(sid, -1)
        xs = []
        for mod in self.modalities:
            df = self.cache[sid][mod]
            window_df = df.iloc[start : start + self.win]
            arr = _get_expanded_data(window_df, mod)

            # 2. Dynamic Normalization Logic
            if mod == 'insole' and arr.shape[1] == 28:
                # --- SPECIAL CASE: 14-Channel Anatomical Mode ---
                # We only have global stats for the first 6 scalars. 
                # The other 8 channels are synthetic sums, so we use Mean=0/Std=1 (Identity).
                
                # A. Get Scalar Stats (Cols 0-5)
                flat_cols = INSOLE_SCALARS + INSOLE_KINEMATICS
                s_means = [self.stats.get(c, (0.0, 1.0))[0] for c in flat_cols]
                s_stds  = [self.stats.get(c, (0.0, 1.0))[1] for c in flat_cols]
                # B. Get Zone Stats (Cols 6-13) -> Default to 0.0 mean, 1.0 std
                z_means = [0.0] * 8
                z_stds  = [1.0] * 8
                means = np.array(s_means + z_means, dtype=np.float32)
                stds  = np.array(s_stds + z_stds, dtype=np.float32)
            else:
                cols = self.feat_map[mod]
                # Safety check: if column count doesn't match, trim or pad stats
                if len(cols) != arr.shape[1]:
                     # Fallback to identity to prevent crash if mismatch occurs
                     means = np.zeros(arr.shape[1], dtype=np.float32)
                     stds  = np.ones(arr.shape[1], dtype=np.float32)
                else:
                    means = np.array([self.stats.get(c, (0.0, 1.0))[0] for c in cols], dtype=np.float32)
                    stds  = np.array([self.stats.get(c, (0.0, 1.0))[1] for c in cols], dtype=np.float32)

            # 3. Apply Normalization
            stds[stds == 0] = 1.0 # Avoid division by zero
            arr = (arr - means) / stds
            tensor = torch.tensor(arr, dtype=torch.float32).transpose(0, 1) 
            if self.augmenter is not None:
                tensor = self.augmenter(tensor)
            xs.append(tensor)
            
        return {"xs": xs, "y": torch.tensor(y, dtype=torch.long), "sid": sid}

# ==================== 5. INTERFACE FUNCTIONS ====================
def prepare_split(
    train_subs: List[str],
    test_subs: List[str],
    data_cache: Dict, # <--- Expects RAM Cache
    data_dir: Path = C.OUTPUT_DIR, # Legacy arg, ignored if cache present
    modalities: Tuple[str, ...] = ("walkway", "insole", "imu"),
    win: int = C.WINDOW_SIZE,
    hop: int = int(C.WINDOW_SIZE * C.STRIDE)
):
    # 1. Calculate Stats on Train Subjects
    stats = calc_fold_stats(train_subs, data_cache, modalities)
    
    # 2. Return data needed for Loader creation
    return {
        "train_subs": train_subs,
        "test_subs": test_subs,
        "stats": stats,
        "cache": data_cache,
        "win": win,
        "hop": hop,
        "modalities": modalities
    }

def make_sync_loaders(prep_data, subj2label, batch_size=C.BATCH_SIZE, num_workers=4, **kwargs):
    """
    Creates DataLoaders from the prep_data dictionary.
    Accepts **kwargs to safely ignore extra args from Trainer.
    """
    # Create Datasets
    train_ds = WearGaitLazyDataset(
        prep_data["train_subs"], prep_data["cache"], prep_data["stats"],
        modalities=prep_data["modalities"], win_len=prep_data["win"], hop_len=prep_data["hop"],
        subj2label=subj2label
    )
    
    test_ds = WearGaitLazyDataset(
        prep_data["test_subs"], prep_data["cache"], prep_data["stats"],
        modalities=prep_data["modalities"], win_len=prep_data["win"], hop_len=prep_data["hop"],
        subj2label=subj2label,
        mode='test'
    )
    
    # Collate Function
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

# ==================== 6. UTILS ====================
def build_subj2label(pd_ids: List[str], hc_ids: List[str]) -> Dict[str, int]:
    return {**{s: 1 for s in pd_ids}, **{s: 0 for s in hc_ids}}

def make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=10, seed=42):
    import random
    rng = random.Random(seed)
    
    pd_pool = sorted(pd_ids); rng.shuffle(pd_pool)
    hc_pool = sorted(hc_ids); rng.shuffle(hc_pool)
    
    n_pd = len(pd_ids) // n_folds
    n_hc = len(hc_ids) // n_folds
    
    folds = []
    for f in range(n_folds):
        te_pd = pd_pool[f*n_pd : (f+1)*n_pd]
        te_hc = hc_pool[f*n_hc : (f+1)*n_hc]
        test_subs = sorted(te_pd + te_hc)
        train_subs = sorted(list(set(pd_ids + hc_ids) - set(test_subs)))
        folds.append((train_subs, test_subs))
    return folds