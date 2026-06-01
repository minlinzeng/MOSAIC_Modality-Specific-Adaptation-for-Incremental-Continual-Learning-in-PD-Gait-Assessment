import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from model.paths import FOG_CACHE, FOG_C3D, FOG_IMU, FOG_LABELS, FOG_POSE

# ==========================================
# 1. Paths and physics constants
# ==========================================
class Config:
    # Raw FOG data paths
    C3D_DIR = FOG_C3D
    IMU_DIR = FOG_IMU
    POSE_DIR = FOG_POSE
    LABEL_PATH = FOG_LABELS

    # Preprocessed cache output
    OUTPUT_DIR = FOG_CACHE
    
    # Sampling rates (strict alignment)
    IMU_HZ = 128.0       
    VIDEO_HZ = 30.0      
    TARGET_HZ = 30.0

# ==========================================
# 2. Label parsing
# ==========================================
def extract_labels(label_path):
    """
    Replicate label logic from dataloaders.py / pdfeReader:
    Subtract 2 from H&Y scores to map classes 0, 1, 2.
    """
    label_df = pd.read_excel(label_path)
    label_df.columns = [str(col).strip() for col in label_df.columns]
    hy_columns = [col for col in label_df.columns if "H&Y" in col]

    subject_labels = {}
    for idx, row in label_df.iterrows():
        if idx == 0: continue
        subject_id = f"SUB{idx:02d}" # Match IMU prefix SUB01, SUB02
        for col in hy_columns:
            try:
                if pd.notna(row[col]):
                    label = int(row[col]) - 2 
                    subject_labels[subject_id] = max(0, label) # Clamp negatives
                    break # Use first valid H&Y column
            except ValueError:
                continue
    return subject_labels

# ==========================================
# 3. Modality alignment (core fix)
# ==========================================
def temporal_alignment(imu_data, skel_t, skel_data):
    """
    Align 30Hz skeleton (missing frames) with 128Hz IMU.
    """
    num_imu_frames = imu_data.shape[0]
    if num_imu_frames == 0 or len(skel_t) < 10:
        return None, None, None
        
    t_imu = np.arange(num_imu_frames) / Config.IMU_HZ
    
    # Intersect timelines; drop IMU without skeleton
    max_t = min(t_imu[-1], skel_t[-1])
    min_t = max(t_imu[0], skel_t[0])
    
    if max_t <= min_t: return None, None, None
    
    # Build 128Hz common time grid
    t_target = np.arange(min_t, max_t, 1.0 / Config.TARGET_HZ)
    
    # 1. Interpolate IMU (truncate ends)
    imu_interpolator = interp1d(t_imu, imu_data, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
    imu_aligned = imu_interpolator(t_target)
    
    # 2. Interpolate skeleton (fill Mmpose gaps, upsample to 128Hz)
    skel_interpolator = interp1d(skel_t, skel_data, axis=0, kind='cubic', bounds_error=False, fill_value="extrapolate")
    skel_aligned = skel_interpolator(t_target)
    
    return imu_aligned, skel_aligned, t_target

# ==========================================
# 4. Main preprocessing pipeline
# ==========================================
def build_fog_cache():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_dict = extract_labels(Config.LABEL_PATH)
    
    imu_files = [f for f in os.listdir(Config.IMU_DIR) if f.endswith(".txt") and "standing" not in f.lower()]
    
    # Drop hard-coded outlier sessions
    bad_subjects = ['SUB19_1_1', 'SUB21_1_1'] 
    
    valid_sessions = 0
    final_subj2label = {}
    
    print(f"🚀 开始重构 FOG 数据流形，扫描到 {len(imu_files)} 个 IMU 序列...")
    
    for imu_file in sorted(imu_files):
        # Example filename: SUB01_1.txt
        base_name = imu_file.replace(".txt", "") 
        subject_id = base_name.split("_")[0] # SUB01
        
        if base_name in bad_subjects: continue
        if subject_id not in labels_dict: continue
        
        # Match pose JSON (SUB01 -> PDFE01)
        pose_prefix = subject_id.replace("SUB", "PDFE")
        trial_suffix = base_name.split("_")[1] if "_" in base_name else "1"
        json_file = Config.POSE_DIR / f"{pose_prefix}_{trial_suffix}_3d_predictions.json"
        
        if not json_file.exists():
            print(f"⚠️ [跳过] 找不到匹配的 JSON 骨架: {json_file.name}")
            continue
            
        try:
            # --- 1. Load IMU ---
            imu_df = pd.read_csv(Config.IMU_DIR / imu_file, sep=r'\s{2,}|\t', engine='python')
            raw_imu = imu_df.iloc[:, 2:8].to_numpy() # (N, 6)
            
            # --- 2. Load skeleton (fix missing-frame collapse) ---
            with open(json_file, 'r') as f:
                pose_data = json.load(f)
            
            skel_t = []
            skel_kps = []
            
            for frame_idx, frame_pred in enumerate(pose_data):
                instances = frame_pred.get('predictions') or []
                if instances:
                    # First 7 joints (0-6) -> 21-d features
                    kp = instances[0][0]['keypoints'][0:7]
                    skel_t.append(frame_idx / Config.VIDEO_HZ) # Store physical timestamps
                    skel_kps.append(np.array(kp).flatten())
            
            if len(skel_t) < 10:
                print(f"⚠️ [跳过] {base_name}: 有效骨架帧太少 ({len(skel_t)} 帧)")
                continue
                
            skel_t = np.array(skel_t)
            raw_skel = np.array(skel_kps)
            
            # --- 3. Strict temporal alignment ---
            imu_aligned, skel_aligned, t_target = temporal_alignment(raw_imu, skel_t, raw_skel)
            if imu_aligned is None or len(t_target) < 128: 
                continue # Skip very short sequences
                
            # --- 4. Split and save PKL ---
            # Naming: sub01-1_modality_raw.pkl
            sid = f"{subject_id.lower()}-{trial_suffix}"
            
            # A. Acc
            df_acc = pd.DataFrame(imu_aligned[:, 0:3], columns=['Acc_X', 'Acc_Y', 'Acc_Z'])
            df_acc.insert(0, 'Time', t_target)
            df_acc.to_pickle(Config.OUTPUT_DIR / f"{sid}_acc_raw.pkl")
            
            # B. Gyr
            df_gyr = pd.DataFrame(imu_aligned[:, 3:6], columns=['Gyr_X', 'Gyr_Y', 'Gyr_Z'])
            df_gyr.insert(0, 'Time', t_target)
            df_gyr.to_pickle(Config.OUTPUT_DIR / f"{sid}_gyr_raw.pkl")
            
            # C. Skeleton (21 dimensions)
            skel_cols = [f'Skel_{i}' for i in range(21)]
            df_skel = pd.DataFrame(skel_aligned, columns=skel_cols)
            df_skel.insert(0, 'Time', t_target)
            df_skel.to_pickle(Config.OUTPUT_DIR / f"{sid}_skeleton_raw.pkl")
            
            final_subj2label[sid] = labels_dict[subject_id]
            valid_sessions += 1
            print(f"✅ 完成: {sid} | 生成时间跨度: {t_target[-1]-t_target[0]:.2f}s | 张量长度: {len(t_target)}")
            
        except Exception as e:
            print(f"❌ 处理 {base_name} 时崩溃: {e}")

    with open(Config.OUTPUT_DIR / "subj2label.json", "w") as f:
        json.dump(final_subj2label, f, indent=4)
        
    print("\n" + "="*50)
    print(f"🏆 物理流形对齐完成！成功生成 {valid_sessions} 个 Sessions 的完美 PKL。")
    print(f"📂 缓存地址: {Config.OUTPUT_DIR.absolute()}")
    print("="*50 + "\n")

if __name__ == "__main__":
    build_fog_cache()