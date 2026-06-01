import os
import glob
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

# Suppress noisy warnings from engine probing
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def parse_args():
    from model.paths import FBG_C3D, FBG_PROCESSED, as_str
    parser = argparse.ArgumentParser(description="Strict Biomechanical Preprocessing Pipeline for FBG Dataset")
    parser.add_argument('--data_root', type=str,
                        default=as_str(FBG_C3D),
                        help="Root directory containing SUBXX_on and SUBXX_off folders")
    parser.add_argument('--out_dir', type=str,
                        default=as_str(FBG_PROCESSED),
                        help="Output directory for the temporally aligned multimodal PKL tensors")
    return parser.parse_args()

def biomechanical_filter(data_matrix, cutoff_freq, fs, order=4):
    """
    Zero-phase 4th-order Butterworth low-pass filter.
    Smooths high-frequency noise and derivative artifacts without breaking temporal causality.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt: forward + backward pass for strict zero-phase filtering
    filtered_data = signal.filtfilt(b, a, data_matrix, axis=0)
    return filtered_data

def auto_trim_deadzone(X_lin, X_ang, X_grf, energy_threshold=1e-3, window=20):
    """
    Joint kinetic-energy profile; trims invalid standing dead-zones at trial start/end.
    """
    # Align time dimension across modalities
    T = min(X_lin.shape[0], X_ang.shape[0], X_grf.shape[0])
    X_lin, X_ang, X_grf = X_lin[:T], X_ang[:T], X_grf[:T]

    # Local temporal variance of motion state
    ang_diff = np.sum(np.abs(np.diff(X_ang, axis=0)), axis=1)
    grf_diff = np.sum(np.abs(np.diff(X_grf, axis=0)), axis=1)
    
    # Fuse kinetic indicators and smooth
    energy_profile = ang_diff + grf_diff
    # Pad one frame lost by diff
    energy_profile = np.append(energy_profile, energy_profile[-1]) 
    energy_profile = np.convolve(energy_profile, np.ones(window)/window, mode='same')
    
    # Find active interval
    active_indices = np.where(energy_profile > energy_threshold)[0]
    
    if len(active_indices) < 100:
        # Skip trim if too short or inactive to avoid manifold collapse
        return X_lin, X_ang, X_grf
        
    start_idx = max(0, active_indices[0] - window)
    end_idx = min(T, active_indices[-1] + window)
    
    return X_lin[start_idx:end_idx], X_ang[start_idx:end_idx], X_grf[start_idx:end_idx]

def robust_manifold_extraction(file_path, modality_type):
    """
    Extract a clean tensor from heterogeneous file layouts.
    Defensive interpolation when data are heavily missing.
    """
    try:
        # 1. Magic-byte detection and load
        with open(file_path, 'rb') as f:
            magic_bytes = f.read(4)
            
        if magic_bytes == b'PK\x03\x04':
            df_raw = pd.read_excel(file_path, engine='openpyxl', header=None)
        elif magic_bytes == b'\xd0\xcf\x11\xe0':
            df_raw = pd.read_excel(file_path, engine='xlrd', header=None)
        else:
            try:
                df_raw = pd.read_csv(file_path, header=None, low_memory=False, on_bad_lines='skip')
            except Exception:
                df_raw = pd.read_csv(file_path, header=None, sep=r'\t|;|,', engine='python', on_bad_lines='skip')

        # 2. Dynamic header row lookup
        header_idx = -1
        for idx, row in df_raw.iterrows():
            row_str = ' '.join(row.dropna().astype(str))
            if 'Frame' in row_str and 'Time' in row_str:
                header_idx = idx
                break
                
        if header_idx == -1:
            raise ValueError("No valid header (Frame/Time) found.")
            
        # 3. Strip metadata and cast to numeric matrix
        df_clean = df_raw.iloc[header_idx + 1:].copy()
        df_clean.columns = df_raw.iloc[header_idx]
        df_clean = df_clean.dropna(axis=1, how='all').dropna(axis=0, how='all')
        data_matrix = df_clean.apply(pd.to_numeric, errors='coerce').values
        
        # ==========================================
        # 4. Continuous manifold reconstruction (interpolate first)
        # ==========================================
        if np.isnan(data_matrix).any():
            df_temp = pd.DataFrame(data_matrix)
            total_frames = len(df_temp)
            
            # Drop channels with low retention ratio (heavily corrupted)
            valid_counts = df_temp.count()
            retention_ratio = valid_counts / total_frames
            bad_cols = retention_ratio[retention_ratio < 0.60].index
            
            if len(bad_cols) > 0:
                df_temp.loc[:, bad_cols] = 0.0
                
            # Per-column interpolation; fall back if spline fails
            # Only columns with NaNs
            for col in df_temp.columns:
                if df_temp[col].isna().any():
                    # Cubic spline if at least 2 valid points
                    if df_temp[col].count() >= 2:
                        try:
                            df_temp[col] = df_temp[col].interpolate(method='cubicspline', limit_direction='both')
                        except (ValueError, np.linalg.LinAlgError):
                            # Fall back to linear if spline singular
                            df_temp[col] = df_temp[col].interpolate(method='linear', limit_direction='both')
            
            # Final pass: fill any remaining NaNs
            df_filled = df_temp.bfill().ffill().fillna(0.0)
            data_matrix = df_filled.values
            
        # ==========================================
        # 5. Impulse suppression (after interpolation)
        # ==========================================
        # Median filter only for temporally continuous optical modalities
        if modality_type in ['linear', 'angular']:
            data_matrix = signal.medfilt(data_matrix, kernel_size=(3, 1))
            
        return data_matrix

    except Exception as e:
        # Re-raise for diagnostic logging
        raise RuntimeError(f"Extraction failure: {str(e)}")
        
import traceback # For stack traces in diagnostic logs

def process_fbg_dataset(data_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    subject_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) 
                    if os.path.isdir(os.path.join(data_root, d)) and d.startswith('SUB')]
    
    success_count, fail_count = 0, 0
    
    # Diagnostic log collector
    diagnostic_logs = []

    for subj_dir in tqdm(subject_dirs, desc="Compiling Biomechanical Manifolds"):
        linear_files = glob.glob(os.path.join(subj_dir, '*_linear_kinematics.csv'))
        
        for lin_file in linear_files:
            base_name = lin_file.replace('_linear_kinematics.csv', '')
            walk_id = os.path.basename(base_name)
            
            ang_file = f"{base_name}_angular_kinematics.csv"
            grf_file = f"{base_name}_grf.csv"
            
            # 1. Missing-file check
            if not (os.path.exists(ang_file) and os.path.exists(grf_file)):
                fail_count += 1
                missing_info = []
                if not os.path.exists(ang_file): missing_info.append("Angular Missing")
                if not os.path.exists(grf_file): missing_info.append("GRF Missing")
                diagnostic_logs.append({
                    "walk_id": walk_id,
                    "error_type": "Missing File",
                    "details": " & ".join(missing_info)
                })
                continue
            
            try:
                # Phase 1: Per-modality manifold extraction
                X_lin = robust_manifold_extraction(lin_file, modality_type='linear')
                X_ang = robust_manifold_extraction(ang_file, modality_type='angular')
                X_grf = robust_manifold_extraction(grf_file, modality_type='grf')
                
                # Phase 2: Resample and align lengths
                T_kin = min(X_lin.shape[0], X_ang.shape[0])
                X_lin, X_ang = X_lin[:T_kin, :], X_ang[:T_kin, :]
                
                if X_grf.shape[0] != T_kin:
                    X_grf = signal.resample(X_grf, T_kin, axis=0)
                
                # Phase 3: Adaptive kinetic trim
                X_lin, X_ang, X_grf = auto_trim_deadzone(X_lin, X_ang, X_grf)
                
                # Phase 4: Zero-phase biomechanical low-pass filter
                X_lin = biomechanical_filter(X_lin, cutoff_freq=6.0, fs=150.0)
                X_ang = biomechanical_filter(X_ang, cutoff_freq=6.0, fs=150.0)
                X_grf = biomechanical_filter(X_grf, cutoff_freq=30.0, fs=150.0)
                
                # Phase 5: Remove absolute position offset
                X_lin = X_lin - X_lin[0, :]
                
                # Phase 6: Per-instance z-score normalization
                X_lin, X_ang, X_grf = instance_level_standardization(X_lin, X_ang, X_grf)
                
                # Serialize to PKL
                modality_dict = {
                    "walk_id": walk_id,
                    "linear_kinematics": X_lin.astype(np.float32),
                    "angular_kinematics": X_ang.astype(np.float32),
                    "translational_kinetics": X_grf.astype(np.float32)
                }
                
                out_path = os.path.join(out_dir, f"{walk_id}.pkl")
                with open(out_path, 'wb') as f:
                    pickle.dump(modality_dict, f)
                    
                success_count += 1
                
            except Exception as e:
                # Record manifold failure reason
                fail_count += 1
                diagnostic_logs.append({
                    "walk_id": walk_id,
                    "error_type": type(e).__name__,
                    "details": str(e)
                })
                
    print(f"\n[Rigorous Processing Complete] Success: {success_count} | Failed: {fail_count}")
    
    # Export diagnostic report
    if diagnostic_logs:
        log_df = pd.DataFrame(diagnostic_logs)
        log_path = os.path.join(data_root, 'preprocessing_diagnostics.csv')
        log_df.to_csv(log_path, index=False)
        print(f"\n[!] {fail_count} failed manifolds. Diagnostic report: {log_path}")
        print("Most common error types:")
        print(log_df['error_type'].value_counts())

def instance_level_standardization(X_lin, X_ang, X_grf):
    """
    Per-sequence instance normalization.
    Removes absolute scale so the 1D-CNN focuses on relative manifold shape.
    """
    def z_score(matrix):
        # Mean/std per channel along time (axis=0)
        # 1e-8 avoids division by zero on silent channels
        mu = np.mean(matrix, axis=0)
        sigma = np.std(matrix, axis=0)
        return (matrix - mu) / (sigma + 1e-8)

    return z_score(X_lin), z_score(X_ang), z_score(X_grf)

if __name__ == "__main__":
    args = parse_args()
    process_fbg_dataset(args.data_root, args.out_dir)