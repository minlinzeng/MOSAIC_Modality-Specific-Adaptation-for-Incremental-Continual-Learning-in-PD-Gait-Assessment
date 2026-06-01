import os
import glob
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

# 屏蔽底层引擎探测时的冗余警告
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
    零相移四阶 Butterworth 低通滤波。
    在不改变时序因果性的前提下，平滑高频底噪与导数伪影。
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt 执行前向和后向滤波，实现严格的零相移
    filtered_data = signal.filtfilt(b, a, data_matrix, axis=0)
    return filtered_data

def auto_trim_deadzone(X_lin, X_ang, X_grf, energy_threshold=1e-3, window=20):
    """
    联合动能梯度计算，自动切除测试首尾的无效站立期 (Dead-zones)。
    """
    # 提取时间维度
    T = min(X_lin.shape[0], X_ang.shape[0], X_grf.shape[0])
    X_lin, X_ang, X_grf = X_lin[:T], X_ang[:T], X_grf[:T]

    # 计算运动状态的局部时间方差 (Local Variance)
    ang_diff = np.sum(np.abs(np.diff(X_ang, axis=0)), axis=1)
    grf_diff = np.sum(np.abs(np.diff(X_grf, axis=0)), axis=1)
    
    # 融合动能指标并进行平滑
    energy_profile = ang_diff + grf_diff
    # 补齐 diff 导致少掉的一帧
    energy_profile = np.append(energy_profile, energy_profile[-1]) 
    energy_profile = np.convolve(energy_profile, np.ones(window)/window, mode='same')
    
    # 寻找激活区间
    active_indices = np.where(energy_profile > energy_threshold)[0]
    
    if len(active_indices) < 100:
        # 如果序列过短或未激活，不进行切除以防流形崩溃
        return X_lin, X_ang, X_grf
        
    start_idx = max(0, active_indices[0] - window)
    end_idx = min(T, active_indices[-1] + window)
    
    return X_lin[start_idx:end_idx], X_ang[start_idx:end_idx], X_grf[start_idx:end_idx]

def robust_manifold_extraction(file_path, modality_type):
    """
    穿透异构文件结构提取纯净张量。
    增加了防御性编程，确保样条插值逻辑在数据严重缺失时依然稳健。
    """
    try:
        # 1. 魔法字节探测与加载
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

        # 2. 动态寻址表头
        header_idx = -1
        for idx, row in df_raw.iterrows():
            row_str = ' '.join(row.dropna().astype(str))
            if 'Frame' in row_str and 'Time' in row_str:
                header_idx = idx
                break
                
        if header_idx == -1:
            raise ValueError("No valid header (Frame/Time) found.")
            
        # 3. 剥离冗余元数据并转化为数值矩阵
        df_clean = df_raw.iloc[header_idx + 1:].copy()
        df_clean.columns = df_raw.iloc[header_idx]
        df_clean = df_clean.dropna(axis=1, how='all').dropna(axis=0, how='all')
        data_matrix = df_clean.apply(pd.to_numeric, errors='coerce').values
        
        # ==========================================
        # 4. 连续流形重建 (插值前置)
        # ==========================================
        if np.isnan(data_matrix).any():
            df_temp = pd.DataFrame(data_matrix)
            total_frames = len(df_temp)
            
            # 绝对防御：信息保留率截断 (剔除重度损坏通道)
            valid_counts = df_temp.count()
            retention_ratio = valid_counts / total_frames
            bad_cols = retention_ratio[retention_ratio < 0.60].index
            
            if len(bad_cols) > 0:
                df_temp.loc[:, bad_cols] = 0.0
                
            # 动态降阶插值：逻辑严密化，防止样条插值触发矩阵异常
            # 仅对存在 NaN 的列进行处理
            for col in df_temp.columns:
                if df_temp[col].isna().any():
                    # 只要存在至少 2 个非 NaN 值，即可尝试三次样条插值
                    if df_temp[col].count() >= 2:
                        try:
                            df_temp[col] = df_temp[col].interpolate(method='cubicspline', limit_direction='both')
                        except (ValueError, np.linalg.LinAlgError):
                            # 若物理奇点导致样条崩塌，降级为线性插值
                            df_temp[col] = df_temp[col].interpolate(method='linear', limit_direction='both')
            
            # 最终兜底：清理一切遗漏的 NaN
            df_filled = df_temp.bfill().ffill().fillna(0.0)
            data_matrix = df_filled.values
            
        # ==========================================
        # 5. 脉冲抑制 (安全后置)
        # ==========================================
        # 仅对具备时序连续性的光学模态执行中值滤波
        if modality_type in ['linear', 'angular']:
            data_matrix = signal.medfilt(data_matrix, kernel_size=(3, 1))
            
        return data_matrix

    except Exception as e:
        # 将具体错误抛出，以便 diagnostic 日志记录
        raise RuntimeError(f"Extraction failure: {str(e)}")
        
import traceback # 在文件开头别忘了加上这个，用于捕获具体报错行数

def process_fbg_dataset(data_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    subject_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) 
                    if os.path.isdir(os.path.join(data_root, d)) and d.startswith('SUB')]
    
    success_count, fail_count = 0, 0
    
    # 🌟 新增：诊断日志追踪器
    diagnostic_logs = []

    for subj_dir in tqdm(subject_dirs, desc="Compiling Biomechanical Manifolds"):
        linear_files = glob.glob(os.path.join(subj_dir, '*_linear_kinematics.csv'))
        
        for lin_file in linear_files:
            base_name = lin_file.replace('_linear_kinematics.csv', '')
            walk_id = os.path.basename(base_name)
            
            ang_file = f"{base_name}_angular_kinematics.csv"
            grf_file = f"{base_name}_grf.csv"
            
            # 1. 物理文件缺失检测
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
                # Phase 1: 独立流形提取
                X_lin = robust_manifold_extraction(lin_file, modality_type='linear')
                X_ang = robust_manifold_extraction(ang_file, modality_type='angular')
                X_grf = robust_manifold_extraction(grf_file, modality_type='grf')
                
                # Phase 2: 频域重采样与全局对齐
                T_kin = min(X_lin.shape[0], X_ang.shape[0])
                X_lin, X_ang = X_lin[:T_kin, :], X_ang[:T_kin, :]
                
                if X_grf.shape[0] != T_kin:
                    X_grf = signal.resample(X_grf, T_kin, axis=0)
                
                # Phase 3: 自适应动能截断
                X_lin, X_ang, X_grf = auto_trim_deadzone(X_lin, X_ang, X_grf)
                
                # Phase 4: 零相移生物力学低通滤波
                X_lin = biomechanical_filter(X_lin, cutoff_freq=6.0, fs=150.0)
                X_ang = biomechanical_filter(X_ang, cutoff_freq=6.0, fs=150.0)
                X_grf = biomechanical_filter(X_grf, cutoff_freq=30.0, fs=150.0)
                
                # Phase 5: 绝对物理域剥离
                X_lin = X_lin - X_lin[0, :]
                
                # Phase 6: 终极量纲统一定理 (确保你已经加入了 z-score 那个函数)
                X_lin, X_ang, X_grf = instance_level_standardization(X_lin, X_ang, X_grf)
                
                # 打包序列化
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
                # 🌟 新增：捕获导致流形崩塌的具体原因
                fail_count += 1
                diagnostic_logs.append({
                    "walk_id": walk_id,
                    "error_type": type(e).__name__,
                    "details": str(e)
                })
                
    print(f"\n[Rigorous Processing Complete] Success: {success_count} | Failed: {fail_count}")
    
    # 🌟 新增：导出诊断报告
    if diagnostic_logs:
        log_df = pd.DataFrame(diagnostic_logs)
        log_path = os.path.join(data_root, 'preprocessing_diagnostics.csv')
        log_df.to_csv(log_path, index=False)
        print(f"\n[!] 发现了 {fail_count} 个崩塌流形。诊断报告已生成: {log_path}")
        print("最常见的错误类型统计：")
        print(log_df['error_type'].value_counts())

def instance_level_standardization(X_lin, X_ang, X_grf):
    """
    独立序列标准化 (Instance Normalization)。
    彻底解耦物理量纲的绝对幅值，迫使 1D-CNN 仅关注病理流形的相对形态变异。
    """
    def z_score(matrix):
        # 沿时间轴 (axis=0) 计算每个独立物理通道的均值与标准差
        # 加上 1e-8 极小值防止静默通道导致的除零错误
        mu = np.mean(matrix, axis=0)
        sigma = np.std(matrix, axis=0)
        return (matrix - mu) / (sigma + 1e-8)

    return z_score(X_lin), z_score(X_ang), z_score(X_grf)

if __name__ == "__main__":
    args = parse_args()
    process_fbg_dataset(args.data_root, args.out_dir)