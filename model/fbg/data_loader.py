import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class FBGIncrementalDataset(Dataset):
    def __init__(self, data_root, target_subjects, mode='train', 
                 window_size=256, step_size=64, modality_drop_prob=0.2):
        super().__init__()
        self.mode = mode
        self.window_size = window_size
        self.step_size = step_size
        self.modality_drop_prob = modality_drop_prob if mode == 'train' else 0.0
        
        self.walks = []   # 存储完整的序列流形
        self.windows = [] # 存储锚点索引
        
        all_pkl_files = glob.glob(os.path.join(data_root, '*.pkl'))
        if len(all_pkl_files) == 0:
            raise FileNotFoundError(f"未在 {data_root} 找到任何 PKL 数据文件！")

        for pkl_file in tqdm(all_pkl_files, desc=f"Loading {mode} dataset"):
            walk_id = os.path.basename(pkl_file).replace('.pkl', '')
            subj_id = walk_id.split('_')[0]
            
            if subj_id not in target_subjects:
                continue
                
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                
            X_lin = data['linear_kinematics']
            X_ang = data['angular_kinematics']
            X_grf = data['translational_kinetics']
            label = 1 if '_on_' in walk_id.lower() else 0
            
            T = X_lin.shape[0]
            if T < self.window_size:
                continue
                
            walk_idx = len(self.walks)
            self.walks.append({
                'linear': X_lin, 'angular': X_ang, 'grf': X_grf, 'label': label, 'T': T
            })
            
            # 生成静态锚点
            for start in range(0, T - self.window_size + 1, self.step_size):
                self.windows.append({'walk_idx': walk_idx, 'start': start})

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        walk = self.walks[item['walk_idx']]
        start_idx = item['start']
        
        # 🌟 方案 B：时域相位抖动 (Temporal Phase Jittering)
        if self.mode == 'train':
            jitter = torch.randint(-15, 16, (1,)).item()
            start_idx = max(0, min(start_idx + jitter, walk['T'] - self.window_size))
            
        end_idx = start_idx + self.window_size
        
        lin_tensor = torch.tensor(walk['linear'][start_idx:end_idx], dtype=torch.float32)
        ang_tensor = torch.tensor(walk['angular'][start_idx:end_idx], dtype=torch.float32)
        grf_tensor = torch.tensor(walk['grf'][start_idx:end_idx], dtype=torch.float32)
        label = torch.tensor(walk['label'], dtype=torch.long)
        
        if self.mode == 'train' and self.modality_drop_prob > 0:
            if torch.rand(1).item() < self.modality_drop_prob:
                drop_target = torch.randint(0, 3, (1,)).item()
                if drop_target == 0: lin_tensor = torch.zeros_like(lin_tensor)
                elif drop_target == 1: ang_tensor = torch.zeros_like(ang_tensor)
                else: grf_tensor = torch.zeros_like(grf_tensor)
                    
        return {'linear': lin_tensor, 'angular': ang_tensor, 'grf': grf_tensor, 'label': label}

def get_fbg_dataloaders(data_root, train_subjects, test_subjects, batch_size=64, window_size=256, step_size=64):
    train_dataset = FBGIncrementalDataset(
        data_root=data_root, target_subjects=train_subjects, mode='train',
        window_size=window_size, step_size=step_size, modality_drop_prob=0.2
    )
    test_dataset = FBGIncrementalDataset(
        data_root=data_root, target_subjects=test_subjects, mode='test',
        window_size=window_size, step_size=step_size, modality_drop_prob=0.0
    )
    
    # 强制单线程，防止 OOM 与 I/O 阻塞
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    return train_loader, test_loader