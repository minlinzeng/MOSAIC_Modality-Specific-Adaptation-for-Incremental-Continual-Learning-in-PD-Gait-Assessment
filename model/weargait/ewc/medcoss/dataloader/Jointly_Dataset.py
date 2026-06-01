import sys
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

project_root = Path(__file__).resolve().parents[5]
sys.path.append(str(project_root))

from model.weargait.ewc.data_loader import preload_all_subjects, prepare_split, make_sync_loaders

class Buffer_Dataset(Dataset):
    def __init__(self, task_modality, args, train_subs, test_subs, subj2label, is_training=True):
        self.mod_map = {'1D_text': 'imu', '2D_xray': 'walkway', '2D_path': 'insole'}
        self.forward_mod_map = {'1D_text': 'text', '2D_xray': '2D image', '2D_path': '2D image'}
        
        self.medcoss_modality = task_modality
        self.weargait_modality = self.mod_map[self.medcoss_modality]
        self.forward_modality = self.forward_mod_map[self.medcoss_modality]

        self.data_cache = preload_all_subjects()
        
        self.needed_mods = ['imu', self.weargait_modality]
        self.needed_mods = list(dict.fromkeys(self.needed_mods)) 
        
        self.prep_data = prepare_split(
            train_subs, test_subs, self.data_cache, 
            win=args.win_len, hop=args.hop_len, 
            modalities=tuple(self.needed_mods)
        )
        
        self.native_mod_order = list(self.prep_data['modalities'])
        
        train_loader, test_loader = make_sync_loaders(self.prep_data, subj2label, batch_size=1, num_workers=0)
        self.native_dataset = train_loader.dataset if is_training else test_loader.dataset

        self.buffer_indices = []
        self.buffer_modalities = [] # Tracks which modality each buffer index belongs to
        
        if is_training:
            base_dir = args.load_current_pretrained_weight.rsplit("/", 1)[0] if getattr(args, 'load_current_pretrained_weight', "") else args.output_dir
            
            # List of all past tasks we want to rehearse
            past_tasks = ['1D_text', '2D_xray'] 
            
            for past_mod in past_tasks:
                # Assuming you save all buffers in the main log dir
                from model.paths import WEARGAIT_MEDCOSS_LOG, as_str
                main_log_dir = as_str(WEARGAIT_MEDCOSS_LOG) + os.sep
                buffer_path = os.path.join(main_log_dir, f"{past_mod}_buffer_indices.json")
                
                if os.path.exists(buffer_path):
                    with open(buffer_path, 'r') as f:
                        raw_indices = json.load(f).get("buffer_indices", [])
                        
                        # Flatten the K-Means clusters just like before
                        for item in raw_indices:
                            if isinstance(item, list):
                                self.buffer_indices.extend(item)
                                self.buffer_modalities.extend([past_mod] * len(item))
                            else:
                                self.buffer_indices.append(item)
                                self.buffer_modalities.append(past_mod)
                                
                    print(f"✅ Rehearsal Buffer Loaded: {past_mod} ")

    def __len__(self):
        return len(self.native_dataset) + len(self.buffer_indices)

    def __getitem__(self, idx):
        is_buffer = idx >= len(self.native_dataset)
        
        if is_buffer:
            # Figure out which old task this buffer index belongs to
            buffer_idx = idx - len(self.native_dataset)
            past_mod = self.buffer_modalities[buffer_idx]
            
            target_mod = self.mod_map[past_mod]       # 'imu' or 'walkway'
            target_fwd = self.forward_mod_map[past_mod] # 'text' or '2D image'
            actual_idx = self.buffer_indices[buffer_idx]
        else:
            target_mod = self.weargait_modality
            target_fwd = self.forward_modality
            actual_idx = idx

        native_item = self.native_dataset[actual_idx]
        
        if isinstance(native_item, (tuple, list)):
            x_package, y_data = native_item[0], native_item[1]
        else:
            x_package = native_item.get('x', native_item)
            y_data = native_item.get('y', native_item.get('label', -1))

        x_target = None
        if isinstance(x_package, (list, tuple)):
            try:
                mod_idx = self.native_mod_order.index(target_mod)
                x_target = x_package[mod_idx]
            except (ValueError, IndexError):
                x_target = x_package[0]
        elif isinstance(x_package, dict):
            x_target = x_package.get(target_mod, list(x_package.values())[0])
        else:
            x_target = x_package

        while isinstance(x_target, (list, tuple)) and len(x_target) > 0:
            x_target = x_target[0]

        if torch.is_tensor(x_target):
            x = x_target.clone().detach().float()
        else:
            x = torch.from_numpy(np.array(x_target, dtype=np.float32)).float()
        
        # MATHEMATICAL FIX: Force geometric reduction to a 2D matrix
        while x.dim() > 2:
            x = x[0]
            
        x = x.unsqueeze(0)
        
        return {
            "data": x,
            "modality": target_fwd,
            "label": y_data 
        }