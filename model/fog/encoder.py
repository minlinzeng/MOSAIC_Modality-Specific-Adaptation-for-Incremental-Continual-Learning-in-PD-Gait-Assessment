# encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
        
WIDTH = 128       # Wider manifold capacity
POOL_SIZE = 1     

class ResBlock1D(nn.Module):
    """
    Standard Residual Block for Time-Series (Large Kernel Enabled).
    Multi-task BN routing for continual learning.
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dilation=1, num_tasks=3):
        super().__init__()
        # Strict padding for causal/equal-length sequences
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1_list = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_tasks)])
        self.act   = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2_list = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_tasks)])
        
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.active_task = 0
    
    def set_task(self, task_id):
        if 0 <= task_id < len(self.bn1_list):
            self.active_task = task_id
        else:
            raise ValueError(f"Task ID {task_id} out of DBN bounds.")

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1_list[self.active_task](out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2_list[self.active_task](out)
        return self.act(out + res)

class KinematicEncoder(nn.Module):
    """
    Kinematic encoder for acc/gyro.
    Input channels: 3
    Receptive field k=15 (~0.5s at 30Hz)
    """
    def __init__(self, in_channels=3, out_channels=WIDTH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.stem(x)


class SkeletonEncoder(nn.Module):
    """
    FOG skeleton encoder (anti anthropometric overfitting).
    Uses frame velocity to remove static anthropometric cues,
    forcing motion/tremor features.
    """
    def __init__(self, in_channels=21, out_channels=WIDTH):
        super().__init__()
        # Weak skeleton signal: RF=15 (0.5s) and strong channel dropout
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout1d(p=0.5) # 50% dropout at bottom to block co-adaptation
        )

    def forward(self, x):
        # x shape: (B, 21, T)
        
        # 🚨 First-order temporal difference (velocity)
        # Maps absolute coords to deltas
        # Static traits vanish after differencing
        velocity = torch.zeros_like(x)
        velocity[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        
        # Keep scaled position + velocity
        # Focuses capacity on motion/tremor
        x_dynamic = velocity + (x * 0.01)
        
        return self.stem(x_dynamic)



class UniversalBackbone(nn.Module):
    def __init__(self, channels=WIDTH, pool_size=POOL_SIZE):
        super().__init__()
        # K=7 dilated conv: RF=127 in 3 layers
        # Anti-overfit with long-range dependency
        dilations = [1, 4, 16]
        
        self.blocks = nn.ModuleList([
            ResBlock1D(channels, channels, kernel_size=7, dilation=d) for d in dilations
        ])
        self.pool = nn.AdaptiveAvgPool1d(pool_size)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x)  # [B, C, P]
        return x.flatten(1)


class CosineLinear(nn.Module):
    """Cosine Similarity Normalization for Latent Clustering"""
    def __init__(self, in_features, out_features, sigma=10.0):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma = Parameter(torch.Tensor(1))
        
        self.reset_parameters()
        self.sigma.data.fill_(sigma)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine_score = F.linear(x_norm, w_norm)
        return self.sigma * cosine_score


class TaskHead(nn.Module):
    def __init__(self, input_dim=WIDTH*POOL_SIZE, num_classes=3):
        super().__init__()
        # 50% dropout blocks overfitting paths
        self.dropout = nn.Dropout(p=0.5) 
        self.fc = CosineLinear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)

class WearGaitUniversal(nn.Module):
    """
    MICL backbone network.
    Encoder dict for FOG three isolated modalities.
    """
    def __init__(self, num_classes=3, disable_dbn=False):
        super().__init__()
        self.disable_dbn = disable_dbn
        
        # Replaced walkway/insole with FOG acc/gyr/skeleton
        self.encoders = nn.ModuleDict({
            "acc":      KinematicEncoder(in_channels=3, out_channels=WIDTH),
            "gyr":      KinematicEncoder(in_channels=3, out_channels=WIDTH),
            "skeleton": SkeletonEncoder(in_channels=21, out_channels=WIDTH),
        })
        
        self.shared_backbone = UniversalBackbone(channels=WIDTH, pool_size=POOL_SIZE)
        self.shared_head = TaskHead(input_dim=WIDTH*POOL_SIZE, num_classes=num_classes)
        self.active_mod = None

    def set_active_modality(self, mod: str):
        if mod not in self.encoders:
            raise ValueError(f"Unknown modality: {mod}. Available: {list(self.encoders.keys())}")
        self.active_mod = mod

    def set_active_task(self, task_id: int):
        target_id = 0 if self.disable_dbn else task_id
        for m in self.modules():
            if hasattr(m, 'set_task'):
                m.set_task(target_id)

    def forward(self, x):
        if self.active_mod is None:
            raise RuntimeError("Active modality not set. Call set_active_modality() first.")
        
        z = self.encoders[self.active_mod](x)   # (B, 64, T)
        features = self.shared_backbone(z)      # (B, 512)
        logits = self.shared_head(features)     # (B, num_classes)
        return logits