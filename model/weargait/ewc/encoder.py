# encoder.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

WIDTH = 64
POOL_SIZE = 8

class ResBlock1D(nn.Module):
    """
    Standard Residual Block for Time-Series.
    Supports Dilation for expanding receptive field without adding parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, num_tasks=3):
        super().__init__()
        # Calculate padding to keep output length same as input (assuming stride=1)
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        # Create N Batch Norms for N Tasks
        self.bn1_list = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_tasks)])
        self.act   = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2_list = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_tasks)])
        self.active_task = 0
    
    def set_task(self, task_id):
        # self.active_task = 0  
        # return

        if 0 <= task_id < len(self.bn1_list):
            self.active_task = task_id
        else:
            raise ValueError(f"Task ID {task_id} is out of range (Num Tasks: {len(self.bn1_list)})")

    def forward(self, x):
        bn1 = self.bn1_list[self.active_task]
        bn2 = self.bn2_list[self.active_task]

        out = self.act(bn1(self.conv1(x)))
        out = self.act(bn2(self.conv2(out)))
        return out


class WalkwayEncoder(nn.Module):
    """
    Input: 8 Channels (Pressure L/R, Contact L/R, X, Vel, Diff, Load)
    Design: Shallow 1-Layer FCN.
    Logic: Uses a very wide kernel (k=11) to capture the 'Bell Curve' shape of a step
           and the 'Slope' of velocity in a single glance.
    """
    def __init__(self, in_channels=8, out_channels=WIDTH):
        super().__init__()
        self.adapter = nn.Sequential(
            # Kernel 11 covers ~0.36s at 30Hz -> Captures full Stance Phase
            nn.Conv1d(in_channels, out_channels, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.adapter(x)


class InsoleEncoder(nn.Module):
    """
    Input: 38 Channels (16 L-Press + 16 R-Press + 2 Force + 4 CoP)
    Design: Shallow 1-Layer FCN.
    Logic: Kernel 7 smooths the pressure map data. 
           Treats the 32 pressure sensors as a spatial map (Channels) to detect
           asymmetry and roll-over patterns immediately.
    """
    def __init__(self, in_channels=28, out_channels=WIDTH):
        super().__init__()

        self.adapter = nn.Sequential(
            # Kernel 7 covers ~0.23s at 30Hz -> Captures Weight Transfer
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        # self.private_block = ResBlock1D(out_channels, out_channels, kernel_size=3)
        
    def forward(self, x): 
        x = self.adapter(x)
        # x = self.private_block(x)
        return x


class IMUEncoder(nn.Module):
    def __init__(self, in_channels=78, out_channels=WIDTH):
        super().__init__()
        
        self.spatial_fusion = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU()
        )
        
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(16, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.spatial_fusion(x)  # (B, 16, T)
        x = self.temporal_stem(x)   # (B, 64, T)
        return x


class UniversalBackbone(nn.Module):
    """
    Input: (Batch, 64, Time) - Coming from ANY of the above encoders.
    Output: (Batch, 512) - Fixed vector for the classifier.
    Design: 2 Standard ResBlocks + Pooling.
    """
    def __init__(self, channels=WIDTH, pool_size=POOL_SIZE, dropout_p=0.5):
        super().__init__()
        self.block1 = ResBlock1D(channels, channels, kernel_size=3)
        self.block2 = ResBlock1D(channels, channels, kernel_size=3)
        self.spatial_dropout = nn.Dropout1d(p=dropout_p)
        self.pool = nn.AdaptiveAvgPool1d(pool_size)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.spatial_dropout(x)
        x = self.pool(x) # (B, 64, 8)
        x = x.view(x.size(0), -1) # Flatten -> (B, 512)
        return x


class CosineLinear(nn.Module):
    """
    Linear layer with Cosine Similarity Normalization (LUCIR style).
    Output = sigma * (x_norm . w_norm)
    """
    def __init__(self, in_features, out_features, sigma=10.0):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma = Parameter(torch.Tensor(1))
        
        # Init
        self.reset_parameters()
        # Initialize sigma to a reasonable value (e.g., 10 or 20) to help convergence
        self.sigma.data.fill_(sigma)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # 1. Normalize Input Features
        x_norm = F.normalize(x, p=2, dim=1)
        # 2. Normalize Weights
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 3. Calculate Cosine Similarity
        cosine_score = F.linear(x_norm, w_norm)
        
        # 4. Scale by Sigma
        return self.sigma * cosine_score


class TaskHead(nn.Module):
    def __init__(self, input_dim=WIDTH*POOL_SIZE, num_classes=2):
        super().__init__()
        # self.fc = nn.Linear(input_dim, num_classes)
        self.fc = CosineLinear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)


class WearGaitUniversal(nn.Module):
    def __init__(self, num_classes=2, disable_dbn=False):
        super().__init__()
        self.disable_dbn = disable_dbn
        self.encoders = nn.ModuleDict({
            "walkway": WalkwayEncoder(in_channels=8, out_channels=WIDTH),
            "insole":  InsoleEncoder(in_channels=28, out_channels=WIDTH),
            "imu":     IMUEncoder(in_channels=78, out_channels=WIDTH),
        })
        self.shared_backbone = UniversalBackbone(channels=WIDTH, pool_size=POOL_SIZE)
        self.shared_head = TaskHead(input_dim=WIDTH*POOL_SIZE, num_classes=num_classes)
        self.active_mod = None

    def set_active_modality(self, mod: str):
        if mod not in self.encoders:
            raise ValueError(f"Unknown modality: {mod}. Available: {list(self.encoders.keys())}")
        self.active_mod = mod

    def set_active_task(self, task_id: int):
        """
        Recursively sets the task ID for all Dual-BN blocks in the model.
        This ensures both the Backbone and any private Encoder blocks use the correct stats.
        """
        # If DBN is disabled, force all tasks to use Task 0's Batch Norm
        target_id = 0 if self.disable_dbn else task_id
        
        # Iterate over all sub-modules (Backbone, Encoders, etc.)
        for m in self.modules():
            if hasattr(m, 'set_task'):
                m.set_task(target_id)

    def forward(self, x):
        if self.active_mod is None:
            raise RuntimeError("Active modality not set. Call set_active_modality() first.")
        z = self.encoders[self.active_mod](x)   # 1. Modality Specific Encoder
        features = self.shared_backbone(z)      # 2. Shared Backbone
        logits = self.shared_head(features)     # 3. Shared Head
        return logits