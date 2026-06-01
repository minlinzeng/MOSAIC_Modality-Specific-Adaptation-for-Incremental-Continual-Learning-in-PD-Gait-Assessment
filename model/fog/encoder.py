# encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
        
WIDTH = 128       # 适度拓宽流形带宽
POOL_SIZE = 1     

class ResBlock1D(nn.Module):
    """
    Standard Residual Block for Time-Series (Large Kernel Enabled).
    具备连续学习多任务 Batch Normalization 路由机制。
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dilation=1, num_tasks=3):
        super().__init__()
        # 严格的 Padding 物理对齐，确保因果序列或等长序列不崩塌
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
    通用物理运动学编码器，适用于 Accelerometer 和 Gyroscope。
    输入通道: 3
    感受野: k=15 (在 30Hz 下覆盖 0.5s 的局部步态动力学)
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
    FOG 专用的抗体型过拟合骨架编码器。
    通过内部强制计算帧间速度 (Velocity)，彻底抹除身高等静态生物特征，
    逼迫网络只能学习“运动和震颤”。
    """
    def __init__(self, in_channels=21, out_channels=WIDTH):
        super().__init__()
        # 骨架的动态特征非常微弱，扩大感受野到 15 (0.5秒)，并加入极强的通道级 Dropout
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout1d(p=0.5) # 在最底层直接随机切断 50% 的神经元，防止协同记忆
        )

    def forward(self, x):
        # x 的维度: (Batch, 21, Time)
        
        # 🚨 核弹级特征工程：计算一阶时间差分 (Velocity)
        # 这会将绝对坐标 [x_t, y_t] 变为相对运动量 [Δx, Δy]
        # 身高、腿长等静态常量在差分后全部归零！
        velocity = torch.zeros_like(x)
        velocity[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        
        # 将原坐标微弱保留（除以 100 压制其权重），与速度特征相加
        # 这样网络 99% 的注意力被迫集中在运动速度（震颤）上
        x_dynamic = velocity + (x * 0.01)
        
        return self.stem(x_dynamic)



class UniversalBackbone(nn.Module):
    def __init__(self, channels=WIDTH, pool_size=POOL_SIZE):
        super().__init__()
        # K=7 配合指数空洞率：在 3 层内达到 Receptive Field = 127
        # 严格防御过拟合，同时保障长程马尔可夫依赖
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
        # 加入 50% 的 Dropout 强行截断过拟合路径
        self.dropout = nn.Dropout(p=0.5) 
        self.fc = CosineLinear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)

class WearGaitUniversal(nn.Module):
    """
    Modality-Incremental Continual Learning (MICL) 主力网络架构。
    模块字典已更新为适配 FOG 物理隔离的三大模态流形。
    """
    def __init__(self, num_classes=3, disable_dbn=False):
        super().__init__()
        self.disable_dbn = disable_dbn
        
        # 核心修改：移除原有的步道与鞋垫模块，替换为 FOG 的三大基础物理源
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