import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import your existing sensor encoders to reuse them
# (Assumes this file is in the same directory as encoder.py)
from .encoder import WalkwayEncoder, InsoleEncoder, IMUEncoder, WIDTH, POOL_SIZE

class ResNetBlock1D(nn.Module):
    """
    Standard ResNet Block adapted for 1D time-series.
    Uses Standard BatchNorm (Shared) to test the Baseline SOTA capacity hypothesis.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    """
    A deep 18-layer Residual Network for 1D signals.
    Input: [Batch, 64, Time] (from the sensor encoders)
    Output: [Batch, 512] (Deep features)
    """
    def __init__(self, input_channels=WIDTH, pool_size=POOL_SIZE):
        super(ResNet18Backbone, self).__init__()
        self.inplanes = 64
        
        # Initial Stage: Matches the output of your sensor encoders (64 channels)
        # We start immediately with the ResNet blocks
        self.layer1 = self._make_layer(ResNetBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNetBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNetBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNetBlock1D, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(pool_size)
        
        # Flattening dimension: 512 channels * pool_size (e.g., 512 * 8 = 4096)
        self.out_dim = 512 * pool_size 

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, 64, Time]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNetTaskHead(nn.Module):
    """
    Modified TaskHead to accept the larger dimension from ResNet (512 * 8).
    Includes Dropout to prevent overfitting since the backbone is larger.
    """
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5) 
        self.fc = nn.Linear(input_dim, num_classes)
        # Note: Using standard Linear here for the Sanity Check to be pure.
        # If you want CosineLinear, you can copy it from encoder.py, 
        # but standard Linear is better for checking capacity issues.

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)


class WearGaitResNet18(nn.Module):
    """
    The main container class.
    Drop-in replacement for WearGaitUniversal.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 1. Reuse your existing Modality Encoders
        self.encoders = nn.ModuleDict({
            "walkway": WalkwayEncoder(in_channels=8, out_channels=WIDTH),
            "insole":  InsoleEncoder(in_channels=14, out_channels=WIDTH),
            "imu":     IMUEncoder(in_channels=90, out_channels=WIDTH),
        })
        
        # 2. Use the Deep ResNet Backbone
        self.shared_backbone = ResNet18Backbone(input_channels=WIDTH, pool_size=POOL_SIZE)
        
        # 3. Use a compatible Head (handles 512 channels)
        self.shared_head = ResNetTaskHead(input_dim=self.shared_backbone.out_dim, num_classes=num_classes)
        
        self.active_mod = None

    def set_active_modality(self, mod: str):
        if mod not in self.encoders:
            raise ValueError(f"Unknown modality: {mod}. Available: {list(self.encoders.keys())}")
        self.active_mod = mod

    def set_active_task(self, task_id: int):
        # This is a dummy method to prevent crashing if the training script calls it.
        # The Baseline ResNet does NOT use Dual-BN, so we don't need to route tasks.
        pass

    def forward(self, x):
        if self.active_mod is None:
            raise RuntimeError("Active modality not set. Call set_active_modality() first.")
        
        # 1. Modality Specific Encoder (Shallow)
        z = self.encoders[self.active_mod](x)   
        
        # 2. Shared Backbone (Deep ResNet-18)
        features = self.shared_backbone(z)      
        
        # 3. Shared Head
        logits = self.shared_head(features)     
        return logits