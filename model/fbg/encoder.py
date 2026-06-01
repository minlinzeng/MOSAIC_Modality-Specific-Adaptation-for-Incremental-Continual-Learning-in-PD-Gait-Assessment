import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_tasks=3, stride=1, drop_rate=0.2, kernel_size=5, disable_msbn=False):
        super().__init__()
        self.disable_msbn = disable_msbn # Fallback: shared BN instead of MSBN
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.ReLU()
        self.drop1d = nn.Dropout1d(p=drop_rate) 
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        # Dynamic BN assignment
        if self.disable_msbn:
            # Baseline: single shared BN across tasks (statistics can mix)
            self.shared_bn1 = nn.BatchNorm1d(out_channels)
            self.shared_bn2 = nn.BatchNorm1d(out_channels)
        else:
            # Ours: task-isolated MSBN
            self.bn1_list = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_tasks)])
            self.bn2_list = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_tasks)])
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, task_id):
        identity = self.shortcut(x)
        x = self.conv1(x)
        
        # BN routing
        if self.disable_msbn:
            x = self.shared_bn1(x)
        else:
            x = self.bn1_list[task_id](x)
            
        x = self.act(x)
        x = self.drop1d(x) 
        x = self.conv2(x)
        
        if self.disable_msbn:
            x = self.shared_bn2(x)
        else:
            x = self.bn2_list[task_id](x)
            
        return self.act(x + identity)

class MICL_CNN_PD_Model(nn.Module):
    def __init__(self, d_model=64, num_tasks=3, dropout=0.3, disable_msbn=False):
        super().__init__()
        self.input_drop = nn.Dropout1d(p=0.2)
        
        self.enc_lin = nn.Conv1d(137, d_model, kernel_size=7, padding=3)
        self.enc_ang = nn.Conv1d(47, d_model, kernel_size=7, padding=3)
        self.enc_grf = nn.Conv1d(8, d_model, kernel_size=7, padding=3)
        
        # Pass MSBN flag into residual blocks
        self.res1 = ResBlock1D(d_model, d_model, num_tasks, stride=2, drop_rate=dropout, kernel_size=5, disable_msbn=disable_msbn)
        self.res2 = ResBlock1D(d_model, d_model, num_tasks, stride=2, drop_rate=dropout, kernel_size=5, disable_msbn=disable_msbn)
        self.res3 = ResBlock1D(d_model, d_model // 2, num_tasks, stride=2, drop_rate=dropout, kernel_size=5, disable_msbn=disable_msbn)
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model // 2, 2)
        self.noise_std = 0.1
        
    def forward(self, x_lin=None, x_ang=None, x_grf=None, current_task=0):
        if x_lin is not None: x = self.enc_lin(self.input_drop(x_lin.permute(0, 2, 1)))
        elif x_ang is not None: x = self.enc_ang(self.input_drop(x_ang.permute(0, 2, 1)))
        elif x_grf is not None: x = self.enc_grf(self.input_drop(x_grf.permute(0, 2, 1)))
        
        x = self.res1(x, current_task)
        x = self.res2(x, current_task)
        x = self.res3(x, current_task)
        
        pooled = torch.mean(x, dim=2)
        if self.training:
            noise = torch.randn_like(pooled) * self.noise_std
            pooled = pooled + noise
            
        logits = self.head(self.dropout(pooled))
        return logits, pooled