import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from timm.models.vision_transformer import Block

# =====================================================================
# 🌟 1. FOG 专属 1D 分词器 (120 帧感受野)
# =====================================================================
class FOG_TokenEmbedding(nn.Module):
    def __init__(self, in_channels, dim=768, max_len=120): 
        super().__init__()
        self.embeddings = nn.Conv1d(in_channels=in_channels, out_channels=dim, kernel_size=1, stride=1)
        self.embeddings_norm = nn.LayerNorm(dim)
        self.embeddings_pos = nn.Embedding(max_len, dim)

    def forward(self, x):
        embeddings = self.embeddings(x)
        embeddings = embeddings.transpose(1, 2)
        pos_ids = torch.arange(embeddings.size(1), dtype=torch.long, device=x.device)
        pos_embeds = self.embeddings_pos(pos_ids).unsqueeze(0)
        
        embeddings = embeddings + pos_embeds
        embeddings = self.embeddings_norm(embeddings)
        return embeddings

# =====================================================================
# 🌟 2. FOG 专属 Unified Model
# =====================================================================
class FOG_Unified_Model(nn.Module):
    def __init__(self, patch_size=1, embed_dim=768, decoder_embed_dim=512, is_teacher=False):
        super().__init__()
        self.is_teacher = is_teacher
        
        # 严格对齐 FOG 物理通道
        self.patch_embed_acc = FOG_TokenEmbedding(in_channels=3, dim=embed_dim)
        self.patch_embed_gyr = FOG_TokenEmbedding(in_channels=3, dim=embed_dim)
        self.patch_embed_skel = FOG_TokenEmbedding(in_channels=21, dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=12, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if not self.is_teacher:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, num_heads=16, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
                for _ in range(8)
            ]) 
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
            
            # 针对不同模态通道数重建
            self.decoder_pred_acc = nn.Linear(decoder_embed_dim, 3, bias=True)
            self.decoder_pred_gyr = nn.Linear(decoder_embed_dim, 3, bias=True)
            self.decoder_pred_skel = nn.Linear(decoder_embed_dim, 21, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        if not self.is_teacher:
            torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _tokenize(self, x, modality):
        if x.dim() == 3:
            x = x.transpose(1, 2) # [B, T, C] -> [B, C, T]
        elif x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1).transpose(1, 2)
            
        if modality == 'acc': return self.patch_embed_acc(x)
        elif modality == 'gyr': return self.patch_embed_gyr(x)
        elif modality == 'skeleton': return self.patch_embed_skel(x)
        else: raise ValueError(f"Unknown FOG modality: {modality}")

    def random_masking(self, x, mask_ratio=0.75, noise=None):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        if noise is None: noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, noise

    def forward_decoder(self, x, modality, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) 
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  
        x = torch.cat([x[:, :1, :], x_], dim=1) 

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        if modality == 'acc': x = self.decoder_pred_acc(x)
        elif modality == 'gyr': x = self.decoder_pred_gyr(x)
        elif modality == 'skeleton': x = self.decoder_pred_skel(x)
            
        x = x[:, 1:, :]
        return x

    def forward(self, data_dict, mask_ratio=0.75, feature=False, noise=None):
        raw_x = data_dict['data'].clone()
        modality = data_dict['modality']
        
        x = self._tokenize(raw_x, modality)
        x, mask, ids_restore, generated_noise = self.random_masking(x, mask_ratio, noise)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if feature or self.is_teacher:
            return x, generated_noise

        pred = self.forward_decoder(x, modality, ids_restore)
        target = raw_x 
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return (loss, None), pred, mask, ids_restore


# =====================================================================
# 🌟 3. FOG 缓冲数据集 (适配列表映射)
# =====================================================================
# =====================================================================
# 🌟 3. FOG 缓冲数据集 (适配列表映射)
# =====================================================================
class FOG_Buffer_Dataset(Dataset):
    # 🌟 修复 1：增加 seed 和 fold 参数进行物理隔离
    def __init__(self, target_modality, native_dataset, mod_order, buffer_json_dir=None, past_tasks=None, seed=42, fold=0):
        self.target_modality = target_modality
        self.native_dataset = native_dataset
        self.mod_order = mod_order 
        
        self.buffer_indices = []
        self.buffer_modalities = []
        
        if buffer_json_dir and past_tasks:
            for past_mod in past_tasks:
                # 🌟 修复 2：读取带有隔离后缀的文件名
                buffer_path = os.path.join(buffer_json_dir, f"{past_mod}_buffer_seed{seed}_fold{fold}.json")
                if os.path.exists(buffer_path):
                    with open(buffer_path, 'r') as f:
                        raw_indices = json.load(f).get("buffer_indices", [])
                        for item in raw_indices:
                            self.buffer_indices.append(item)
                            self.buffer_modalities.append(past_mod)

    def __len__(self):
        return len(self.native_dataset) + len(self.buffer_indices)

    def __getitem__(self, idx):
        is_buffer = idx >= len(self.native_dataset)
        
        if is_buffer:
            buffer_idx = idx - len(self.native_dataset)
            actual_modality = self.buffer_modalities[buffer_idx]
            actual_idx = self.buffer_indices[buffer_idx]
        else:
            actual_modality = self.target_modality
            actual_idx = idx

        raw_data = self.native_dataset[actual_idx]
        mod_idx = self.mod_order.index(actual_modality)
        
        # FOGLazyDataset 返回 {"xs": [...], "y": tensor, "sid": ...}
        # 需在获取之后剥离多余维度，恢复至 [T, C] (120 长度在前以符合 raw_x 格式)
        x_tensor = raw_data['xs'][mod_idx].clone().detach().float()
        
        # FOG 的 Tensor 已经是 [Channels, Time] (从 data_loader 看来)
        # Unified 模型期望的是 [Time, Channels] 供 Transpose 使用，需翻转
        if x_tensor.size(0) in [3, 21] and x_tensor.size(1) == 120:
            x_tensor = x_tensor.transpose(0, 1)

        return {
            "data": x_tensor,
            "modality": actual_modality,
            "label": raw_data['y']
        }