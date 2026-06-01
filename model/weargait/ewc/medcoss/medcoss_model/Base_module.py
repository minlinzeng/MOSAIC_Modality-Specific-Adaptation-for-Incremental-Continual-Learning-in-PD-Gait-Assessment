import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, Callable
from torch import Tensor
import math
import timm.optim as optim_factory
from einops import rearrange
from timm.layers import DropPath
# Try to use APEX for faster LayerNorm if available, otherwise fallback
try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    has_fused_layernorm = True
    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda: return super().forward(x)
            else:
                with torch.cuda.device(x.device): return super().forward(x)
except ImportError:
    has_fused_layernorm = False

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, drop_path_ratio: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_scale: bool = False, ls_init_values: float = 1e-3,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
        super(TransformerEncoderLayer, self).__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.batch_first = batch_first

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma_1 = nn.Parameter(ls_init_values * torch.ones((d_model)), requires_grad=True)
            self.gamma_2 = nn.Parameter(ls_init_values * torch.ones((d_model)), requires_grad=True)

        self.activation = gelu_new if activation == "gelu" else F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        x = self.drop_path1(self.dropout1(x))
        if self.layer_scale: x = self.gamma_1 * x
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.drop_path2(self.dropout2(x))
        if self.layer_scale: x = self.gamma_2 * x
        return x

class NNEmbeddingEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(NNEmbeddingEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)

    def forward(self, x, start_time=0):
        if isinstance(x, int):
            return self.position_embeddings(torch.tensor([x], dtype=torch.long, device='cuda'))
        elif isinstance(x, torch.Tensor) and x.dim() == 1:
            return self.position_embeddings(x)
        else:
            position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device) + start_time
            return self.position_embeddings(position_ids)


class TokenBaseEmbedding(nn.Module):
    def __init__(self, dim=768, **kwargs):
        super(TokenBaseEmbedding, self).__init__()
        # WearGait IMU Adapter: Updated to 78 channels
        self.embeddings = nn.Conv1d(in_channels=78, out_channels=dim, kernel_size=1, stride=1)
        self.embeddings_norm = nn.LayerNorm(dim)
        self.embeddings_pos = NNEmbeddingEncoding(dim, 512)
        self.pos_before = True

    def forward(self, input_ids):
        # 1. Squeeze the dummy dimension: [Batch, 1, 78, 120] -> [Batch, 78, 120]
        if input_ids.dim() == 4:
            input_ids = input_ids.squeeze(1) 
            
        # Error check before the conv call
        if input_ids.size(1) != 78:
            raise ValueError(f"Conv1d expected 78 channels, got {input_ids.size(1)}. Input shape: {input_ids.shape}")

        # 2. Apply Conv1d: [Batch, 78, 120] -> [Batch, Dim, 120]
        embeddings = self.embeddings(input_ids)

        # 3. Transpose for the Vision Transformer: [Batch, Dim, Length] -> [Batch, Length, Dim]
        embeddings = embeddings.transpose(1, 2)

        if self.embeddings_pos is not None:
            # Add positional embeddings along the sequence length (120)
            position_embeddings = self.embeddings_pos(embeddings)
            embeddings = embeddings + position_embeddings.to(embeddings.dtype)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        return embeddings


class VideoBaseEmbedding(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, patch_size=8, input_size_3D=None, input_size_2D=None):
        super(VideoBaseEmbedding, self).__init__()
        self.patch_size = patch_size
        self.time_span = 1
        self.pos_before = True

        # Calculate spatial grid size for 2D
        max_spatial_size_2D = (input_size_2D[0] // patch_size) * (input_size_2D[1] // patch_size)
        self.embeddings_norm = nn.LayerNorm(out_dim)
        
        self.embeddings_st_pos_2D = Divide_ST_POS(max_spatial_size_2D, 8, out_dim, True)
        
        # WearGait Spatial Adapter: 1 channel (pressure mat) instead of 3 (RGB)
        self.embeddings = nn.Conv2d(1, out_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, data):
        # --- SAFETY INITIALIZATION ---
        embeddings = None
        
        # 1. Handle 3D Volumes (CT/MR) [Batch, 1, Depth, H, W] by taking middle slice or pooling
        if data.dim() == 5:
            # If we get 3D data but only have a 2D embedder, we take the middle slice
            # This prevents a crash while allowing the pipeline to move forward
            data = data[:, :, data.size(2)//2, :, :] 
            
        # 2. Handle 2D Images (Walkway/Insole) [Batch, 1, H, W]
        if data.dim() == 4:
            bs = data.size(0)
            x = self.embeddings(data) 
            x = x.flatten(2) 
            # Project to latent space
            embeddings = rearrange(x, '(b t s) c hw -> b t hw (s c)', b=bs, s=self.time_span)
            
            # Apply spatio-temporal positional encoding
            embeddings_pos = self.embeddings_st_pos_2D(embeddings).unsqueeze(0).flatten(1, 2)
            embeddings = embeddings.flatten(1, 2)
            
            if self.pos_before:
                embeddings = embeddings + embeddings_pos.to(embeddings.dtype)
                
            if self.embeddings_norm is not None:
                embeddings = self.embeddings_norm(embeddings)
        
        # 3. Final Safety Check
        if embeddings is None:
            raise ValueError(f"VideoBaseEmbedding received invalid data shape: {data.shape}. "
                             f"Expected 4D [B, C, H, W] or 5D [B, C, D, H, W].")

        return embeddings

class Divide_ST_POS(nn.Module):
    def __init__(self, num_patches, max_time_len, out_dim, random_temporal_pos):
        super(Divide_ST_POS, self).__init__()
        self.spatial_pos_embed = nn.Embedding(num_patches, out_dim)
        self.temporal_pos_embed = nn.Embedding(max_time_len, out_dim)
        self.spatial_pos_embed_index = 0 
        self.max_frames = max_time_len
        self.random_temporal_pos = random_temporal_pos

    def forward(self, x):
        dtype = x.dtype
        temp_len, spatial_size = x.size(1), x.size(2)

        if self.training and self.random_temporal_pos:
            temporal_pos_ids = torch.arange(temp_len, dtype=torch.long, device=x.device) + \
                torch.randint(0, self.max_frames - temp_len + 1, size=(1,), dtype=torch.long, device=x.device)
        else:
            temporal_pos_ids = torch.arange(temp_len, dtype=torch.long, device=x.device)
            
        pos_embed = self.temporal_pos_embed(temporal_pos_ids).unsqueeze(1).to(dtype=dtype) + \
            self.spatial_pos_embed(torch.arange(start=self.spatial_pos_embed_index, end=spatial_size + self.spatial_pos_embed_index, dtype=torch.long, device=x.device)).unsqueeze(0).to(dtype=dtype)
        return pos_embed


