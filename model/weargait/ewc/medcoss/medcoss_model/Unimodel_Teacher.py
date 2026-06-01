import torch
from torch import nn
from medcoss_model.Base_module import TokenBaseEmbedding, VideoBaseEmbedding
from timm.models.vision_transformer import Block

class Teacher_Unified_Model(nn.Module):
    def __init__(self, 
                 now_1D_input_size=(112, 1), 
                 now_2D_input_size=(512, 512), 
                 now_3D_input_size=(16, 192, 192), 
                 patch_size=8, 
                 embed_dim=768, 
                 norm_pix_loss=False, **kwargs):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_chans = 1
        self.now_input_size_2D = now_2D_input_size
        self.now_input_size_3D = now_3D_input_size
        self.norm_pix_loss = norm_pix_loss

        # 1. ENCODER COMPONENTS
        self.patch_embed_1D = TokenBaseEmbedding(dim=embed_dim)
        self.video_embed = VideoBaseEmbedding(
            patch_size=patch_size, 
            input_size_2D=now_2D_input_size,
            input_size_3D=now_3D_input_size,
            out_dim=embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=12, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(12)])
        self.norm = nn.LayerNorm(embed_dim)

        self.pos_embed_2D = nn.Parameter(torch.zeros(1, 4097, embed_dim), requires_grad=True)
        self.pos_embed_1D = nn.Parameter(torch.zeros(1, 121, embed_dim), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed_2D, std=.02)
        torch.nn.init.normal_(self.pos_embed_1D, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _determine_true_modality(self, x, modality_string):
        if x.dim() == 3 and x.size(1) == 78: return "text"
        if x.dim() == 4 and x.size(2) == 78: return "text"
        if x.dim() == 4 and x.size(2) != 78: return "2D image"
        if x.dim() == 5: return "3D image"
        return modality_string

    def _tokenize(self, data):
        modality = data.get('modality', None)
        if isinstance(modality, (list, tuple)): modality = modality[0]
        x = data['data']
        
        true_modality = self._determine_true_modality(x, modality)
        data['modality'] = true_modality 

        if true_modality == 'text':
            if x.dim() == 4: x = x.squeeze(1) 
            data['data'] = self.patch_embed_1D(x)
        elif true_modality == '2D image':
            data['data'] = self.video_embed(x)
        elif true_modality == '3D image':
            data['data'] = self.video_embed(x)
        else:
            raise ValueError(f"Teacher routing failed. Modality: {true_modality}, Shape: {x.shape}")

    def random_masking(self, x, mask_ratio=0.75, noise=None):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # Teacher uses the exact noise passed by the student to match the mask
        if noise is None:
            noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, noise

    def forward(self, data, mask_ratio=0.75, feature=False, noise=None):
        # Shape Interceptor for ER batch bugs
        if data['data'].dim() == 4 and data['data'].size(2) == 78 and data['data'].size(1) != 1:
            data['data'] = data['data'][:, 0:1, :, :]
            
        self._tokenize(data)
        x = data['data']
        modality = data['modality'] 

        if modality != "text":
            if x.shape[1] != self.pos_embed_2D.shape[1]:
                x = x + self.pos_embed_2D[:, :x.shape[1], :]
            else:
                x = x + self.pos_embed_2D

        # Pass the synchronized noise in
        x, mask, ids_restore, _ = self.random_masking(x, mask_ratio, noise)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if feature:
            return x

        return x