import torch
from torch import nn
from medcoss_model.Base_module import TokenBaseEmbedding, VideoBaseEmbedding
from timm.models.vision_transformer import Block

class Unified_Model(nn.Module):
    def __init__(self, 
                 now_1D_input_size=(112, 1), 
                 now_2D_input_size=(512, 512), 
                 now_3D_input_size=(16, 192, 192), 
                 patch_size=8, 
                 embed_dim=768, 
                 decoder_embed_dim=512, 
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

        # 2. DECODER COMPONENTS
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads=16, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(8)]) 
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        self.decoder_pred_1D = nn.Linear(decoder_embed_dim, 78, bias=True)
        self.decoder_pos_embed_2D = nn.Parameter(torch.zeros(1, 4097, decoder_embed_dim), requires_grad=True)
        self.decoder_pos_embed_1D = nn.Parameter(torch.zeros(1, 121, decoder_embed_dim), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed_2D, std=.02)
        torch.nn.init.normal_(self.pos_embed_1D, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_2D, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_1D, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
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

    def _determine_true_modality(self, x, modality_string):
        """The Ultimate Safety Net: Routes purely by tensor dimensions to prevent collate bugs."""
        if x.dim() == 3 and x.size(1) == 78:
            return "text"
        if x.dim() == 4 and x.size(2) == 78:
            return "text"
        if x.dim() == 4 and x.size(2) != 78:
            return "2D image"
        if x.dim() == 5:
            return "3D image"
        return modality_string

    def _tokenize(self, data):
        # Clean the tuple if PyTorch wrapped it
        modality = data.get('modality', None)
        if isinstance(modality, (list, tuple)):
            modality = modality[0]

        x = data['data']
        
        # Override with physical shape truth
        true_modality = self._determine_true_modality(x, modality)
        data['modality'] = true_modality 

        if true_modality == 'text':
            if x.dim() == 4: x = x.squeeze(1) # Squeeze dummy dim back out for 1D CNN
            data['data'] = self.patch_embed_1D(x)
            
        elif true_modality == '2D image':
            data['data'] = self.video_embed(x)
            
        elif true_modality == '3D image':
            data['data'] = self.video_embed(x)
            
        else:
            raise ValueError(f"Routing failed entirely. Modality: {true_modality}, Shape: {x.shape}")

    def random_masking(self, x, mask_ratio=0.75, noise=None):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

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

    def forward_decoder(self, x, modality, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) 
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  
        x = torch.cat([x[:, :1, :], x_], dim=1) 

        pos_embed = self.decoder_pos_embed_1D if modality == "text" else self.decoder_pos_embed_2D

        if x.shape[1] != pos_embed.shape[1]:
            x = x + pos_embed[:, :x.shape[1], :]
        else:
            x = x + pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # --- NEW: Route to the correct prediction head ---
        if modality == "text":
            x = self.decoder_pred_1D(x)  # Outputs 78 values for IMU
        else:
            x = self.decoder_pred(x)     # Outputs 64 values for 2D Mats
            
        x = x[:, 1:, :] # Remove CLS token

        return x
    
    def forward(self, data, mask_ratio=0.75, feature=False):
        raw_imgs = data['data'].clone()
        
        self._tokenize(data)
        x = data['data']
        modality = data['modality'] # Now contains the bulletproof string

        if modality != "text":
            if x.shape[1] != self.pos_embed_2D.shape[1]:
                x = x + self.pos_embed_2D[:, :x.shape[1], :]
            else:
                x = x + self.pos_embed_2D

        x, mask, ids_restore, generated_noise = self.random_masking(x, mask_ratio)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if feature:
            return x, generated_noise

        pred = self.forward_decoder(x, modality, ids_restore)
        loss = self.forward_loss(raw_imgs, pred, mask, modality)

        # We add a 4th return value to satisfy engine_pretrain_er.py
        return loss, pred, mask, ids_restore
 
    def forward_loss(self, imgs, pred, mask, modality):
        target = self.patchify(imgs, modality)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, (None, None)

    def patchify(self, imgs, modality):
        if modality == "text":
            if imgs.dim() == 4:
                imgs = imgs.squeeze(1)
            x = imgs.transpose(1, 2)
            
        elif modality == "2D image":
            c = self.in_chans 
            h_img, w_img = imgs.shape[2], imgs.shape[3]
            h, w = h_img // self.patch_size, w_img // self.patch_size
            
            x = imgs.reshape(shape=(imgs.shape[0], c, h, self.patch_size, w, self.patch_size))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, self.patch_size ** 2 * c))
            
        elif modality == "3D image":
            c = self.in_chans
            d_img, h_img, w_img = imgs.shape[2], imgs.shape[3], imgs.shape[4]
            d, h, w = d_img // self.patch_size, h_img // self.patch_size, w_img // self.patch_size
            
            x = imgs.reshape(shape=(imgs.shape[0], c, d, self.patch_size, h, self.patch_size, w, self.patch_size))
            x = torch.einsum('ncdkhpwq->ndhwkpqc', x)
            x = x.reshape(shape=(imgs.shape[0], d * h * w, self.patch_size ** 3 * c))
            
        return x