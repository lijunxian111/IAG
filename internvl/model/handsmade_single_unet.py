import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (normalized_shape
                                 if isinstance(normalized_shape, tuple)
                                 else (normalized_shape,))
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        x = x / (rms + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x


class SimpleCrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim_q // heads) ** -0.5

        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

        self.to_q = nn.Linear(dim_q, dim_q)
        self.to_k = nn.Linear(dim_kv, dim_q)
        self.to_v = nn.Linear(dim_kv, dim_q)
        self.to_out = nn.Sequential(
            nn.Linear(dim_q, dim_q),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        B, N, C = x.shape
        H = self.heads
        residual = x
        x = self.norm_q(x)
        context = self.norm_kv(context)

        q = self.to_q(x).reshape(B, N, H, -1).transpose(1, 2)
        k = self.to_k(context).reshape(B, context.size(1), H, -1).transpose(1, 2)
        v = self.to_v(context).reshape(B, context.size(1), H, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, H, N, C//H)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        return residual + self.to_out(out)  # residual connection
    
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        return self.pool(x), x  # pooled, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)
class UNetWithTextCond(nn.Module):

    def __init__(self, img_ch: int = 3, base_ch: int = 16, text_dim: int = 768):
        super().__init__()

        # ---------- Encoder ----------
        self.down1 = DownBlock(img_ch, base_ch)                    # 16
        self.down2 = DownBlock(base_ch, base_ch * 2)               # 32
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)           # 64

        # ---------- Bottleneck ----------
        self.middle_conv = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, padding=1),     # 64 → 128
            nn.ReLU(),
            nn.Conv2d(base_ch * 8, base_ch * 4, 3, padding=1),     # 128 → 64
        )

        # ---------- Decoder ----------
        self.up3 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)  # 64 → 32
        self.up2 = UpBlock(base_ch * 2, base_ch * 2, base_ch)      # 32 → 16
        self.up1 = UpBlock(base_ch, base_ch, base_ch)              # 16 → 16

        self.out_conv = nn.Conv2d(base_ch, img_ch, 1)

        # ---------- Cross‑attention (text conditioning) ----------
        self.cross_attn_mid = SimpleCrossAttention(dim_q=base_ch * 4, dim_kv=text_dim)   # 64‑d queries
        self.cross_attn_up3 = SimpleCrossAttention(dim_q=base_ch * 2, dim_kv=text_dim)   # 32‑d queries
        self.cross_attn_up2 = SimpleCrossAttention(dim_q=base_ch,     dim_kv=text_dim)   # 16‑d queries
        self.cross_attn_up1 = SimpleCrossAttention(dim_q=base_ch,     dim_kv=text_dim)   # 16‑d queries

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : (B, img_ch,  H,  W)          input image / feature
            text_emb : (B, N_txt, text_dim)         embedded text tokens
        Returns:
            (B, img_ch, H, W)                        predicted image
        """

        # ---------- Encoder ----------
        x, skip1 = self.down1(x)     # (B,  16, H/2 , W/2 )
        x, skip2 = self.down2(x)     # (B,  32, H/4 , W/4 )
        x, skip3 = self.down3(x)     # (B,  64, H/8 , W/8 )

        # ---------- Bottleneck with attention ----------
        x = self.middle_conv(x)      # (B,  64, H/8 , W/8 )
        B, C, H_, W_ = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)          # (B, N, 64)
        x_flat = self.cross_attn_mid(x_flat, text_emb)
        x = x_flat.transpose(1, 2).view(B, C, H_, W_)

        # ---------- Decoder level 3 ----------
        x = self.up3(x, skip3)       # (B,  32, H/4 , W/4 )
        B, C, H_, W_ = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)
        x_flat = self.cross_attn_up3(x_flat, text_emb)
        x = x_flat.transpose(1, 2).view(B, C, H_, W_)

        # ---------- Decoder level 2 ----------
        x = self.up2(x, skip2)       # (B,  16, H/2 , W/2 )
        B, C, H_, W_ = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)
        x_flat = self.cross_attn_up2(x_flat, text_emb)
        x = x_flat.transpose(1, 2).view(B, C, H_, W_)

        # ---------- Decoder level 1 ----------
        x = self.up1(x, skip1)       # (B,  16, H   , W   )
        B, C, H_, W_ = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)
        x_flat = self.cross_attn_up1(x_flat, text_emb)
        x = x_flat.transpose(1, 2).view(B, C, H_, W_)

        return self.out_conv(x)
