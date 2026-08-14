# nnunetv2/model_sharing/lhunetV2/modules/parallel_attention.py

import torch
import torch.nn as nn


class ParallelDualAttention(nn.Module):
    """
    Parallel Spatial + Channel Attention using original LHUNet's attention modules
    """
    def __init__(
        self,
        input_size: int,
        dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        print(f"[ParallelDualAttention] dim={dim}, num_heads={num_heads}, input_size={input_size}")
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) should be divisible by num_heads ({num_heads})")
        proj_size = dim
        # Spatial attention (from original LHUNet)
        self.spatial_attention = SpatialAttention(
            input_size=input_size,
            hidden_size=dim,
            proj_size=proj_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate / 2,
            use_norm=True,
            use_temperature=True,
        )

        # Channel attention (from original LHUNet)
        self.channel_attention = ChannelAttention(
            input_size=input_size,
            hidden_size=dim,
            proj_size=proj_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate / 2,
            use_norm=True,
            use_temperature=True,
        )

        # Learnable fusion weights
        self.spatial_weight = nn.Parameter(torch.tensor(0.5))
        self.channel_weight = nn.Parameter(torch.tensor(0.5))

        # LayerNorm and positional embedding
        self.norm = nn.LayerNorm(dim)
        self.pos_embed = nn.Parameter(1e-6 + torch.zeros(1, input_size, dim))
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        N = D * H * W
        
        # Flatten and add positional embedding
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x_flat = x_flat + self.pos_embed
        x_norm = self.norm(x_flat)
        
        # Parallel attention
        x_spatial = self.spatial_attention(x_norm, B, C, D, H, W)
        x_channel = self.channel_attention(x_norm, B, C, D, H, W)
        
        # Learnable fusion
        x_fused = (self.spatial_weight * x_spatial + 
                   self.channel_weight * x_channel)
        
        # Residual + projection
        x_out = x_flat + x_fused
        x_out = self.out_proj(x_out)
        
        # Reshape back
        x_out = x_out.transpose(1, 2).reshape(B, C, D, H, W)
        
        return x_out


class ChannelAttention(nn.Module):
    """Channel attention from original LHUNet (unchanged)"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        use_norm: bool = False,
        use_temperature: bool = False,
        dropout_rate: float = 0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.use_norm = use_norm
        self.use_dropout = dropout_rate > 0
        self.use_temperature = use_temperature

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out = nn.Linear(hidden_size, hidden_size)

        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape
        head_dim = C // self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, v_CA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attn_CA = query @ key.transpose(-2, -1)
        if self.use_temperature:
            attn_CA *= self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        if self.use_dropout:
            attn_CA = self.dropout(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        
        if self.use_norm:
            x_CA = self.norm(x_CA)
        x_CA = self.out(x_CA)
        return x_CA


class SpatialAttention(nn.Module):
    """Spatial attention from original LHUNet (unchanged)"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        use_norm: bool = False,
        use_temperature: bool = False,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_norm = use_norm
        self.use_dropout = dropout_rate > 0
        self.use_temperature = use_temperature

        self.E = self.F = nn.Linear(input_size, proj_size)

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out = nn.Linear(hidden_size, hidden_size)

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, B_in, C_in, H, W, D):
        B, N, C = x.shape
        head_dim = C // self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, v_SA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_projected = self.E(key)
        v_SA_projected = self.F(v_SA)

        query = torch.nn.functional.normalize(query, dim=-1)

        attn_SA = query.permute(0, 1, 3, 2) @ k_projected
        if self.use_temperature:
            attn_SA *= self.temperature
        attn_SA = attn_SA.softmax(dim=-1)
        if self.use_dropout:
            attn_SA = self.dropout(attn_SA)
        x_SA = (
            (attn_SA @ v_SA_projected.transpose(-2, -1))
            .permute(0, 3, 1, 2)
            .reshape(B, N, C)
        )
        
        if self.use_norm:
            x_SA = self.norm(x_SA)
        x_SA = self.out(x_SA)
        return x_SA