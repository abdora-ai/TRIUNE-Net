import torch
import torch.nn as nn
from ..cnn import *

class MultiScaleContextAggregator(nn.Module):
    def __init__(self, dim, dilations=[2, 3, 4]):
        super().__init__()
        self.dilations = dilations
        num_groups = len(dilations)

        self.split_size = dim // num_groups  # FIX: was [dim // num_groups] (a list)

        self.norm = nn.LayerNorm(dim)        # norm lives here now

        self.ccu = CCU3D(dim)

        self.dw_convs = nn.ModuleList()
        for i, d in enumerate(dilations):
            self.dw_convs.append(
                nn.Conv3d(self.split_size, self.split_size, kernel_size=3,
                          padding=d, dilation=d, groups=self.split_size)
            )

        self.gate = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.SiLU(inplace=True)
        )
        self.value_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.SiLU(inplace=True)
        )
        self.fusion = nn.Conv3d(dim, dim, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        shortcut = x

        # Norm after shortcut, before processing
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)

        x = self.ccu(x)

        splits = torch.split(x, self.split_size, dim=1)
        outputs = [dw_conv(split) for split, dw_conv in zip(splits, self.dw_convs)]
        multi_scale = torch.cat(outputs, dim=1)

        g = self.gate(x)
        v = self.value_proj(multi_scale)
        x = self.fusion(g * v)

        return x + shortcut

class CCU3D(nn.Module):
    """
    Channel Calibration Unit from CENet
    Uses max, mean, and std pooling for rich channel statistics
    """
    def __init__(self, channels, hidden_scale=3):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, hidden_scale * channels, kernel_size=3, 
                             groups=channels, bias=False, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden_scale * channels, channels, kernel_size=1, 
                             groups=channels, bias=False, padding=0)
        self.bn = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Global pooling across spatial dimensions
        x_max = torch.max(x.view(B, C, -1), dim=2)[0]  # (B, C)
        x_mean = torch.mean(x, dim=(2, 3, 4))          # (B, C)
        x_std = torch.std(x, dim=(2, 3, 4), unbiased=False)  # (B, C)
        
        # Stack: (B, C, 3)
        u = torch.stack([x_max, x_mean, x_std], dim=-1)
        
        # Channel calibration via 1D convolutions
        z = self.fc2(self.act(self.fc1(u))).view(B, C)
        if B > 1:
            z = self.bn(z)
        g = torch.sigmoid(z).reshape(B, C, 1, 1, 1)
        
        return x * g

class MLPWithDWConv(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        self.norm = nn.LayerNorm(dim)        # norm lives here now
        self.fc1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3,
                                padding=1, groups=hidden_dim)
        self.fc2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x

        # Norm after shortcut, before processing
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)

        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)

        return x + shortcut


class HybAttnBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        dim: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        cnn_block_code="d",
        vit_block_code="c",
        use_rb=False,
        use_r=False,
        stage_id=None,
    ) -> None:
        super().__init__()

        if stage_id == 3:
            dilations = [2, 3, 4, 5]
        elif stage_id == 4:
            dilations = [1, 2]
        else:
            dilations = [1]

        print("=" * 50)
        print(f"MCALeaderBlock Stage {stage_id} (Serial: MCA + LKAd + MLP)")
        print(f"  - dim: {dim}")
        print(f"  - dilations: {dilations}")
        print(f"  - LKAd: {cnn_block_code}")
        print("=" * 50)

        # Each block owns its norm internally now
        self.mca = MultiScaleContextAggregator(dim, dilations=dilations)
        self.lkad = LKAd3D_Block(d_model=dim)
        self.mlp = MLPWithDWConv(dim, expansion_ratio=4)

        # Kept for output stability
        self.bnorm = nn.BatchNorm3d(dim)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        # Each sub-block handles its own pre-norm + residual internally
        x = self.mca(x)
        x = self.lkad(x)
        x = self.mlp(x)

        x = self.dropout(x)
        x = self.bnorm(x)

        return x