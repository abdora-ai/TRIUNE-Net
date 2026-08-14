from typing import Optional, Sequence, Tuple, Union, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm


__all__ = ["BaseBlock", "get_conv_layer", "get_padding", "get_output_padding", "DownsampleWithSpaceToDepth"]


class BaseBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError(
            "padding value should not be negative, please change the kernel size and/or stride."
        )
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError(
            "out_padding value should not be negative, please change the kernel size and/or stride."
        )
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]

# class DownsampleWithSpaceToDepth(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        assert in_channels == out_channels, \
            "Gated residual requires in_channels == out_channels"
        expanded = in_channels * 8

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.space_to_depth = SpaceToDepth3D()

        self.compress = nn.Conv3d(
            expanded, out_channels,
            kernel_size=1, bias=True,
            groups=out_channels,
        )

        # Per-channel gate — one scalar per channel
        # init at -6: sigmoid(-6) ≈ 0.002 → pure MaxPool at epoch 0
        self.gate = nn.Parameter(torch.full((out_channels, 1, 1, 1), -6.0))

        with torch.no_grad():
            self.compress.weight.fill_(1.0 / 8)
            self.compress.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mp = self.maxpool(x)                                  # baseline
        learned = self.compress(self.space_to_depth(x))       # correction candidate
        alpha = torch.sigmoid(self.gate)                      # (C, 1, 1, 1), ≈0 at init
        return mp + alpha * (learned - mp)                    # gated residual

class SpaceToDepth3D(nn.Module):
    """
    Rearranges (B, C, D, H, W) → (B, C * 8, D/2, H/2, W/2)
    by folding each 2×2×2 spatial block into the channel dim.
    Assumes all spatial dims are divisible by 2.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = x.reshape(B, C, D//2, 2, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.reshape(B, C * 8, D//2, H//2, W//2)
        return x


class DownsampleWithSpaceToDepth(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        expanded = in_channels * 8  # bs=2, bs³=8

        self.space_to_depth = SpaceToDepth3D()

        # Grouped conv: each output channel only sees its own spatial copies
        # params: in_channels × (out_channels // 8)  vs  in_channels*8 × out_channels
        #self.compress = nn.Sequential(
        #    nn.Conv3d(expanded, out_channels, kernel_size=1, bias=False, groups=out_channels),
        #    nn.BatchNorm3d(out_channels),
        #    nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #)

        # Mean-pool init: each output channel averages its 8 spatial copies
        #with torch.no_grad():
        #    self.compress[0].weight.fill_(1.0 / 8)
        self.compress = nn.Conv3d(
            expanded, out_channels,
            kernel_size=1, bias=True,  # bias=True since no BN
            groups=out_channels
        )

        with torch.no_grad():
            self.compress.weight.fill_(1.0 / 8)
            self.compress.bias.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.space_to_depth(x)
        x = self.compress(x)
        return x

class SpaceToDepth3D_old(nn.Module):
    """
    Rearranges (B, C, D, H, W) → (B, C * bs³, D/bs, H/bs, W/bs)
    by folding each (bs × bs × bs) spatial block into the channel dim.
    Pads with zeros when spatial dims are not divisible by block_size.
    """
    def __init__(self, block_size: int = 2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        bs = self.block_size

        # --- Pad (not truncate) so dims are divisible by block_size ---
        pad_d = (bs - D % bs) % bs
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        if pad_d or pad_h or pad_w:
            # F.pad order: (W_last, W_first, H_last, H_first, D_last, D_first)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        _, _, D, H, W = x.shape          # refresh after padding
        D_new, H_new, W_new = D // bs, H // bs, W // bs

        # Fold spatial blocks into channels
        x = x.reshape(B, C, D_new, bs, H_new, bs, W_new, bs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.reshape(B, C * (bs ** 3), D_new, H_new, W_new)
        return x


class DownsampleWithSpaceToDepth_old(nn.Module):
    def __init__(self, in_channels, out_channels=None, block_size=2):
        super().__init__()
        self.block_size = block_size
        out_channels = out_channels if out_channels is not None else in_channels
        expanded = in_channels * (block_size ** 3)

        self.space_to_depth = SpaceToDepth3D(block_size=block_size)

        # Project expanded channels back down
        self.compress = nn.Sequential(
            nn.Conv3d(expanded, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Initialize compress conv as a max-approximating mean
        # so early training behaves close to avgpool, not random noise
        self._init_compress_as_mean(in_channels, out_channels, block_size)

    def _init_compress_as_mean(self, in_channels, out_channels, block_size):
        """
        Initialize weights so each output channel is the mean of its
        corresponding input channel across the block_size^3 copies.
        This makes the module behave like AvgPool at init, giving a
        stable starting point rather than random compression.
        """
        factor = block_size ** 3
        with torch.no_grad():
            w = self.compress[0].weight  # (out_channels, expanded, 1, 1, 1)
            w.zero_()
            # Each output channel i gets 1/factor weight on channels i, i+out, i+2*out...
            for i in range(min(in_channels, out_channels)):
                for k in range(factor):
                    w[i, i + k * in_channels, 0, 0, 0] = 1.0 / factor

    def forward(self, x):
        x = self.space_to_depth(x)
        x = self.compress(x)
        return x