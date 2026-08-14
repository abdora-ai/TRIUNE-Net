import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F


class CrossPSA3D(nn.Module):
    """
    Boundary-enhanced decoder-guided skip connection alignment.

    Pipeline:
        1. Extract boundary signal via downsample→upsample→subtract (unsharp mask)
           - scale adapted automatically based on spatial resolution
           - skipped entirely at 6³ (too small, signal is noise)
        2. CrossPSA gates the enhanced skip using decoder as query:
           - spatial path: decoder queries which spatial locations in skip matter
           - channel path: decoder queries which channels in skip matter

    Args:
        skip_channels    : channels of encoder skip connection
        decoder_channels : channels of decoder (upsampled) feature
    """
    def __init__(self, skip_channels: int, decoder_channels: int):
        super().__init__()
        self.inter_channels = max(skip_channels // 2, 1)

        # Learnable per-channel boundary enhancement weight
        # >0: enhance boundaries, =0: ignore, <0: suppress
        # self.boundary_weight = nn.Parameter(torch.ones(skip_channels, 1, 1, 1))
        self.boundary_weight = nn.Parameter(torch.zeros(skip_channels, 1, 1, 1))
        self.blend = nn.Parameter(torch.tensor(-6.0))          
        # ---- Spatial attention path ----
        self.q_spatial       = nn.Conv3d(decoder_channels, 1, kernel_size=1, bias=False)
        self.v_spatial       = nn.Conv3d(skip_channels, self.inter_channels, kernel_size=1, bias=False)
        self.proj_spatial    = nn.Conv3d(self.inter_channels, skip_channels, kernel_size=1, bias=False)
        self.softmax_spatial = nn.Softmax(dim=2)

        # ---- Channel attention path ----
        self.q_channel       = nn.Conv3d(decoder_channels, self.inter_channels, kernel_size=1, bias=False)
        self.avg_pool        = nn.AdaptiveAvgPool3d(1)
        self.v_channel       = nn.Conv3d(skip_channels, self.inter_channels, kernel_size=1, bias=False)
        self.softmax_channel = nn.Softmax(dim=2)

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_boundary(self, skip: torch.Tensor) -> torch.Tensor:
        """
        Unsharp-mask boundary extraction with resolution-adaptive scale.

            low_freq = upsample(downsample(skip, scale))
            boundary = skip - low_freq
            output   = skip + boundary_weight * boundary

        Scale selection based on minimum spatial dimension:
            <= 6  : skip enhancement entirely (signal is noise at this resolution)
            <= 12 : scale=0.5  (coarse boundary extraction)
            >  12 : scale=0.75 (fine boundary extraction)
        """
        size    = skip.shape[2:]
        min_dim = min(size)

        if min_dim <= 6:
            return skip

        scale = 0.5 if min_dim <= 12 else 0.75

        low_freq = F.interpolate(
            F.interpolate(skip, scale_factor=scale, mode='trilinear', align_corners=False),
            size=size, mode='trilinear', align_corners=False,
        )

        boundary = skip - low_freq                              # high-freq residual
        return skip + self.boundary_weight * boundary

    def spatial_guided_mask(self, skip, decoder):
        B, C, D, H, W = skip.shape
        v = self.v_spatial(skip).view(B, self.inter_channels, D * H * W)
        q = self.q_spatial(decoder)
        if q.shape[2:] != skip.shape[2:]:
            q = F.interpolate(q, size=skip.shape[2:], mode='trilinear', align_corners=False)
        q = self.softmax_spatial(q.view(B, 1, D * H * W))
        context = torch.matmul(v, q.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1)
        return self.sigmoid(self.proj_spatial(context))          # (B, C, 1, 1, 1)

    def channel_guided_mask(self, skip, decoder):
        B, C, D, H, W = skip.shape
        q = self.q_channel(decoder)
        if q.shape[2:] != skip.shape[2:]:
            q = F.interpolate(q, size=skip.shape[2:], mode='trilinear', align_corners=False)
        q = self.avg_pool(q).view(B, self.inter_channels, 1).permute(0, 2, 1)
        v = self.v_channel(skip).view(B, self.inter_channels, D * H * W)
        context = self.softmax_channel(torch.matmul(q, v))
        return self.sigmoid(context.view(B, 1, D, H, W))        # (B, 1, D, H, W)

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        skip_enhanced = self.extract_boundary(skip)

        spatial_mask = self.spatial_guided_mask(skip_enhanced, decoder)  # channel gate
        channel_mask = self.channel_guided_mask(skip_enhanced, decoder)  # spatial gate

        # Multiplicative gating: both must agree to pass signal through
        # range: [0, skip_enhanced] — no amplification, only suppression
        recalibrated = skip_enhanced * spatial_mask * channel_mask

        # Gated blend: start as pure skip passthrough, learn to recalibrate
        alpha = torch.sigmoid(self.blend)                        # scalar
        return (1 - alpha) * skip + alpha * recalibrated
