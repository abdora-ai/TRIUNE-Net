import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryDoULoss3D(nn.Module):
    """
    3D BoundaryDoU loss — direct port of the 2D version for volumetric
    segmentation (e.g. MSD pancreas, 96³ patches).

    The 2D cross kernel (4-connectivity) becomes a 6-connectivity 3D
    face-adjacent kernel. Boundary voxels are those with < 6 fully
    foreground face-neighbours (i.e. they touch background in at least
    one axis direction).  Everything else is identical to the 2D paper.

    Args:
        n_classes  : total classes including background.
        smooth     : numerical stability epsilon.
        alpha_cap  : upper bound on alpha (paper recommends 0.8).
        ignore_bg  : if True, class 0 is excluded from the loss sum.
    """

    def __init__(
        self,
        n_classes: int,
        smooth: float = 1e-5,
        alpha_cap: float = 0.8,
        ignore_bg: bool = True,
    ):
        super().__init__()
        self.n_classes   = n_classes
        self.smooth      = smooth
        self.alpha_cap   = alpha_cap
        self.start_class = 1 if ignore_bg else 0

        # ── 6-connectivity face-adjacent kernel (3D analogue of 2D cross) ──
        # Center + 6 face neighbours = 7 positions
        # A voxel is INTERIOR iff all 7 positions are foreground (neighbour_sum == 7)
        # A voxel is BOUNDARY iff it is foreground AND neighbour_sum < 7
        k = torch.zeros(1, 1, 3, 3, 3)
        k[0, 0, 1, 1, 1] = 1   # center
        k[0, 0, 0, 1, 1] = 1   # z−
        k[0, 0, 2, 1, 1] = 1   # z+
        k[0, 0, 1, 0, 1] = 1   # y−
        k[0, 0, 1, 2, 1] = 1   # y+
        k[0, 0, 1, 1, 0] = 1   # x−
        k[0, 0, 1, 1, 2] = 1   # x+
        self.register_buffer("kernel", k)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _one_hot_encoder(target: torch.Tensor, n_classes: int) -> torch.Tensor:
        """
        (B, D, H, W) or (B, 1, D, H, W) long → (B, C, D, H, W) float one-hot.
        Handles the nnUNet convention of targets with a channel dim.
        """
        if target.dim() == 5:
            target = target.squeeze(1)
        B, D, H, W = target.shape
        one_hot = torch.zeros(
            B, n_classes, D, H, W,
            device=target.device, dtype=torch.float32
        )
        one_hot.scatter_(1, target.unsqueeze(1).long(), 1.0)
        return one_hot

    # ------------------------------------------------------------------ #

    def _get_boundary(self, target_binary: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary voxels from a binary 3D mask.

        Strategy (mirrors the 2D original):
          neighbour_sum = conv3d(target, kernel)   # sums center + 6 neighbours
          interior      = voxels where sum == 7    # all neighbours foreground
          boundary      = foreground − interior

        This is morphological erosion with the 6-connectivity element;
        boundary = foreground XOR eroded_foreground.

        Args:
            target_binary : (B, D, H, W) float {0, 1}
        Returns:
            boundary      : (B, D, H, W) float {0, 1}
            C             : scalar — number of boundary voxels (for alpha)
            S             : scalar — number of foreground voxels (for alpha)
        """
        t = target_binary.unsqueeze(1)                           # (B,1,D,H,W)
        neighbour_sum = F.conv3d(t, self.kernel, padding=1)      # (B,1,D,H,W)

        # Interior: center IS foreground AND all 6 face-neighbours ARE foreground
        interior = (neighbour_sum >= 7.0).float()
        boundary = (t - interior).clamp(min=0.0).squeeze(1)      # (B,D,H,W)

        C = boundary.sum()
        S = target_binary.sum()
        return boundary, C, S

    # ------------------------------------------------------------------ #

    def _adaptive_size(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        BoundaryDoU loss for one class.

        Loss = (z² + y² − 2I) / (z² + y² − (1+α)·I)

        where I, y², z² are computed only over boundary voxels,
        and α = clip(2·(1 − C/S) − 1,  0,  alpha_cap).

        Args:
            score  : (B, D, H, W) predicted probability for this class.
            target : (B, D, H, W) binary GT for this class.
        Returns:
            Scalar loss value.
        """
        boundary, C, S = self._get_boundary(target)

        # ── Geometry-adaptive alpha ──
        raw_alpha = 2.0 * (1.0 - (C + self.smooth) / (S + self.smooth)) - 1.0
        alpha = float(raw_alpha.clamp(0.0, self.alpha_cap))

        # ── Intersection terms restricted to boundary voxels ──
        # Multiplying by `boundary` zeroes out interior contributions,
        # focusing gradient signal exactly where the 2D version focused it.
        score_b  = score  * boundary
        target_b = target * boundary

        intersect = (score_b * target_b).sum()
        y_sum     = (target_b * target_b).sum()
        z_sum     = (score_b  * score_b ).sum()

        loss = (
            (z_sum + y_sum - 2.0 * intersect + self.smooth)
            / (z_sum + y_sum - (1.0 + alpha) * intersect + self.smooth)
        )
        return loss

    # ------------------------------------------------------------------ #

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs : (B, C, D, H, W) raw logits.
            target : (B, D, H, W) or (B, 1, D, H, W) integer class labels.
        Returns:
            Scalar loss averaged over active classes.
        """
        probs     = torch.softmax(inputs, dim=1)
        target_oh = self._one_hot_encoder(target, self.n_classes)

        assert probs.shape == target_oh.shape, (
            f"Shape mismatch: probs {probs.shape} vs one-hot {target_oh.shape}"
        )

        loss = sum(
            self._adaptive_size(probs[:, i], target_oh[:, i])
            for i in range(self.start_class, self.n_classes)
        )
        return loss / (self.n_classes - self.start_class)