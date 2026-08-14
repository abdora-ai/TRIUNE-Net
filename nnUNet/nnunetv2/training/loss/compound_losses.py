import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from nnunetv2.training.loss.boundary_dou import BoundaryDoULoss3D

class DC_and_CE_and_BDoU_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        bdou_kwargs=None,
        weight_ce=1,
        weight_dice=1,
        weight_bdou=0.5,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Extension of DC_and_CE_loss with an optional BoundaryDoU3D term.

        Weights do not need to sum to one.

        Args:
            soft_dice_kwargs : kwargs forwarded to SoftDiceLoss (or dice_class).
            ce_kwargs        : kwargs forwarded to RobustCrossEntropyLoss.
            bdou_kwargs      : kwargs forwarded to BoundaryDoULoss3D.
                               Must contain at least {'n_classes': <int>}.
                               Pass None or {} to use defaults (n_classes inferred
                               from net_output at runtime — see note below).
            weight_ce        : scalar weight for CE term.   Set 0 to disable.
            weight_dice      : scalar weight for Dice term. Set 0 to disable.
            weight_bdou      : scalar weight for BDoU term. Set 0 to disable.
            ignore_label     : label index to ignore (forwarded to CE and masked
                               out of BDoU as well).
            dice_class       : Dice implementation to use.
        """
        super().__init__()

        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_ce   = weight_ce
        self.weight_dice = weight_dice
        self.weight_bdou = weight_bdou
        self.ignore_label = ignore_label

        # ── base losses ──────────────────────────────────────────────────
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        # ── boundary DoU ─────────────────────────────────────────────────
        if weight_bdou != 0:
            if bdou_kwargs is None:
                bdou_kwargs = {}
            # n_classes is required by BoundaryDoULoss3D; if the caller did not
            # supply it we defer instantiation to the first forward pass.
            if 'n_classes' in bdou_kwargs:
                self.bdou = BoundaryDoULoss3D(**bdou_kwargs)
            else:
                # lazy init — n_classes will be read from net_output.shape[1]
                self._bdou_kwargs  = bdou_kwargs
                self.bdou          = None
        else:
            self.bdou = None

    # ------------------------------------------------------------------ #

    def _get_bdou(self, n_classes: int, device) -> BoundaryDoULoss3D:
        """Lazy initialisation when n_classes was not provided up front."""
        if self.bdou is None:
            self.bdou = BoundaryDoULoss3D(
                n_classes=n_classes, **self._bdou_kwargs
            ).to(device)
        return self.bdou

    # ------------------------------------------------------------------ #

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Args:
            net_output : (B, C, D, H, W) raw logits.
            target     : (B, 1, D, H, W) integer class labels.
        Returns:
            Scalar combined loss.
        """
        # ── ignore-label masking (identical to DC_and_CE_loss) ───────────
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                'ignore_label is not supported for one-hot encoded targets.'
            )
            mask    = (target != self.ignore_label)
            target_dice = torch.where(mask, target, 0)
            num_fg  = mask.sum()
        else:
            target_dice = target
            mask        = None
            num_fg      = None   # not needed when ignore_label is None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0 else 0
        )

        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0
            and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        if self.weight_bdou != 0:
            bdou_fn = self._get_bdou(net_output.shape[1], net_output.device)

            if self.ignore_label is not None and num_fg == 0:
                # no valid voxels in this batch — skip to avoid NaN
                bdou_loss = 0
            else:
                # Apply the same ignore mask: zero out ignored voxels in target.
                # BoundaryDoULoss3D expects (B,1,D,H,W) or (B,D,H,W) long.
                bdou_target = target_dice if self.ignore_label is not None else target
                bdou_loss   = bdou_fn(net_output, bdou_target)
        else:
            bdou_loss = 0

        return (
              self.weight_ce   * ce_loss
            + self.weight_dice * dc_loss
            + self.weight_bdou * bdou_loss
        )

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
