from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional


class SoftBCEWithLogitsLoss(nn.Module):

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
        "apply_sample_weight",
    ]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
        apply_sample_weight: bool = False,
    ):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index, label_smoothing,
            and sample_weight

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.apply_sample_weight = apply_sample_weight

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (
                1 - self.smooth_factor
            )
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred,
            soft_targets,
            self.weight,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if self.apply_sample_weight:
            loss = loss * sample_weight

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, apply_sample_weight=False, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.apply_sample_weight = apply_sample_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, sample_weight=None):

        BCE = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        if self.apply_sample_weight:
            focal_loss = focal_loss * sample_weight

        return focal_loss.mean()


def loss_fn(y_pred, y_true, sample_weight=None):
    loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    if sample_weight is not None:
        loss = loss * sample_weight
    loss = criterion(y_pred, y_true)
    return loss.mean()
