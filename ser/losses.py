from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    ラベルスムージングとクラスごとの alpha ウェイトに対応したマルチクラス Focal Loss。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.register_buffer("alpha", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if targets.dim() == 1:
            target_dist = F.one_hot(targets, num_classes=num_classes).float()
        else:
            target_dist = targets

        if self.label_smoothing > 0.0:
            target_dist = target_dist * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        pt = (probs * target_dist).sum(dim=-1).clamp_min(1e-6)
        ce_loss = -(target_dist * log_probs).sum(dim=-1)

        if self.alpha is not None:
            if self.alpha.numel() == num_classes:
                alpha_factor = (target_dist * self.alpha.unsqueeze(0)).sum(dim=-1)
            else:
                alpha_factor = self.alpha.squeeze()
        else:
            alpha_factor = 1.0

        focal_factor = (1.0 - pt) ** self.gamma
        loss = alpha_factor * focal_factor * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _balanced_class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    counts = class_counts.float().clamp_min(1.0)
    total = counts.sum()
    num_classes = counts.numel()
    weights = total / (num_classes * counts)
    return weights


def build_loss(cfg, class_counts: Optional[torch.Tensor], device: torch.device) -> nn.Module:
    loss_cfg = cfg.training.loss
    label_smoothing = loss_cfg.label_smoothing
    class_weights = None

    if isinstance(loss_cfg.class_weights, Sequence) and not isinstance(loss_cfg.class_weights, str):
        class_weights = torch.tensor(list(loss_cfg.class_weights), dtype=torch.float32)
    elif isinstance(loss_cfg.class_weights, str) and loss_cfg.class_weights.lower() == "balanced" and class_counts is not None:
        class_weights = _balanced_class_weights(class_counts)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    name = loss_cfg.name.lower()
    if name == "focal":
        alpha = None
        if isinstance(loss_cfg.focal_alpha, Sequence) and not isinstance(loss_cfg.focal_alpha, str):
            alpha = torch.tensor(list(loss_cfg.focal_alpha), dtype=torch.float32)
        elif isinstance(loss_cfg.focal_alpha, str) and loss_cfg.focal_alpha.lower() == "balanced" and class_counts is not None:
            alpha = _balanced_class_weights(class_counts)
        elif isinstance(loss_cfg.focal_alpha, (float, int)):
            alpha = torch.tensor([loss_cfg.focal_alpha], dtype=torch.float32)

        module = FocalLoss(
            gamma=loss_cfg.focal_gamma,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )
    else:
        module = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    return module.to(device)
