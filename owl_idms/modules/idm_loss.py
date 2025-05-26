from torch import Tensor
import torch.nn as nn

from owl_idms._types import ActionGroundTruth, ActionPrediction


class IDMLoss(nn.Module):
    """
    BCEWithLogitsLoss for key presses + Gaussian NLL for mouse deltas.
    """
    def __init__(self, mouse_weight: float = 0.3,
                 eps: float = 1e-6, full: bool = True,
                 reduction: str = "mean") -> None:
        super().__init__()
        self.key_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.mouse_loss = nn.GaussianNLLLoss(full=full, eps=eps, reduction=reduction)
        self.mouse_weight = mouse_weight
        self.eps = eps

    def forward(self, pred: ActionPrediction, target: ActionGroundTruth) -> tuple[Tensor, Tensor]:
        loss_keys = self.key_loss(pred.buttons, target.buttons.float())
        # expect strictly positive variance for Gaussian NLL
        var = (pred.mouse_log_sigma.exp() + self.eps) ** 2
        loss_mouse = self.mouse_loss(pred.mouse_mu, target.mouse, var)
        return loss_keys, loss_mouse * self.mouse_weight
