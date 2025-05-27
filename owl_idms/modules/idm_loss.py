import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from owl_idms._types import ActionGroundTruth, ActionPrediction
from owl_idms.constants import KEYBINDS

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

class IDM_Focal_Loss(nn.Module):
    """
    Improved loss for IDM that handles class imbalance and encourages better predictions.
    """
    def __init__(self, 
                 mouse_weight: float = 0.3,
                 eps: float = 1e-6, 
                 full: bool = True,
                 reduction: str = "mean",
                 # New parameters
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25,
                 any_button_weight: float = 0.2,
                 button_smoothing: float = 0.01) -> None:
        super().__init__()
        
        self.mouse_loss = nn.GaussianNLLLoss(full=full, eps=eps, reduction=reduction)
        self.mouse_weight = mouse_weight
        self.eps = eps
        self.reduction = reduction
        
        # Focal loss parameters
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        # Auxiliary loss weights
        self.any_button_weight = any_button_weight
        
        # Label smoothing
        self.button_smoothing = button_smoothing

    def focal_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Focal loss for handling extreme class imbalance.
        Focuses learning on hard examples (rare button presses).
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal term: (1-p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t) ** self.focal_gamma
        
        # Optional alpha weighting
        if self.focal_alpha is not None:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
    
    def forward(self, pred: ActionPrediction, target: ActionGroundTruth) -> tuple[Tensor, Tensor]:
        loss_keys = self.focal_loss(pred.buttons, target.buttons.float())
        var = (pred.mouse_log_sigma.exp() + self.eps) ** 2
        loss_mouse = self.mouse_loss(pred.mouse_mu, target.mouse, var)
        return loss_keys, loss_mouse * self.mouse_weight

class BCEMSE_IDM_Loss(nn.Module):
    """
    BCE-with-logits on button channels + MSE on mouse deltas.
    The loss is computed on the **centre frame** (t = T//2).
    """

    def __init__(
        self,
        *,
        n_controls: int = len(KEYBINDS),
        btn_weight: float = 1.0,
        mouse_weight: float = 1.0,
        threshold: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.n_controls   = n_controls
        self.btn_weight   = btn_weight
        self.mouse_weight = mouse_weight
        self.threshold    = threshold
        self.reduction    = reduction

        self._btn_loss   = nn.BCEWithLogitsLoss(reduction=reduction)
        self._mouse_loss = nn.MSELoss(reduction=reduction)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred:   ActionPrediction,
        target: ActionGroundTruth,
    ) -> tuple[Tensor, dict[str, float]]:
        btn_pred   = pred.buttons            # [B, N_keys] logits
        mouse_pred = pred.mouse             # [B, 2]

        btn_gt     = target.buttons.float()  # 0/1
        mouse_gt   = target.mouse           # [B, 2]

        # -------------------------------------------------------------- #
        # losses                                                         #
        # -------------------------------------------------------------- #
        btn_loss  = self._btn_loss(btn_pred, btn_gt)
        mse_loss  = self._mouse_loss(mouse_pred, mouse_gt)
        total     = self.btn_weight * btn_loss + self.mouse_weight * mse_loss

        # -------------------------------------------------------------- #
        # metrics (no-grad)                                              #
        # -------------------------------------------------------------- #
        with torch.no_grad():
            btn_bin = (torch.sigmoid(btn_pred) > self.threshold).float()

            btn_acc = (btn_bin == btn_gt).float().mean()

            # “Sensitivity” = accuracy on the non-zero channels
            nz_mask = btn_gt != 0
            nz_acc  = (
                (btn_bin[nz_mask] == btn_gt[nz_mask]).float().mean()
                if nz_mask.any()
                else torch.tensor(float("nan"), device=btn_gt.device)
            )

            all_zero_btn  = (btn_gt.sum(dim=-1) == 0).float().mean()

        metrics = {
            "total_loss"         : total.item(),
            "button_loss"        : btn_loss.item(),
            "mouse_loss"         : mse_loss.item(),
            "button_accuracy"    : btn_acc.item(),
            "button_sensitivity" : nz_acc.item(),   # ← original name restored
            "p(all_zero_buttons)": all_zero_btn.item(),
        }

        return total, metrics
