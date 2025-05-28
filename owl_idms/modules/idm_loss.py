import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from owl_idms._types import ActionGroundTruth, ActionPrediction
from owl_idms.constants import KEYBINDS
from torchvision.ops import sigmoid_focal_loss

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


# --------------------------------------------------------------------------- #
# Helper focal & asymmetric losses
# --------------------------------------------------------------------------- #
class FocalBCEWithLogits(nn.Module):
    """Sigmoid focal BCE (Lin et al., γ>0) – supports per-class α."""
    def __init__(self, gamma: float = 2.0, alpha: Tensor | float | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        loss = sigmoid_focal_loss(
            logits, target, gamma=self.gamma,
            alpha=self.alpha, reduction=self.reduction  # TV handles None
        )
        return loss

class AsymmetricLoss(nn.Module):
    """Ridnik et al. ICCV-21 – γ_pos≈0, γ_neg≈4, optional prob-clip."""
    def __init__(self, g_pos=0, g_neg=4, clip=0.05, eps=1e-8,
                 reduction="mean"):
        super().__init__()
        self.g_pos, self.g_neg, self.clip, self.eps, self.reduction = \
            g_pos, g_neg, clip, eps, reduction
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        x_sig = torch.sigmoid(logits)
        xs_pos = torch.clamp(x_sig, self.eps, 1.0 - self.eps)
        xs_neg = torch.clamp(1.0 - x_sig + self.clip, 0.0, 1.0)
        loss_pos = target * torch.log(xs_pos) * (1 - xs_pos).pow(self.g_pos)
        loss_neg = (1 - target) * torch.log(xs_neg) * xs_neg.pow(self.g_neg)
        loss = -(loss_pos + loss_neg)
        return loss.mean() if self.reduction == "mean" else loss.sum()

# --------------------------------------------------------------------------- #
# Main IDM loss
# --------------------------------------------------------------------------- #
class BCEMSE_IDM_Loss(nn.Module):
    """
    Button loss = focal / asymmetric / BCE-with-pos-weight  (choose via kwargs)
    Mouse  loss = MSE on Δx/Δy.
    Both computed on centre frame (t = T//2).
    """

    def __init__(
        self,
        *,
        n_controls: int = len(KEYBINDS),
        btn_weight: float = 1.0,
        mouse_weight: float = 1.0,
        threshold: float = 0.5,
        mode: str = "focal",          # 'focal' | 'asymmetric' | 'bce'
        gamma: float = 2.0,           # focal γ
        pos_weight: Tensor | None = None,  # for mode='bce'
        alpha: float | Tensor | None = 0.25,  # focal α (balance factor)
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.btn_weight   = btn_weight
        self.mouse_weight = mouse_weight
        self.threshold    = threshold

        if mode == "focal":
            self._btn_loss = FocalBCEWithLogits(
                gamma=gamma, alpha=alpha, reduction=reduction
            )
        elif mode == "asymmetric":
            self._btn_loss = AsymmetricLoss(
                g_pos=0, g_neg=gamma, reduction=reduction
            )
        elif mode == "bce":
            self._btn_loss = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction=reduction
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self._mouse_loss = nn.MSELoss(reduction=reduction)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred:   ActionPrediction,
        target: ActionGroundTruth,
    ) -> tuple[Tensor, dict[str, float]]:
        btn_loss  = self._btn_loss(pred.buttons, target.buttons.float())
        mse_loss  = self._mouse_loss(pred.mouse,  target.mouse)
        total     = self.btn_weight * btn_loss + self.mouse_weight * mse_loss

        # --------------------------- metrics --------------------------- #
        with torch.no_grad():
            btn_bin  = (torch.sigmoid(pred.buttons) > self.threshold).float()
            btn_acc  = (btn_bin == target.buttons).float().mean()
            nz_mask  = target.buttons != 0
            nz_acc   = (btn_bin[nz_mask] == target.buttons[nz_mask]
                        ).float().mean() if nz_mask.any() else torch.tensor(
                            float('nan'), device=btn_bin.device)
            all_zero = (target.buttons.sum(dim=-1) == 0).float().mean()

        return total, {
            "total_loss": total.item(), "button_loss": btn_loss.item(),
            "mouse_loss": mse_loss.item(), "button_accuracy": btn_acc.item(),
            "button_sensitivity": nz_acc.item(),
            "p(all_zero_buttons)": all_zero.item(),
        }