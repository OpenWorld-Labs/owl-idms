from dataclasses import dataclass
from torch import Tensor


@dataclass
class ActionGroundTruth:
    """
    Ground-truth controls for an **entire clip**.

    buttons : binary tensor  – shape [B, T, N_keys]
    mouse   : float tensor   – shape [B, T, 2]  (dx, dy)
    """
    buttons: Tensor          # 0 / 1
    mouse:   Tensor          # real-valued


@dataclass
class ActionPrediction:
    """
    Model outputs for an **entire clip**.

    buttons : logits tensor  – shape [B, T, N_keys]
    mouse   : float tensor   – shape [B, T, 2]  (predicted dx, dy)
    """
    buttons: Tensor
    mouse:   Tensor
