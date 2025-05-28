from dataclasses import dataclass
from torch import Tensor


@dataclass
class ActionGroundTruth:
    """
    Ground-truth controls for an **entire clip**.

    mouse   : float tensor   – shape [B, T, 2]  (dx, dy)
    """
    mouse:   Tensor          # real-valued


@dataclass
class ActionPrediction:
    """
    Model outputs for an **entire clip**.

    mouse   : float tensor   – shape [B, T, 2]  (predicted dx, dy)
    """
    mouse:   Tensor
