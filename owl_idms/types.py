from dataclasses import dataclass
from torch import Tensor


@dataclass
class ActionGroundTruth:
    buttons: Tensor          # [B, N_keys]  binary 0/1
    mouse: Tensor         # [B, 2]       delta x, delta y  (ground-truth)

@dataclass
class ActionPrediction:
    buttons: Tensor           # [B, N_keys]  logits
    mouse_mu: Tensor          # [B, 2]       mean delta x, delta y
    mouse_log_sigma: Tensor   # [B, 2]       log sigma  (unconstrained)
