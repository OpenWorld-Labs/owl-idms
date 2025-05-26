import torch
from torch import nn

from owl_idms.modules.utils import PositionalEncoding
from owl_idms._types import ActionPrediction
from owl_idms.constants import KEYBINDS

class LatentVPTInverseDynamics(nn.Module):
    """
    VPT-style inverse-dynamics model that takes a window of *latent tokens*
    (no pixels) and predicts keyboard logits + mouse deltas (mu, log sigma).

    Input
    -----
    latents : Tensor[B, T, L, Z_in]   — T frames x L tokens each.

    Output
    ------
    dict | { "keys": Tensor[B, n_keys],
             "mouse_mu": Tensor[B, 2],
             "mouse_log_sigma": Tensor[B, 2] }
    """

    def __init__(
        self,
        zin: int,
        n_keys: int,
        n_frames: int,
        embed_dim: int = 512,
        n_layers: int = 6, n_heads: int = 8,
        ff_dim: int = 2048, dropout: float = 0.1,
    ):
        super().__init__()

        self.token_proj = nn.Linear(zin, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        # learnable cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_ln = nn.LayerNorm(embed_dim)

        self.key_head = nn.Linear(embed_dim, n_keys * n_frames)
        self.mouse_mu_head = nn.Linear(embed_dim, 2 * n_frames)
        self.mouse_logsigma_head = nn.Linear(embed_dim, 2 * n_frames)

        self._init_weights()

    def _init_weights(self, std: float = 0.02):
        nn.init.trunc_normal_(self.cls_token, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, latents: torch.Tensor) -> ActionPrediction:
        """
        latents : [B, T, L, Z_in]  →  flatten tokens → [B, S, D]
        """
        b, t, l, _ = latents.shape
        x = latents.reshape(b, t*l, -1)
        x = self.token_proj(x)
        x = self.pos_enc(x)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        pooled = self.final_ln(x[:, 0])

        return ActionPrediction(
            buttons=self.key_head(pooled).reshape(b, t, -1), # [B, T, n_keys]
            mouse_mu=self.mouse_mu_head(pooled).reshape(b, t, -1), # [B, T, 2]
            mouse_log_sigma=self.mouse_logsigma_head(pooled).reshape(b, t, -1), # [B, T, 2]
        )


def vpt_pico(zin: int = 32, n_keys: int = len(KEYBINDS), n_frames: int = 1, **kwargs):
    return LatentVPTInverseDynamics(zin, n_keys, n_frames, **kwargs)



# def vpt_base(zin: int = 4096, n_keys: int = len(KEYBINDS), **kwargs):
def vpt_base(zin: int = 256, n_keys: int = len(KEYBINDS), n_frames: int = 1, **kwargs):
    return LatentVPTInverseDynamics(zin, n_keys, n_frames, **kwargs)
