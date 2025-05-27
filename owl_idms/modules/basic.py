import torch
from torch import nn

from owl_idms.modules.utils import PositionalEncoding
from owl_idms._types import ActionPrediction
from owl_idms.constants import KEYBINDS



class BasicInverseDynamics(nn.Module):
    """
    Basic IDM from a bunch of Conv3Ds and a final transformer encoder block or two.
    """

    def __init__(self,
                 n_keys: int,
                 n_frames: int,
                 in_channels: int = 3,
                 embed_dim: int = 128,
                 **kwargs): 
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(embed_dim, embed_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(embed_dim, embed_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.pos_enc = PositionalEncoding(embed_dim)
        self._encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self._encoder_layer, num_layers=2)

        self.final_ln = nn.LayerNorm(embed_dim)
        self.key_head = nn.Linear(embed_dim, n_keys * n_frames)
        self.mouse_mu_head = nn.Linear(embed_dim, 2 * n_frames)
        self.mouse_logsigma_head = nn.Linear(embed_dim, 2 * n_frames)

        self._init_weights()

    def _init_weights(self, std: float = 0.02):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)

    def forward(self, video: torch.Tensor) -> ActionPrediction:
        x = self.conv1(video)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transformer_encoder(x)  # TODO [B, T, C, H, W] but need to patchify so we dont attend 128x128
        x = self.final_ln(x)
        return ActionPrediction(
            keys=self.key_head(x),
            mouse_mu=self.mouse_mu_head(x),
            mouse_log_sigma=self.mouse_logsigma_head(x)
        )


def basic_idm_base(n_keys: int = len(KEYBINDS), n_frames: int = 1, **kwargs):
    return BasicInverseDynamics(n_keys, n_frames, **kwargs)

