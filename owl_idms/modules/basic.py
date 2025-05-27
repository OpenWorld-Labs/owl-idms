import torch
from torch import nn
import torch.nn.functional as F

from owl_idms.modules.utils import PositionalEncoding
from owl_idms._types import ActionPrediction
from owl_idms.constants import KEYBINDS


class BasicInverseDynamics(nn.Module):
    def __init__(self, 
                 n_keys: int,
                 n_frames: int,
                 in_channels: int = 3,
                 embed_dim: int = 128,
                 **kwargs):
        super().__init__()
        
        self.n_frames = n_frames
        self.embed_dim = embed_dim
        
        # Conv3D layers with temporal awareness
        self.conv1 = nn.Conv3d(in_channels, 64, 
                              kernel_size=(5, 7, 7),  # Large temporal kernel
                              stride=(1, 2, 2),       # Only downsample spatially
                              padding=(2, 3, 3))      # Maintain temporal dimension
        
        self.conv2 = nn.Conv3d(64, 128, 
                              kernel_size=(3, 5, 5), 
                              stride=(1, 2, 2), 
                              padding=(1, 2, 2))
        
        self.conv3 = nn.Conv3d(128, embed_dim, 
                              kernel_size=(3, 3, 3), 
                              stride=(1, 1, 1), 
                              padding=(1, 1, 1))
        
        # BatchNorm for stability
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(embed_dim)
        
        # Global average pooling (2D, applied per frame)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.cls_button = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_mouse  = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Prediction heads (per-frame)
        self.key_head = nn.Linear(embed_dim, n_keys)
        self.mouse_delta_head = nn.Linear(embed_dim, 2)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize conv layers
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # Initialize linear layers
        for m in [self.key_head, self.mouse_delta_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, video: torch.Tensor) -> ActionPrediction:
        """
        video: [B, T, C, H, W] - e.g., [64, 32, 3, 128, 128]
        outputs action prediction for the middle frame, e.g. 16 to 17
        """
        B, T, C, H, W = video.shape
        
        # Rearrange for Conv3D: [B, C, T, H, W]
        video = video.transpose(1, 2)
        
        # Apply Conv3D layers with non-causal temporal processing
        x = F.relu(self.bn1(self.conv1(video)))  # [B, 64, T, H/2, W/2]
        x = F.relu(self.bn2(self.conv2(x)))      # [B, 128, T, H/4, W/4]
        x = F.relu(self.bn3(self.conv3(x)))      # [B, embed_dim, T, H/4, W/4]
        
        # Get dimensions after convolutions
        _, D, T, H, W = x.shape
        
        # Rearrange for spatial pooling: process each frame independently and apply 2D GAP to each frame
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, embed_dim, H, W]
        x = x.reshape(B * T, self.embed_dim, H, W)  # [B*T, embed_dim, H, W]
        x = self.gap(x)  # [B*T, embed_dim, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B*T, embed_dim]
        # Reshape back to sequence
        x = x.reshape(B, T, self.embed_dim)  # [B, T, embed_dim]
                
        # Append cls tokens and apply bidirectional transformer across time
        cls_button = self.cls_button.expand(B, -1, -1)
        cls_mouse  = self.cls_mouse.expand(B, -1, -1)
        x = torch.cat([cls_button, cls_mouse, x], dim=1)
        x = self.temporal_transformer(x)  # [B, T, embed_dim]
        x = self.final_norm(x)
  
        
        # Generate per-frame predictions
        button_tok = x[:, 0]
        mouse_tok  = x[:, 1]
        buttons = self.key_head(button_tok)  # [B, n_keys]
        mouse = self.mouse_delta_head(mouse_tok)  # [B, 2]
        
        return ActionPrediction(
            buttons=buttons,
            mouse=mouse,
        )

def basic_idm_base(n_keys: int = len(KEYBINDS), n_frames: int = 1, **kwargs):
    return BasicInverseDynamics(n_keys, n_frames, **kwargs)