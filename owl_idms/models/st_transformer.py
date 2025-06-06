from __future__ import annotations
"""Spatio-Temporal Transformer-based Inverse Dynamics Model (IDM).

This module plugs into the *owl_idms* code-base and replaces the vanilla
``nn.TransformerEncoder`` in ``action_predictor`` with Genie-style
**bidirectional** ST blocks.  It predicts:

* ``buttons`` - raw logits for each key in ``KEYBINDS``
* ``mouse``   - 2-D continuous Δx / Δy (pixels)

The network operates on VAE latents of shape **[B, T, C, H', W']** where
*B*  = batch, *T*=16 frames by default, *C*=latent channels.

Key design choices
------------------
1. **Bidirectional** temporal attention → richer context for inverse dynamics.
2. **CLS tokens per frame** (mouse + buttons) - copied across time so heads can
   attend locally and temporally.
3. Lightweight projection (``Conv1x1``) + learned positional encoding.

Based on Genie ST-Transformer (§4, Fig 4) but with causal mask removed.
"""


import torch
from torch import nn, Tensor
from einops import rearrange

from owl_idms.configs import STTransformerConfig, ControlPredConfig
from owl_idms.nn.mlp import MLPSimple as MLP


# -----------------------------------------------------------------------------
# ST-Blocks (borrowed + minor tweak: causal=False)
# -----------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            # qk normalization https://arxiv.org/pdf/2302.05442
            # Note that LN is done in fp32, so they have to be
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, d_model: int, ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(d_model * ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class STBlock(nn.Module):
    """Bidirectional Genie ST block (spatial → temporal → MLP)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.s_attn = SelfAttention(d_model=d_model, num_heads=num_heads, attn_drop=attn_drop)
        self.t_attn = SelfAttention(d_model=d_model, num_heads=num_heads, attn_drop=attn_drop)
        self.norm_m = nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model, ratio=mlp_ratio, drop=mlp_drop)

    def forward(self, x: Tensor) -> Tensor:  # x: [B,T,S,C]
        B, T, S, C = x.shape
        # spatial
        xs = rearrange(x, 'B T S C -> (B T) S C')
        xs = xs + self.s_attn(self.norm_s(xs))
        x = rearrange(xs, '(B T) S C -> B T S C', B=B)
        # temporal (bidirectional)
        xt = rearrange(x, 'B T S C -> (B S) T C')
        xt = xt + self.t_attn(self.norm_t(xt), causal=False)
        x = rearrange(xt, '(B S) T C -> B T S C', S=S)
        # MLP
        x = x + self.mlp(self.norm_m(x))
        return x


class STBackbone(nn.Module):
    def __init__(self, depth: int, d_model: int, num_heads: int, **h):
        super().__init__()
        self.layers = nn.ModuleList([
            STBlock(d_model, num_heads, **h) for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.layers:
            x = blk(x)
        return x

# -----------------------------------------------------------------------------
# Patch embedding + CLS + positional encodings
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2-D conv patchifier used frame-wise."""

    def __init__(self, in_ch: int = 3, d_model: int = 512, patch: int = 16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x: Tensor) -> Tensor:  # [B*T,3,H,W]
        x = self.proj(x)                    # [B*T, d_model, H', W']
        x = x.flatten(2).transpose(1, 2)    # [B*T, S, d_model]
        return x

# -----------------------------------------------------------------------------
# Full IDM
# -----------------------------------------------------------------------------

class STInverseDynamics(nn.Module):
    """RGB-video → action prediction via bidirectional ST-Transformer."""

    def __init__(
        self,
        config: ControlPredConfig
    ) -> None:
        super().__init__()
        st_config: STTransformerConfig = config.st_transformer_config
        H, W = st_config.img_size
        patch = st_config.patch
        frames = st_config.frames
        d_model = st_config.d_model
        depth = st_config.depth
        num_heads = st_config.num_heads
        token_drop = st_config.token_drop
        n_keys = config.n_buttons
        n_mouse_axes = config.n_mouse_axes
        assert H % patch == 0 and W % patch == 0, 'Image dims must be divisible by patch'
        self.frames = frames
        self.S = (H // patch) * (W // patch)
        self.n_keys = n_keys
        # 1. Patch embed
        self.embed = PatchEmbed(3, d_model, patch)

        # 2. CLS tokens
        self.mouse_cls = nn.Parameter(torch.randn(1, d_model)) # [1,D]. at fwd we unsqueeze batch dim and repeat.
        self.key_cls   = nn.Parameter(torch.randn(1, d_model)) # [1,D]. at fwd we unsqueeze batch dim and repeat.
        self.n_cls_tokens = self.mouse_cls.shape[0] + self.key_cls.shape[0]
        # 3. Positional encodings (learned) - separate spatial & temporal
        self.temporal_pe = nn.Parameter(torch.randn(1, frames, 1, d_model))
        self.spatial_pe  = nn.Parameter(torch.randn(1, 1, self.S + self.n_cls_tokens, d_model))
        
        # 4. Token dropout
        self.token_drop = nn.Dropout(token_drop)

        # 4. Backbone
        self.backbone = STBackbone(depth, d_model, num_heads)
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Heads
        self.btn_head  = MLP(d_model, n_keys)
        self.mouse_head_mu = MLP(d_model, n_mouse_axes)
        self.mouse_head_logvar = MLP(d_model, n_mouse_axes)


    # ------------------------------------------------------------------
    def forward(self, video: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """video: [B, T, 3, H, W] where *T* == ``self.frames``."""
        B, T, C, H, W = video.shape
        assert T == self.frames, f"Expect {self.frames} frames, got {T}"

        # flatten batch & frames for patchification
        x = rearrange(video, 'B T C H W -> (B T) C H W')  # [B*T,3,H,W]
        x = self.embed(x)                                 # [B*T,S,d_model]
        x = rearrange(x, '(B T) S C -> B T S C', B=B, T=T)

        # add CLS per frame - an extra cls token at the spatial dimension
        mouse_tok = self.mouse_cls\
            .unsqueeze(0)\
            .unsqueeze(0)\
            .repeat(B, self.frames, 1, 1)
        key_tok   = self.key_cls\
            .unsqueeze(0)\
            .unsqueeze(0)\
            .repeat(B, self.frames, 1, 1)
        x = torch.cat([x, mouse_tok, key_tok], dim=2) # [B,T,S+2,C] ?

        # add positional embeddings
        x = x + self.temporal_pe[:, :T] + self.spatial_pe[:, :, : x.size(2)]
        x = self.token_drop(x)

        # ST blocks
        x = self.backbone(x)
        x = x[:, -1, -self.n_cls_tokens:] # take cls token from last frame
        x = self.final_norm(x)

        mouse_tok = x[:, 0, :]
        key_tok = x[:, 1, :]

        # mouse token is the 2nd to last spatial token in the last frame
        # out_mouse_tok = x[:, -1, -2, :]
        # button tokens are the last two spatial tokens in the last frame
        # out_btn_tok = x[:, -1, -1, :]

        b_logits    = self.btn_head(key_tok)
        m_mu        = self.mouse_head_mu(mouse_tok)
        m_logvar    = self.mouse_head_logvar(mouse_tok)

        return (m_mu, m_logvar), b_logits


def st_idm():
    return STInverseDynamics(
        img_size=(128, 128),
        patch=16,
        frames=32,
        d_model=128,
        depth=4,
        num_heads=4
    )