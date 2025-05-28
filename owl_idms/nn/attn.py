import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from .mlp import MLP
from .normalization import LayerNorm, QKNorm

torch.backends.cuda.enable_flash_sdp(enabled=True)

class Attn(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)

        self.causal = False

    def forward(self, x):
        # x is [b, n, d]
        qkv = self.qkv(x) # [b, n, 3*d]
        
        # Split into q,k,v and reshape
        chunk_size = qkv.shape[-1] // 3
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.n_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, b, h, n, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.qk_norm(q, k)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        
        # Reshape back
        x = x.permute(0, 2, 1, 3).contiguous() # [b, n, h, d]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [b, n, h*d]
        
        x = self.out(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config)
        self.mlp = MLP(config)

    def forward(self, x):
        res1 = x.clone()
        x = self.norm1(x)
        x = self.attn(x)
        x = res1 + x

        res2 = x.clone()
        x = self.norm2(x)
        x = self.mlp(x)
        x = res2 + x

        return x

class StackedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for _ in range(config.n_layers):
            blocks.append(Transformer(config))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x