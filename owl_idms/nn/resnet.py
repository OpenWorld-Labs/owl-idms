from torch import nn
import torch.nn.functional as F

from .normalization import RMSNorm3d

"""
Building blocks for any ResNet based model
"""

class ResBlock3D(nn.Module):
    """
    3D Res Block

    :param ch: Channel count to use for the block
    :param total_res_blocks: How many res blocks are there in the entire model?
    """
    def __init__(self, ch, total_res_blocks):
        super().__init__()

        grp_size = 16
        n_grps = (2*ch) // grp_size

        self.conv1 = nn.Conv3d(ch, 2*ch, 1, 1, 0)

        self.norm1 = RMSNorm3d(2*ch)
        self.conv2 = nn.Conv3d(2*ch, 2*ch, 3, 1, 1, groups = n_grps)
        self.norm2 = RMSNorm3d(2*ch)

        self.conv3 = nn.Conv3d(2*ch, ch, 1, 1, 0, bias=False)

        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        # Fix up init
        scaling_factor = total_res_blocks ** -.25

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.conv1.weight.data *= scaling_factor

        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.conv2.weight.data *= scaling_factor

        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        res = x.clone()
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv3(x)

        return x + res

class SameBlock(nn.Module):
    """
    General block with no up/down
    """
    def __init__(self, ch_in, ch_out, num_res, total_blocks):
        super().__init__()

        blocks = []
        num_total = num_res * total_blocks
        for _ in range(num_res):
            blocks.append(ResBlock3D(ch_in, num_total))
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Conv3d(ch_in, ch_out, 1, 1, 0, bias=False) if ch_in != ch_out else nn.Sequential()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv3d(ch_in, ch_out, 3, 1, 1, bias=False)
        self.norm = RMSNorm3d(ch_out)
        self.act = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
