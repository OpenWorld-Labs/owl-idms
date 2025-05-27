from ..nn.attn import StackedTransformer
from ..nn.embeddings import LearnedPosEnc
from ..nn.resnet import SameBlock
from ..nn.mlp import MLPSimple

import torch
from torch import nn

class SpatialPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        # x: [b,c,t,h,w] -> [b,c,t,h/2,w/2]
        return self.pool(x)

class SpatioTemporalPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        # x: [b,c,t,h,w] -> [b,c,t/2,h/2,w/2]
        return self.pool(x)

class TemporalPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        # x: [b,c,t,h,w] -> [b,c,t/2,h,w]
        return self.pool(x)

class ControlPredictor(nn.Module):
    def __init__(self, config : 'ControlPredConfig'):
        super().__init__()

        self.config = config
        self.t_config = config.transformer_config
        self.r_config = config.resnet_config

        self.conv_in = nn.Conv3d(3, self.r_config.ch_0, 3, 1, 1, bias = False)

        self.transformer = StackedTransformer(self.t_config)

        self.mouse_tokens = nn.Parameter(torch.randn(self.config.n_mouse_axes, self.t_config.d_model)*0.02)
        self.button_tokens = nn.Parameter(torch.randn(self.config.n_buttons, self.t_config.d_model)*0.02)
        self.n_cls_toks = self.config.n_mouse_axes + self.config.n_buttons

        self.proj_t_in = nn.Linear(self.r_config.ch_max, self.t_config.d_model)
        self.pos_enc = LearnedPosEnc(256, self.t_config.d_model)

        n_per_block = 1
        total_blocks = 5

        def get_block(ch_in, ch_out):
            return SameBlock(ch_in, ch_out, n_per_block, n_per_block * total_blocks)
        ch_0 = self.r_config.ch_0
        ch_max = self.r_config.ch_max

        # input assumed [32,128,128]
        blocks = [
            get_block(ch_0, min(ch_0*2, ch_max)),
            SpatioTemporalPool(), # -> [16, 64, 64]
            get_block(min(ch_0*2, ch_max), min(ch_0*4, ch_max)),
            SpatioTemporalPool(), # -> [8, 32, 32]
            get_block(min(ch_0*4, ch_max), min(ch_0*8, ch_max)),
            SpatialPool(), # -> [8, 16, 16]
            get_block(min(ch_0*8, ch_max), ch_max),
            SpatialPool(), # -> [8, 8, 8]
            get_block(ch_max, ch_max),
            SpatialPool() # -> [8, 4, 4]
        ]
        self.blocks = nn.Sequential(*blocks)

        self.m_head = MLPSimple(self.t_config.d_model, 1)
        self.b_head = MLPSimple(self.t_config.d_model, 1)
        

    def forward(self, x):
        b = x.shape[0]

        # x is [b,n,c,h,w]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv_in(x)
        x = self.blocks(x)

        # x is [b,c,8,4,4]
        x = x.flatten(2).permute(0, 2, 1) # -> [b,8*4*4,c]
        x = self.proj_t_in(x)
        x = self.pos_enc(x)

        m_tok = self.mouse_tokens.unsqueeze(0).repeat(b,1,1)
        b_tok = self.button_tokens.unsqueeze(0).repeat(b,1,1)

        x = torch.cat([x, m_tok, b_tok], dim = 1)
        x = self.transformer(x)
        x = x[:,-self.n_cls_toks:,:]

        m_emb = x[:,:self.config.n_mouse_axes,:]
        b_emb = x[:,self.config.n_mouse_axes:,:]

        m_preds = self.m_head(m_emb).squeeze(-1) # -> [b,n_mouse_axes]
        b_logits = self.b_head(b_emb).squeeze(-1) # -> [b,n_buttons]

        return m_preds, b_logits

if __name__ == "__main__":
    from ..configs import Config

    cfg = Config.from_yaml("configs/basic.yml").model

    model = ControlPredictor(cfg).bfloat16().cuda()
    x = torch.randn(1,32,3,128,128).bfloat16().cuda()

    with torch.no_grad():
        m_pred, b_logits = model(x)
        print(m_pred.shape)
        print(b_logits.shape)