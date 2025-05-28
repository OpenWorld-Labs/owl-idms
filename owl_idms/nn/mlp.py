from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x

class MLPSimple(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.fc1 = nn.Linear(dim_in, 4 * dim_in)
        self.fc2 = nn.Linear(4 * dim_in, dim_out)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x
