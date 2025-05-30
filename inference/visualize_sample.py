import torch
import random

from owl_idms.data.cod_data import CoDDataset
from inference.visualize_overlay_actions import overlay_infer_actions


# Set random seeds
random.seed(43)
torch.manual_seed(43)

ds              = CoDDataset(window_length = 1000)
ds_index        = random.randint(0, 10)

cfg_path        = 'configs/basic_adamw.yml'
model_path      = '17k_ema_idm.pt'
mp4_save_path   = 'sample.mp4'
device          = 'cuda'

overlay_infer_actions(
    model_path = model_path,
    cfg_path = cfg_path,
    use_dataset = True,
    dataset = ds,
    dataset_index = ds_index,
    save_path = mp4_save_path,
    device=device,
)
