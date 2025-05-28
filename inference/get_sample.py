from owl_idms.data.cod_data import CoDDataset

import random
import torch

from inference.vis import write_video
from tqdm import tqdm

# Set random seeds
random.seed(43)
torch.manual_seed(43)

ds = CoDDataset(window_length = 1000)
vid,_,_ = ds.get_item()

from owl_idms.configs import Config
from owl_idms.models.cnn_3d import ControlPredictor

cfg = Config.from_yaml("configs/basic_adamw.yml").model
model = ControlPredictor(cfg)

ckpt = torch.load('checkpoints/v0/step_17000.pt', weights_only=False,map_location='cpu')['ema']
prefix = "ema_model."
ckpt = {k[len(prefix):] : v for (k,v) in ckpt.items() if k.startswith(prefix)}

model.load_state_dict(ckpt)
model = model.bfloat16().cuda().eval()
#model = torch.compile(model)
print("Success")
print(vid.shape)

def unscale_mouse(mouse):
    # mouse is [b,2]
    # Apply inverse of symlog to each component
    # symlog^-1(x) = sign(x) * (exp(|x|) - 1)
    return torch.sign(mouse) * torch.expm1(torch.abs(mouse)) # [b,2]

# Inference with sliding window
WINDOW_LENGTH = 32
N_FRAMES = len(vid)
# Calculate padding sizes
pad_start = WINDOW_LENGTH // 2
pad_end = WINDOW_LENGTH // 2 - 1

# Get first and last frames
first_frame = vid[0:1].expand(pad_start, -1, -1, -1)  # Repeat first frame
last_frame = vid[-1:].expand(pad_end, -1, -1, -1)     # Repeat last frame

# Concatenate padding with original video
padded_vid = torch.cat([first_frame, vid, last_frame], dim=0)
print(padded_vid.shape) # 543 expected

mouse_preds = []
btn_preds = []

with torch.no_grad():
    for i in tqdm(range(N_FRAMES)):
        sample = padded_vid[i:i+WINDOW_LENGTH].unsqueeze(0)
        (mouse_pred,_),button_logits = model(sample.bfloat16().cuda())

        button_pred = (button_logits.sigmoid() > 0.5)

        mouse_preds.append(mouse_pred)
        btn_preds.append(button_pred)

    mouse_preds = torch.cat(mouse_preds)
    mouse_preds = unscale_mouse(mouse_preds)

    btn_preds = torch.cat(btn_preds)

write_video(vid.float(), mouse_preds, btn_preds)