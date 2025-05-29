import torch
from owl_idms.configs import Config
from owl_idms.models import get_model_cls

# -----------------------------------------------------------------------------
# Defaults for model loading
# -----------------------------------------------------------------------------
DEFAULT_MODEL_PATH = '/home/sami/owl_idms/inference/17k_ema_idm.pt'
DEFAULT_CFG_PATH   = '/home/sami/owl_idms/inference/shab.yml'
WINDOW_SIZE = 32

def infer_actions(
    model: torch.nn.Module,
    video: torch.Tensor,
    window_size: int = WINDOW_SIZE,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the model on sliding windows of the video in batches.
    
    Args:
        model: a PyTorch model that takes input of shape
               (B, window_size, C, H, W) and returns
               ((m_mu_preds, m_logvar_preds), button_logits),
               where each prediction tensor has time‐dim ≥ window_size+1.
        video: tensor of shape (T, C, H, W)
        window_size: number of frames per window
        batch_size: how many windows to process at once
    
    Returns:
        windows:      Tensor of shape (N, window_size, C, H, W)
        buttons:      Tensor of shape (N, num_buttons)
        mouse_vec:    Tensor of shape (N, vec_dim)
        mouse_std:    Tensor of shape (N, vec_dim)
    """
    #--- 
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # Number of sliding windows
    T = video.size(0)
    N = T - window_size
    assert N > 0, f"Video length {T} must exceed window_size {window_size}"

    # Build all sliding windows on CPU:
    #   unfold dim=0 into shape (T-window_size+1, window_size, C, H, W),
    #   then drop the last one so we have exactly N = T - window_size
    windows: torch.Tensor = (
        video
        .unfold(0, window_size, 1)   # (T-window_size+1, window_size, C, H, W)
        [:N]                         # (N, window_size, C, H, W)
        .contiguous()
        .permute(0,4,1,2,3)          # (B, N, C, H, W)
        .to(device=device, dtype=dtype)
    )

    all_buttons = []
    all_mouse_vec = []
    all_mouse_std = []

    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype):
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            batch = windows[i:j].to(device)  # (B, window_size, C, H, W)

            # Forward pass
            (m_mu_preds, m_logvar_preds), button_logits = model(batch)
            on_buttons = (torch.sigmoid(button_logits) > 0.5).float()

            # Extract the t = window_size prediction from each sequence
            #   m_mu_preds: (B, ≥window_size+1, vec_dim)
            #   button_logits: (B, ≥window_size+1, num_buttons)
            b_vec = m_mu_preds.cpu()          # (B, vec_dim)
            b_std = torch.exp(m_logvar_preds).cpu()  # (B, vec_dim)
            b_btn = on_buttons.cpu()          # (B, num_buttons)

            all_mouse_vec.append(b_vec)
            all_mouse_std.append(b_std)
            all_buttons.append(b_btn)

    buttons   = torch.cat(all_buttons,   dim=0).bool()  # (N, num_buttons)
    mouse_vec = torch.cat(all_mouse_vec, dim=0)         # (N, vec_dim)
    mouse_std = torch.cat(all_mouse_std, dim=0)         # (N, vec_dim)

    return windows[:, window_size//2], buttons, mouse_vec, mouse_std


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    cfg_path:   str = DEFAULT_CFG_PATH,
    *,
    device:      torch.device   = torch.device('cuda'),
    dtype:       torch.dtype     = torch.bfloat16,
    weights_only: bool           = False,
) -> torch.nn.Module:
    """
    Load and prepare an IDM model for inference.

    Args:
        model_path (str): Path to the .pt checkpoint.
        cfg_path   (str): Path to the YAML config for model instantiation.
        device     (torch.device): Target device (e.g. 'cpu' or 'cuda').
        dtype      (torch.dtype):  Desired parameter dtype after loading.
        weights_only(bool):       If True, ignore extra keys in the checkpoint.

    Returns:
        torch.nn.Module: The model on `device` in eval mode.
    """
    # 1. Load configuration and instantiate
    cfg        = Config.from_yaml(cfg_path)
    ModelClass = get_model_cls(cfg.model.model_id)
    model      = ModelClass(cfg.model)

    # 2. Load checkpoint (to CPU first, then transfer)
    ckpt = torch.load(
        model_path,
        map_location='cpu',
        weights_only=weights_only
    )
    model.load_state_dict(ckpt)

    # 3. Cast and move to device, then switch to eval
    model = model.to(dtype=dtype, device=device)
    model.eval()
    return model

if __name__ == "__main__":
    pass