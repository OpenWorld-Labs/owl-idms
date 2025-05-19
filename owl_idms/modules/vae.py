import torch

from torch import nn
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

def get_vae(vae_id: str, mode: str = "encoder"):
    """
    Get a specific autoencoder then get either its encoder or decoder
    """
    if vae_id == "sd":
        return VAE("madebyollin/taesd", mode)
    if vae_id == "sdxl":
        return VAE("madebyollin/taesdxl", mode)
    if vae_id == "flux":
        return VAE("madebyollin/taef1", mode)
    raise ValueError(f"Unknown VAE ID: {vae_id}")


class VAE(nn.Module):
    def __init__(self, model_id, mode="encoder", do_compile=True):
        super().__init__()

        vae = AutoencoderTiny.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        if mode == "encoder":
            self.model = vae.encoder
            del vae.decoder
        if mode == "decoder":
            self.model = vae.decoder
            del vae.encoder

        if do_compile:
            self.model = torch.compile(self.model)

    def forward(self, x):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return self.model(x)

class IdentityModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): # pretend it outputs a latent haha 
        return torch.randn(10, 16, 16, 32)


# for giggles for now
def vae_debug():
    return IdentityModule()

def sd():
    return get_vae("madebyollin/taesd", "encoder")

def sdxl():
    return get_vae("madebyollin/taesdxl", "encoder")

def flux():
    return get_vae("madebyollin/taef1", "encoder")