import torch

from torch import nn
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

def get_vae(vae_id: str, mode: str = "encoder", do_compile: bool = True, requires_grad: bool = True):
    """
    Get a specific autoencoder then get either its encoder or decoder
    """
    return VAE(vae_id, mode=mode, do_compile=do_compile, requires_grad=requires_grad)


class VAE(nn.Module):
    def __init__(self, model_id, mode="encoder", do_compile=True, requires_grad=True):
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

        for p in self.model.parameters():
            p.requires_grad = requires_grad

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

def sd(do_compile: bool = True, requires_grad: bool = True):
    return get_vae("madebyollin/taesd", "encoder", do_compile, requires_grad)

def sdxl(do_compile: bool = True, requires_grad: bool = True):
    return get_vae("madebyollin/taesdxl", "encoder", do_compile, requires_grad)

def flux(do_compile: bool = True, requires_grad: bool = True):
    return get_vae("madebyollin/taef1", "encoder", do_compile, requires_grad)


if __name__ == "__main__":
    vaes = [sd(), sdxl(), flux()]
    names = ["SD", "SDXL", "Flux"]
    
    for name, vae in zip(names, vaes):
        param_count = sum(p.numel() for p in vae.parameters())
        print(f"{name} VAE parameter count: {param_count:,}")
