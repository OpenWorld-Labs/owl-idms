from __future__ import annotations
import os, random
from functools import cache
from typing import Callable, Literal, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset
import kornia.augmentation as K


DEFAULT_TRANSFORM_CPU =  K.VideoSequential(
        K.RandomAffine(degrees=0.0, translate=0.05, scale=(0.9,1.1), p=1.0),
        K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=1.0),
        K.RandomGaussianNoise(mean=0.0, std=0.02, p=1.0),
        data_format="BTCHW",       # expects (B,T,C,H,W)
        same_on_frame=True,
)

from torchvision.transforms import Compose, RandomAffine, ColorJitter, Lambda
TORCHVISION_TRANSFORMS = Compose([
    # RandomAffine(degrees=0.0, translate=(0.05, 0.05), scale=(0.9,1.1)),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    Lambda(lambda x: x + (torch.randn_like(x) * 0.02)),
])


@cache
def get_cod_paths(root: str = "/home/shared/cod_data/") -> list[tuple[str, str, str]]:
    paths: list[tuple[str, str, str]] = []
    for root_dir in os.listdir(root):
        splits_dir = os.path.join(root, root_dir, "splits")
        if not os.path.isdir(splits_dir):
            continue
        for name in os.listdir(splits_dir):
            if "_mouse" in name or "_buttons" in name:
                continue
            stem = os.path.splitext(name)[0]
            v = os.path.join(splits_dir, name)
            m = os.path.join(splits_dir, f"{stem}_mouse.pt")
            b = os.path.join(splits_dir, f"{stem}_buttons.pt")
            if os.path.exists(m) and os.path.exists(b):
                paths.append((v, m, b))
    return paths

MOUSE_STATS = {
  "clip_mag": 10.883796691894531,
  "median": 0.8615849614143372,
  "iqr": 1.0,
  "x_mean": 0.009292899630963802,
  "x_std": 1.8915449380874634
}
_CLIP = MOUSE_STATS["clip_mag"]
_MED  = MOUSE_STATS["median"]
_IQR  = MOUSE_STATS["iqr"] + 1e-6     # avoid /0

def normalise_mouse(dx_dy: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    dx_dy : Tensor[2] – raw integer deltas
    Returns
    -------
    Tensor[2] – clipped-log-robust-scaled deltas in roughly [-2, 2]
    """
    # 1) clip by magnitude (vectorially, not per-component)
    mag   = dx_dy.norm()
    if mag > _CLIP:
        dx_dy = dx_dy * (_CLIP / mag)

    # 2) log-compress component-wise (keeps sign)
    dx_dy = torch.sign(dx_dy) * torch.log1p(dx_dy.abs())

    # 3) robust scale with global median/IQR **per component** or on |Δ|
    dx_dy = (dx_dy - _MED) / _IQR

    # 4) optional squash to [-1,1] to match tanh in the network head
    dx_dy = torch.tanh(dx_dy)

    return dx_dy

class CoDDataset(IterableDataset):
    def __init__(
        self,
        window_length: int = 32,
        root: str = "/home/shared/cod_data/",
        split: Literal["train", "val"] = "train",
        transform: Callable = DEFAULT_TRANSFORM_CPU # TORCHVISION_TRANSFORMS,
    ):
        super().__init__()
        self.window = window_length
        paths = get_cod_paths(root)
        cut = int(len(paths) * 0.85)
        self.paths = paths[:cut] if split == "train" else paths[cut:]
        self.transform = transform

    # ------------------------------------------------------------------
    def _random_clip(self):
        v_path, m_path, b_path = random.choice(self.paths)
        vid = torch.load(v_path, map_location="cpu", mmap=True)     # [N,C,H,W]
        mouse = torch.load(m_path, map_location="cpu", mmap=True)
        buttons = torch.load(b_path, map_location="cpu", mmap=True)

        s = random.randint(0, min(len(vid), len(mouse), len(buttons)) - self.window)
        vid = vid[s : s + self.window].float()                      # [-1,1]
        vid = (vid + 1.0) * 0.5                                     # → [0,1]

        mouse = mouse[s : s + self.window]
        buttons = buttons[s : s + self.window]

        if self.transform is not None:
            vid = self.transform(vid.unsqueeze(0)).squeeze(0)
            # vid = torch.stack([self.transform(frame) for frame in vid])
            # vid = (vid - vid.min()) / (vid.max() - vid.min())

        return (
            vid.to(torch.bfloat16), # [b,t,c,h,w]
            normalise_mouse(mouse),
            buttons
        )

    def __iter__(self):
        while True:
            yield self._random_clip()


def collate_fn(batch: Sequence):
    v, m, b = zip(*batch)
    return torch.stack(v), torch.stack(m), torch.stack(b)


def get_loader(batch_size: int, split: Literal["train", "val"] = "train",
               **dl_kwargs):
    return DataLoader(
        CoDDataset(split=split),
        batch_size=batch_size,
        pin_memory=True,
        persistent_workers=False, # TODO 
        # persistent_workers=True if dl_kwargs.get("num_workers", 0) > 0 else False,
        collate_fn=collate_fn,
        prefetch_factor=24,
        **dl_kwargs,
    )


# quick smoke-test -----------------------------------------------------------
if __name__ == "__main__":
    import time, torch.backends.cuda
    torch.backends.cuda.matmul.allow_tf32 = True

    loader = get_loader(32, split="val")
    t0 = time.time()
    vids, mouse, buttons = next(iter(loader))
    print(f"{time.time() - t0:.3f}s  →", vids.shape, mouse.shape, buttons.shape)
