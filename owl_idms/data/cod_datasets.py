from __future__ import annotations
import os, random
from functools import cache
from typing import Callable, Literal, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset
import kornia.augmentation as K


DEFAULT_TRANSFORM = K.VideoSequential(                # lives on GPU
    K.RandomAffine(degrees=0.0,
                   translate=0.05,
                   scale=(0.9, 1.1),
                   p=1.0),
    K.ColorJitter(brightness=0.1,
                  contrast=0.1,
                  saturation=0.1,
                  hue=0.0,
                  p=1.0),
    K.RandomGaussianNoise(mean=0.0, std=0.02, p=1.0),
    data_format="BTCHW",        # (B,T,C,H,W)
    same_on_frame=True,        # ➀ same params for all frames in a clip
)


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


class CoDDataset(IterableDataset):
    def __init__(
        self,
        window_length: int = 32,
        root: str = "/home/shared/cod_data/",
        split: Literal["train", "val"] = "train",
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.window = window_length
        paths = get_cod_paths(root)
        cut = int(len(paths) * 0.75)
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

        if self.transform is not None:
            vid = self.transform(vid)

        return vid.to(torch.bfloat16), mouse[s : s + self.window], buttons[s : s + self.window]

    def __iter__(self):
        while True:
            yield self._random_clip()


def collate_fn(batch: Sequence):
    v, m, b = zip(*batch)
    return torch.stack(v), torch.stack(m), torch.stack(b)


def get_loader(batch_size: int, split: Literal["train", "val"] = "train", **dl_kwargs):
    return DataLoader(
        CoDDataset(split=split),
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=collate_fn,
        # prefetch_factor=12,
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
