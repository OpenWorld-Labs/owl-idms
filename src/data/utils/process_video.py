from __future__ import annotations

from pathlib import Path
from typing import Callable

import os, PIL
import torch, numpy as np, decord ; decord.bridge.set_bridge('torch')
from torchvision import transforms

from src.constants import FPS, ROOT_DIR
from src.data.utils import str_to_dtype

def create_transforms(aspect_ratio: str,
                      resize: int) -> Callable[[torch.Tensor], torch.Tensor]:
    
    _transforms = [transforms.Resize(resize),
                   transforms.Lambda(lambda x: x.float()),
                   transforms.Lambda(lambda x: x.div_(255.))]

    if aspect_ratio != "1:1":
        w, h = aspect_ratio.split(":") ; w, h = int(w), int(h)
        w, h = w * resize // max(w, h), h * resize // max(w, h)
        _transforms += [transforms.CenterCrop((h, w))]

    return transforms.Compose(_transforms)


def get_frames(video_path: Path, *,
               num_threads: int = 0,
               device: torch.device = torch.device('cpu'),
               cap_frames: int = None) -> torch.Tensor:

    vr = decord.VideoReader(str(video_path), num_threads=num_threads, ctx=device)
    frames: torch.Tensor = vr.get_batch(np.arange(min(cap_frames or len(vr),
                                                      len(vr))))
    return frames.permute(0,3,1,2)


def _save_sample(sample_chw: torch.Tensor, path: Path):

    c,h,w = sample_chw.shape
    sample = sample_chw.cpu().numpy()
    
    if c == 1: sample = np.repeat(sample, 3, axis=0)
    if c == 2: sample = np.concatenate([sample, np.zeros((1,h,w), dtype=sample.dtype)], axis=0)

    multiplier = 255. if sample.max() <= 1 else 1.
    sample = PIL.Image.fromarray((sample.transpose(1, 2, 0) * multiplier).astype(np.uint8))
    print(f'Saving sample to {path}') ; sample.save(path)


def preprocess_video_frames(video_path: str | os.PathLike, *,
                            transform: Callable[[torch.Tensor], torch.Tensor] = None,
                            num_threads: int = 0,
                            device: torch.device = torch.device('cpu'),
                            save_dtype: torch.dtype = torch.float8_e4m3fn,
                            keep_channels: list[int] = [0,1,2],
                            save_sample: bool = False,
                            cap_frames: int = None) -> torch.Tensor:
    
    assert len(keep_channels) > 0, "Must keep at least one channel"

    print(f'Processing video {video_path}')
    frames = get_frames(video_path, num_threads=num_threads, device=device, cap_frames=cap_frames)
    frames = frames[:, keep_channels, :, :]

    if transform: frames = transform(frames)
    if save_sample: _save_sample(frames[len(frames) // 2] , video_path / 'sample.png')

    # NOTE Convert to dtype here since a lot of transforms don't work with fp8
    frames = frames.to(dtype=save_dtype)

    return frames


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    # -- dir
    p.add_argument('--root', type=str, default=ROOT_DIR)
    p.add_argument('--out_dir', type=str, default=ROOT_DIR / 'processed')
    # -- video transforms
    p.add_argument('--fps', type=int, default=FPS)
    p.add_argument('--resize', type=int, default=512)
    p.add_argument('--discard_channel', nargs='+', default=[], choices=['r', 'g', 'b', 'R', 'G', 'B'])
    p.add_argument('--aspect_ratio', type=str, default='1:1', choices=['1:1', '16:9', '4:3', '1:2', '2:1'])
    p.add_argument('--normalize', action='store_true', default=False)
    p.add_argument('--save_dtype', type=str, default='fp8', choices=['fp8', 'fp16', 'fp32', 'bfloat16'])
    # -- processing
    p.add_argument('--num_threads', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--skip_existing', action='store_true', default=False)
    p.add_argument('--save_sample', action='store_true', default=False)
    return p.parse_args()


def main():
    args        = parse_args()
    device      = decord.gpu(0) if args.device == 'gpu' else decord.cpu(0)
    save_dtype  = str_to_dtype(args.save_dtype)

    c2i: dict[str, int] = dict(zip('rgb', range(3)))
    discard_channel: set[int] = set([c2i[c.lower()]
                                     for c in args.discard_channel])
    assert 0 <= len(discard_channel) <= 2, "Must discard 0, 1, or 2 channels"

    transform   = create_transforms(args.aspect_ratio,
                                    args.resize)
    out_dir     = args.out_dir
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    preprocess_video_frames(args.root,
                      transform=transform, device=device,
                      save_dtype=save_dtype, save_sample=args.save_sample,
                      keep_channels=list(set(range(3)) - discard_channel),
                      out_dir=out_dir)

if __name__ == "__main__":


    out_dir = './data_dumps/processed' 
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    dtype = str_to_dtype('fp8')
    device = decord.gpu(0)
    transform = create_transforms('1:1', 512)
    preprocess_video_frames(ROOT_DIR,
                      transform=transform, device=device, save_dtype=dtype,
                      save_sample=True,
                      keep_channels=list(set(range(3)) - set([2])),
                      out_dir=out_dir)

