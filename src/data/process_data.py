from __future__ import annotations

import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Literal
import os, torch, decord, json, csv, pandas as pd
from toolz import groupby, valfilter, valmap, partial

from src.constants import FPS, ROOT_DIR
from src.data.utils import (preprocess_video_frames, create_transforms,
                            preprocess_actions, seek_video_dirs, str_to_dtype)

def process_raw_data(src_dir: os.PathLike, dst_dir: os.PathLike, *,
                     aspect_ratio: str = '1:1',
                     resize: int = 512,
                     keep_channels: list[int] = [0,1,2],
                     fps: int = FPS,
                     device: str = Literal['cpu', 'gpu'],
                     save_dtype: torch.dtype,
                     cap_frames: int = None # NOTE for debugging so we dont go oom
                ) -> tuple[list[Path], list[Path], list[Path]]:

    device = decord.gpu(0) if device == 'gpu' else decord.cpu(0)
    dst_dir, src_dir = Path(dst_dir), Path(src_dir)
    video_transform = create_transforms(aspect_ratio, resize)

    video_dirs = tqdm(seek_video_dirs(src_dir), desc="Processing videos")
    paths = []

    for idx, video_dir in enumerate(video_dirs):
        if not (videos := list(video_dir.glob("*.mp4"))) or len(videos) != 1:
            print(f"Found {len(videos)} videos in {video_dir}, expected 1") ; continue

        video_path  = videos[0].absolute()
        action_path = video_dir / "inputs.csv"

        if not action_path.exists():
            print(f'Missing action file: {action_path}') ; continue

        video_dst, mouse_dst, buttons_dst = (
            Path(dst_dir) / f"video_{idx:06d}.pt",
            Path(dst_dir) / f"mouse_{idx:06d}.pt",
            Path(dst_dir) / f"buttons_{idx:06d}.pt"
        )

        video_frames = preprocess_video_frames(video_path,
                                               transform=video_transform,
                                               keep_channels=keep_channels,
                                               save_dtype=save_dtype,
                                               device=device,
                                               cap_frames=cap_frames)
        
        button_data, mouse_data = preprocess_actions(action_path, fps=fps)

        if not dst_dir.exists():
            print(f'Creating directory: {dst_dir}') ; dst_dir.mkdir(parents=True, exist_ok=True)

        torch.save(video_frames, video_dst)
        torch.save(button_data,  buttons_dst)
        torch.save(mouse_data,   mouse_dst)
        
        print(f'Saved {video_dst}, {buttons_dst}, {mouse_dst}')
        paths.append((video_dst, buttons_dst, mouse_dst))

    return list(zip(*paths)) # video_paths, buttons_paths, mouse_paths


def seek_tensors(src_dir: os.PathLike) -> dict[int, list[Path, # video, buttons, mouse
                                                         Path,
                                                         Path]]:
    src_dir = Path(src_dir)

    def _sorting(path: Path) -> int:
        if path.stem.startswith('video'):   return 0
        if path.stem.startswith('buttons'): return 1
        if path.stem.startswith('mouse'):   return 2

    def _tensor_key(path: Path) -> int: return int(path.stem.split('_')[1])

    tensors:       dict[int, list[Path]] = groupby(_tensor_key, src_dir.glob("*.pt"))
    valid_tensors: dict[int, list[Path]] = valfilter(lambda x: len(x) == 3, tensors)

    for key in tensors.keys() - valid_tensors.keys():
        print(f"Found {len(tensors[key])} tensors for video {key}, expected 3")
    
    return valmap(partial(sorted, key=_sorting), valid_tensors)


def process_data_bounds(src_dir: os.PathLike, dst_dir: os.PathLike, *,
                        clip_size: int,
                        frame_skip: int,
                        stride: int,
                        allow_overlap: bool = False):
    assert (frame_skip * clip_size * stride) > 0
    if not allow_overlap: 
        if stride >= clip_size: print(f"{allow_overlap=}, but stride is larger than (clip size x frame_skip) = {clip_size}, so clips won't overlap anyways.")
        stride += (clip_size if stride < clip_size else 0)

    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    out_path, meta_path = dst_dir / "datalist.csv", dst_dir / "metadata.json"
    if not dst_dir.exists():
        print(f'Creating directory: {dst_dir}') ; dst_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(clip_size=clip_size, frame_skip=frame_skip, stride=stride)

    rows: list[tuple[str, str, str]] = []

    for idx, (video_path, *_ ) in seek_tensors(src_dir).items():
        n_frames, *_ = torch.load(video_path, mmap=True).shape
        rows.extend((idx, start, clip_size)
                    for start in range(0,(n_frames- clip_size)+1,stride))

    with open(meta_path, "w+") as fmeta, open(out_path, "w+") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames:=["tensor_idx", "start", "clip_size"])
        writer.writeheader()
        writer.writerows([dict(zip(fieldnames, row)) for row in rows])
        json.dump(meta, fmeta)
        print(f"Wrote {len(rows)} rows to {out_path}, and metadata to {meta_path}")


def add_tensor_pargs(argparse: argparse.ArgumentParser):
    argparse.add_argument("--tensor_dir",   type=str,   default="processed/")
    argparse.add_argument("--aspect-ratio", type=str,   default="1:1")
    argparse.add_argument("--resize",       type=int,   default=512)
    argparse.add_argument("--keep-channels",type=list,  default=[0,1,2])
    argparse.add_argument("--fps",          type=int,   default=FPS)
    argparse.add_argument("--device",       type=str,   default="gpu", choices=["cpu", "gpu"])
    argparse.add_argument("--save-dtype",   type=str,   default="fp8", choices=["fp8", "fp16", "fp32", "bfloat16"])
    argparse.add_argument("--cap-frames",   type=int,   default=None)
    return argparse

def add_bounds_pargs(argparse: argparse.ArgumentParser):
    argparse.add_argument("--datalist_dir", type=str,   default="datalist/")
    argparse.add_argument("--clip-size",    type=int,   default=6)
    argparse.add_argument("--frame-skip",   type=int,   default=1)
    argparse.add_argument("--stride",       type=int,   default=1)
    argparse.add_argument("--allow-overlap",type=bool,  default=False)
    return argparse

def parse_args():
    p = argparse.ArgumentParser()
    add_tensor_pargs(p)
    add_bounds_pargs(p)
    return p.parse_args()

def main():
    args = parse_args()
    save_dtype  = str_to_dtype(args.save_dtype)

    process_raw_data(ROOT_DIR, dst_dir=args.tensor_dir,
                     aspect_ratio=args.aspect_ratio,
                     resize=args.resize,
                     keep_channels=args.keep_channels,
                     fps=args.fps,
                     device=args.device,
                     save_dtype=save_dtype,
                     cap_frames=args.cap_frames)

    process_data_bounds(src_dir=args.tensor_dir, dst_dir=args.datalist_dir,
                        clip_size=args.clip_size,
                        frame_skip=args.frame_skip,
                        stride=args.stride,
                        allow_overlap=args.allow_overlap)


if __name__ == "__main__":
    import sys
    sys.argv[1:] = ["--tensor_dir", "processed/", "--datalist_dir", "datalist/", "--cap-frames", "100"]
    main()

