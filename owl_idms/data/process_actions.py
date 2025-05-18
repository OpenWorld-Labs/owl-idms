from __future__ import annotations

import os, torch, argparse, pandas as pd
from toolz import pipe

from src.data import utils
from src.data.utils import code_from_ascii
from src.constants import FPS, ROOT_DIR, KEYBINDS


def to_button_data_tensor(action_data: pd.DataFrame, *,
                          fps: int = FPS, filter_out_keys: tuple[str, ...] = ("LMB", "RMB")) -> torch.Tensor:

    valid_codes: list[int] = [code_from_ascii(k) for k in KEYBINDS if k not in filter_out_keys]
    return pipe(
        action_data,
        utils._normalize_timestamps,
        utils._filter_event_types,
        utils._filter_keys(valid_codes),
        utils._convert_events,
        utils._add_frame_column(fps),
        utils._simplify_event_types,
        utils._collapse_by_frame,
        utils._events_to_tensor(KEYBINDS),
    )


def mouse_data_to_tensor(action_data: pd.DataFrame, *, fps: int = FPS) -> torch.Tensor:
    return pipe(
        action_data,
        utils._normalize_timestamps,
        utils._filter_mouse_moves,
        utils._add_frame_column(fps),
        utils._parse_mouse_args,
        utils._aggregate_mouse_by_frame,
        utils._mouse_to_tensor,
    )


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=ROOT_DIR)
    p.add_argument("--out_dir", type=str, default=ROOT_DIR / 'processed')
    p.add_argument("--fps", type=int, default=FPS)
    # TODO Maybe add split size here optionally and we can chunk shit in preprocess_videos
    return p.parse_args()


def preprocess_actions(csv_path: str | os.PathLike, *,
                       fps: int = FPS) -> tuple[torch.Tensor, torch.Tensor]:
    data = pd.read_csv(csv_path)
    return to_button_data_tensor(data, fps=fps), mouse_data_to_tensor(data, fps=fps)


def main():
    args = _parse_args()
    global FPS, ROOT_DIR, OUT_DIR
    FPS, ROOT_DIR, OUT_DIR = args.fps, args.root, args.out_dir
    preprocess_actions(ROOT_DIR, OUT_DIR)


if __name__ == "__main__":
    main()