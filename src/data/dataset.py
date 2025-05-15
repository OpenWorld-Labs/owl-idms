from pathlib import Path
from typing import Sequence, Callable, Literal

import torch, pandas as pd, json
from torch.utils.data import Dataset

from src.data.process_data import seek_tensors

__all__ = ["InverseDynamicsDataset"]


class InverseDynamicsDataset(Dataset):
    def __init__(self, datalist_filepath: str | Path, tensor_dir: str | Path, *,
                 transform: Callable = None,
                 map_location: torch.device = None):
        self.datalist_filepath = Path(datalist_filepath)
        self.tensor_dir = Path(tensor_dir)
        self.datalist   = pd.read_csv(self.datalist_filepath)
        self.meta       = json.load(open(self.datalist_filepath.parent / "metadata.json"))
        self.transform  = transform
        
        self.map_location = map_location
        self.paths: dict[int, list[Path, Path, Path]] = seek_tensors(self.tensor_dir)

    def _get_video(self, path: Path) -> torch.Tensor: 
        return torch.load(path, map_location=self.map_location, mmap=True)

    def _get_actions(self, btn_path: Path, mouse_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.load(btn_path,   map_location=self.map_location, mmap=True),
                torch.load(mouse_path, map_location=self.map_location, mmap=True))

    def __len__(self): return len(self.datalist)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        row                = self.datalist.iloc[idx]
        start_f, clip_size = int(row.start), int(row.clip_size)
        frame_idxs: Sequence[int] = range(start_f, start_f + clip_size) # (t,h,w,3) uint8

        video_path, btn_path, mouse_path = self.paths[row.tensor_idx]
        frames_tchw: torch.Tensor        = self._get_video(video_path)[frame_idxs]
        btn, mouse                       = self._get_actions(btn_path, mouse_path)
        # FIXME Sometimes they are not the same length! 
        btn, mouse                       = btn[frame_idxs], mouse[frame_idxs]
        act_vec: torch.Tensor            = torch.cat((btn, mouse), dim=1)
        
        if self.transform:
            frames_tchw = self.transform(frames_tchw)
        # TODO Questions 5/15/2025:
        # - Should we only be reading in frames that have actions?
        # - Should there be a no-action option in the act_vec?
        # - - If so, how should that affect the datalist creation?
        # - There are often multiple actions per frame, not just multiple keys per action. How should we handle this?
        # - - we end up in a situation where our button and mouse tensors are like [42,6] and [36,2] yet our frames are always [6, ...]
        # - - so how do what is the action target in this case?
        # - - And how do we handle the action dimension mismatch?
        return {
            'obs': frames_tchw, # (t,c,h,w)
            'action': act_vec, # (action_dim,), which is like 8 or something
            'meta': {'video': video_path, 'start': start_f, 'clip_size': clip_size}
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.constants import ROOT_DIR
    ds = InverseDynamicsDataset(
        datalist_filepath='./datalist/datalist.csv',
        tensor_dir=ROOT_DIR, transform=None)

    print(ds[0])

    dl = DataLoader(ds, batch_size=3, shuffle=True)
    for batch in dl:
        print(batch)
        break