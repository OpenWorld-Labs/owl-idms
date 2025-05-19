import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal

from owl_idms.constants import KEYBINDS

class DummyDataset(Dataset):
    def __init__(
        self,
        dataset_size=1000,
        seq_length=10,
        channels=3,
        height=224,
        width=224,
        num_buttons=len(KEYBINDS),
        video_dtype=torch.bfloat16,
    ):
        """
        Args:
            dataset_size: Number of samples in the dataset
            seq_length: Number of frames in each video sample
            channels: Number of channels in video frames
            height: Height of video frames
            width: Width of video frames
            num_buttons: Number of buttons for key inputs
            video_dtype: Data type for video tensors
        """
        self.dataset_size = dataset_size
        self.seq_length = seq_length
        self.channels = channels
        self.height = height
        self.width = width
        self.num_buttons = num_buttons
        self.video_dtype = video_dtype

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Generate random video tensor (n,c,h,w)
        video = torch.rand(
            self.seq_length, 
            self.channels, 
            self.height, 
            self.width
        ).to(self.video_dtype)
        
        # Generate random key presses (n,num_buttons)
        keys = torch.randint(
            0, 2, 
            (self.seq_length, self.num_buttons)
        ).bool()
        
        # Generate random mouse movements (n,2) in range (-1,1)
        mouse = torch.rand(self.seq_length, 2) * 2 - 1
        
        return video, keys, mouse

def dummy_dataloader(
    batch_size=4,
    seq_length=10,
    channels=3,
    height=64,
    width=64,
    num_buttons=8,
    num_workers=0,
    video_dtype=torch.bfloat16,
    split: Literal["train", "val"] = "train",
):
    dataset = DummyDataset(
        dataset_size=8_000 if split == "train" else 1_000,
        seq_length=seq_length,
        channels=channels,
        height=height,
        width=width,
        num_buttons=num_buttons,
        video_dtype=video_dtype,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        num_workers=num_workers,
    )
