from __future__ import annotations

import io
import tarfile
from toolz import curry
from pathlib import Path
from typing import TypeVar

import torch
from torch import Tensor
from collections.abc import Sized
from torch.utils.data import SequentialSampler
from torchdata.nodes import (
    SamplerWrapper,
    ParallelMapper,
    Unbatcher,
    Batcher,
    Loader,
    BaseNode,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from owl_idms.data.tigris import get_filesystem, glob_shards

T = TypeVar("T")

def _load_pt(buf: bytes) -> Tensor:
    """Load a `.pt` tensor saved via `torch.save(tensor, file)`."""
    return torch.load(io.BytesIO(buf), map_location="cpu")


@curry
def _process_tar(url: str, fs) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Download + parse a shard into ``[(video, button, mouse), ...]`` pairs."""
    data: list[tuple[Tensor, Tensor, Tensor]] = []
    stash = {}

    with fs.open(url, "rb") as fh, tarfile.open(fileobj=fh, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name)    # 000012.video.pt or 000012.button.pt or 000012.mouse.pt
            stem = name.stem            # 000012.video    or 000012.button    or 000012.mouse

            raw = tar.extractfile(member).read()

            if ".video" in stem:
                base_id = stem.replace(".video", "")
                tensor = _load_pt(raw)
                stash.setdefault(base_id, {})["video"] = tensor
            elif ".button" in stem:
                base_id = stem.replace(".button", "")
                tensor = _load_pt(raw)
                stash.setdefault(base_id, {})["button"] = tensor
            elif ".mouse" in stem:
                base_id = stem.replace(".mouse", "")
                tensor = _load_pt(raw)
                stash.setdefault(base_id, {})["mouse"] = tensor
            else:
                continue

            rec = stash.get(base_id)
            if rec and {"video", "button", "mouse"} <= rec.keys():
                data.append((rec["video"], rec["button"], rec["mouse"]))
                del stash[base_id]

    return data


def make_tensor_video_loader(
    shard_urls: list[str],
    fs,
    *,
    batch_size: int = 8,
    num_workers: int = 8,
    drop_last: bool = False,
) -> StatefulDataLoader:
    """Return a Loader streaming (video, button, mouse) tensors from shards."""
    node = SamplerWrapper(shard_urls)
    node = ParallelMapper(node, map_fn=_process_tar(fs=fs), num_workers=num_workers, method="thread")
    node = Unbatcher(node)
    node = Batcher(node, batch_size=batch_size, drop_last=drop_last)
    return Loader(node, restart_on_stop_iteration=False)  # TODO should be True or False? False cause we  iterate over entire dataset per epoch and not over fixed number of batches
    return StatefulDataLoader(node, restart_on_stop_iteration=False)  # TODO should be True or False? False cause we  iterate over entire dataset per epoch and not over fixed number of batches

class SequentialSamplerWrapper(SequentialSampler):
    def __init__(self, data_source: Sized, size: int,):
        super().__init__(data_source)
        self.size = size

    def __len__(self):
        # NOTE This is not actually used but is just there for compatibility with the base class and DDP
        # in stable-ssl. Our actual sampler is iterable, not indexable.
        return self.size


class SequentialSamplerAdapter(Loader):
    def __init__(
        self,
        root: BaseNode[T],
        restart_on_stop_iteration: bool = True,
        size: int = 1e6, # NOTE This is not actually used but is just there for compatibility with the stable-ssl base class and DDP
    ):
        """This class just adds a sampler to the loader so that it will be wrapped by stable-ssl's init."""
        super().__init__(root, restart_on_stop_iteration)
        self.sampler = SequentialSampler(range(size))


def init_dataloaders(
    bucket: str,
    prefix: str,
    *,
    endpoint_url: str,
    batch_size: int = 8,
    num_workers: int = 8,
    cache_dir: str = "/tmp/tigris_cache",
    **s3_credentials,
):
    """Create a tensor-centric Loader that streams from Tigris."""
    fs = get_filesystem(endpoint_url, cache_dir=cache_dir, **s3_credentials)
    shard_urls = glob_shards(fs, bucket, prefix)
    loader = make_tensor_video_loader(
        shard_urls,
        fs,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return SequentialSamplerAdapter(loader)

# NOTE Initially I opted to use torchdata because of the lower overhead of batching and the fact that it's more idiomatic for loading from blobstores.
# However it does not play as nice with stable-ssl, because stable-ssl assumes usage of torch.utils.data. 

# NOTE The reason this is very unlikely to work because I have no faith in Stable-SSL's BaseTrainer to handle DDP properly after taking a closer look.
# It checks to see if our dataloader's sampler is an instance of Sequential or Random Sampler, and then uses that to wrap the DataLoader with
# stable-ssl's own DDP Sampler Wrapper. However it seems that the sampler wrapper constructor expects a sampler and instead the loader is passed in.
# Either:
# 1) stable-ssl doesn't do DDP properly yet: in this case, I can either fix stable-ssl myself or replace it by using my own trainer Base Classes.
# 2) stable-ssl DOES do DDP properly, in which case, we might be able to keep wrangling torchdata's approaches to make-do with stable-ssl.
# 3) Shab uses torch.utils.data, in which case, we can just use that, still predicated on #2 and assuming we are fine with the batching overhead of torch.utils.data.
# 4) Remove stable-ssl completely, in which case we need to make DDP work.
# TL;DR this is all kind of a mess for now.