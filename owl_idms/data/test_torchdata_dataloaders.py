import itertools
from typing import Iterator, Generator

from torch import Tensor

from torchdata.nodes import (
    IterableWrapper,
    ParallelMapper,
    Batcher,
    Loader,
)
from toolz import curry

RANK = 0
def get_rank():
    return RANK

def get_world_size():
    return 4

@curry
def _process_tar(url: str, fs) -> Generator[tuple[str, str, str], None, None]:
    """Download + parse a shard into ``[(video, button, mouse), ...]`` pairs."""
    for i in range(10):
        yield (f'url_{url}', f'fs_{fs}', f'i_{i}')

def make_tensor_video_loader(
    shard_urls: list[str],
    fs,
    *,
    batch_size: int = 8,
    num_workers: int = 8,
    drop_last: bool = False,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Return a Loader streaming (video, button, mouse) tensors from shards."""
    # rank=1 world_size=4, yields 1, 5, 9, 13
    urls = itertools.islice(shard_urls, get_rank(), None, get_world_size())
    node = IterableWrapper(urls)
    node = ParallelMapper(node, map_fn=_process_tar(fs=fs),# yields (vid, key, mouse)
                          num_workers=num_workers, method="process")
    node = Batcher(node, batch_size=batch_size, drop_last=drop_last)
    node = Loader(node, restart_on_stop_iteration=True)
    return node

if __name__ == "__main__":
    shard_urls = [f"url_{i}" for i in range((4*8)+3)]
    fs = "myfilesys"
    loader = make_tensor_video_loader(shard_urls, fs, batch_size=4, num_workers=2)
    for rank in range(4):
        RANK = rank
        print(f'rank {rank}')
        for batch in loader:
            print(batch)
