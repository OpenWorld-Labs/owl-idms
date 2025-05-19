from __future__ import annotations

import fsspec
from functools import cache

DEFAULT_CACHE_DIR = "/tmp/tigris_cache"


@cache
def get_filesystem(
    endpoint_url: str,
    *,
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_pool_connections: int = 64,
    **s3_kwargs,
) -> fsspec.AbstractFileSystem:
    """Return an fsspec FS ready for streaming shards from Tigris.

    Parameters
    ----------
    endpoint_url:
        The ``https://...`` endpoint for Tigris deployment or Fly.io region.
    cache:
        When *True* (default) wrap the S3 filesystem in a local *filecache* so
        that each shard is only pulled over the network once per machine.
    cache_dir:
        Where to store cached shards. Defaults to `/tmp/tigris_cache`.
    max_pool_connections:
        Passed through to botocore. 64 is usually enough to saturate a 10 GbE
        link when using 8MiB multipart/Range GETs.
    **s3_kwargs:
        Extra keyword args forwarded to the boto3 client (e.g. credentials).
    """
    # Raw S3 filesystem
    s3_fs = fsspec.filesystem(
        "s3",
        client_kwargs={"endpoint_url": endpoint_url, **s3_kwargs},
        config_kwargs={"max_pool_connections": max_pool_connections},
    )

    if not cache:
        return s3_fs

    # Transparent ondisk cache, replace with "simplecache" for writethrough
    cache_fs = fsspec.filesystem(
        "filecache",
        target_protocol=s3_fs,
        cache_storage=cache_dir,
        same_names=True,
    )
    return cache_fs  # type: ignore[return-value]


def glob_shards(
    fs: fsspec.AbstractFileSystem,
    bucket: str,
    prefix: str = "",
    pattern: str = "*.tar",
) -> list[str]:
    return fs.glob(f"s3://{bucket}/{prefix}{pattern}")
