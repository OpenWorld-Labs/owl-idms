from .dummy_dataset import dummy_dataloader
from .cod_datasets import get_loader as get_cod_loader

__all__ = ["dummy_dataloader", "get_cod_loader"]