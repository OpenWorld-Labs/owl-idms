import os
from dataclasses import dataclass

import yaml
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("env", lambda k: os.environ.get(k))

@dataclass
class ResNetConfig:
    ch_0 : int = 256
    ch_max : int = 2048

    blocks_per_stage : list = None

@dataclass
class TransformerConfig:
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 384

@dataclass
class ControlPredConfig:
    model_id : str = "resnet3d"

    resnet_config: ResNetConfig = None
    transformer_config: TransformerConfig = None

    n_buttons: int = 8 # count LMB and RMB
    n_mouse_axes: int = 2

@dataclass
class TrainingConfig:
    trainer_id : str = None
    data_id : str = None

    target_batch_size : int = 128
    batch_size : int = 2

    epochs : int = 200

    opt : str = "AdamW"
    opt_kwargs : dict = None

    loss_weights : dict = None

    scheduler : str = None
    scheduler_kwargs : dict = None

    checkpoint_dir : str = "checkpoints/v0" # Where checkpoints saved
    resume_ckpt : str = None

    sample_interval : int = 1000
    save_interval : int = 1000

@dataclass
class WANDBConfig:
    name : str = None
    project : str = None
    run_name : str = None

@dataclass
class Config:
    model: ControlPredConfig
    train: TrainingConfig
    wandb: WANDBConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)

        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))