import os
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from owl_idms.constants import CHECKPOINT_DIR

def _vae_id(cfg: DictConfig) -> str:
    return cfg.module.vae_encoder['_target_'].split('.')[-1]

def _idm_id(cfg: DictConfig) -> str:
    return cfg.module.action_predictor['_target_'].split('.')[-1]

def idm_pretrain_id(cfg: DictConfig) -> str:
    hydra_cfg = HydraConfig.get()
    path =  [hydra_cfg.runtime.cwd]
    path += [CHECKPOINT_DIR]
    path += [_vae_id(cfg)+'-'+_idm_id(cfg)]

    return os.path.join(*path)


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("idm_pretrain_id", idm_pretrain_id, replace=True)