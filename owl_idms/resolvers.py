from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

def idm_pretrain_id(cfg: DictConfig) -> str:
    hydra_cfg = HydraConfig.get()
    path =  [hydra_cfg.runtime.cwd]
    path += [(cfg.vae_id, cfg.depth_model_id, cfg.optical_flow_model_id)]
    path += [cfg.action_model_id]

    return '/'.join(
        '-'.join(item) if isinstance(item, tuple) else item
        for item in path
    )


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("idm_pretrain_id", idm_pretrain_id, replace=True)