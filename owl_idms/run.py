import os
import hydra
from torch.nn import Module
from omegaconf import DictConfig

from owl_idms.trainers.base import HardwareConfig, OptimizationConfig, LoggingConfig


@hydra.main(config_path='../configs', config_name='idm.yaml', version_base=None)
def main(cfg: DictConfig):
    # Extract DDP info from environment (set by torch.distributed.launch)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    global_rank = int(os.environ.get('RANK', '0'))
    
    # -- logging
    logging: LoggingConfig = hydra.utils.instantiate(cfg.logging, _convert_="object", _recursive_=False)
    # -- optim
    optim: OptimizationConfig = hydra.utils.instantiate(cfg.optim, _convert_="object", _recursive_=False)
    # -- hardware
    hardware: HardwareConfig = hydra.utils.instantiate(cfg.hardware, _convert_="object", _recursive_=False)
    hardware['local_rank'] = local_rank
    hardware['global_rank'] = global_rank
    hardware['world_size'] = world_size

    # -- loss
    loss: Module = hydra.utils.instantiate(cfg.loss, _convert_="object", _recursive_=False)
    # -- modules
    modules: dict[str, Module] = hydra.utils.instantiate(cfg.modules, _convert_="object", _recursive_=False)
    # -- dataloaders
    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader, _convert_="object", _recursive_=False)
    val_dataloader = hydra.utils.instantiate(cfg.val_dataloader, _convert_="object", _recursive_=False)

    # -- trainer
    from owl_idms.trainers.idm_trainer_custom import IDMTrainer
    trainer = IDMTrainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        modules=modules,
        loss=loss,
        hardware=hardware,
        optim_config=optim,
        logging_config=logging,
    )
    trainer()


if __name__ == '__main__':
    main()