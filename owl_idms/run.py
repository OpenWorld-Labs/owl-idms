import os
import hydra
from omegaconf import DictConfig
from owl_idms import resolvers  # noqa

@hydra.main(config_path='../configs', config_name='idm.yaml', version_base=None)
def main(cfg: DictConfig):
    # Extract DDP info from environment (set by torch.distributed.launch)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    # Only modify the config if we're in a distributed setting
    if world_size > 1:
        print(f"Initializing DDP training with world_size={world_size}")
        # Update the world_size in config
        cfg.trainer.hardware.world_size = world_size
        # Enable DDP debugging
        cfg.trainer.ddp_debug = True
    
# Instantiate and run the trainer
    trainer = hydra.utils.instantiate(cfg.trainer, _convert_="object", _recursive_=False)
    trainer()


if __name__ == '__main__':
    main()