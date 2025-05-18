import hydra
from omegaconf import DictConfig


@hydra.main(config_path='configs', config_name='idm.yaml')
def main(cfg: DictConfig):
    pass


if __name__ == '__main__':
    main()
