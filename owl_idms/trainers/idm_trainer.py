import torch
from torch import Tensor
import hydra
from typing_extensions import override
from stable_ssl.base import BaseTrainer

from owl_idms.optim.weight_decay import CosineWDSchedule
from owl_idms.modules.idm_loss import IDMLoss
from owl_idms.types import ActionPrediction, ActionGroundTruth
from owl_idms.modules.vpt import LatentVPTInverseDynamics


class InverseDynamicsTrainer(BaseTrainer):

    required_modules = {
        "vae_encoder": torch.nn.Module,
        "action_predictor": LatentVPTInverseDynamics,
    }

    def __init__(self,
                 data, module, hardware, optim, logger,
                 loss: IDMLoss = IDMLoss(),
                 ddp_debug: bool = False,
                 **kwargs):

        self.weight_decay = optim.pop('weight_decay')
        super().__init__(data, module, hardware, optim, logger, loss, **kwargs)
        self.ddp_debug = ddp_debug
        
        _tmp_data = hydra.utils.instantiate(self._data, _convert_="object")
        self.ipe = self._calculate_ipe(_tmp_data)
        self.total_steps = self.ipe * self._optim['epochs']
        self._optim["scheduler"]["total_steps"] = self.total_steps


    def _calculate_ipe(self, _tmp_data: dict) -> int:
        return sum(len(loader) for k, loader in _tmp_data.items()
                   if isinstance(loader, torch.utils.data.DataLoader) and k == 'train')

    def before_fit(self):
        self.wd_scheduler = CosineWDSchedule(
            self.optim["optimizer"],
            ref_wd=self.weight_decay[0],
            final_wd=self.weight_decay[1],
            T_max=self.total_steps,
        )


    def compute_loss(self):
        video, buttons, mouse = self.batch
        action: ActionGroundTruth = ActionGroundTruth(buttons, mouse)
        latent = self.module['vae_encoder'](video)
        
        action_pred: ActionPrediction = self.module['action_predictor'](latent)
        keypress_loss, mouse_loss = self.loss(action_pred, action)

        return {
            'action_loss': keypress_loss + mouse_loss,
            'keypress_loss': keypress_loss,
            'mouse_loss': mouse_loss,
        }


    def after_fit_step(self):
        self.wd_scheduler.step()
        self._log({'train/wd': self.wd_scheduler[-1]}, commit=False)


    @override
    def _set_device(self, hardware):
        import logging, os
        from stable_ssl.utils import get_gpu_info
        import socket
        import torch.distributed as dist
        
        # Check if CUDA is available, otherwise set to CPU
        use_gloo_backend = False

        if self.ddp_debug:
            hardware["world_size"] = 2

        if not torch.cuda.is_available() or hardware["device"] == "cpu":
            logging.warning("CUDA is not available or device is set to CPU.")
            use_gloo_backend = True
            self._device = "cpu"
        else:
            self._device = hardware["device"]

        if hardware["world_size"] <= 1:
            logging.info(f"Not using DDP, world_size={hardware['world_size']}")
            return

        # DDP setup (world_size > 1)
        logging.info("Setting up Distributed model.")
        
        # For local testing, use environment variables if available
        # Otherwise, set up defaults for local testing
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", str(hardware["world_size"])))
        
        # Set master address/port if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            # Find a free port if not specified
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
                os.environ["MASTER_PORT"] = str(port)
        
        # Log DDP setup
        logging.info(f"\tMASTER_ADDR: {os.environ['MASTER_ADDR']}")
        logging.info(f"\tMASTER_PORT: {os.environ['MASTER_PORT']}")
        logging.info(f"\tlocal rank: {local_rank}")
        logging.info(f"\tglobal rank: {rank}")
        logging.info(f"\tworld size: {world_size}")
        
        # Initialize process group with appropriate backend
        backend = "gloo" if use_gloo_backend else "nccl"
        logging.info(f"Using {backend} backend for DDP.")
        
        try:
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                rank=rank,
                world_size=world_size,
            )
            
            # Set device based on backend
            if backend == "nccl":
                self._device = f"cuda:{local_rank}"
            else:
                self._device = "cpu"
                
            logging.info(f"Process group initialized successfully. Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
        except Exception as e:
            logging.error(f"Error initializing process group: {e}")
            raise
        
        # Log device status
        logging.info("Device status at start of process:")
        if 'cuda' in self._device:
            get_gpu_info()
            torch.cuda.set_device(self._device)
        else:
            logging.info(f"Device is CPU - {self.ddp_debug=}")