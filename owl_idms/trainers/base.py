import torch
from pathlib import Path
from typing import TypedDict
from abc import abstractmethod

from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb
import time
import functools
from hydra.core.hydra_config import HydraConfig

from owl_idms.constants import WANDB_ENTITY, WANDB_PROJECT

class HardwareConfig(TypedDict):
    device: torch.device
    world_size: int
    local_rank: int
    global_rank: int

class OptimizationConfig(TypedDict):
    epochs: int
    grad_max_norm: float | None
    batch_size: int
    accumulation_steps: int
    optimizer: Optimizer
    iterations_per_epoch: int

class LoggingConfig(TypedDict):
    save_epoch_frequency: int
    log_step_frequency: int
    log_vis_frequency: int
    eval_every_n_epochs: int
    save_path: Path = Path("./checkpoints")
    load_path: Path | None = None

class BaseTrainer(torch.nn.Module):

    PORT = 29500

    def __init__(self,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 modules: dict[str, Module],
                 loss: Module,
                 hardware_config: HardwareConfig,
                 optim_config: OptimizationConfig,
                 logging_config: LoggingConfig,
                 gpu_transform: Module | None = None,
                 eval_first: bool = False):

        super().__init__()
        self.epochs = optim_config['epochs']
        self.iterations_per_epoch = optim_config['iterations_per_epoch']
        self.eval_first = eval_first
        # -- model
        self._modules = modules
        # -- dataloaders
        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloader: DataLoader   = val_dataloader
        # -- loss
        self.loss: Module = loss
        # -- optimization - all the muon shit
        self.optimizer: Optimizer = optim_config['optimizer']
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        self.grad_max_norm = optim_config['grad_max_norm']
        self.batch_size = train_dataloader.batch_size
        self.accumulation_steps = optim_config['accumulation_steps']
        self.batch_idx      = 0
        self.register_buffer("global_idx", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.register_buffer("current_epoch", torch.zeros(1, dtype=torch.int64), persistent=True)
        # -- hardware
        self.device         = hardware_config['device']
        self.world_size     = hardware_config['world_size']
        self.local_rank     = hardware_config['local_rank']
        self.global_rank    = hardware_config['global_rank']
        # -- logging
        self.save_epoch_frequency = logging_config['save_epoch_frequency']
        self.log_step_frequency   = logging_config['log_step_frequency']
        self.log_vis_frequency    = logging_config['log_vis_frequency'] # Used in subclasses only
        self.save_path            = Path(HydraConfig.get().runtime.output_dir)
        self.eval_every_n_epochs  = logging_config['eval_every_n_epochs']
        self.log_buffer           = {}
        self.load_path            = logging_config['load_path'] and Path(logging_config['load_path'])
        # -- after all attributes are set we can start setting up ddp and devices
        self.setup_hardware()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_modules()
        self.transform: Module = gpu_transform.to(self.device) if gpu_transform is not None else None
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.trainable_parameters()):,}')
        print(f'Number of untrainable parameters: {sum(p.numel() for p in self.parameters() if not p.requires_grad):,}')
        print(f'Number of total parameters: {sum(p.numel() for p in self.parameters()):,}')
        self.global_idx.to(torch.device('cpu'))
        self.current_epoch.to(torch.device('cpu'))

    def setup_modules(self):
        for module in self._modules.values():
            print(f'Setting up module {module.__class__.__name__} on device {self.device}')
            module.to(self.device)

        if self.load_path is not None:
            print(f'Loading checkpoint from {self.load_path}')
            self.load_checkpoint(self.load_path)

    def setup_hardware(self):
        if not torch.cuda.is_available() or self.device == "cpu":
            self.device = "cpu"
            return
        
        if self.world_size > 1:
            dist_url = f'tcp://localhost:{self.PORT}'
            print(f"Initializing DDP training with world_size={self.world_size} and dist_url={dist_url}")
            torch.distributed.init_process_group('nccl', init_method=dist_url, rank=self.global_rank, world_size=self.world_size)

        self.device = torch.device(f'cuda:{self.local_rank}')

    def setup_optimizer(self):
        if isinstance(self.optimizer, functools.partial):
            self.optimizer = self.optimizer(self.trainable_named_parameters(), rank=self.global_rank, world_size=self.world_size)

    def setup_logging(self):
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def trainable_named_parameters(self):
        yield from ((n,p) for n,p in self.named_parameters() if p.requires_grad)
    def trainable_parameters(self):
        yield from (p for n,p in self.named_parameters() if p.requires_grad)

    def __call__(self):
        return self.train_()

    @abstractmethod
    def forward(self):
        pass

    def train_(self):
        self.before_train()
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            if self.eval_first:
                self.eval_epoch()
            for epoch in range(self.epochs):
                print(f'Training epoch {epoch} of {self.epochs}, started at {time.time()}')
                self.current_epoch[0] = torch.tensor(epoch, device=self.device)
                self.before_epoch()
                self.train_epoch()
                if epoch % self.eval_every_n_epochs == 0:
                    self.eval_epoch()
                self.after_epoch()

        self.after_train()

    def train_epoch(self):
        self.train() # set to train mode
        for idx, batch in enumerate(tqdm(self.train_dataloader,
                                        total=self.iterations_per_epoch,
                                        desc=f"Training: {self.current_epoch}")):
            self.batch_idx = idx
            self.before_step(batch)
            self.train_step(batch)
            self.after_step(batch)
            if idx + 1 == self.iterations_per_epoch:
                break

    def train_step(self, batch: dict):
        loss = self.forward_step(batch)
        print(f'Loss: {loss.item():.4f}')
        self.scaler.scale(loss).backward()

        if (self.batch_idx + 1) % self.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            if self.grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), self.grad_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if self.global_idx.item() % self.log_step_frequency == 0:
            self.log(None, flush=True)

    def log(self, batch: dict | None = None, flush: bool = False):
        self.log_buffer.update(batch or {})

        if not flush:
            return
        
        wandb.log(self.log_buffer, step=self.global_idx.item())
        self.log_buffer = {}


    def forward_step(self, batch: dict) -> torch.Tensor:
        pass

    def before_step(self, batch: dict):
        pass
    
    def after_step(self, batch: dict):
        self.global_idx += 1

    def after_epoch(self):
        if self.current_epoch % self.save_epoch_frequency == 0:
            self.save_checkpoint(path=self.save_path / f"epoch_{self.current_epoch.item()}.pt")

    def after_train(self):
        pass

    def before_train(self):
        for module in self._modules.values():
            module.to(self.device)
    
    def before_epoch(self):
        pass
    
    def save_checkpoint(self, path: Path):
        print(f'Saving checkpoint at epoch {self.current_epoch} to {path}')
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: Path):
        state_dict = torch.load(path, map_location=self.device)
        # Patch current_epoch shape if needed
        if 'current_epoch' in state_dict:
            curr = state_dict['current_epoch']
            if curr.dim() == 0:  # scalar, shape []
                state_dict['current_epoch'] = curr.unsqueeze(0)  # shape [1]
        self.load_state_dict(state_dict)

    # ---- evaluation
    def eval_epoch(self):
        self.eval()
        for idx, batch in enumerate(tqdm(self.val_dataloader,
                                        total=self.iterations_per_epoch,
                                        desc=f"Evaluating: {self.current_epoch}")):
            self.batch_idx = idx
            self.before_step(batch)
            self.eval_step(batch)
            self.after_step(batch)
            if idx + 1 == self.iterations_per_epoch:
                break

    def eval_step(self, batch: dict):
        pass

    def after_eval(self):
        pass
    
    