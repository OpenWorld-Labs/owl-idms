import torch
from pathlib import Path
from typing import TypedDict

from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import wandb

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
    weight_decay: tuple[float, float]

class LoggingConfig(TypedDict):
    save_epoch_frequency: int
    log_step_frequency: int
    save_path: Path = Path("./checkpoints")

class BaseTrainer(torch.nn.Module):
    def __init__(self,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 modules: dict[str, Module],
                 loss: Module,
                 hardware: HardwareConfig,
                 optim_config: OptimizationConfig,
                 logging_config: LoggingConfig):

        super().__init__()
        self.epochs = optim_config['epochs']
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
        self.batch_size = optim_config['batch_size']
        self.accumulation_steps = optim_config['accumulation_steps']
        self.batch_idx      = 0
        self.global_idx     = self.register_buffer("global_idx", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.current_epoch  = self.register_buffer("current_epoch", torch.zeros(1, dtype=torch.int64), persistent=True)
        self.weight_decay   = optim_config['weight_decay']
        # -- hardware
        self.device         = hardware['device']
        self.world_size     = hardware['world_size']
        self.local_rank     = hardware['local_rank']
        self.global_rank    = hardware['global_rank']
        # -- logging
        self.save_epoch_frequency = logging_config['save_epoch_frequency']
        self.log_step_frequency   = logging_config['log_step_frequency']
        self.save_path            = logging_config['save_path']
        self.log_buffer           = {}

    def train(self):
        self.before_train()
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                self.before_epoch()
                self.train_epoch()
                self.after_epoch()

        self.after_train()

    def train_epoch(self):
        for idx, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader), desc=f"Training: {self.current_epoch}")):
            self.batch_idx = idx
            self.before_step(batch)
            self.train_step(batch)
            self.after_step(batch)

    def train_step(self, batch: dict):
        loss = self.forward_step(batch)
        self.scaler.scale(loss).backward()

        if (self.batch_idx + 1) % self.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            if self.grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if self.global_idx % self.log_step_frequency == 0:
            self.log(batch, flush=True)

    def log(self, batch: dict | None = None, flush: bool = False):
        if not flush:
            self.log_buffer.update(batch or {})
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
            self.save_checkpoint(path=self.save_path / f"epoch_{self.current_epoch}.pt")

    def after_train(self):
        pass

    def before_train(self):
        pass
    
    def before_epoch(self):
        pass
    
    def save_checkpoint(self, path: Path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: Path):
        self.load_state_dict(torch.load(path))
