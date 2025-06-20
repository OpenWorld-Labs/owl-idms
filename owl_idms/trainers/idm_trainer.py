"""
Trainer for reconstruction only
"""

import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import math

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..schedulers import get_scheduler_cls
from ..utils import Timer
from ..utils.logging import LogHelper
from .base import BaseTrainer

def rescale_mouse(mouse):
    # mouse is [b,2]
    # Apply symlog to each component
    # symlog(x) = sign(x) * log(1 + |x|)
    return torch.sign(mouse) * torch.log1p(torch.abs(mouse)) # [b,2]

def unscale_mouse(mouse):
    # mouse is [b,2]
    # Apply inverse of symlog to each component
    # symlog^-1(x) = sign(x) * (exp(|x|) - 1)
    return torch.sign(mouse) * torch.expm1(torch.abs(mouse)) # [b,2]

def gaussian_nll_loss(mouse_true, pred_mouse_mu, pred_mouse_logvar):
    # mouse_true: true dx, dy movements [batch, 2]
    # pred_mouse_mu: predicted mean dx, dy [batch, 2]  
    # pred_mouse_logvar: predicted log variance dx, dy [batch, 2]

    pred_mouse_var = torch.exp(pred_mouse_logvar)
    
    nll = 0.5 * (math.log(2 * torch.pi) + 
                 pred_mouse_logvar +
                 torch.square(mouse_true - pred_mouse_mu) / pred_mouse_var)
    
    # Sum across dx,dy dimensions (dim=1) then take mean across batch
    return torch.mean(torch.sum(nll, dim=1))

class IDMTrainer(BaseTrainer):
    """
    Trainer for IDM

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg)

        if self.rank == 0:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {param_count:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
            
        super().save(save_dict)

    def load(self):
        if self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
        else:
            return

        self.model.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model and ema
        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank,world_size=self.world_size,**self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        # Timer reset
        timer = Timer()
        timer.reset()

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size)
        logger = LogHelper()

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for videos, mouse_input, button_input in loader:
                # Move data to device
                videos = videos.cuda().bfloat16()
                button_target = button_input.cuda()[:,16]
                mouse_target = mouse_input.float().cuda()[:,16]
                mouse_target = rescale_mouse(mouse_target)

                with ctx:
                    # Forward pass
                    (m_mu_preds, m_logvar_preds), button_logits = self.model(videos)

                    # Gaussian NLL loss
                    mouse_loss = gaussian_nll_loss(mouse_target, m_mu_preds, m_logvar_preds) / accum_steps
                    button_loss = F.binary_cross_entropy_with_logits(button_logits, button_target.float()) / accum_steps

                    total_loss = 0.0*mouse_loss + button_loss

                    logger.log('mouse_loss', mouse_loss)
                    logger.log('button_loss', button_loss)
                    logger.log('total_loss', total_loss)

                # Calculate sensitivity (true positive rate) for buttons
                with torch.no_grad():
                    button_preds = (torch.sigmoid(button_logits) > 0.5).float()
                    true_positives = (button_preds * button_target).sum()
                    total_positives = button_target.sum()
                    sensitivity = true_positives / (total_positives + 1e-8)  # Avoid div by 0
                    logger.log('button_sensitivity', sensitivity/accum_steps)

                self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Get reduced metrics and log to wandb
                    metrics = logger.pop()
                    metrics['time_per_step'] = timer.hit()
                    wandb.log(metrics, step=self.total_step_counter)
                    timer.reset()

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()