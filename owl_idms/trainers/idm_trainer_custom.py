import torch
from torch.nn import Module
from owl_idms.modules.idm_loss import IDMLoss
from owl_idms._types import ActionGroundTruth, ActionPrediction
from owl_idms.optim.weight_decay import CosineWDSchedule
from owl_idms.trainers.base import BaseTrainer, HardwareConfig, OptimizationConfig, LoggingConfig
from torch.utils.data import DataLoader
from typing import Callable
from owl_idms.data.cod_datasets import DEFAULT_TRANSFORM
import pathlib
import datetime

class IDMTrainer(BaseTrainer):
    _do_profile = False

    def __init__(self,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 modules: dict[str, Module],    
                 loss: IDMLoss,
                 hardware: HardwareConfig,
                 optim_config: OptimizationConfig,
                 logging_config: LoggingConfig,
                 transform: Callable = DEFAULT_TRANSFORM):
        super().__init__(train_dataloader, val_dataloader, modules, loss, hardware, optim_config, logging_config, transform)

    def _compute_loss(self, video, buttons, mouse):
        videos_as_frames = video.view(-1, *video.shape[2:])
        latent = self._modules['vae_encoder'](videos_as_frames)
        # TODO fix this ugly shit
        latent_as_videos = latent.view(video.shape[0], video.shape[1], latent.shape[1], latent.shape[2], latent.shape[3])
        flattened_latent = latent_as_videos.view(latent_as_videos.shape[0], latent_as_videos.shape[1], latent_as_videos.shape[2], -1)
        pred: ActionPrediction = self._modules['action_predictor'](flattened_latent)
        gt = ActionGroundTruth(buttons=buttons, mouse=mouse)        
        kp_loss, mouse_loss = self.loss(pred, gt)
        return {
            "loss": kp_loss + mouse_loss,
            "kp":   kp_loss,
            "mouse": mouse_loss,
        }


    def forward_step(self, batch):
        self._ensure_profiler()

        video, mouse, buttons = batch
        device = self.device

        with torch.autograd.profiler.record_function("to_device"):
            video   = video.to(device, dtype=torch.float32, non_blocking=True)
            mouse   = mouse.to(device, non_blocking=True)
            buttons = buttons.to(device, non_blocking=True)

        if self.transform is not None:
            with torch.autograd.profiler.record_function("augment"):
                video = self.transform(video)         # GPU augments

        with torch.autograd.profiler.record_function("cast_bf16"):
            video = video.to(dtype=torch.bfloat16)

        with torch.autograd.profiler.record_function("fwd_and_loss"):
            loss_dict = self._compute_loss(video, buttons, mouse)

        self.log(loss_dict, flush=False)

        # 3.  Advance the profiler
        if self._prof is not None:
            self._prof.step()

        return loss_dict["loss"]

    def after_epoch(self):
        super().after_epoch()

    def _ensure_profiler(self):
        if hasattr(self, "_prof"):
            return

        if not self._do_profile:
            self._prof = None
            return

        logdir = (
            pathlib.Path(self.save_path)
            / "tb_profiler"
            / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        logdir.mkdir(parents=True, exist_ok=True)

        self._prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,     # skip the very first it. (dataloader warm-up)
                warmup=1,   # profile but don't record
                active=3,   # record 3 iterations
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
            record_shapes=True,
            with_stack=False,          # True = bigger trace, includes Python stack
            profile_memory=True,
        )
        print(f"[Profiler] will write chrome traces to {logdir}")
