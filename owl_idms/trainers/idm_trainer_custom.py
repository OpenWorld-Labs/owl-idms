import torch
from torch.nn import Module
from owl_idms.types import ActionGroundTruth, ActionPrediction
from owl_idms.optim.weight_decay import CosineWDSchedule
from owl_idms.trainers.base import BaseTrainer, HardwareConfig, OptimizationConfig, LoggingConfig
from torch.utils.data import DataLoader


class IDMTrainer(BaseTrainer):
    def __init__(self,  # TPDP 05/25/2025: Fix configs.yaml to use this trainer
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 modules: dict[str, Module],
                 loss: Module,
                 hardware: HardwareConfig,
                 optim_config: OptimizationConfig,
                 logging_config: LoggingConfig):
        super().__init__(train_dataloader, val_dataloader, modules, loss, hardware, optim_config, logging_config)
        self.ipe = 500_000  # NOTE Start off with an estimate, adjust this after first epoch

    def _compute_loss(self, video, buttons, mouse):
        latent = self._modules['vae_encoder'](video)
        pred: ActionPrediction = self._modules['action_predictor'](latent)
        gt = ActionGroundTruth(keys=buttons, mouse=mouse)        
        kp_loss, mouse_loss = self.loss(pred, gt)
        return {
            "loss": kp_loss + mouse_loss,
            "kp":   kp_loss,
            "mouse": mouse_loss,
        }

    def _create_wd_scheduler(self):
        return CosineWDSchedule(
            self.optimizer,
            ref_wd=self.weight_decay[0],
            final_wd=self.weight_decay[1],
            T_max=self.ipe * self.epochs,
        )

    def before_fit(self):
        self.wd_scheduler = self._create_wd_scheduler()

    def forward_step(self, batch: dict) -> torch.Tensor:
        video, buttons, mouse = batch
        loss_dict = self._compute_loss(video, buttons, mouse)
        self.log(loss_dict, flush=False)
        return loss_dict["loss"]

    def after_epoch(self):
        if self.current_epoch == 0:
            self.ipe = self.global_idx.item()
            self.wd_scheduler = self._create_wd_scheduler()
            for _ in range(self.global_idx.item()):
                self.wd_scheduler.step()

        super().after_epoch()
