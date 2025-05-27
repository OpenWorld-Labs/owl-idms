import torch
from torch.nn import Module
from owl_idms.modules.idm_loss import IDMLoss
from owl_idms._types import ActionGroundTruth, ActionPrediction
from owl_idms.optim.weight_decay import CosineWDSchedule
from owl_idms.trainers.base import BaseTrainer, HardwareConfig, OptimizationConfig, LoggingConfig
from torch.utils.data import DataLoader
from typing import Callable
from owl_idms.data.cod_datasets import DEFAULT_TRANSFORM_GPU
import pathlib
import datetime
from stable_ssl import BaseTrainer as BT
import wandb
from owl_idms.utils import draw_frame_groundtruth, draw_frame_predicted
import numpy as np
import cv2

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
                 gpu_transform: Callable = None):
        super().__init__(train_dataloader, val_dataloader, modules, loss, hardware, optim_config, logging_config, gpu_transform)

    def _compute_loss(self, video, buttons, mouse):
        pred: ActionPrediction = self._modules['action_predictor'](video)
        gt = ActionGroundTruth(buttons=buttons, mouse=mouse)        
        total_loss, metrics = self.loss(pred, gt)
        return {
            "loss": total_loss,
            "metrics": metrics,
            "data": {
                "pred": pred,
                "gt": gt,
            },
        }


    def forward_step(self, batch):
        video, mouse, buttons = batch
        device = self.device

        video   = video.to(device, dtype=torch.float32, non_blocking=True)
        mouse   = mouse.to(device, non_blocking=True)
        buttons = buttons.to(device, non_blocking=True)

        if self.transform is not None:
            video = self.transform(video)
            video = (video - video.min()) / (video.max() - video.min())

        video = video.to(dtype=torch.bfloat16)
        data = self._compute_loss(video, buttons, mouse)
        
        if self.global_idx.item() % self.log_step_frequency == 0:
            self.log(data["metrics"], flush=False)
            self.log({'train/loss': data["loss"]}, flush=True)

        if self.global_idx.item() % self.log_vis_frequency == 0:
            gt_mouse, gt_buttons = data["data"]["gt"].mouse, data["data"]["gt"].buttons
            pred_mouse, pred_buttons = data["data"]["pred"].mouse, data["data"]["pred"].buttons
            vis_vid, vis_gt, vis_pred = self.visualize_predictions(video[0],
                                                          gt_mouse[0], gt_buttons[0], 
                                                          pred_mouse[0], pred_buttons[0])
            self.log({ # NOTE take first video in batch
                "vis/predictions": vis_pred,
                "vis/groundtruth": vis_gt,
                "vis/raw": vis_vid,
            }, flush=True)
        return data["loss"]

    def after_epoch(self):
        super().after_epoch()

    def visualize_predictions(
        self,
        video: torch.Tensor,                     # [T, 3, H, W]  float16/32  (0-1 or 0-255)
        gt_mouse: torch.Tensor,
        gt_buttons: torch.Tensor,
        pred_mouse: torch.Tensor,
        pred_buttons: torch.Tensor,
    ) -> tuple[wandb.Video, wandb.Video]:
        """
        Render per-frame overlays for both GT and prediction.

        Returns
        -------
        (gt_video, pred_video)  – two independent `wandb.Video` objects.
        """
        from owl_idms.utils import draw_frame_groundtruth, draw_frame_predicted_new
        import numpy as np
        import cv2
        import torch

        # ------------------------------------------------------------------ move to CPU
        video = video.float().cpu()                      # [T, 3, H, W]
        gt_mouse    = gt_mouse.cpu()    # [T, 2]
        gt_buttons  = gt_buttons.cpu()  # [T, N_keys]

        pred_mouse  = pred_mouse.detach().cpu()       # [T, 2]
        pred_buttons   = pred_buttons.detach().float().cpu()        # [T, N_keys]

        # ------------------------------------------------------------------ iterate frames
        raw_frames, gt_frames, pred_frames = [], [], []
        T = video.shape[0]
        for t in range(T):
            # 1. tensor ➜ uint8 BGR H×W×3 (OpenCV wants BGR)
            frame = video[t]
            if frame.max() <= 1.0:                       # 0-1 range → 0-255
                frame = frame * 255.0
            frame_uint8 = (
                frame.clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)                   # HWC RGB
                    .numpy()
            )
            frame_bgr = frame_uint8[:, :, ::-1].copy()    # RGB → BGR

            # 2. overlays
            gt_frame = draw_frame_groundtruth(
                frame_bgr,
                gt_mouse=gt_mouse[t].tolist(),
                gt_buttons=gt_buttons[t].bool().tolist(),
            )
            pred_frame = draw_frame_predicted_new(
                frame_bgr,
                pred_mouse=pred_mouse[t].tolist(),
                pred_buttons=pred_buttons[t],
            )

            # 3. back to RGB for WandB
            gt_frames.append(cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB))
            pred_frames.append(cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB))
            raw_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        # ------------------------------------------------------------------ stack ➜ wandb.Video
        raw_frames_np = np.ascontiguousarray(np.stack(raw_frames, axis=0)).astype(np.uint8).transpose(0, 3, 1, 2)
        gt_frames_np = np.ascontiguousarray(np.stack(gt_frames, axis=0)).astype(np.uint8).transpose(0, 3, 1, 2)
        pred_frames_np = np.ascontiguousarray(np.stack(pred_frames, axis=0)).astype(np.uint8).transpose(0, 3, 1, 2)

        raw_video  = wandb.Video(raw_frames_np, fps=30, format="mp4")
        gt_video   = wandb.Video(gt_frames_np,   fps=30, format="mp4")
        pred_video = wandb.Video(pred_frames_np, fps=30, format="mp4")
        return raw_video, gt_video, pred_video




    def eval_step(self, batch: dict):
        with torch.no_grad():
            loss = self.forward_step(batch)
            if self.global_idx.item() % self.log_step_frequency == 0:
                self.log({'val/loss': loss.item()}, flush=True)

