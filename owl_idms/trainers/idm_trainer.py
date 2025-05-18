import torch
from torch import Tensor
import hydra
import torchvision
import stable_ssl
from stable_ssl.trainer import BaseTrainer
from utils import CosineWDSchedule


class InverseDynamicsTrainer(BaseTrainer):

    required_modules = {
        "vae_encoder": torch.nn.Module,
        "vae_decoder": torch.nn.Module,
        "optical_flow_predictor": torch.nn.Module,
        "depth_map_predictor": torch.nn.Module,
        "action_predictor": torch.nn.Module,
    }

    def __init__(self,
                 data, module, hardware, optim, logger, loss, # base
                 weight_decay: tuple[float, float],
                 **kwargs):

        super().__init__(data, module, hardware, optim, logger, loss, **kwargs)
        self.weight_decay = weight_decay
        _tmp_data = hydra.utils.instantiate(self._data, _convert_="object")
        self.ipe = self._calculate_ipe(_tmp_data)
        self.total_steps = self.ipe * self._optim['epochs']

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

    def format_batch(self, batch: dict) -> dict:
        batch = {
            'image': batch['image'],
            'latent': self.module['vae_encoder'](batch['image']), # FIXME
            'action': batch['action'],
        }
        for item in batch.values():
            if isinstance(item, torch.Tensor):
                item = item.to(self.hardware['device'])

        return batch

    def decompose_action_loss(self, action_loss: list[Tensor]) -> tuple[Tensor, Tensor]:
        pass

    def compute_loss(self):
        batch = self.format_batch(self.batch)
        latent, image, action = batch['latent'], batch['image'], batch['action']
        depth = self.module['depth_map_predictor'](image)
        flow = self.module['optical_flow_predictor'](image)

        action_pred = self.module['action_predictor'](latent, depth, flow)
        action_loss = self.loss['action_predictor'](action_pred, action)
        keypress_loss, mouse_loss = self.decompose_action_loss(action_loss)

        return {
            'action_loss': sum(action_loss),
            'keypress_loss': sum(keypress_loss),
            'mouse_loss': sum(mouse_loss),
        }


    def after_fit_step(self):
        self.wd_scheduler.step()
        self._log({'train/wd': self.wd_scheduler[-1]}, commit=False)

