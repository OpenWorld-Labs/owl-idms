import torch.optim.lr_scheduler as lr_scheduler

class LinearWarmupScheduler(lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps=1000, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lrs = None
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_steps:
            return self.base_lrs

        # Linear interpolation from min_lr to base_lr
        scale = self.last_epoch / self.warmup_steps
        return [self.min_lr + (base_lr - self.min_lr) * scale
                for base_lr in self.base_lrs]

def get_scheduler_cls(scheduler_id):
    if scheduler_id == "LinearWarmup":
        return LinearWarmupScheduler
    raise ValueError(f"Unknown scheduler {scheduler_id}")