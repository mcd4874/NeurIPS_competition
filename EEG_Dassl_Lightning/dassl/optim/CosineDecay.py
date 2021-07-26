import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import weakref
import torch.optim.lr_scheduler

class CosineDecay(_LRScheduler):
    def __init__(self, optimizer, max_lr, warmup, total_epochs=100, warm_drop=1.0, last_epoch=-1):
        self.real_eps = total_epochs - warmup
        self.max_lr = max_lr
        self.warmup = warmup
        self.warm_drop = warm_drop
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        progress = self.last_epoch - self.warmup + 1
        if progress <= 0:
            return [self.max_lr * (self.last_epoch + 1) / self.warmup for _ in self.base_lrs]
        else:
            return [self.warm_drop * self.max_lr * 0.5 * (1 + np.cos(progress * np.pi / self.real_eps))
                    for _ in self.base_lrs]


