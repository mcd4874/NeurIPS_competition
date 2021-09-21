from dassl.engine import TRAINER_REGISTRY,TrainerBase
from dassl.data import DataManager
from dassl.utils import MetricMeter
from torch.utils.data import Dataset as TorchDataset
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.engine.trainer import SimpleNet
import numpy as np


@TRAINER_REGISTRY.register()
class BaseModel(TrainerBase):
    """
    Base model that use 1 classifier +backbone.
    Default choice is vanilla EEGNet
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)

    def build_model(self):
        super(BaseModel, self).build_model()
        freeze_common_feature = self.cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.FREEZE if self.cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.FREEZE else False
        if freeze_common_feature:
            for parameter in self.model.backbone.parameters():
                parameter.requires_grad = False
            print("freeze base model backbone --- ")



