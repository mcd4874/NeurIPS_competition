from dassl.engine import TRAINER_REGISTRY
from dassl.engine.trainer import TrainerMultiAdaptation
from dassl.data import DataManager
from dassl.utils import MetricMeter
from torch.utils.data import Dataset as TorchDataset
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.engine.trainer_tmp import SimpleNet
import numpy as np
from dassl.modeling import build_layer





@TRAINER_REGISTRY.register()
class MultiDatasetAdaptation(TrainerMultiAdaptation):
    """
    Apply EEGNet for multi-source dataset and 1 target dataset
    Each dataset has its own 1st temporal filter layer and spatial filter layer
    The 2nd temporal filter layer is shared among all dataset
    Each dataset has its own classifier
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)




