"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import warnings
import torch
import torch.nn as nn

from .radam import RAdam
from torch_optimizer.radam import RAdam
from .eadam import EAdam

AVAI_OPTIMS = ['adam', 'amsgrad', 'sgd', 'rmsprop', 'radam','adamW','EAdam']

def build_optimizer(params, optim_cfg):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module): model.
        optim_cfg (CfgNode): optimization config.
    """
    optim = optim_cfg.OPTIMIZER.NAME
    optim_params = optim_cfg.OPTIMIZER.PARAMS

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            'Unsupported optim: {}. Must be one of {}'.format(
                optim, AVAI_OPTIMS
            )
        )

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            params,**optim_params

        )
    elif  optim == 'adamW':
        optimizer = torch.optim.AdamW(
            params,
            **optim_params
        )
    elif optim == 'EAdam':
        optimizer = EAdam(
            params,
            **optim_params
        )
    # elif optim == 'amsgrad':
    #     optimizer = torch.optim.Adam(
    #         params,
    #         lr=lr,
    #         weight_decay=weight_decay,
    #         betas=(adam_beta1, adam_beta2),
    #         amsgrad=True,
    #     )
    #
    # elif optim == 'sgd':
    #     optimizer = torch.optim.SGD(
    #         params,
    #         lr=lr,
    #         momentum=momentum,
    #         weight_decay=weight_decay,
    #         dampening=sgd_dampening,
    #         nesterov=sgd_nesterov,
    #     )
    #
    # elif optim == 'rmsprop':
    #     optimizer = torch.optim.RMSprop(
    #         params,
    #         lr=lr,
    #         momentum=momentum,
    #         weight_decay=weight_decay,
    #         alpha=rmsprop_alpha,
    #     )
    #
    #
    # elif optim == 'radam':
    #     optimizer = RAdam(
    #         params,
    #         lr=lr,
    #         weight_decay=weight_decay
    #         # betas=(adam_beta1, adam_beta2)
    #     )
    return optimizer
