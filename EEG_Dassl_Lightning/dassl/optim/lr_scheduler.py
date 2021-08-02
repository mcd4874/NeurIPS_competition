"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import torch
AVAI_SCHEDS = ['single_step', 'multi_step', 'cosine','exponential','cosine_decay']


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    # stepsize = optim_cfg.STEPSIZE
    # gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    lr_scheduler = optim_cfg.SCHEDULER.NAME
    scheduler_params = optim_cfg.SCHEDULER.PARAMS




    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            'Unsupported scheduler: {}. Must be one of {}'.format(
                lr_scheduler, AVAI_SCHEDS
            )
        )

    if lr_scheduler == 'single_step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **scheduler_params
        )

    elif lr_scheduler == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, **scheduler_params
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
    elif lr_scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,**scheduler_params
        )
    elif lr_scheduler == 'cosine_decay':
        from dassl.optim.CosineDecay import CosineDecay
        # max_lr = optim_cfg.COSINDECAY.MAX_LR
        # warmup = optim_cfg.COSINDECAY.WARM_UP
        # warm_drop = optim_cfg.COSINDECAY.WARM_DROP
        # last_epoch = optim_cfg.COSINDECAY.LAST_EPOCH

        scheduler = CosineDecay(
            optimizer,**scheduler_params
        )
    return scheduler
