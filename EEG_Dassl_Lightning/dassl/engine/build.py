from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry('TRAINER')


def build_trainer(cfg, **kwargs):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.LIGHTNING_MODEL.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print('Loading trainer: {}'.format(cfg.LIGHTNING_MODEL.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.LIGHTNING_MODEL.TRAINER.NAME)(cfg, **kwargs)
