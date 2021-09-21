from dassl.engine.base.base_model import BaseModel
from dassl.engine import TRAINER_REGISTRY,TrainerBase



@TRAINER_REGISTRY.register()
class DeepConvNet(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
