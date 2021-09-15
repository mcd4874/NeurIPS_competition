
from dassl.engine import TRAINER_REGISTRY,DANN
from dassl.data.datasets import build_dataset
@TRAINER_REGISTRY.register()
class CUSTOM_DANN(DANN):
    def __init__(self, cfg):
        # self.cfg = cfg
        super().__init__(cfg)

    def build_data_loader(self):
        """Create essential data-related attributes."""
        # Load dataset
        dataset = build_dataset(self.cfg)
