from .build import DATASET_REGISTRY, build_dataset # isort:skip

# from .da import *
# from .da import BCI_IV
# from .da import kaggle_bci
# from .dg import *
# from .ssl import *
from .general_dataset import GENERAL_DATASET
from .general_whole_dataset import GENERAL_WHOLE_DATASET
# from .dataset_adaptation import ADAPTATION_DATASET
# from .increment_adaptation import INCREMENT_ADAPTATION_DATASET
from .increment_adaptation_v1 import INCREMENT_ADAPTATION_DATASET_V1
# from .general_trial_dataset import GENERAL_TRIAL_DATASET
# from .general_dataset_temp import GENERAL_DATASET
from .increment_database_setup import INCREMENT_DATASET_SETUP
from .NeurIPS_BCI import MultiDataset
from .NeurIPS_Sleep import MultiDatasetV1