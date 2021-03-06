from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1


# Directory to save the output files
_C.OUTPUT_DIR = './output'
_C.output_dir = ''
_C.history_dir = ''
# Path to a directory where the files were saved
_C.RESUME = ''
# Set seed to negative value to random everything
# Set seed to positive value to use a fixed seed
# _C.SEED = -1
_C.USE_CUDA = True
# Print detailed information (e.g. what trainer,
# dataset, backbone, etc.)
_C.VERBOSE = True

####################
#Display information
##################
_C.DISPLAY_INFO = CN()
_C.DISPLAY_INFO.DATASET = True
_C.DISPLAY_INFO.TRAINER = True
_C.DISPLAY_INFO.DataManager = True
#no need to create writer for tensorboard visualization
_C.DISPLAY_INFO.writer = False
###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
# For available choices please refer to transforms.py
# available options = ['z_transform']
_C.INPUT.TRANSFORMS = []
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = True



_C.DATAMANAGER = CN()
_C.DATAMANAGER.DATASET = CN()

# Directory where datasets are stored
_C.DATAMANAGER.DATASET.ROOT = ''
_C.DATAMANAGER.DATASET.DIR = ''
_C.DATAMANAGER.DATASET.FILENAME = ''
_C.DATAMANAGER.DATASET.NAME = ''
_C.DATAMANAGER.DATASET.EA = False

#split the dataset into train/valid and test fold setup
_C.DATAMANAGER.DATASET.SETUP = CN()

#split whole dataset into whole_train and test data
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD = CN()
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.N_TEST_FOLDS = 1
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_FOLD_PREFIX = 'test_fold'
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.START_TEST_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.END_TEST_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.CURRENT_TEST_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_SUBJECT_INDEX = []
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.NUM_TEST_SUBJECTS = -1


_C.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD = CN()
_C.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS = 5
_C.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_FOLD_PREFIX = 'shuffle_fold'
_C.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.START_SHUFFLE_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.END_SHUFFLE_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.CURRENT_SHUFFLE_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SET_FIX_SEED = False
_C.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_SEED = [0]

#split whole_train data into train and valid data
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD = CN()
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.N_VALID_FOLDS = 5
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.VALID_FOLD_PREFIX = 'valid_fold'
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.START_VALID_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.END_VALID_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.CURRENT_VALID_FOLD=1
#choose to use session between subject as valid data to pick best model
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.WITHIN_SUBJECTS = True
#choose to use cross subject as a valid data to pick best model
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.CROSS_SUBJECTS = False

#calculate the label weight for each train subject in the target train dataset
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.DOMAIN_CLASS_WEIGHT = True
#calculate the label weight for the target train dataset of all subject in train data
_C.DATAMANAGER.DATASET.SETUP.VALID_FOLD.TOTAL_CLASS_WEIGHT = True

_C.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD = CN()
_C.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_NUM_TRAIN_SUGJECT = -1
_C.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.INCREMENT_TRAIN_SUGJECT = 0
_C.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.N_INCREMENT_FOLDS = 5
_C.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.INCREMENT_FOLD_PREFIX = 'increment_fold'
_C.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_INCREMENT_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.END_INCREMENT_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.CURRENT_INCREMENT_FOLD=1

# _C.DATAMANAGER.DATASET.SETUP.HISTORY_FOLDER = 'history_folder'
# _C.DATAMANAGER.DATASET.SETUP.RESULT_FOLDER = 'result_folder'
_C.DATAMANAGER.RESULT_FOLDER = 'result_folder'
_C.DATAMANAGER.MANAGER_TYPE = 'single_dataset' #either single dataset or multi_dataset

_C.DATAMANAGER.DATASET.AUGMENTATION = CN()

_C.DATAMANAGER.DATASET.AUGMENTATION.NAME = ""
_C.DATAMANAGER.DATASET.AUGMENTATION.PARAMS = CN()
_C.DATAMANAGER.DATASET.AUGMENTATION.PARAMS.N_SEGMENT=4
_C.DATAMANAGER.DATASET.AUGMENTATION.PARAMS.MAX_TRIAL_MUL = 3
_C.DATAMANAGER.DATASET.AUGMENTATION.PARAMS.MAX_FIX_TRIAL = -1
_C.DATAMANAGER.DATASET.AUGMENTATION.PARAMS.DATASET_NAME = "BCI_IV"


###########################
# Dataloader
###########################
_C.DATAMANAGER.DATALOADER = CN()
_C.DATAMANAGER.DATALOADER.NUM_WORKERS = 0
# Setting for train_x data-loader
_C.DATAMANAGER.DATALOADER.TRAIN_X = CN()
_C.DATAMANAGER.DATALOADER.TRAIN_X.SAMPLER = 'RandomSampler'
_C.DATAMANAGER.DATALOADER.TRAIN_X.BATCH_SIZE = 64
# Parameter for RandomDomainSampler
# 0 or -1 means sampling from all domains
_C.DATAMANAGER.DATALOADER.TRAIN_X.N_DOMAIN = 0

# Setting for train_u data-loader
_C.DATAMANAGER.DATALOADER.TRAIN_U = CN()
# Set to false if you want to have unique
# data loader params for train_u
_C.DATAMANAGER.DATALOADER.TRAIN_U.SAME_AS_X = True
_C.DATAMANAGER.DATALOADER.TRAIN_U.SAMPLER = 'RandomSampler'
_C.DATAMANAGER.DATALOADER.TRAIN_U.BATCH_SIZE = 64
_C.DATAMANAGER.DATALOADER.TRAIN_U.N_DOMAIN = 0

# Setting for test data-loader
_C.DATAMANAGER.DATALOADER.TEST = CN()
_C.DATAMANAGER.DATALOADER.TEST.SAMPLER = 'SequentialSampler'
_C.DATAMANAGER.DATALOADER.TEST.BATCH_SIZE = 32

# Setting for valid data-loader
_C.DATAMANAGER.DATALOADER.VALID = CN()
_C.DATAMANAGER.DATALOADER.VALID.SAMPLER = 'SequentialSampler'
_C.DATAMANAGER.DATALOADER.VALID.BATCH_SIZE = 32
_C.DATAMANAGER.DATALOADER.VALID.N_DOMAIN = 0