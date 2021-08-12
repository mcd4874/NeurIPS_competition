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
# Default mean and std come from ImageNet

###########################
# Dataset
###########################
# _C.DATASET = CN()
# Directory where datasets are stored
# _C.DATASET.ROOT = ''
# _C.DATASET.DIR = ''
# _C.DATASET.FILENAME = ''
# _C.DATASET.NAME = ''
# List of names of source domains
# _C.DATASET.SOURCE_DOMAINS = ()
# List of names of target domains
# _C.DATASET.TARGET_DOMAINS = ()
# _C.DATASET.TEST_K_FOLDS = False
# _C.DATASET.K_FOLD_TEST = 1
# _C.DATASET.TEST_FOLD_PREFIX = 'test_fold_'
# _C.DATASET.VALID_FOLD_TEST = 1
# _C.DATASET.TEST_RANDOM_SEEDS = []
# _C.DATASET.TEST_NUM_SUBJECTS = -1


# _C.DATASET.K_FOLD = 5
# _C.DATASET.VALID_FOLD = 1
# _C.DATASET.DOMAIN_CLASS_WEIGHT = True
# _C.DATASET.TOTAL_CLASS_WEIGHT = True
# _C.DATASET.WITHIN_SUBJECTS = True
# _C.DATASET.CROSS_SUBJECTS = False
# _C.DATASET.TRAIN_K_FOLDS = True
# _C.DATASET.EA = False
# _C.DATASET.NUM_TRAIN_VALID_SUBJECTS = -1
# _C.DATASET.EXTRA_CONFIG = CN()
# _C.DATASET.EXTRA_CONFIG.SHUFFLE_SEED = 10
# _C.DATASET.EXTRA_CONFIG.SET_FIX_SEED = False
# _C.DATASET.EXTRA_CONFIG.TEST_SUBJECT_INDEX = []

_C.DATAMANAGER = CN()
_C.DATAMANAGER.DATASET = CN()

# Directory where datasets are stored
_C.DATAMANAGER.DATASET.ROOT = ''
_C.DATAMANAGER.DATASET.DIR = ''
_C.DATAMANAGER.DATASET.FILENAME = ''
_C.DATAMANAGER.DATASET.extra_files = None
_C.DATAMANAGER.DATASET.TEST_DIR = ''
_C.DATAMANAGER.DATASET.TEST_DATA_FILE = ''
_C.DATAMANAGER.DATASET.NAME = ''
_C.DATAMANAGER.DATASET.EA = False
_C.DATAMANAGER.DATASET.target_dataset_relabelled = False
_C.DATAMANAGER.DATASET.source_dataset_LA = False

#split the dataset into train/valid and test fold setup
_C.DATAMANAGER.DATASET.SETUP = CN()

#split whole dataset into whole_train and test data
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD = CN()
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.N_TEST_FOLDS = 5
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_FOLD_PREFIX = 'test_fold'
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.START_TEST_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.END_TEST_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.CURRENT_TEST_FOLD=1
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_SUBJECT_INDEX = []
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.NUM_TEST_SUBJECTS = -1
_C.DATAMANAGER.DATASET.SETUP.TEST_FOLD.SAME_AS_VALID = False


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


#use filter bank
_C.DATAMANAGER.DATASET.FILTERBANK= CN()
_C.DATAMANAGER.DATASET.FILTERBANK.USE_FILTERBANK = False
_C.DATAMANAGER.DATASET.FILTERBANK.freq_interval = 4

#use Euclidean Alignment
_C.DATAMANAGER.DATASET.USE_Euclidean_Aligment = False

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





_C.LIGHTNING_MODEL = CN()
_C.LIGHTNING_MODEL.COMPONENTS = CN()

#Define a basic backbone component
_C.LIGHTNING_MODEL.COMPONENTS.BACKBONE = CN()
_C.LIGHTNING_MODEL.COMPONENTS.BACKBONE.NAME = ''
_C.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS = CN()
_C.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PRETRAINED = False
_C.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PRETRAINED_PATH = ''


# Definitions of last layer classifier
_C.LIGHTNING_MODEL.COMPONENTS.LAST_FC = CN()
_C.LIGHTNING_MODEL.COMPONENTS.LAST_FC.NAME= 'max_norm'
_C.LIGHTNING_MODEL.COMPONENTS.LAST_FC.max_norm= 0.5

_C.LIGHTNING_MODEL.COMPONENTS.LAYER = CN()
_C.LIGHTNING_MODEL.COMPONENTS.LAYER.NAME = 'EEGNetConv3'
_C.LIGHTNING_MODEL.COMPONENTS.LAYER.PARAMS = CN()
_C.LIGHTNING_MODEL.COMPONENTS.LAYER.PARAMS.samples = 256
_C.LIGHTNING_MODEL.COMPONENTS.LAYER.PARAMS.F2 = 16
_C.LIGHTNING_MODEL.COMPONENTS.LAYER.PARAMS.drop_prob = 0.25
_C.LIGHTNING_MODEL.COMPONENTS.LAYER.PARAMS.avg_pool_1 = 4
_C.LIGHTNING_MODEL.COMPONENTS.LAYER.PARAMS.avg_pool_2 = 8
_C.LIGHTNING_MODEL.COMPONENTS.LAYER.PARAMS.sep_kern_length = 16

_C.LIGHTNING_MODEL.TRAINER = CN()
_C.LIGHTNING_MODEL.TRAINER.NAME = 'BaseModel'
_C.LIGHTNING_MODEL.TRAINER.PARAMS = CN()
_C.LIGHTNING_MODEL.TRAINER.PARAMS.pretrain = False
_C.LIGHTNING_MODEL.TRAINER.PARAMS.pretrain_epochs = 0

#MultiDataset
_C.LIGHTNING_MODEL.TRAINER.EXTRA = CN()
_C.LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO = 0.8
_C.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO = 0.2
_C.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_PRE_TRAIN_EPOCHS = 10

#DANN
_C.LIGHTNING_MODEL.TRAINER.DANN = CN()
_C.LIGHTNING_MODEL.TRAINER.DANN.lmda = 1.0

#CDAN
_C.LIGHTNING_MODEL.TRAINER.CDAN = CN()
_C.LIGHTNING_MODEL.TRAINER.CDAN.lmda = 1.0
_C.LIGHTNING_MODEL.TRAINER.CDAN.use_projection = False
_C.LIGHTNING_MODEL.TRAINER.CDAN.use_entropy = False

#DAN
_C.LIGHTNING_MODEL.TRAINER.DAN = CN()
_C.LIGHTNING_MODEL.TRAINER.DAN.lmda = 1.0
_C.LIGHTNING_MODEL.TRAINER.DAN.trade_off=1.0
_C.LIGHTNING_MODEL.TRAINER.DAN.linear=False
_C.LIGHTNING_MODEL.TRAINER.DAN.GaussianKernel = CN()
_C.LIGHTNING_MODEL.TRAINER.DAN.GaussianKernel.track_running_stats=True
_C.LIGHTNING_MODEL.TRAINER.DAN.GaussianKernel.alpha = [0.5,1.0,2.0]
_C.LIGHTNING_MODEL.TRAINER.DAN.GaussianKernel.sigma = []

#MLDG
_C.LIGHTNING_MODEL.TRAINER.MLDG = CN()
_C.LIGHTNING_MODEL.TRAINER.MLDG.alpha = 0.1
_C.LIGHTNING_MODEL.TRAINER.MLDG.inner_lr = 0.1
_C.LIGHTNING_MODEL.TRAINER.MLDG.percent_test_subject = 0.0
_C.LIGHTNING_MODEL.TRAINER.MLDG.num_test_subject = 1
_C.LIGHTNING_MODEL.TRAINER.MLDG.num_inner_loop = 1



# #MetaSGD
# _C.TRAINER.MetaSGD = CN()
# _C.TRAINER.MetaSGD.num_test_subject = 1
# _C.TRAINER.MetaSGD.inner_lr = 0.001
# _C.TRAINER.MetaSGD.warm_up = 20

#EpiDG
_C.LIGHTNING_MODEL.TRAINER.EpiDG = CN()
_C.LIGHTNING_MODEL.TRAINER.EpiDG.loss_weight_epir = 0.8
_C.LIGHTNING_MODEL.TRAINER.EpiDG.loss_weight_epif = 0.8
_C.LIGHTNING_MODEL.TRAINER.EpiDG.loss_weight_epic = 0.8
_C.LIGHTNING_MODEL.TRAINER.EpiDG.warn_up_AGG = 35
_C.LIGHTNING_MODEL.TRAINER.EpiDG.warm_up_DS = 25
_C.LIGHTNING_MODEL.TRAINER.EpiDG.start_train_feature = 35
_C.LIGHTNING_MODEL.TRAINER.EpiDG.start_train_classifier = 35
# MCD
# _C.TRAINER.MCD = CN()
# _C.TRAINER.MCD.N_STEP_F = 4
#
# #CustomMCD
# _C.TRAINER.CustomMCD = CN()
# _C.TRAINER.CustomMCD.N_STEP_F = 4
#
# # MME
# _C.TRAINER.MME = CN()
# _C.TRAINER.MME.LMDA = 0.1
# # SelfEnsembling
# _C.TRAINER.SE = CN()
# _C.TRAINER.SE.EMA_ALPHA = 0.999
# _C.TRAINER.SE.CONF_THRE = 0.95
# _C.TRAINER.SE.RAMPUP = 300
#
# # M3SDA
# _C.TRAINER.M3SDA = CN()
# _C.TRAINER.M3SDA.LMDA = 0.5
# _C.TRAINER.M3SDA.N_STEP_F = 4
# # DAEL
# _C.TRAINER.DAEL = CN()
# _C.TRAINER.DAEL.WEIGHT_U = 0.5
# _C.TRAINER.DAEL.CONF_THRE = 0.95
# _C.TRAINER.DAEL.STRONG_TRANSFORMS = ()
#
# # CrossGrad
# _C.TRAINER.CG = CN()
# _C.TRAINER.CG.EPS_F = 1.
# _C.TRAINER.CG.EPS_D = 1.
# _C.TRAINER.CG.ALPHA_F = 0.5
# _C.TRAINER.CG.ALPHA_D = 0.5
# # DDAIG
# _C.TRAINER.DDAIG = CN()
# _C.TRAINER.DDAIG.G_ARCH = ''
# _C.TRAINER.DDAIG.LMDA = 0.3
# _C.TRAINER.DDAIG.CLAMP = False
# _C.TRAINER.DDAIG.CLAMP_MIN = -1.
# _C.TRAINER.DDAIG.CLAMP_MAX = 1.
# _C.TRAINER.DDAIG.WARMUP = 0
# _C.TRAINER.DDAIG.ALPHA = 0.5
#
# # EntMin
# _C.TRAINER.ENTMIN = CN()
# _C.TRAINER.ENTMIN.LMDA = 1e-3
# # Mean Teacher
# _C.TRAINER.MEANTEA = CN()
# _C.TRAINER.MEANTEA.WEIGHT_U = 1.
# _C.TRAINER.MEANTEA.EMA_ALPHA = 0.999
# # MixMatch
# _C.TRAINER.MIXMATCH = CN()
# _C.TRAINER.MIXMATCH.WEIGHT_U = 100.
# _C.TRAINER.MIXMATCH.TEMP = 2.
# _C.TRAINER.MIXMATCH.MIXUP_BETA = 0.75
# _C.TRAINER.MIXMATCH.RAMPUP = 20000
# # FixMatch
# _C.TRAINER.FIXMATCH = CN()
# _C.TRAINER.FIXMATCH.WEIGHT_U = 1.
# _C.TRAINER.FIXMATCH.CONF_THRE = 0.95
# _C.TRAINER.FIXMATCH.STRONG_TRANSFORMS = ()
#
#
# ###########################
# # Trainer specifics
# ###########################
# _C.TRAINER = CN()
# _C.TRAINER.NAME = ''
# _C.TRAINER.PARAMS = CN()
# _C.TRAINER.PARAMS.pretrain = False
# _C.TRAINER.PARAMS.pretrain_epochs = 0

#lightning trainer setup
_C.LIGHTNING_TRAINER = CN()
_C.LIGHTNING_TRAINER.LOGGER = CN()
# _C.LIGHTNING_TRAINER.LOGGER.TENSORBOARD = CN()

_C.LIGHTNING_TRAINER.CHECKPOINT = CN()
_C.LIGHTNING_TRAINER.CHECKPOINT.monitor = 'val_loss'
_C.LIGHTNING_TRAINER.CHECKPOINT.filename = 'best'
_C.LIGHTNING_TRAINER.CHECKPOINT.save_top_k = 1
_C.LIGHTNING_TRAINER.CHECKPOINT.save_last = True
_C.LIGHTNING_TRAINER.CHECKPOINT.every_n_val_epochs = 1

_C.LIGHTNING_TRAINER.early_stop = CN()
_C.LIGHTNING_TRAINER.early_stop.params = CN()
_C.LIGHTNING_TRAINER.early_stop.use_early_stop = False

_C.LIGHTNING_TRAINER.multiple_trainloader_mode = 'max_size_cycle'
_C.LIGHTNING_TRAINER.progress_bar_refresh_rate = 100
_C.LIGHTNING_TRAINER.num_sanity_val_steps = 0
_C.LIGHTNING_TRAINER.profiler = 'simple'
_C.LIGHTNING_TRAINER.stochastic_weight_avg = False


_C.EXTRA_FIELDS = CN()

########################
#TRAIN PROCEDuRE PROCESS
########################

# _C.TRAIN_EVAL_PROCEDURE = CN()
# _C.TRAIN_EVAL_PROCEDURE.HISTORY_FOLDER = 'history_folder'
# _C.TRAIN_EVAL_PROCEDURE.RESULT_FOLDER = 'result_folder'
# _C.TRAIN_EVAL_PROCEDURE.TRAIN = CN()
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.TEST_FOLD_PREFIX = 'test_fold'
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.START_TEST_FOLD = 1
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.END_TEST_FOLD = 4
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.VALID_FOLD_PREFIX = 'valid_fold'
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.START_VALID_FOLD = 1
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.END_VALID_FOLD = 1
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL = CN()
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.INCREMENT_FOLD_PREFIX = 'increment_fold'
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.START_NUM_TRAIN_SUGJECT = -1
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.CURRENT_FOLD = 1
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.START_INCREMENT_FOLD = 1
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.END_INCREMENT_FOLD = 1
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.INCREMENT_UPDATE = 0
# _C.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.NUM_INCREMENT_FOLDS = 1


###########################
# # Dataloader
# ###########################
# _C.DATALOADER = CN()
# _C.DATALOADER.NUM_WORKERS = 0
# # Apply transformations to an image K times (during training)
# # _C.DATALOADER.K_TRANSFORMS = 1
# # Setting for train_x data-loader
# _C.DATALOADER.TRAIN_X = CN()
# _C.DATALOADER.TRAIN_X.SAMPLER = 'RandomSampler'
# _C.DATALOADER.TRAIN_X.BATCH_SIZE = 32
# # Parameter for RandomDomainSampler
# # 0 or -1 means sampling from all domains
# _C.DATALOADER.TRAIN_X.N_DOMAIN = 0
#
# # Setting for train_u data-loader
# _C.DATALOADER.TRAIN_U = CN()
# # Set to false if you want to have unique
# # data loader params for train_u
# _C.DATALOADER.TRAIN_U.SAME_AS_X = True
# _C.DATALOADER.TRAIN_U.SAMPLER = 'RandomSampler'
# _C.DATALOADER.TRAIN_U.BATCH_SIZE = 64
# _C.DATALOADER.TRAIN_U.N_DOMAIN = 0
#
# # Setting for test data-loader
# _C.DATALOADER.TEST = CN()
# _C.DATALOADER.TEST.SAMPLER = 'SequentialSampler'
# _C.DATALOADER.TEST.BATCH_SIZE = 32
#
# # Setting for valid data-loader
# _C.DATALOADER.VALID = CN()
# _C.DATALOADER.VALID.SAMPLER = 'SequentialSampler'
# _C.DATALOADER.VALID.BATCH_SIZE = 32
# _C.DATALOADER.VALID.N_DOMAIN = 0


###########################
# Model
###########################
# _C.MODEL = CN()
# # Path to model weights for initialization
# _C.MODEL.INIT_WEIGHTS = ''
# _C.MODEL.BACKBONE = CN()
# _C.MODEL.BACKBONE.NAME = ''
# _C.MODEL.BACKBONE.PRETRAINED = False
# _C.MODEL.BACKBONE.PRETRAINED_PATH = ''

#domain adaptor source models params
# _C.MODEL.MODULELIST = CN()


#BACKBONE Params info
# _C.MODEL.BACKBONE.PARAMS = CN()

# Definition of embedding layer
# _C.MODEL.HEAD = CN()
# If none, no embedding layer will be constructed
# _C.MODEL.HEAD.NAME = ''
# Structure of hidden layers which is a list, e.g. [512, 512]
# If not defined, no embedding layer will be constructed
# _C.MODEL.HEAD.HIDDEN_LAYERS = ()
# _C.MODEL.HEAD.ACTIVATION = 'relu'
# _C.MODEL.HEAD.BN = True
# _C.MODEL.HEAD.DROPOUT = 0.

# Define custome layer
# _C.MODEL.LAYER = CN()
# _C.MODEL.LAYER.NAME = ''
# _C.MODEL.LAYER.PARAMS = CN()


# _C.MODEL.LAYER = CN()
# _C.MODEL.LAYER.NAME = 'EEGNetConv3'
# _C.MODEL.LAYER.PARAMS = CN()
# _C.MODEL.LAYER.PARAMS.samples = 256
# _C.MODEL.LAYER.PARAMS.F2 = 16
# _C.MODEL.LAYER.PARAMS.drop_prob = 0.25
# _C.MODEL.LAYER.PARAMS.avg_pool_1 = 4
# _C.MODEL.LAYER.PARAMS.avg_pool_2 = 8
# _C.MODEL.LAYER.PARAMS.sep_kern_length = 16
# _C.MODEL.LAYER.PARAMS.total_domain = 2


# Definitions of last layer classifier
# _C.MODEL.LAST_FC = CN()
# _C.MODEL.LAST_FC.NAME = ''
# _C.MODEL.LAST_FC.max_norm = -1.0

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.OPTIMIZER = CN()
_C.OPTIM.SCHEDULER = CN()

# _C.OPTIM.NAME = 'adam'


# _C.OPTIM.LR = 0.001 #default to 1e-3
# _C.OPTIM.WEIGHT_DECAY = 5e-4
# _C.OPTIM.MOMENTUM = 0.9
# _C.OPTIM.SGD_DAMPNING = 0
# _C.OPTIM.SGD_NESTEROV = False
# _C.OPTIM.RMSPROP_ALPHA = 0.99
# _C.OPTIM.ADAM_BETA1 = 0.9
# _C.OPTIM.ADAM_BETA2 = 0.999

# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.OPTIM.STAGED_LR = False
_C.OPTIM.NEW_LAYERS = ()
_C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = 'single_step'
_C.OPTIM.STEPSIZE = (10, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 10

#CosinDecay Optimizer
_C.OPTIM.COSINDECAY = CN()
_C.OPTIM.COSINDECAY.MAX_LR = 0.01
_C.OPTIM.COSINDECAY.WARM_UP = 10
_C.OPTIM.COSINDECAY.WARM_DROP = 1.0
_C.OPTIM.COSINDECAY.LAST_EPOCH = -1

###########################
# Train
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to disable
_C.TRAIN.CHECKPOINT_FREQ = 0
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 10
# Whether to save last epoch model during train process
_C.TRAIN.SAVE_LAST_EPOCH = True
#save the history record of valdiation and test information of every epochs
_C.TRAIN.SAVE_HISTORY_RECORD = True
# Use 'train_x', 'train_u' or 'smaller_one' to count
# the number of iterations in an epoch (for DA and SSL)
_C.TRAIN.COUNT_ITER = 'bigger_one'

###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = 'Classification'
_C.TEST.PER_CLASS_RESULT = False
# Compute confusion matrix, which will be saved
# to $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
# How often (epoch) to do testing during training
# Set to 0 or negative value to disable
_C.TEST.EVAL_FREQ = 1
# Use 'test' set or 'val' set for evaluation
_C.TEST.SPLIT = 'test'

###########################
# Trainer specifics
###########################
# _C.TRAINER = CN()
# _C.TRAINER.NAME = ''
# _C.TRAINER.PARAMS = CN()
# _C.TRAINER.PARAMS.pretrain = False
# _C.TRAINER.PARAMS.pretrain_epochs = 0

# #ADV
# _C.TRAINER.ADV = CN()
# _C.TRAINER.ADV.lmda = 0.1
#
# #
# _C.TRAINER.ShareLabelDANN = CN()
# _C.TRAINER.ShareLabelDANN.lmda = 0.5
#
# #
# _C.TRAINER.HeterogeneousDANN = CN()
# _C.TRAINER.HeterogeneousDANN.lmda = 1.0
#
# #CDAN
# _C.TRAINER.HeterogeneousCDAN = CN()
# _C.TRAINER.HeterogeneousCDAN.lmda = 1.0
# _C.TRAINER.HeterogeneousCDAN.use_projection = False
# _C.TRAINER.HeterogeneousCDAN.use_entropy = False
# #MDD
#
#
# _C.TRAINER.HeterogeneousMDD = CN()
# _C.TRAINER.HeterogeneousMDD.fix_lmda = False
# _C.TRAINER.HeterogeneousMDD.lmda = 1.0
# _C.TRAINER.HeterogeneousMDD.mdd_weight = 5.0
#
# #DAN
# _C.TRAINER.HeterogeneousDAN = CN()
# _C.TRAINER.HeterogeneousDAN.trade_off = 1.0
#
#
# _C.TRAINER.DANN = CN()
# _C.TRAINER.DANN.lmda=1.0
#
# _C.TRAINER.DAN = CN()
# _C.TRAINER.DAN.trade_off=1.0
# _C.TRAINER.DAN.linear=False
# _C.TRAINER.DAN.GaussianKernel = CN()
# _C.TRAINER.DAN.GaussianKernel.track_running_stats=True
# _C.TRAINER.DAN.GaussianKernel.alpha = [0.5,1.0,2.0]
# _C.TRAINER.DAN.GaussianKernel.sigma = []
# #MLDG
# _C.TRAINER.MLDG = CN()
# _C.TRAINER.MLDG.alpha = 0.1
# _C.TRAINER.MLDG.inner_lr = 0.1
# _C.TRAINER.MLDG.percent_test_subject = 0.0
# _C.TRAINER.MLDG.num_test_subject = 1
# _C.TRAINER.MLDG.num_inner_loop = 1
#
# #MetaSGD
# _C.TRAINER.MetaSGD = CN()
# _C.TRAINER.MetaSGD.num_test_subject = 1
# _C.TRAINER.MetaSGD.inner_lr = 0.001
# _C.TRAINER.MetaSGD.warm_up = 20
# #MASF
# _C.TRAINER.MASF = CN()
# _C.TRAINER.MASF.beta1 = 0.1
# _C.TRAINER.MASF.beta2 = 0.1
# _C.TRAINER.MASF.inner_lr = 0.001
#
# #MetaReg
# _C.TRAINER.MetaReg = CN()
# _C.TRAINER.MetaReg.train_full_model_epoch = 20
# _C.TRAINER.MetaReg.regularize_ratio = 0.005
# _C.TRAINER.MetaReg.meta_train_step = 10
# _C.TRAINER.MetaReg.inner_lr = 0.0001
#
# #EpiDG
# _C.TRAINER.EpiDG = CN()
# _C.TRAINER.EpiDG.loss_weight_epir = 0.8
# _C.TRAINER.EpiDG.loss_weight_epif = 0.8
# _C.TRAINER.EpiDG.loss_weight_epic = 0.8
# _C.TRAINER.EpiDG.warn_up_AGG = 35
# _C.TRAINER.EpiDG.warm_up_DS = 25
# _C.TRAINER.EpiDG.start_train_feature = 35
# _C.TRAINER.EpiDG.start_train_classifier = 35
# # MCD
# _C.TRAINER.MCD = CN()
# _C.TRAINER.MCD.N_STEP_F = 4
#
# #CustomMCD
# _C.TRAINER.CustomMCD = CN()
# _C.TRAINER.CustomMCD.N_STEP_F = 4
#
# # MME
# _C.TRAINER.MME = CN()
# _C.TRAINER.MME.LMDA = 0.1
# # SelfEnsembling
# _C.TRAINER.SE = CN()
# _C.TRAINER.SE.EMA_ALPHA = 0.999
# _C.TRAINER.SE.CONF_THRE = 0.95
# _C.TRAINER.SE.RAMPUP = 300
#
# # M3SDA
# _C.TRAINER.M3SDA = CN()
# _C.TRAINER.M3SDA.LMDA = 0.5
# _C.TRAINER.M3SDA.N_STEP_F = 4
# # DAEL
# _C.TRAINER.DAEL = CN()
# _C.TRAINER.DAEL.WEIGHT_U = 0.5
# _C.TRAINER.DAEL.CONF_THRE = 0.95
# _C.TRAINER.DAEL.STRONG_TRANSFORMS = ()
#
# # CrossGrad
# _C.TRAINER.CG = CN()
# _C.TRAINER.CG.EPS_F = 1.
# _C.TRAINER.CG.EPS_D = 1.
# _C.TRAINER.CG.ALPHA_F = 0.5
# _C.TRAINER.CG.ALPHA_D = 0.5
# # DDAIG
# _C.TRAINER.DDAIG = CN()
# _C.TRAINER.DDAIG.G_ARCH = ''
# _C.TRAINER.DDAIG.LMDA = 0.3
# _C.TRAINER.DDAIG.CLAMP = False
# _C.TRAINER.DDAIG.CLAMP_MIN = -1.
# _C.TRAINER.DDAIG.CLAMP_MAX = 1.
# _C.TRAINER.DDAIG.WARMUP = 0
# _C.TRAINER.DDAIG.ALPHA = 0.5
#
# # EntMin
# _C.TRAINER.ENTMIN = CN()
# _C.TRAINER.ENTMIN.LMDA = 1e-3
# # Mean Teacher
# _C.TRAINER.MEANTEA = CN()
# _C.TRAINER.MEANTEA.WEIGHT_U = 1.
# _C.TRAINER.MEANTEA.EMA_ALPHA = 0.999
# # MixMatch
# _C.TRAINER.MIXMATCH = CN()
# _C.TRAINER.MIXMATCH.WEIGHT_U = 100.
# _C.TRAINER.MIXMATCH.TEMP = 2.
# _C.TRAINER.MIXMATCH.MIXUP_BETA = 0.75
# _C.TRAINER.MIXMATCH.RAMPUP = 20000
# # FixMatch
# _C.TRAINER.FIXMATCH = CN()
# _C.TRAINER.FIXMATCH.WEIGHT_U = 1.
# _C.TRAINER.FIXMATCH.CONF_THRE = 0.95
# _C.TRAINER.FIXMATCH.STRONG_TRANSFORMS = ()
