# EEG-Dassl

EEG-Dassl is a [PyTorch](https://pytorch.org) toolbox for domain adaptation and semi-supervised learning with EEG data. It is an improvement version of Dassl for EEG data
You can use Dassl as a library for the following research:

- Domain adaptation
- Domain generalization

## Overview

Dassl has implemented the following papers:

- Vanilla Model
    - [[dassl/engine/base/base_model.py](dassl/engine/base/base_model.py)]

- Multi-source domain adaptation
    - [[dassl/engine/da/heterogeneous/multi_dataset_adaptation.py](dassl/engine/da/heterogeneous/multi_dataset_adaptation.py)]
    - [[dassl/engine/da/heterogeneous/multi_dataset_component_adapt.py](dassl/engine/da/heterogeneous/multi_dataset_component_adapt.py)]

Dassl supports the following datasets.

- [BCI_IV]()
- [Cho2017]()
- [Physionet]()


Dassl current support the following TRAINER

- [[BaseModel](dassl/engine/base/base_model.py)]
- [[MultiDatasetAdaptation](dassl/engine/da/heterogeneous/multi_dataset_adaptation.py)]
- [[ComponentNet](dassl/engine/da/heterogeneous/multi_dataset_component_adapt.py)]


Dassl current support the following BACKBONE

- [[eegnet](dassl/modeling/backbone/EEGBackBone/EEGNET.py)]
- [[extractor](dassl/modeling/backbone/EEGBackBone/FeatureExtractor.py)]
- [[componentExtractor](dassl/modeling/backbone/EEGBackBone/ComponentExtractor.py)]

Dassl current support the following DATAMANAGER

- [[dassl/data/data_manager_v1.py](dassl/data/data_manager_v1.py)]

Dassl current support the following DATASET process

- [[MultiDataset](dassl/data/datasets/NeurIPS_BCI.py)]

## Get started

### Installation

Make sure [conda](https://www.anaconda.com/distribution/) is installed properly.

```bash
# Clone this repo
cd EEG_Dassl_Lightning

# Create a conda environment
conda create -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch and torchvision (select a version that suits your machine)
pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```


### Training

The main interface is implemented in `tools/train.py` or `train_temp.py`, which has 2 main components:

1. call `trainer.train()` for training and evaluating the model for single run
2. Perform k-folds training to train k models for cross-validation


Below we provide an example for training a vanilla EEGNet on a transfer learning experiment with 2 source dataset (Cho2017 and Physionet)
and 1 target dataset BCI_IV

Step 1: Generate a matlab file that contains the data in a specific format. 

Follow the instructions in [DATASETS.md](./DATASETS.md) for more detail on how to setup the datasets format.

locate the file BCI_competition.py in EEG_Dassl_Lightning\da_dataset\NeurIPS_competition\BCI_competition.py

run  python BCI_competition.py 

a matlab file name NeurIPS_TL.mat is generated in folder case_1.

Step 2: Generate a set of experiment .yaml config files

locate the file generate_yaml.py in EEG_Dassl_Lightning\generate_experiment\generate_yaml.py

run  python generate_yaml.py 

a folder name experiment_1 is generated. You can go through the folder to see the transfer_learning.yaml file in main_config folder. Copy the file and
move it to EEG_Dassl_LIghtning\experiment_1


Here is a general format of transfer_adaptation.yaml for main-config

Here is an example of a transfer_learning.yaml file: 

```YAML
INPUT:
  SIZE: (22, 256)
  TRANSFORMS: []
  NO_TRANSFORM: true
USE_CUDA: true
DATAMANAGER:
  RESULT_FOLDER: result_folder
  MANAGER_TYPE: single_dataset
  DATASET:
    NAME: MultiDataset
    DIR: ''
    FILENAME: NeurIPS_TL.mat
    SETUP:
      TARGET_DATASET_NAME: BCI_IV
      TEST_FOLD:
        N_TEST_FOLDS: 1
        TEST_FOLD_PREFIX: test_fold
        START_TEST_FOLD: 1
        END_TEST_FOLD: 1
        CURRENT_TEST_FOLD: 1
        WITHIN_SUBJECTS: true
      SHUFFLE_TRAIN_VALID_FOLD:
        SHUFFLE_FOLD_PREFIX: shuffle_fold
        N_SHUFFLE_FOLDS: 4
        START_SHUFFLE_FOLD: 1
        END_SHUFFLE_FOLD: 4
        CURRENT_SHUFFLE_FOLD: 1
        SHUFFLE_SEED:
          - 0
          - 10
          - 20
          - 30
      INCREMENT_FOLD:
        INCREMENT_FOLD_PREFIX: increment_fold
        N_INCREMENT_FOLDS: 3
        START_NUM_TRAIN_SUGJECT: 2
        INCREMENT_TRAIN_SUGJECT: 2
        START_INCREMENT_FOLD: 1
        END_INCREMENT_FOLD: 3
        CURRENT_INCREMENT_FOLD: 1
      VALID_FOLD:
        VALID_FOLD_PREFIX: valid_fold
        N_VALID_FOLDS: 3
        START_VALID_FOLD: 1
        END_VALID_FOLD: 3
        CURRENT_VALID_FOLD: 1
        WITHIN_SUBJECTS: true
        TRAIN_VALID_DATA_RATIO: 0.3
        TRAIN_DATA_RATIO: 0.8
    AUGMENTATION:
      NAME: ''
      PARAMS:
        N_SEGMENT: 4
        MAX_TRIAL_MUL: 3
        MAX_FIX_TRIAL: -1
        DATASET_NAME: BCI_IV
  DATALOADER:
    NUM_WORKERS: 0
    TRAIN_X:
      SAMPLER: RandomSampler
      BATCH_SIZE: 64
    VALID:
      SAMPLER: SequentialSampler
      BATCH_SIZE: 64
    TEST:
      SAMPLER: SequentialSampler
      BATCH_SIZE: 64
OPTIM:
  NAME: adam
  LR: 0.001
  MAX_EPOCH: 50
  LR_SCHEDULER: exponential
  STEPSIZE: (1, )
  WEIGHT_DECAY: 0.0
  GAMMA: 1.0
TRAIN:
  CHECKPOINT_FREQ: 0
  PRINT_FREQ: 10
TEST:
  EVALUATOR: Classification
  PER_CLASS_RESULT: true
LIGHTNING_MODEL:
  TRAINER:
    NAME: BaseModel
  COMPONENTS:
    BACKBONE:
      NAME: eegnet
      PARAMS:
        kern_legnth: 64
        num_ch: 22
        samples: 256
        F1: 8
        D: 2
        F2: 16
        drop_prob: 0.25
        sep_kern_length: 16
      LAST_FC:
        NAME: max_norm
        max_norm: 0.5
EXTRA_FIELDS:
  target_dataset: BCI_IV
  source_dataset: []
  normalize: no_norm
  aug: no_aug
  model: BaseModel
  source_label_space: []
  target_label_space: 3

```

We can define the YAML file with 3 important sections. The datamanager, the LIGHTNING_MODEL and EXTRA_FIELDS.

- INPUT.SIZE: Define the (channels,samples) of input EEG data
- INPUT.TRANSFORMS: contains a list of possible data normalization
    - ['cross_channel_norm'] : if we want to apply cross channel normalization. We also need NO_TRANSFORM: True
- INPUT.NO_TRANSFORM: A boolean value to see if there are any data normalization 
 
- DATAMANAGER.RESULT_FOLDER: define the name of result folder where we store experiment result. EX: 'result_folder'
- DATAMANAGER.MANAGER_TYPE: define the type of data manager setup. 
    - 'single_dataset' if we only need to use 1 target dataset.
    - 'multi_dataset' if we use 1 target datast and couple source datasets
- DATAMANAGER.DATASET.NAME: The data manager type that will process the data. 
    - Only support 'MultiDataset' at the moment
- DATAMANAGER.DATASET.DIR: directory that contains the matlab file. Default = ''
- DATAMANAGER.DATASET.FILENAME: the matlab file contains the dataset.
    - Ex: \data_folder_path\case_1\NeurIPS_TL.mat 
- DATAMANAGER.DATASET.SETUP.TEST_FOLD.N_TEST_FOLDS: number of different test fold. Each test fold has 1 unique set of test data.
- DATAMANAGER.DATASET.SETUP.TEST_FOLD.START_TEST_FOLD: indicate the start of test fold
- DATAMANAGER.DATASET.SETUP.TEST_FOLD.END_TEST_FOLD: indicate the end of test fold
- DATAMANAGER.DATASET.SETUP.TEST_FOLD.CURRENT_TEST_FOLD: indicate the current test fold
- DATAMANAGER.DATASET.SETUP.TEST_FOLD.WITHIN_SUBJECTS: A bool True mean we conduct cross session evaluation for same subject. 
EX: if we have 5 subject, then we use 70% data of each subject as test data while the other 30% as train/valid data. 
- DATAMANAGER.DATASET.SETUP.TEST_FOLD.TRAIN_VALID_DATA_RATIO: ratio to split the train/valid and test data for within subject scenario

- DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS: number of different shuffle fold. Each fold is a unique shuffle. E
X: for 5 subjects, we have shuffle 1 indices as \[1,5,3,4,2], and shuffle 2 indices as \[2,3,1,5,4]
- DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.START_SHUFFLE_FOLD: indicate the start of shuffle fold
- DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.END_SHUFFLE_FOLD: indicate the end of shuffle fold
- DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.CURRENT_SHUFFLE_FOLD: indicate the current shuffle fold
- DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_SEED: a list of shuffle seed. Same number as N_shuffle folds.

- DATAMANAGER.DATASET.SETUP.VALID_FOLD.N_VALID_FOLDS: number of different valid fold. 
- DATAMANAGER.DATASET.SETUP.VALID_FOLD.START_VALID_FOLD: indicate the start of valid fold
- DATAMANAGER.DATASET.SETUP.VALID_FOLD.END_VALID_FOLD: indicate the end of valid fold
- DATAMANAGER.DATASET.SETUP.VALID_FOLD.CURRENT_VALID_FOLD: indicate the current valid fold

- DATAMANAGER.DATASET.SETUP.VALID_FOLD.WITHIN_SUBJECTS: A bool True mean we conduct cross session evaluation for same subject.
- DATAMANAGER.DATASET.SETUP.VALID_FOLD.TRAIN_DATA_RATIO: ratio to split the train and valid data for within subject scenario

- DATAMANAGER.DATASET.AUGMENTATION.NAME: Define the type of augmentation. There are 3 types
    - "temporal_segment" mean we use time domain augmentation
    - "temporal_segment_T_F"  mean we use time-frequency domain augmentation
    - "" mean no augmentation 
- DATAMANAGER.DATASET.AUGMENTATION.PARAMS.N_SEGMENT: number of divided segment for augmentation.
EX: N_SEGMENT=4 for trial (22,256) mean we split the trial into 4 segment of 64.
- DATAMANAGER.DATASET.AUGMENTATION.PARAMS.MAX_TRIAL_MUL: multiply number of trials. 
EX MAX_TRIAL_MUL=3 for 500 trial mean we generate an extra 1500 trials beside original 500.

- LIGHTNING_MODEL.TRAINER.NAME:  Define a trainer technique to train different components. Options include :  ['BaseModel','MultiDatasetAdaptation','ComponentAdaptation']
    - 'BaseModel' : a basic trainer that use 1 backbone with 1 classifier (set BACKBONE.NAME: 'eegnet'')
    - 'MultiDatasetAdaptation' : a multi-task trainer that use 1 target dataset and mult-source datasets. 
- LIGHTNING_MODEL.COMPONENTS.BACKBONE.NAME: Define a backbone model to learn the feature representation of EEG data. Options include : ['eegnet','extractor','componentExtractor']
- LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS: Provide a list of parameter for the given backbone model. 
- OPTIM.MAX_EPOCH: Total epochs to train the model

- TRAIN.CHECKPOINT_FREQ: Define the frequenly to save a model. ex: CHECKPOINT_FREQ = 1 mean save a model at the end of every epoch. 
- TRAIN.PRINT_FREQ: Define the frequenly to print out information of train process during an epoch

- EXTRA_FIELDS: We define extra variable to record for the experiment. This can be used to distinguish different experiments

Step 3: train model

1) train a single sub experiment

```bash
python tools/train.py \
--root da_dataset/NeurIPS_competition
--output-dir experiment_1/no_aug/chan_norm/vanilla/BCI_IV\model
--main-config-file experiment_1/no_aug/chan_norm/vanilla/BCI_IV/main_config/transfer_adaptation.yaml
--train-k-folds
```

2) Get test result from a single sub experiment

```bash
python tools/train.py 
--root da_dataset/NeurIPS_competition
--output-dir experiment_1/no_aug/chan_norm/vanilla/BCI_IV\model
--main-config-file experiment_1/no_aug/chan_norm/vanilla/BCI_IV/main_config/transfer_adaptation.yaml
--train-k-folds
--eval-only 
--model-dir experiment_1/no_aug/chan_norm/vanilla/BCI_IV/model
```

The result is located in 'result_folder'

2) train multiple models in an experiment

Find bash script files in train_script folder. Modify path to the dataset in common_script.sh
Modify the specific model, augmentation and normalization for experiment_script in gpu_0,1,2,3_script

run  bash main_script.sh to run multiple bash script for different model.

Step 4: Analyze result
 
run anaconda enviroment
open the jupyter notebook analyze_experiment\NeurIPS_Analysis.ipynb




Example yaml config for multi source domain and target domain experiment,

```YAML
INPUT:
  SIZE: (22, 256)
  TRANSFORMS: []
  NO_TRANSFORM: true
USE_CUDA: true
DATAMANAGER:
  RESULT_FOLDER: result_folder
  MANAGER_TYPE: multi_dataset
  DATASET:
    NAME: MultiDataset
    DIR: case_1
    FILENAME: NeurIPS_TL.mat
    SETUP:
      TARGET_DATASET_NAME: BCI_IV
      TEST_FOLD:
        N_TEST_FOLDS: 3
        TEST_FOLD_PREFIX: test_fold
        START_TEST_FOLD: 1
        END_TEST_FOLD: 3
        CURRENT_TEST_FOLD: 1
        WITHIN_SUBJECTS: true
        TRAIN_VALID_DATA_RATIO: 0.3
      SHUFFLE_TRAIN_VALID_FOLD:
        SHUFFLE_FOLD_PREFIX: shuffle_fold
        N_SHUFFLE_FOLDS: 4
        START_SHUFFLE_FOLD: 1
        END_SHUFFLE_FOLD: 4
        CURRENT_SHUFFLE_FOLD: 1
        SHUFFLE_SEED:
          - 0
          - 10
          - 20
          - 30
      INCREMENT_FOLD:
        INCREMENT_FOLD_PREFIX: increment_fold
        N_INCREMENT_FOLDS: 3
        START_NUM_TRAIN_SUGJECT: 2
        INCREMENT_TRAIN_SUGJECT: 2
        START_INCREMENT_FOLD: 1
        END_INCREMENT_FOLD: 3
        CURRENT_INCREMENT_FOLD: 1
      VALID_FOLD:
        VALID_FOLD_PREFIX: valid_fold
        N_VALID_FOLDS: 1
        START_VALID_FOLD: 1
        END_VALID_FOLD: 1
        CURRENT_VALID_FOLD: 1
        WITHIN_SUBJECTS: true
        TRAIN_DATA_RATIO: 0.8
    AUGMENTATION:
      NAME: ''
      PARAMS:
        N_SEGMENT: 4
        MAX_TRIAL_MUL: 3
        MAX_FIX_TRIAL: -1
        DATASET_NAME: BCI_IV
  DATALOADER:
    NUM_WORKERS: 0
    TRAIN_X:
      SAMPLER: RandomSampler
      BATCH_SIZE: 64
    LIST_TRAIN_U:
      SAMPLERS:
        - RandomSampler
        - RandomSampler
      BATCH_SIZES:
        - 64
        - 64
    VALID:
      SAMPLER: SequentialSampler
      BATCH_SIZE: 64
    TEST:
      SAMPLER: SequentialSampler
      BATCH_SIZE: 64
OPTIM:
  NAME: adam
  LR: 0.001
  MAX_EPOCH: 30
  LR_SCHEDULER: exponential
  STEPSIZE: (1, )
  WEIGHT_DECAY: 0.0
  GAMMA: 1.0
TRAIN:
  CHECKPOINT_FREQ: 0
  PRINT_FREQ: 10
TEST:
  EVALUATOR: Classification
  PER_CLASS_RESULT: true
LIGHTNING_MODEL:
  TRAINER:
    NAME: MultiDatasetAdaptation
    EXTRA:
      TARGET_LOSS_RATIO: 0.8
      SOURCE_LOSS_RATIO: 0.2
      SOURCE_PRE_TRAIN_EPOCHS: 10
  COMPONENTS:
    LAST_FC:
      NAME: max_norm
      max_norm: 0.5
    LAYER:
      NAME: EEGNetConv3
      PARAMS:
        samples: 256
        F2: 16
        drop_prob: 0.25
        avg_pool_1: 4
        avg_pool_2: 8
        sep_kern_length: 16
    BACKBONE:
      NAME: extractor
      PARAMS:
        kern_legnth: 64
        num_ch: 22
        samples: 256
        F1: 8
        D: 2
        F2: 16
        drop_prob: 0.25
EXTRA_FIELDS:
  target_dataset: BCI_IV
  source_dataset:
    - cho2017
    - physionet
  normalize: no_norm
  aug: no_aug
  model: MultiDatasetAdaptation
  source_label_space:
    - 2
    - 3
  target_label_space: 3

```

## Available example
You can find example config file in 'experiment_1' folder.


## Citation
Please cite original Dassl framework.

```
@article{zhou2020domain,
  title={Domain Adaptive Ensemble Learning},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={arXiv preprint arXiv:2003.07325},
  year={2020}
}
```