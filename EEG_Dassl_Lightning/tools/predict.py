import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
import pandas as pd
from dassl.utils import ( generate_path_for_multi_sub_model)
from pytorch_lightning import LightningModule,Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from dassl.data.data_manager_v1 import DataManagerV1, MultiDomainDataManagerV1
from torch.utils.data import DataLoader

import pytorch_lightning as pl
def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATAMANAGER.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir


def setup_cfg(args):
    cfg = get_cfg_default()
    reset_cfg(cfg, args)
    #allowed to add new keys for config
    cfg.set_new_allowed(True)
    if args.main_config_file:
        cfg.merge_from_file(args.main_config_file)
    cfg.merge_from_list(args.opts)
    return cfg

from yacs.config import CfgNode as CN
def convert_to_dict(cfg_node, key_list):
    def _valid_type(value, allow_cfg_node=False):
        return (type(value) in _VALID_TYPES) or (
                allow_cfg_node and isinstance(value, CN)
        )
    def _assert_with_logging(cond, msg):
        if not cond:
            logger.debug(msg)
        assert cond, msg
    import logging
    logger = logging.getLogger(__name__)
    _VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
    if not isinstance(cfg_node, CN):
        _assert_with_logging(
            _valid_type(cfg_node),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES
            ),
        )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

from typing import Any, Dict, Optional, Union
from collections import defaultdict
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn
class CustomExperimentWriter(object):
    NAME_HPARAMS_FILE = 'hparams.yaml'
    NAME_METRICS_FILE = 'metrics.csv'
    def __init__(self,log_dir: str,step_key='step'):
        # super().__init__(log_dir)
        self.metrics = defaultdict(dict)
        self.hparams = {}
        self.step_key = step_key
        # self.metrics = []

        self.log_dir = log_dir
        if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

        # print("use custome writer ")
        if os.path.exists(self.metrics_file_path):
            history = pd.read_csv(self.metrics_file_path)
            for k, row in history.iterrows():
                # print(row.to_dict())
                self.metrics[row[self.step_key]] = row.to_dict()

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams"""
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics"""

        def _handle_value(value):
            if isinstance(value, torch.Tensor):
               return value.item()
            return value

        if step is None:
            step = len(self.metrics)
        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics[self.step_key] = step

        self.metrics[step].update(metrics)
    def save(self) -> None:
        """Save recorded hparams and metrics into files"""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)

        if not self.metrics:
           return

        last_m =  [self.metrics[i] for i in sorted(self.metrics.keys())]
        record = pd.DataFrame(last_m)
        record.to_csv(self.metrics_file_path, index=False)

class CustomeCSVLogger(CSVLogger):
    def __init__(self,save_dir: str,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        prefix: str = '',experiment_writer=None,step_key='step'):
        super().__init__(save_dir=save_dir,name=name,version=version,prefix=prefix)
        if experiment_writer:
            os.makedirs(self.root_dir, exist_ok=True)
            self._experiment = experiment_writer(self.log_dir,step_key=step_key)

class CustomModelCheckPoint(ModelCheckpoint):
    def on_load_checkpoint(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', callback_state: Dict[str, Any]
    ) -> None:
        """
        Fix a bug in ModelCheckpoint for pytorch lightning
        If we interrupt the model during the training and resume it, the model checkpoint does not load all the neccessary from last.cktp
        which contains information about the best model
        """
        super(CustomModelCheckPoint, self).on_load_checkpoint(trainer,pl_module,callback_state)
        self.kth_best_model_path = callback_state["best_model_path"]
        self.best_k_models[self.best_model_path] = self.best_model_score


def process_target_data(epoch_data, f_min=4, f_max=36, resample=128, t_min=0, t_max=3):
    epoch_f = epoch_data.copy().filter(
        f_min, f_max, method="iir")
    # if bmin < tmin or bmax > tmax:
    epoch_f = epoch_f.crop(tmin=t_min, tmax=t_max)
    # if self.resample is not None:
    epoch_f = epoch_f.resample(resample)
    return epoch_f

def modify_data(data, time=256):
    return data[:, :, :time]
def _expand_data_dim(data):
    if isinstance(data, list):
        for idx in range(len(data)):
            new_data = np.expand_dims(data[idx], axis=1)
            data[idx] = new_data
        return data
    elif isinstance(data, np.ndarray):
        return np.expand_dims(data, axis=2)
    else:
        raise ValueError("the data format during the process section is not correct")

def normalization(X):
    # assert len(X) == len(y)
    # Normalised, you could choose other normalisation strategy
    mean = np.mean(X, axis=1, keepdims=True)
    # here normalise across channels as an example, unlike the in the sleep kit
    std = np.std(X, axis=1, keepdims=True)
    X = (X - mean) / std
    return X

def dataset_norm(data):
    new_data = list()
    for subject_idx in range(len(data)):
        subject_data = data[subject_idx]
        subject_data = normalization(subject_data)
        new_data.append(subject_data)
    return new_data

import mne
import pickle
from numpy.random import RandomState
def get_dataset_B(norm=False):


    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cuda = torch.cuda.is_available()
    print('gpu: ', cuda)
    device = 'cuda' if cuda else 'cpu'

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    rng = RandomState(seed)
    # process Dataset B (S1, S2, S3)
    # get train target data
    path = '/Users/wduong/mne_data/MNE-beetlmileaderboard-data/'
    sfreq = 128
    tmin = 0
    tmax = 3
    fmin, fmax = 4, 36

    max_time_length = int((tmax - tmin) * sfreq)
    X_MIB_test_data = []
    subject_ids = []

    for subj in range(3, 6):
        savebase = os.path.join(path, "S{}".format(subj), "testing")
        subject_test_data = []
        with open(os.path.join(savebase, "testing_s{}X.npy".format(subj)), 'rb') as f:
            subject_test_data.append(pickle.load(f))
        subject_test_data = np.concatenate(subject_test_data)

        total_trials = len(subject_test_data)
        n_channels = 32
        sampling_freq = 200  # in Hertz
        # info = mne.create_info(n_channels, sfreq=sampling_freq)
        ch_names = ['Fp1', 'Fp2', 'F3',
                    'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C5', 'C3',
                    'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                    'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz',
                    'P2', 'P4', 'P6', 'P8']
        ch_types = ['eeg'] * 32

        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
        # print("events : ",events)
        # event_dict = dict(left_hand=0, right_hand=1,feet=2,rest=3)
        mne_data = mne.EpochsArray(subject_test_data, info, tmin=0)

        new_mne_data = process_target_data(mne_data, f_min=fmin, f_max=fmax, resample=sfreq, t_min=tmin, t_max=tmax)

        # print(new_mne_data.get_data().shape)
        subject_test_data = new_mne_data.get_data()

        subject_test_data = modify_data(subject_test_data, time=max_time_length)

        print("dataset B label : ", subject_test_data)
        X_MIB_test_data.append(subject_test_data)
        subject_id = [subj] * len(subject_test_data)
        subject_ids.extend(subject_id)

    for subj in range(len(X_MIB_test_data)):
        # print("subject {}".format(subj + 1))
        subject_train_data = X_MIB_test_data[subj]
        print("There are {} trials with {} electrodes and {} time samples".format(*subject_train_data.shape))
        # print("label shape : ", subject_train_label.shape)
    # print(subject_ids)
    dataset_B_meta = pd.DataFrame(
        {"subject": subject_ids, "session": ["session_0"] * len(subject_ids), "run": ["run_0"] * len(subject_ids)})
    # print("A meta : ",dataset_B_meta)
    # X_MIB_test_data = np.concatenate(X_MIB_test_data)
    if norm:
        X_MIB_test_data = dataset_norm(X_MIB_test_data)
    X_MIB_test_data = _expand_data_dim(X_MIB_test_data)

    #format into trials,channels,sample
    X_MIB_test_data = np.concatenate(X_MIB_test_data)
    return X_MIB_test_data

def get_dataset_A(norm=False):


    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cuda = torch.cuda.is_available()
    print('gpu: ', cuda)
    device = 'cuda' if cuda else 'cpu'

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    rng = RandomState(seed)
    # process Dataset B (S1, S2, S3)
    # get train target data
    path = '/Users/wduong/mne_data/MNE-beetlmileaderboard-data/'
    sfreq = 128
    tmin = 0
    tmax = 3
    fmin, fmax = 4, 36
    max_time_length = int((tmax - tmin) * sfreq)
    X_MIA_test_data = []
    subject_ids = []
    for subj in range(1, 3):
        savebase = os.path.join(path, "S{}".format(subj), "testing")
        subject_test_data = []
        for i in range(6, 16):
            with open(os.path.join(savebase, "race{}_padsData.npy".format(i)), 'rb') as f:
                subject_test_data.append(pickle.load(f))
        subject_test_data = np.concatenate(subject_test_data)

        total_trials = len(subject_test_data)
        n_channels = 63
        sampling_freq = 500  # in Hertz
        # info = mne.create_info(n_channels, sfreq=sampling_freq)
        ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7',
                    'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
                    'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8',
                    'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
                    'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5',
                    'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
                    'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
                    'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
        ch_types = ['eeg'] * 63
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
        mne_data = mne.EpochsArray(subject_test_data, info, tmin=0)

        new_mne_data = process_target_data(mne_data, f_min=fmin, f_max=fmax, resample=sfreq, t_min=tmin, t_max=tmax)

        # print(new_mne_data.get_data().shape)
        subject_test_data = new_mne_data.get_data()

        print("dataset A label : ", subject_test_data)
        X_MIA_test_data.append(subject_test_data)
        subject_id = [subj] * len(subject_test_data)
        subject_ids.extend(subject_id)

    for subj in range(len(X_MIA_test_data)):
        # print("subject {}".format(subj + 1))
        subject_train_data = X_MIA_test_data[subj]
        print("There are {} trials with {} electrodes and {} time samples".format(*subject_train_data.shape))
        # print("label shape : ", subject_train_label.shape)
    if norm:
        X_MIA_test_data = dataset_norm(X_MIA_test_data)
    X_MIA_test_data = _expand_data_dim(X_MIA_test_data)

    #format into trials,channels,sample
    X_MIA_test_data = np.concatenate(X_MIA_test_data)
    return X_MIA_test_data

def generate_pred_MI_label(test_fold_preds,test_fold_probs,output_dir,predict_folder = "predict_folder"):
    for test_fold in range(len(test_fold_preds)):
        valid_fold_pred = test_fold_preds[test_fold]
        current_pred = valid_fold_pred[0]

        valid_fold_prob = test_fold_probs[test_fold]
        current_prob = valid_fold_prob[0]
        for idx in range(1,len(valid_fold_pred)):
            current_pred = current_pred + valid_fold_pred[idx]
            current_prob = current_prob + valid_fold_prob[idx]
        print("result current pred : ",current_pred)
        pred_output = list()
        for trial_idx in range(len(current_pred)):
            # print("trial {} has pred {} ".format(trial_idx,current_pred[trial_idx]))
            # print("trial {} has probs {} ".format(trial_idx,current_prob[trial_idx]))

            preds = current_pred[trial_idx]
            probs = current_prob[trial_idx]
            best_idx = -1
            best_pred = -1
            best_prob = -1
            for idx in range(len(preds)):
                pred=preds[idx]
                prob = probs[idx]
                if pred > best_pred:
                    best_pred = pred
                    best_idx = idx
                    best_prob = prob
                elif pred == best_pred:
                    if prob > best_prob:
                        best_idx = idx
                        best_prob = prob
            pred_output.append(best_idx)
        pred_output = np.array(pred_output)
        print("pred output : ",pred_output)
        print("just arg pred output : ",np.argmax(current_pred,axis=1))
        combine_folder = os.path.join(output_dir, predict_folder)
        np.savetxt(os.path.join(combine_folder,"pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")
def generate_assemble_result(test_fold_preds,test_fold_probs,test_fold_labels,output_dir,predict_folder = "predict_folder"):
    test_fold_acc = list()
    for test_fold in range(len(test_fold_preds)):
        valid_fold_pred = test_fold_preds[test_fold]
        current_pred = valid_fold_pred[0]

        valid_fold_prob = test_fold_probs[test_fold]
        current_prob = valid_fold_prob[0]

        valid_fold_label = test_fold_labels[test_fold]
        current_label = valid_fold_label[0]

        for idx in range(1,len(valid_fold_pred)):
            current_pred = current_pred + valid_fold_pred[idx]
            current_prob = current_prob + valid_fold_prob[idx]
        print("result current pred : ",current_pred)
        pred_output = list()
        for trial_idx in range(len(current_pred)):
            # print("trial {} has pred {} ".format(trial_idx,current_pred[trial_idx]))
            # print("trial {} has probs {} ".format(trial_idx,current_prob[trial_idx]))

            preds = current_pred[trial_idx]
            probs = current_prob[trial_idx]
            best_idx = -1
            best_pred = -1
            best_prob = -1
            for idx in range(len(preds)):
                pred=preds[idx]
                prob = probs[idx]
                if pred > best_pred:
                    best_pred = pred
                    best_idx = idx
                    best_prob = prob
                elif pred == best_pred:
                    if prob > best_prob:
                        best_idx = idx
                        best_prob = prob
            pred_output.append(best_idx)
        pred_output = np.array(pred_output)
        acc = np.mean(pred_output == current_label)
        test_fold_acc.append(acc)
        print("pred output : ",pred_output)
        print("current label : ",current_label)
        print(" valid_fold_label : ",valid_fold_label)
        print("test fold {} has acc {} ".format(test_fold,acc))
        # print("just arg pred output : ",np.argmax(current_pred,axis=1))
        # combine_folder = os.path.join(output_dir, predict_folder)
        # np.savetxt(os.path.join(combine_folder,"pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")


def main(args):
    benchmark = False
    deterministic = False  # this can help to reproduce the result
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        print("use determinstic ")
        benchmark = False
        deterministic = True #this can help to reproduce the result

    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    print("Experiment setup ...")
    # cross test fold setup
    N_TEST_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.N_TEST_FOLDS
    START_TEST_FOLD = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.START_TEST_FOLD
    END_TEST_FOLD = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.END_TEST_FOLD
    TEST_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_FOLD_PREFIX

    #shuffle fold
    SHUFFLE_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_FOLD_PREFIX
    N_SHUFFLE_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS
    START_SHUFFLE_FOLD = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.START_SHUFFLE_FOLD
    END_SHUFFLE_FOLD = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.END_SHUFFLE_FOLD
    USE_SHUFFLE = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS > 1


    #increment fold setup
    # conduct incremental subject experiments
    N_INCREMENT_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.N_INCREMENT_FOLDS
    INCREMENT_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.INCREMENT_FOLD_PREFIX
    START_INCREMENT_FOLD = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_INCREMENT_FOLD
    END_INCREMENT_FOLD = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.END_INCREMENT_FOLD
    USE_INCREMENT = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_NUM_TRAIN_SUGJECT > 0

    #valid fold setup
    N_VALID_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.N_VALID_FOLDS
    START_VALID_FOLD = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.START_VALID_FOLD
    END_VALID_FOLD = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.END_VALID_FOLD
    VALID_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.VALID_FOLD_PREFIX

    data_manager_type = cfg.DATAMANAGER.MANAGER_TYPE

    """Apply data transformation/normalization"""
    norm = False
    if not cfg.INPUT.NO_TRANSFORM:
        normalization = cfg.INPUT.TRANSFORMS[0]
        if normalization == 'cross_channel_norm':
            norm = True
    generate_predict = False
    use_assemble_test_dataloader = True
    dataset_type = cfg.DATAMANAGER.DATASET.SETUP.TARGET_DATASET_NAME
    if generate_predict and not use_assemble_test_dataloader:
        if dataset_type == 'dataset_B':
            dataset = get_dataset_B(norm=norm)
        else:
            dataset = get_dataset_A(norm=norm)

        # dataset_B = get_dataset_B(norm=norm)


    combine_prefix = dict()

    test_fold_preds = list()
    test_fold_probs = list()
    test_fold_label = list()
    for current_test_fold in range(START_TEST_FOLD, END_TEST_FOLD + 1):
        cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD['CURRENT_TEST_FOLD'] = current_test_fold
        combine_prefix[TEST_FOLD_PREFIX] = TEST_FOLD_PREFIX+"_"+str(current_test_fold)
        for current_shuffle_fold in range(START_SHUFFLE_FOLD,END_SHUFFLE_FOLD+1):
            cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD['CURRENT_SHUFFLE_FOLD'] = current_shuffle_fold
            shuffle_fold_prefix = ""
            if USE_SHUFFLE:
                shuffle_fold_prefix = SHUFFLE_FOLD_PREFIX + "_" + str(current_shuffle_fold)
                combine_prefix[SHUFFLE_FOLD_PREFIX] = shuffle_fold_prefix
            for current_increment_fold in range(START_INCREMENT_FOLD,END_INCREMENT_FOLD+1):
                cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD['CURRENT_INCREMENT_FOLD'] = current_increment_fold
                increment_fold_prefix=""
                if USE_INCREMENT:
                    increment_fold_prefix = INCREMENT_FOLD_PREFIX + "_" + str(current_increment_fold)
                    combine_prefix[INCREMENT_FOLD_PREFIX] = increment_fold_prefix

                valid_fold_preds = list()
                valid_fold_probs = list()
                valid_fold_label = list()
                for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD + 1):
                    combine_prefix[VALID_FOLD_PREFIX] = VALID_FOLD_PREFIX+"_"+str(current_valid_fold)
                    cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD['CURRENT_VALID_FOLD'] = current_valid_fold

                    output_dir = cfg.OUTPUT_DIR
                    generate_path = generate_path_for_multi_sub_model(cfg,
                                                                   test_fold_prefix=combine_prefix[TEST_FOLD_PREFIX],
                                                                   shuffle_fold_prefix=shuffle_fold_prefix,
                                                                   increment_fold_prefix=increment_fold_prefix,
                                                                   valid_fold_prefix=combine_prefix[VALID_FOLD_PREFIX]
                                                                   )
                    output_dir = os.path.join(output_dir,generate_path)
                    if not os.path.isdir(output_dir):
                        os.makedirs(output_dir)
                    cfg.merge_from_list( ["output_dir",output_dir])
                    print("current output dir : ",output_dir)
                    pl.seed_everything(42)
                    cfg_dict = convert_to_dict(cfg,[])
                    print("cfg dict : ",cfg_dict)
                    print("test LR variable  : ", cfg_dict['OPTIM']['LR'])



                    # generate_predict= False
                    if generate_predict:
                        if data_manager_type == "single_dataset":
                            data_manager = DataManagerV1(cfg)
                        else:
                            print("check multi process")
                            data_manager = MultiDomainDataManagerV1(cfg)

                        data_manager.prepare_data()
                        data_manager.setup()
                        require_parameter = data_manager.get_require_parameter()
                        trainer_model = build_trainer(cfg,require_parameter=require_parameter)
                        monitor = 'val_loss'
                        checkpoint_callback = CustomModelCheckPoint(
                        # checkpoint_callback = ModelCheckpoint(
                            verbose=True,
                            monitor = monitor,
                            dirpath = output_dir,
                            filename = 'checkpoint',
                            save_top_k=1,
                            save_last=True,
                            every_n_val_epochs=1,
                            auto_insert_metric_name = False)

                        # early_stopping = EarlyStopping(monitor='val_loss',patience=10)

                        csv_logger = CustomeCSVLogger(
                            save_dir=output_dir,
                            version=0,
                            experiment_writer=CustomExperimentWriter,
                            # step_key='epoch'
                        )
                        tensorboard_logger = TensorBoardLogger(
                            save_dir=output_dir,
                            version=1
                        )


                        resume_dir = os.path.join(output_dir,'last.ckpt')
                        # resume_dir = os.path.join(output_dir,'best.ckpt')

                        if os.path.exists(resume_dir):
                            resume = resume_dir
                        else:
                            resume = None
                        trainer_lightning = Trainer(
                            gpus=1,
                            default_root_dir=output_dir,
                            benchmark=benchmark,
                            deterministic=deterministic,
                            max_epochs=cfg.OPTIM.MAX_EPOCH,
                            resume_from_checkpoint=resume,
                            multiple_trainloader_mode='max_size_cycle',
                            # callbacks=[early_stopping,checkpoint_callback],
                            callbacks=[checkpoint_callback],
                            logger=[csv_logger,tensorboard_logger],
                            progress_bar_refresh_rate=100,
                            profiler='simple',
                            num_sanity_val_steps=0
                            # stochastic_weight_avg=True
                        )
                        # trainer_lightning.checkpoin
                        model = torch.load(os.path.join(output_dir,'checkpoint.ckpt'),map_location='cuda:0')
                        print("save checkpoint keys : ",model.keys())
                        trainer_model.load_state_dict(model['state_dict'])
                        trainer_model.eval()
                        # test_result = trainer_lightning.test(trainer_model,datamodule=data_manager)[0]
                        # result = trainer_lightning.predict(trainer_model,dataloaders=predict_dataloader)
                        # print("pred result : ",result                                                                                                                                                                     )
                        # test_result.update(combine_prefix)
                        # test_folds_results.append(test_result)



                        # subject_probs_list = []
                        # subject_preds_list = []
                        # for subject_idx in range(len(dataset_B)):
                        #     test_data = dataset_B[subject_idx]


                        probs_list =[]
                        preds_list = []
                        label_list = []
                        if use_assemble_test_dataloader:
                            def parser(test_input):
                                input, label, domain = test_input
                                label = label.numpy()
                                return input,label
                            test_dataloader = data_manager.test_dataloader()
                            # for step, test_input in enumerate(test_dataloader):
                            #     input, label, domain = test_input

                        else:
                            test_data = dataset
                            test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
                            def parser(test_input):
                                input = test_input
                                label = np.array([None])
                                return input,label

                        for step, test_input in enumerate(test_dataloader):
                            input , label = parser(test_input)
                            input = input.float()
                            probs = trainer_model(input)
                            probs = probs.detach().numpy()
                            new_probs = np.zeros_like(probs)
                            new_probs[np.arange(len(probs)), probs.argmax(1)] = 1
                            probs_list.append(probs)
                            preds_list.append(new_probs)
                            label_list.append(label)
                        print(label_list)
                        label_list = np.concatenate(label_list)
                        probs_list = np.concatenate(probs_list)
                        probs_list = np.around(probs_list, decimals=4)
                        preds_list = np.concatenate(preds_list).astype(int)
                        print("len of prob list : ",probs_list)
                        print("len of prob list : ",preds_list)

                        predict_folder = "predict_folder"
                        combine_folder = os.path.join(cfg.OUTPUT_DIR,predict_folder,generate_path)
                        if not os.path.exists(combine_folder):
                            os.makedirs(combine_folder)

                        if use_assemble_test_dataloader:
                            np.savetxt(os.path.join(combine_folder, 'ensemble_label.txt'), label_list, delimiter=',', fmt="%d")
                            np.savetxt(os.path.join(combine_folder, 'ensemble_pred.txt'), preds_list, delimiter=',', fmt="%d")
                            np.savetxt(os.path.join(combine_folder, 'ensemble_prob.txt'), probs_list, delimiter=',', fmt='%1.4f')
                        else:
                            np.savetxt(os.path.join(combine_folder, 'pred.txt'), preds_list, delimiter=',', fmt="%d")
                            np.savetxt(os.path.join(combine_folder, 'prob.txt'), probs_list, delimiter=',', fmt='%1.4f')
                    else:
                        predict_folder = "predict_folder"
                        combine_folder = os.path.join(cfg.OUTPUT_DIR,predict_folder,generate_path)
                        if use_assemble_test_dataloader:
                            pred = np.loadtxt(os.path.join(combine_folder, 'ensemble_pred.txt'), delimiter=',')
                            probs = np.loadtxt(os.path.join(combine_folder, 'ensemble_prob.txt'), delimiter=',')
                            labels = np.loadtxt(os.path.join(combine_folder,'ensemble_label.txt'),delimiter=',')
                            valid_fold_label.append(labels)
                        else:
                            pred = np.loadtxt(os.path.join(combine_folder,'pred.txt'),delimiter=',')
                            probs = np.loadtxt(os.path.join(combine_folder,'prob.txt'),delimiter=',')
                        print("pred : ",pred)
                        valid_fold_preds.append(pred)
                        valid_fold_probs.append(probs)
                test_fold_preds.append(valid_fold_preds)
                test_fold_probs.append(valid_fold_probs)
                test_fold_label.append(valid_fold_label)


            # winner = np.argwhere(current_pred[trial_idx] == np.amax(current_pred[trial_idx]))
            # winner = winner.flatten().tolist()
            # if len(winner) > 1:
            #     probs_options = [current_prob[trial_idx][idx] for idx in winner]
            #     np.amax(probs_options)


                    # subject_probs_list.append(probs_list)
                    # subject_preds_list.append(preds_list)
                # data_result = pd.DataFrame({
                #     "test_fold":combine_prefix[TEST_FOLD_PREFIX],
                #     "valid_fold": combine_prefix[VALID_FOLD_PREFIX],
                #     "pred_list":subject_preds_list,
                #     "prob_list":subject_probs_list
                # })
                # data_result_org = data_result_org.append(data_result)
                # print("update data result table : ",data_result_org)
    # predict_folder = "predict_folder"
    # predict_folder = os.path.join(cfg.OUTPUT_DIR,predict_folder)
    # if not os.path.exists(predict_folder):
    #     os.makedirs(predict_folder)
    # data_result_org.to_csv(os.path.join(predict_folder,"temp_result.csv"),index=False)
    if not generate_predict:
        if not use_assemble_test_dataloader:
            generate_pred_MI_label(test_fold_preds,test_fold_probs,output_dir=cfg.OUTPUT_DIR)
        else:
            generate_assemble_result(test_fold_preds,test_fold_probs,test_fold_label,output_dir=cfg.OUTPUT_DIR)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--main-config-file',
        type=str,
        default='',
        help='path to main config file for full setup'
    )

    parser.add_argument(
        '--gpu-id', type=int, default=0, help='gpu '
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )
    args = parser.parse_args()
    main(args)
