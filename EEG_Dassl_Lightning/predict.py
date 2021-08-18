import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
import pandas as pd
from dassl.utils import (generate_path_for_multi_sub_model)
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from dassl.data.data_manager_v1 import DataManagerV1, MultiDomainDataManagerV1
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from NeurIPS_competition.util.support import (
    expand_data_dim, normalization, generate_common_chan_test_data, load_Cho2017, load_Physionet, load_BCI_IV,
    correct_EEG_data_order, relabel, process_target_data, relabel_target, load_dataset_A, load_dataset_B, modify_data
)

from dassl.data.datasets.data_util import EuclideanAlignment


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
    # allowed to add new keys for config
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

    def __init__(self, log_dir: str, step_key='step'):
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

        last_m = [self.metrics[i] for i in sorted(self.metrics.keys())]
        record = pd.DataFrame(last_m)
        record.to_csv(self.metrics_file_path, index=False)


class CustomeCSVLogger(CSVLogger):
    def __init__(self, save_dir: str,
                 name: Optional[str] = "default",
                 version: Optional[Union[int, str]] = None,
                 prefix: str = '', experiment_writer=None, step_key='step'):
        super().__init__(save_dir=save_dir, name=name, version=version, prefix=prefix)
        if experiment_writer:
            os.makedirs(self.root_dir, exist_ok=True)
            self._experiment = experiment_writer(self.log_dir, step_key=step_key)


class CustomModelCheckPoint(ModelCheckpoint):
    def on_load_checkpoint(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', callback_state: Dict[str, Any]
                           ) -> None:
        """
        Fix a bug in ModelCheckpoint for pytorch lightning
        If we interrupt the model during the training and resume it, the model checkpoint does not load all the neccessary from last.cktp
        which contains information about the best model
        """
        super(CustomModelCheckPoint, self).on_load_checkpoint(trainer, pl_module, callback_state)
        self.kth_best_model_path = callback_state["best_model_path"]
        self.best_k_models[self.best_model_path] = self.best_model_score











from numpy.random import RandomState
import scipy.signal as signal
import copy

class filterBank(object):
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank, fs, filtAllowance=2, axis=-1, filtType='filter'):
        self.filtBank = filtBank
        self.fs = fs
        self.filtAllowance = filtAllowance
        self.axis = axis
        self.filtType = filtType

    def bandpassFilter(self, data, bandFiltCutF, fs, filtAllowance=2, axis=-1, filtType='filter'):
        """
         Filter a signal using cheby2 iir filtering.
        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'
        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30  # stopband attenuation
        aPass = 3  # passband attenuation
        nFreq = fs / 2  # Nyquist frequency

        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
                bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data

        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass = bandFiltCutF[1] / nFreq
            fStop = (bandFiltCutF[1] + filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')

        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass = bandFiltCutF[0] / nFreq
            fStop = (bandFiltCutF[0] - filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')

        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass = (np.array(bandFiltCutF) / nFreq).tolist()
            fStop = [(bandFiltCutF[0] - filtAllowance) / nFreq, (bandFiltCutF[1] + filtAllowance) / nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut

    def __call__(self, data1):

        data = copy.deepcopy(data1)
        d = data
        # d = data['data']

        # initialize output
        out = np.zeros([*d.shape, len(self.filtBank)])
        # print("out shape : ",out.shape)
        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            filter = self.bandpassFilter(d, filtBand, self.fs, self.filtAllowance,
                                         self.axis, self.filtType)
            # print("filter shape : ",filter.shape)
            out[:, :, :, i] = filter

        # remove any redundant 3rd dimension
        if len(self.filtBank) <= 1:
            out = np.squeeze(out, axis=2)

        # data['data'] = torch.from_numpy(out).float()
        return out



def generate_pred_MI_label(test_fold_preds, test_fold_probs, output_dir, predict_folder="predict_folder",
                           relabel=False):
    final_pred = np.zeros(test_fold_preds[0][0].shape)
    final_prob = np.zeros(test_fold_probs[0][0].shape)
    # print("test fold preds : ",test_fold_preds)
    # print("len test fold : ",len(test_fold_preds))
    # print("val fold size : ",len(test_fold_preds[0]))
    # print("val pred size : ",test_fold_preds[0][0].shape)
    # print("org final pred shape : ",final_pred.shape)
    print("generate MI label")
    for test_fold in range(len(test_fold_preds)):
        current_fold_preds = test_fold_preds[test_fold]
        current_fold_probs = test_fold_probs[test_fold]
        for valid_fold in range(len(current_fold_preds)):
            current_valid_pred = current_fold_preds[valid_fold]
            current_valid_prob = current_fold_probs[valid_fold]
            # print("current valid pred shape : ",current_valid_pred.shape)
            # print("final pred shape : ",final_pred.shape)
            final_pred = final_pred + current_valid_pred
            final_prob = final_prob + current_valid_prob


        # valid_fold_pred = test_fold_preds[test_fold]

    # print("result current pred : ", current_pred)
    pred_output = list()
    for trial_idx in range(len(final_pred)):
        trial_pred = final_pred[trial_idx]
        trial_prob = final_prob[trial_idx]
        best_idx = -1
        best_pred = -1
        best_prob = -1
        for idx in range(len(trial_pred)):
            pred = trial_pred[idx]
            prob = trial_prob[idx]
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
    if relabel:
        pred_output = np.array([relabel_target(l) for l in pred_output])
        print("update pred output : ",pred_output)
    combine_folder = os.path.join(output_dir, predict_folder)
    np.savetxt(os.path.join(combine_folder, "pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")


# def generate_pred_MI_label(test_fold_preds, test_fold_probs, output_dir, predict_folder="predict_folder",
#                            relabel=False):
#     for test_fold in range(len(test_fold_preds)):
#         valid_fold_pred = test_fold_preds[test_fold]
#         current_pred = valid_fold_pred[0]
#
#         valid_fold_prob = test_fold_probs[test_fold]
#         current_prob = valid_fold_prob[0]
#         for idx in range(1, len(valid_fold_pred)):
#             current_pred = current_pred + valid_fold_pred[idx]
#             current_prob = current_prob + valid_fold_prob[idx]
#         print("result current pred : ", current_pred)
#         pred_output = list()
#         if not relabel:
#             for trial_idx in range(len(current_pred)):
#                 preds = current_pred[trial_idx]
#                 probs = current_prob[trial_idx]
#                 best_idx = -1
#                 best_pred = -1
#                 best_prob = -1
#                 for idx in range(len(preds)):
#                     pred = preds[idx]
#                     prob = probs[idx]
#                     if pred > best_pred:
#                         best_pred = pred
#                         best_idx = idx
#                         best_prob = prob
#                     elif pred == best_pred:
#                         if prob > best_prob:
#                             best_idx = idx
#                             best_prob = prob
#                 pred_output.append(best_idx)
#             pred_output = np.array(pred_output)
#         else:
#             update_preds = np.zeros((current_pred.shape[0], 3))
#             update_preds[:, :2] = current_pred[:, :2]
#             update_preds[:, 2] = current_pred[:, 2] + current_pred[:, 3]
#             pred_output = np.argmax(update_preds, axis=1)
#         combine_folder = os.path.join(output_dir, predict_folder)
#         np.savetxt(os.path.join(combine_folder, "pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")
#

def generate_assemble_result(test_fold_preds, test_fold_probs, test_fold_labels, output_dir,
                             predict_folder="predict_folder", relabel=False):
    # test_fold_acc = list()
    test_fold_prefix = 'test_fold_'
    test_fold_result = list()
    for test_fold in range(len(test_fold_preds)):
        final_pred = np.zeros(test_fold_preds[0][0].shape)
        final_prob = np.zeros(test_fold_probs[0][0].shape)
        final_label = test_fold_labels[test_fold][0]
        current_fold_preds = test_fold_preds[test_fold]
        current_fold_probs = test_fold_probs[test_fold]
        for valid_fold in range(len(current_fold_preds)):
            current_valid_pred = current_fold_preds[valid_fold]
            current_valid_prob = current_fold_probs[valid_fold]

            final_pred = final_pred + current_valid_pred
            final_prob = final_prob + current_valid_prob

        pred_output = list()
        for trial_idx in range(len(final_pred)):
            trial_pred = final_pred[trial_idx]
            trial_prob = final_prob[trial_idx]
            best_idx = -1
            best_pred = -1
            best_prob = -1
            for idx in range(len(trial_pred)):
                pred = trial_pred[idx]
                prob = trial_prob[idx]
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
        if relabel:
            pred_output = np.array([relabel_target(l) for l in pred_output])
            final_label = np.array([relabel_target(l) for l in final_label])
        acc = np.mean(pred_output == final_label)

        # test_fold_acc.append(acc)
        # print("pred output : ", pred_output)
        # print("current label : ", current_label)
        # print(" valid_fold_label : ", valid_fold_label)
        print("test fold {} has acc {} ".format(test_fold, acc))
        current_test_fold = test_fold_prefix + str(test_fold + 1)
        result = {
            "test_fold": current_test_fold,
            "test_acc": acc
        }
        test_fold_result.append(result)
    result = pd.DataFrame.from_dict(test_fold_result)
    result_output_dir = os.path.join(output_dir, predict_folder)
    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)
    result_filename = 'ensemble_result.xlsx'
    result.to_excel(os.path.join(result_output_dir, result_filename), index=False)

# def generate_assemble_result(test_fold_preds, test_fold_probs, test_fold_labels, output_dir,
#                              predict_folder="predict_folder", relabel=False):
#     # test_fold_acc = list()
#     test_fold_prefix = 'test_fold_'
#     test_fold_result = list()
#     for test_fold in range(len(test_fold_preds)):
#         valid_fold_pred = test_fold_preds[test_fold]
#         current_pred = valid_fold_pred[0]
#
#         valid_fold_prob = test_fold_probs[test_fold]
#         current_prob = valid_fold_prob[0]
#
#         valid_fold_label = test_fold_labels[test_fold]
#         print("valid fold label : ",valid_fold_label)
#         current_label = valid_fold_label[0]
#
#         print("current label : ",current_label)
#         for idx in range(1, len(valid_fold_pred)):
#             # check valid fold result
#             print("current valid fold : ", test_fold)
#             temp_pred = np.argmax(current_pred)
#             print("temp pred ", temp_pred[:10])
#             print("current label : ", valid_fold_prob[idx][:10])
#             print("acc : ", (temp_pred == valid_fold_prob[idx]))
#
#             current_pred = current_pred + valid_fold_pred[idx]
#             current_prob = current_prob + valid_fold_prob[idx]
#         print("result current pred : ", current_pred)
#         pred_output = list()
#         if not relabel:
#             for trial_idx in range(len(current_pred)):
#                 # print("trial {} has pred {} ".format(trial_idx,current_pred[trial_idx]))
#                 # print("trial {} has probs {} ".format(trial_idx,current_prob[trial_idx]))
#
#                 preds = current_pred[trial_idx]
#                 probs = current_prob[trial_idx]
#                 best_idx = -1
#                 best_pred = -1
#                 best_prob = -1
#                 for idx in range(len(preds)):
#                     pred = preds[idx]
#                     prob = probs[idx]
#                     if pred > best_pred:
#                         best_pred = pred
#                         best_idx = idx
#                         best_prob = prob
#                     elif pred == best_pred:
#                         if prob > best_prob:
#                             best_idx = idx
#                             best_prob = prob
#                 pred_output.append(best_idx)
#             pred_output = np.array(pred_output)
#             acc = np.mean(pred_output == current_label)
#         if relabel:
#             update_preds = np.zeros((current_pred.shape[0], 3))
#             update_preds[:, :2] = current_pred[:, :2]
#             update_preds[:, 2] = current_pred[:, 2] + current_pred[:, 3]
#             pred_output = np.argmax(update_preds, axis=1)
#             update_label = list()
#             for trial_idx in range(len(current_label)):
#                 label = current_label[trial_idx]
#                 if label == 2 or label == 3:
#                     update_label.append(2)
#                 else:
#                     update_label.append(label)
#             update_label = np.array(update_label)
#             acc = np.mean(pred_output == update_label)
#             current_label = update_label
#         # test_fold_acc.append(acc)
#         print("pred output : ", pred_output)
#         print("current label : ", current_label)
#         print(" valid_fold_label : ", valid_fold_label)
#         print("test fold {} has acc {} ".format(test_fold, acc))
#         # print("just arg pred output : ",np.argmax(current_pred,axis=1))
#         # combine_folder = os.path.join(output_dir, predict_folder)
#         # np.savetxt(os.path.join(combine_folder,"pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")
#         current_test_fold = test_fold_prefix + str(test_fold + 1)
#         result = {
#             "test_fold": current_test_fold,
#             "test_acc": acc
#         }
#         test_fold_result.append(result)
#     result = pd.DataFrame.from_dict(test_fold_result)
#     result_output_dir = os.path.join(output_dir, predict_folder)
#     if not os.path.isdir(result_output_dir):
#         os.makedirs(result_output_dir)
#     result_filename = 'ensemble_result.xlsx'
#     result.to_excel(os.path.join(result_output_dir, result_filename), index=False)
from scipy.io import loadmat


def load_test_data_from_file(provide_path,dataset_type):
    temp = loadmat(provide_path)
    datasets = temp['datasets'][0]
    target_dataset = None
    list_r_op =  None
    for dataset in datasets:
        dataset = dataset[0][0]
        dataset_name = dataset['dataset_name'][0]
        if dataset_name == dataset_type:
            target_dataset = dataset

    data = target_dataset['data'].astype(np.float32)

    if dataset_type=="dataset_A":
        potential_r_op="dataset_A_r_op.mat"
    elif dataset_type=="dataset_B":
        potential_r_op="dataset_B_r_op.mat"
    # provide_path = provide_path.split("\\")[:-1]
    # provide_path = "\\".join(provide_path)
    provide_path = provide_path.split("/")[:-1]
    provide_path = "/".join(provide_path)
    # print("provide path : ",provide_path)
    exist_r_op_file = os.path.join(provide_path,potential_r_op)
    print("current r_op file : ",exist_r_op_file)
    if os.path.exists(exist_r_op_file):
        print("path {} exist ".format(exist_r_op_file))
        temp = loadmat(exist_r_op_file)
        dataset = temp['datasets'][0]
        dataset = list(dataset)
        dataset = dataset[0][0]
        # print("dataset : ",dataset)
        if 'r_op_list' in list(dataset.dtype.names):
            r_op = dataset['r_op_list'][0]
            list_r_op = np.array(r_op).astype(np.float32)
            print("use the r-op list")
    return data,list_r_op

def print_info(source_data,dataset_name):
    print("current dataset {}".format(dataset_name))
    for subject_idx in range(len(source_data)):
        print("source_data subject_idx {} has shape : {}, with range scale ({},{}) ".format(
            subject_idx, source_data[subject_idx].shape,
            np.max(source_data[subject_idx]), np.min(source_data[subject_idx])))

def get_test_data(dataset_type, norm, provide_data_path = None,use_filter_bank=False, freq_interval=4, EA=False):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cuda = torch.cuda.is_available()
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    rng = RandomState(seed)
    list_r_op = None
    # get correct chans order
    if provide_data_path is None:
        target_channels = generate_common_chan_test_data()
        fmin, fmax = 4, 36
        epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin, fmax=fmax, selected_chans=target_channels,
                                                        subjects=[1])
        print("cho2017 current chans : ", epoch_X_src1.ch_names)
        print("size : ", len(epoch_X_src1.ch_names))
        target_channels = epoch_X_src1.ch_names
        if dataset_type == 'dataset_B':
            test_data = load_dataset_B(train=False, norm=norm, selected_chans=target_channels)
            n_subjects = 3
        else:
            test_data = load_dataset_A(train=False, norm=norm, selected_chans=target_channels)
            n_subjects = 2
        # if EA:
        print("{} subjects to split : ".format(n_subjects))
        test_data = np.split(test_data,n_subjects)
    else:
        print("load test data from file ")
        test_data,list_r_op = load_test_data_from_file(provide_data_path,dataset_type=dataset_type)
        if list_r_op is None:
            print("generate new list r op")
        else:
            print("use r_op")
    if EA:
        test_EA = EuclideanAlignment(list_r_op=list_r_op)
        test_data = test_EA.convert_subjects_data_with_EA(test_data)
        print("load test predict data ------------------")
    print_info(test_data,dataset_type)
    test_data = np.concatenate(test_data)
    if norm:
        test_data = normalization(test_data)
    test_data = expand_data_dim(test_data)
    print("data shape before predict : ",test_data.shape)
    return test_data


def main(args):
    benchmark = False
    deterministic = False  # this can help to reproduce the result
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        print("use determinstic ")
        benchmark = False
        deterministic = True  # this can help to reproduce the result

    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    print("Experiment setup ...")
    # cross test fold setup
    N_TEST_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.N_TEST_FOLDS
    START_TEST_FOLD = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.START_TEST_FOLD
    END_TEST_FOLD = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.END_TEST_FOLD
    TEST_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_FOLD_PREFIX

    # shuffle fold
    SHUFFLE_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_FOLD_PREFIX
    N_SHUFFLE_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS
    START_SHUFFLE_FOLD = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.START_SHUFFLE_FOLD
    END_SHUFFLE_FOLD = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.END_SHUFFLE_FOLD
    USE_SHUFFLE = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS > 1

    # increment fold setup
    # conduct incremental subject experiments
    N_INCREMENT_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.N_INCREMENT_FOLDS
    INCREMENT_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.INCREMENT_FOLD_PREFIX
    START_INCREMENT_FOLD = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_INCREMENT_FOLD
    END_INCREMENT_FOLD = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.END_INCREMENT_FOLD
    USE_INCREMENT = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_NUM_TRAIN_SUGJECT > 0

    # valid fold setup
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
    EA = cfg.DATAMANAGER.DATASET.USE_Euclidean_Aligment
    use_filter = cfg.DATAMANAGER.DATASET.FILTERBANK.USE_FILTERBANK
    diff = cfg.DATAMANAGER.DATASET.FILTERBANK.freq_interval
    print("use filter bank {} with diff interval {} ".format(use_filter,diff))
    print("use cross channel norm : ",norm)
    generate_predict = args.generate_predict
    use_assemble_test_dataloader = args.use_assemble_test_dataloader
    relabel = args.relabel
    # relabel = False
    print("generate predict : ", generate_predict)
    dataset_type = cfg.DATAMANAGER.DATASET.SETUP.TARGET_DATASET_NAME

    test_file_path = args.test_data if args.test_data != '' else None
    if generate_predict and not use_assemble_test_dataloader:
        dataset = get_test_data(dataset_type, norm, use_filter_bank=use_filter, freq_interval=diff,EA=EA,provide_data_path=test_file_path)

    combine_prefix = dict()
    test_fold_preds = list()
    test_fold_probs = list()
    test_fold_label = list()
    for current_test_fold in range(START_TEST_FOLD, END_TEST_FOLD + 1):
        cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD['CURRENT_TEST_FOLD'] = current_test_fold
        combine_prefix[TEST_FOLD_PREFIX] = TEST_FOLD_PREFIX + "_" + str(current_test_fold)
        for current_shuffle_fold in range(START_SHUFFLE_FOLD, END_SHUFFLE_FOLD + 1):
            cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD['CURRENT_SHUFFLE_FOLD'] = current_shuffle_fold
            shuffle_fold_prefix = ""
            if USE_SHUFFLE:
                shuffle_fold_prefix = SHUFFLE_FOLD_PREFIX + "_" + str(current_shuffle_fold)
                combine_prefix[SHUFFLE_FOLD_PREFIX] = shuffle_fold_prefix
            for current_increment_fold in range(START_INCREMENT_FOLD, END_INCREMENT_FOLD + 1):
                cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD['CURRENT_INCREMENT_FOLD'] = current_increment_fold
                increment_fold_prefix = ""
                if USE_INCREMENT:
                    increment_fold_prefix = INCREMENT_FOLD_PREFIX + "_" + str(current_increment_fold)
                    combine_prefix[INCREMENT_FOLD_PREFIX] = increment_fold_prefix

                valid_fold_preds = list()
                valid_fold_probs = list()
                valid_fold_label = list()
                for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD + 1):
                    combine_prefix[VALID_FOLD_PREFIX] = VALID_FOLD_PREFIX + "_" + str(current_valid_fold)
                    cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD['CURRENT_VALID_FOLD'] = current_valid_fold

                    output_dir = cfg.OUTPUT_DIR
                    generate_path = generate_path_for_multi_sub_model(cfg,
                                                                      test_fold_prefix=combine_prefix[TEST_FOLD_PREFIX],
                                                                      shuffle_fold_prefix=shuffle_fold_prefix,
                                                                      increment_fold_prefix=increment_fold_prefix,
                                                                      valid_fold_prefix=combine_prefix[
                                                                          VALID_FOLD_PREFIX]
                                                                      )
                    output_dir = os.path.join(output_dir, generate_path)
                    if not os.path.isdir(output_dir):
                        os.makedirs(output_dir)
                    cfg.merge_from_list(["output_dir", output_dir])
                    print("current output dir : ", output_dir)
                    pl.seed_everything(42)
                    cfg_dict = convert_to_dict(cfg, [])
                    print("cfg dict : ", cfg_dict)

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
                        trainer_model = build_trainer(cfg, require_parameter=require_parameter)
                        monitor = 'val_loss'
                        checkpoint_callback = CustomModelCheckPoint(
                            # checkpoint_callback = ModelCheckpoint(
                            verbose=True,
                            monitor=monitor,
                            dirpath=output_dir,
                            filename='checkpoint',
                            save_top_k=1,
                            save_last=True,
                            every_n_val_epochs=1,
                            auto_insert_metric_name=False)

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

                        resume_dir = os.path.join(output_dir, 'last.ckpt')
                        # resume_dir = os.path.join(output_dir,'best.ckpt')

                        if os.path.exists(resume_dir):
                            resume = resume_dir
                        else:
                            resume = None
                        # trainer_lightning.checkpoin
                        model = torch.load(os.path.join(output_dir, 'checkpoint.ckpt'), map_location='cuda:0')
                        print("save checkpoint keys : ", model.keys())
                        trainer_model.load_state_dict(model['state_dict'])
                        trainer_model.eval()

                        probs_list = []
                        preds_list = []
                        label_list = []
                        if use_assemble_test_dataloader:
                            def parser(test_input):
                                input, label, domain = test_input
                                label = label.numpy()
                                return input, label

                            test_dataloader = data_manager.test_dataloader()

                        else:
                            test_data = dataset
                            test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

                            def parser(test_input):
                                input = test_input
                                label = np.array([None])
                                return input, label

                        for step, test_input in enumerate(test_dataloader):
                            input, label = parser(test_input)
                            input = input.float()
                            probs = trainer_model(input)
                            probs = probs.detach().numpy()
                            new_probs = np.zeros_like(probs)
                            new_probs[np.arange(len(probs)), probs.argmax(1)] = 1
                            probs_list.append(probs)
                            preds_list.append(new_probs)
                            label_list.append(label)
                        # print(label_list)
                        label_list = np.concatenate(label_list)
                        probs_list = np.concatenate(probs_list)
                        probs_list = np.around(probs_list, decimals=4)
                        preds_list = np.concatenate(preds_list).astype(int)
                        # print("len of prob list : ",probs_list)
                        # print("len of prob list : ",preds_list)

                        predict_folder = "predict_folder"
                        combine_folder = os.path.join(cfg.OUTPUT_DIR, predict_folder, generate_path)
                        if not os.path.exists(combine_folder):
                            os.makedirs(combine_folder)

                        if use_assemble_test_dataloader:
                            np.savetxt(os.path.join(combine_folder, 'ensemble_label.txt'), label_list, delimiter=',',
                                       fmt="%d")
                            np.savetxt(os.path.join(combine_folder, 'ensemble_pred.txt'), preds_list, delimiter=',',
                                       fmt="%d")
                            np.savetxt(os.path.join(combine_folder, 'ensemble_prob.txt'), probs_list, delimiter=',',
                                       fmt='%1.4f')
                        else:
                            np.savetxt(os.path.join(combine_folder, 'pred.txt'), preds_list, delimiter=',', fmt="%d")
                            np.savetxt(os.path.join(combine_folder, 'prob.txt'), probs_list, delimiter=',', fmt='%1.4f')
                    else:
                        predict_folder = "predict_folder"
                        combine_folder = os.path.join(cfg.OUTPUT_DIR, predict_folder, generate_path)
                        if use_assemble_test_dataloader:
                            pred = np.loadtxt(os.path.join(combine_folder, 'ensemble_pred.txt'), delimiter=',')
                            probs = np.loadtxt(os.path.join(combine_folder, 'ensemble_prob.txt'), delimiter=',')
                            labels = np.loadtxt(os.path.join(combine_folder, 'ensemble_label.txt'), delimiter=',')
                            valid_fold_label.append(labels)
                        else:
                            pred = np.loadtxt(os.path.join(combine_folder, 'pred.txt'), delimiter=',')
                            probs = np.loadtxt(os.path.join(combine_folder, 'prob.txt'), delimiter=',')
                        # print("pred : ",pred)
                        valid_fold_preds.append(pred)
                        valid_fold_probs.append(probs)
                test_fold_preds.append(valid_fold_preds)
                test_fold_probs.append(valid_fold_probs)
                test_fold_label.append(valid_fold_label)

    if not generate_predict:
        if not use_assemble_test_dataloader:
            generate_pred_MI_label(test_fold_preds, test_fold_probs, output_dir=cfg.OUTPUT_DIR, relabel=relabel)
        else:
            generate_assemble_result(test_fold_preds, test_fold_probs, test_fold_label, output_dir=cfg.OUTPUT_DIR,
                                     relabel=relabel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument('--test-data', type=str, default='', help='path to test data')
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
        '--generate-predict',
        # type=bool,
        # default=True,
        action='store_true',
        help='generate predict result '
    )
    parser.add_argument(
        '--use-assemble-test-dataloader',
        # type=bool,
        # default=True,
        action='store_true',
        help='use ensemble of multi model to make prediction'
    )

    parser.add_argument(
        '--relabel',
        # type=bool,
        # default=True,
        action='store_true',
        help='relabel predict to be 3 categories'
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
