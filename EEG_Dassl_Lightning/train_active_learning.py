import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
import pandas as pd
import json
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, resume_from_checkpoint, load_pretrained_weights, generate_path_for_multi_sub_model
)
from pytorch_lightning import LightningModule,Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger,TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger,ExperimentWriter
from pytorch_lightning.callbacks import EarlyStopping
from dassl.data.data_manager_v1 import DataManagerV1, MultiDomainDataManagerV1,MultiDomainDataManagerV2
from dassl.data.datasets.build import build_dataset
from NeurIPS_competition.util.support import (
    expand_data_dim, normalization, generate_common_chan_test_data, load_Cho2017, load_Physionet, load_BCI_IV,
    correct_EEG_data_order, relabel, process_target_data, relabel_target, load_dataset_A, load_dataset_B, modify_data,reformat
)

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

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed


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


def generate_excel_report(results,output_dir,result_folder="result_folder"):
    result_filename = 'model_result.xlsx'
    result = pd.DataFrame.from_dict(results)

    result_output_dir = os.path.join(output_dir, result_folder)
    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)

    result.to_excel(os.path.join(result_output_dir, result_filename), index=False)

def generate_model_info_config(cfg,output_dir,result_folder="result_folder"):
    model_info = {
        "EXTRA_FIELDS": cfg.EXTRA_FIELDS
    }

    info_filename = 'model_info.json'
    result_output_dir = os.path.join(output_dir, result_folder)

    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)
    full_path = os.path.join(result_output_dir, info_filename)
    with open(full_path, "w") as outfile:
        json.dump(model_info, outfile,indent=4)



def generate_detail_report(list_results,output_dir,cfg):
    result_filename = 'model_detail_result.xlsx'
    result_folder = cfg.TRAIN_EVAL_PROCEDURE.RESULT_FOLDER

    result_output_dir = os.path.join(output_dir, result_folder)
    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)
    result = pd.DataFrame.from_dict(list_results)
    result.to_excel(os.path.join(result_output_dir, result_filename), index=False)
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

from util_AL import load_test_data_from_file,load_prediction


# def load_test_data(dataset_type="dataset_A"):

def update_datamanager_with_active_learning(cfg, test_file_path, data_manager):
    dataset_type = cfg.DATAMANAGER.DATASET.SETUP.TARGET_DATASET_NAME
    test_file_path = args.test_data if args.test_data != '' else None

    model_list_prefix = [
        # 'adaptationV1',
        # 'dannV1',
        'mcdV1',
        # 'addaV1',
        # 'SRDA'
    ]
    target_dataset_list_prefix = [
        dataset_type
    ]
    augmentation_list_prefix = [
        'no_aug',
    ]
    norm_list_prefix = [
        'no_norm',
    ]
    experiment_type = ["final_result_14_3_1"]
    # common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\sub\\{}\\{}\\{}\\{}\\model"
    # common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\sub\\{}\\{}\\{}\\{}\\model"
    common_path = "/data1/wduong_experiment_data/EEG_Dassl_Lightning/NeurIPS_competition/{}/sub/{}/{}/{}/{}/model"

    test_data, test_label, test_meta = load_test_data_from_file(test_file_path, dataset_type)
    query_pred_label = load_prediction(dataset_type, common_path, experiment_type, model_list_prefix,
                                       augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
    test_data, query_pred_label, test_meta = reformat(test_data, query_pred_label, test_meta)

    dataset = build_dataset(cfg)

    train_x_data, train_x_label = dataset.train_x
    update_x_data = list()
    update_x_label = list()
    for subject_idx in range(len(train_x_data)):
        subject_train_x_data, subject_train_x_label = train_x_data[subject_idx], train_x_label[subject_idx]
        subject_test_data, subject_query_pred_label = test_data[subject_idx], query_pred_label[subject_idx]
        # print("subject query label : ",subject_query_pred_label)
        update_test_data, update_query_label = list(), list()
        for trial in range(len(subject_test_data)):
            trial_test_data = subject_test_data[trial]
            trial_test_label = subject_query_pred_label[trial]
            if trial_test_label > -1:
                update_test_data.append(trial_test_data)
                update_query_label.append(trial_test_label)
        update_test_data = np.array(update_test_data)
        update_query_label = np.array(update_query_label)

        # print("update query label : ",update_query_label)
        # print("shape test data {}, shape test label {}".format(update_test_data.shape,update_query_label.shape))
        # print("shape train data {}, shape train label {}".format(subject_train_x_data.shape,subject_train_x_label.shape))

        combine_subject_data = np.concatenate([subject_train_x_data, update_test_data])
        combine_subject_label = np.concatenate([subject_train_x_label, update_query_label])
        update_x_data.append(combine_subject_data)
        update_x_label.append(combine_subject_label)
    dataset.set_train_x((update_x_data, update_x_label))
    data_manager.update_dataset(dataset=dataset)

def main(args):
    benchmark = False
    deterministic = False  # this can help to reproduce the result
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
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

    if args.train_k_folds:
        generate_detail = True
        test_folds_results = []
        test_fold_detail_results = []
        combine_prefix = dict()

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

                for current_test_fold in range(START_TEST_FOLD, END_TEST_FOLD + 1):
                    cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD['CURRENT_TEST_FOLD'] = current_test_fold
                    combine_prefix[TEST_FOLD_PREFIX] = TEST_FOLD_PREFIX + "_" + str(current_test_fold)
                    for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD + 1):
                        combine_prefix[VALID_FOLD_PREFIX] = VALID_FOLD_PREFIX+"_"+str(current_valid_fold)
                        cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD['CURRENT_VALID_FOLD'] = current_valid_fold

                        output_dir = cfg.OUTPUT_DIR
                        generate_path = generate_path_for_multi_sub_model(cfg,
                                                                          test_fold_prefix=combine_prefix[TEST_FOLD_PREFIX],
                                                                          shuffle_fold_prefix=shuffle_fold_prefix,
                                                                          increment_fold_prefix=increment_fold_prefix,
                                                                          valid_fold_prefix=combine_prefix[VALID_FOLD_PREFIX])
                        output_dir = os.path.join(output_dir, generate_path)
                        if not os.path.isdir(output_dir):
                            os.makedirs(output_dir)
                        cfg.merge_from_list( ["output_dir",output_dir])
                        print("current output dir : ",output_dir)
                        pl.seed_everything(42)
                        cfg_dict = convert_to_dict(cfg,[])
                        print("cfg dict : ",cfg_dict)




                        if data_manager_type == "single_dataset":
                            data_manager = DataManagerV1(cfg)
                        elif data_manager_type == "multi_datasetV2":
                            print("use data manager for domain adaptation")
                            data_manager = MultiDomainDataManagerV2(cfg)
                        else:
                            print("check multi process")
                            data_manager = MultiDomainDataManagerV1(cfg)


                        #load up
                        test_file_path = args.test_data if args.test_data != '' else None

                        update_datamanager_with_active_learning(cfg,test_file_path,data_manager)





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

                        early_stopping = EarlyStopping(monitor='val_loss',patience=10)

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
                            multiple_trainloader_mode=cfg.LIGHTNING_TRAINER.multiple_trainloader_mode,
                            # callbacks=[early_stopping,checkpoint_callback],
                            callbacks=[checkpoint_callback],
                            logger=[csv_logger,tensorboard_logger],
                            progress_bar_refresh_rate=cfg.LIGHTNING_TRAINER.progress_bar_refresh_rate,
                            profiler=cfg.LIGHTNING_TRAINER.profiler,
                            num_sanity_val_steps=cfg.LIGHTNING_TRAINER.num_sanity_val_steps,
                            stochastic_weight_avg=cfg.LIGHTNING_TRAINER.stochastic_weight_avg

                        )


                        if not args.eval_only:
                            trainer_lightning.fit(trainer_model,datamodule=data_manager)

                        else:
                            # trainer_lightning.checkpoin
                            model = torch.load(os.path.join(output_dir,'checkpoint.ckpt'),map_location='cuda:0')
                            # model = torch.load(os.path.join(output_dir,'last.ckpt'))

                            print("save checkpoint keys : ",model.keys())
                            trainer_model.load_state_dict(model['state_dict'])
                            # model = trainer_model.load_from_checkpoint(checkpoint_path=os.path.join(output_dir,'best.ckpt'))
                            # print("load model ",model)
                            # model = pl.load_from_checkpoint(checkpoint_path=os.path.join(output_dir,'best.ckpt'))
                            test_result = trainer_lightning.test(trainer_model,datamodule=data_manager)[0]
                            print("test result : ",test_result                                                                                                                                                                     )
                            test_result.update(combine_prefix)
                            test_folds_results.append(test_result)
            if args.eval_only:
                generate_excel_report(test_folds_results, args.output_dir, result_folder='result_folder')
                generate_model_info_config(cfg, args.output_dir,result_folder='result_folder')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument('--test-data', type=str, default='', help='path to test data')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--main-config-file',
        type=str,
        default='',
        help='path to main config file for full setup'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )

    parser.add_argument(
        '--pretrain-dir',
        type=str,
        default='',
        help='load pre-train model from this directory'
    )

    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        '--train-k-folds', action='store_true', help='call trainer.train()'
    )
    parser.add_argument(
        '--tune-models', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        '--param-tuning-file',
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
