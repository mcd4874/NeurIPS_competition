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

import pytorch_lightning as pl
from typing import Any, Dict, Optional, Union
from collections import defaultdict
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn

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

    # if args.resume:
    #     cfg.RESUME = args.resume
    #
    # if args.seed:
    #     cfg.SEED = args.seed


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
    print("current result before convert excel : ",results)
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



def trainer_setup(output_dir,cfg,benchmark=False,deterministic=False,seed=42,update_dataset=None):

    # experiments = generate_setup(org_cfg)
    # for experiment in experiments:
    # output_dir = experiment["generate_sub_exp_path"]
    # combine_prefix = experiment["sub_exp_prefix"]
    # cfg = experiment["cfg"]


    pl.seed_everything(seed)
    data_manager_type = cfg.DATAMANAGER.MANAGER_TYPE

    if data_manager_type == "single_dataset":
        data_manager = DataManagerV1(cfg)
    elif data_manager_type == "multi_datasetV2":
        print("use data manager for domain adaptation")
        data_manager = MultiDomainDataManagerV2(cfg)
    else:
        print("check multi process")
        data_manager = MultiDomainDataManagerV1(cfg)
    if update_dataset:

        data_manager.update_dataset(dataset=update_dataset)
        print("update data manager with customize dataset")
    data_manager.prepare_data()
    data_manager.setup()
    require_parameter = data_manager.get_require_parameter()
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

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

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
        logger=[csv_logger, tensorboard_logger],
        progress_bar_refresh_rate=cfg.LIGHTNING_TRAINER.progress_bar_refresh_rate,
        profiler=cfg.LIGHTNING_TRAINER.profiler,
        num_sanity_val_steps=cfg.LIGHTNING_TRAINER.num_sanity_val_steps,
        stochastic_weight_avg=cfg.LIGHTNING_TRAINER.stochastic_weight_avg

    )

    trainer_model = build_trainer(cfg, require_parameter=require_parameter)

    return trainer_model,trainer_lightning,data_manager


def generate_setup(org_cfg):
    cfg= org_cfg.clone()
    experiments = list()

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

    # generate_detail = True
    # test_folds_results = []
    # test_fold_detail_results = []
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
                    combine_prefix[VALID_FOLD_PREFIX] = VALID_FOLD_PREFIX + "_" + str(current_valid_fold)
                    cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD['CURRENT_VALID_FOLD'] = current_valid_fold

                    output_dir = cfg.OUTPUT_DIR
                    generate_path = generate_path_for_multi_sub_model(cfg,
                                                                      test_fold_prefix=combine_prefix[
                                                                          TEST_FOLD_PREFIX],
                                                                      shuffle_fold_prefix=shuffle_fold_prefix,
                                                                      increment_fold_prefix=increment_fold_prefix,
                                                                      valid_fold_prefix=combine_prefix[
                                                                          VALID_FOLD_PREFIX])
                    output_dir = os.path.join(output_dir, generate_path)
                    if not os.path.isdir(output_dir):
                        os.makedirs(output_dir)
                    cfg.merge_from_list(["output_dir", output_dir])

                    experiments.append({
                        "sub_exp_prefix":combine_prefix.copy(),
                        "generate_sub_exp_path":generate_path,
                        "cfg":cfg.clone(),
                        "output_dir":output_dir
                    })
    return experiments

def train_full_experiment(cfg,experiments_setup,update_dataset_func=None,benchmark=False,deterministic=True,eval=False,use_best_model_pretrain=True,pretrain_dir="",seed=42):
    print("cfg file setup : ",cfg)
    pl.seed_everything(seed)
# def train_full_experiment(cfg,experiments_setup,benchmark=False,deterministic=True,eval=False,use_best_model_pretrain=True,pretrain_dir="",seed=42):
    fold_results = list()
    common_dir = cfg.OUTPUT_DIR
    for experiment in experiments_setup:
        sub_exp_path = experiment["generate_sub_exp_path"]
        output_dir = experiment["output_dir"]
        combine_prefix = experiment["sub_exp_prefix"]
        cfg = experiment["cfg"]
        if update_dataset_func is not None:
            update_dataset = update_dataset_func(cfg, experiments_setup)
        else:
            update_dataset = None
        # trainer_model,trainer_lightning,data_manager = trainer_setup(output_dir,cfg,benchmark,deterministic,seed=seed)
        trainer_model,trainer_lightning,data_manager = trainer_setup(output_dir,cfg,benchmark,deterministic,seed=seed,update_dataset=update_dataset)


        if eval:
            model_state = torch.load(os.path.join(output_dir, 'checkpoint.ckpt'), map_location='cuda:0')
            print("save checkpoint keys : ", model_state.keys())
            # print("state dict : ", model['state_dict'])
            best_epoch = model_state['epoch']
            trainer_model.load_state_dict(model_state['state_dict'])
            test_result = trainer_lightning.test(trainer_model, datamodule=data_manager)[0]
            if len(test_result) > 1 and isinstance(test_result, list):
                test_result = test_result[0]
            print("test result : ", test_result)
            test_result.update(combine_prefix)
            test_result.update({
                'epoch':best_epoch
            })
            fold_results.append(test_result)
        else:
            if pretrain_dir != "":
                current_pretrain_dir = os.path.join(pretrain_dir, sub_exp_path)
                print("current potential pretrain dir : ",current_pretrain_dir)
                if os.path.exists(current_pretrain_dir):
                    print("path exist")
                    if use_best_model_pretrain:
                        current_pretrain_file = os.path.join(current_pretrain_dir, 'checkpoint.ckpt')
                    else:
                        current_pretrain_file = os.path.join(current_pretrain_dir, 'last.ckpt')
                    pretrain_model_state = torch.load(current_pretrain_file, map_location='cuda:0')
                    trainer_model.load_state_dict(pretrain_model_state['state_dict'])
                    # require_parameter = data_manager.get_require_parameter()
                    # trainer_model.load_from_checkpoint(pretrain_dir,cfg=cfg,require_parameter=require_parameter)
                    print("load pretrain model from {}".format(current_pretrain_file))
            trainer_lightning.fit(trainer_model, datamodule=data_manager)

    if eval:
        generate_excel_report(fold_results, common_dir, result_folder='result_folder')
        generate_model_info_config(cfg, common_dir, result_folder='result_folder')
