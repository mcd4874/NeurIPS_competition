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
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    # if args.source_domains:
    #     cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    # if args.target_domains:
    #     cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.train_k_folds:
        cfg.DATASET.TRAIN_K_FOLDS = args.train_k_folds

    # if args.transforms:
    #     cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head




def setup_cfg(args):
    cfg = get_cfg_default()
    reset_cfg(cfg, args)
    #allowed to add new keys for config
    cfg.set_new_allowed(True)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.main_config_file:
        cfg.merge_from_file(args.main_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
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



def save_cfg(cfg):
    import os
    save_path = cfg.OUTPUT_DIR
    save_file_path = os.path.join(save_path,'config.yaml')
    self_as_dict = convert_to_dict(cfg, [])
    custom_save_config = dict(
        {
            "DATASET":self_as_dict["DATASET"],
            "DATALOADER":self_as_dict["DATALOADER"],
            "OPTIM": self_as_dict["OPTIM"],
            "MODEL": self_as_dict["MODEL"],
            "TRAINER": self_as_dict["TRAINER"],
        }
    )
    import yaml
    with open(save_file_path, 'w') as output:
        yaml.safe_dump(custom_save_config,output)
    # print("dump dataset info : ",self_as_dict)


def generate_excel_report(results,output_dir,cfg):
    result_filename = 'model_result..xlsx'
    result_folder = cfg.TRAIN_EVAL_PROCEDURE.RESULT_FOLDER
    result = pd.DataFrame.from_dict(results)

    result_output_dir = os.path.join(output_dir, result_folder)
    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)

    result.to_excel(os.path.join(result_output_dir, result_filename), index=False)

def generate_model_info_config(cfg,output_dir):
    model_info = {
        "BACKBONE_NAME": [cfg.MODEL.BACKBONE.NAME],
        "TRAINER_NAME": [cfg.TRAINER.NAME],
        "DATASET_NAME": [cfg.DATASET.NAME],
        "DATASET_DIR": [cfg.DATASET.DIR],
        "DATASET_TEST_NUM_K_FOLDS": [cfg.DATASET.K_FOLD_TEST],
        "DATASET_VALID_NUM_K_FOLDS": [cfg.DATASET.K_FOLD],
        "DATASET_SPLIT": ['cross_subject ' if cfg.DATASET.CROSS_SUBJECTS else 'within_subject'],
        "DATALOADER_SAMPLER": [cfg.DATALOADER.TRAIN_X.SAMPLER],
        "OPTIM_NAME": [cfg.OPTIM.NAME],
        "OPTIM_LR_SCHEDULER": [cfg.OPTIM.LR_SCHEDULER],
        "EXTRA_FIELDS": cfg.EXTRA_FIELDS
    }

    info_filename = 'model_info.json'
    result_folder = cfg.TRAIN_EVAL_PROCEDURE.RESULT_FOLDER
    result_output_dir = os.path.join(output_dir, result_folder)

    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)
    full_path = os.path.join(result_output_dir, info_filename)
    with open(full_path, "w") as outfile:
        json.dump(model_info, outfile,indent=4)

    # model_info = pd.DataFrame.from_dict(model_info)
    # model_info.to_csv(), index=False)


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

            # print("resume history : ",self.metrics)
            # self.metrics = defaultdict(list)
            # for key, val in history.items():
            #    self.metrics[key] = history[key]
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

        # print("current metrics dict : ", metrics_dict)
        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics[self.step_key] = step
        # self.metrics.append(metrics)
        # self.metrics.update(metrics)
        # for k, v in metrics.items():
        #     print("key {} - val {}".format(k,v))
        #     self.metrics[k].add(v)
        self.metrics[step].update(metrics)
    def save(self) -> None:
        """Save recorded hparams and metrics into files"""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)

        if not self.metrics:
           return

        # last_m = {}
        # print("before save list metrics : ", self.metrics)
        # for m in self.metrics:
        #     last_m.update(m)
        # last_m = self.metrics
        # metrics_keys = list(last_m.keys())
        # print("after update before save : ", self.metrics)

        # for i in sorted(self.metrics.keys()):
        last_m =  [self.metrics[i] for i in sorted(self.metrics.keys())]
        record = pd.DataFrame(last_m)
        # for key, val in self.metrics.items():
        #     self.record[key].append(val)
        # record = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.metrics.items()]))
        # record = pd.DataFrame.from_dict(self.metrics)
        record.to_csv(self.metrics_file_path, index=False)

        # with io.open(self.metrics_file_path, 'w', newline='') as f:
        #     self.writer = csv.DictWriter(f, fieldnames=metrics_keys)
        #     self.writer.writeheader()
        #     self.writer.writerows(self.metrics)

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



def main(args):
    gpu_id = args.gpu_id
    benchmark = False
    deterministic = False  # this can help to reproduce the result
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        print("use determinstic ")
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.benchmark = True
        benchmark = False
        deterministic = True #this can help to reproduce the result

    # trainer = Trainer(
    #     benchmark = benchmark,
    #     deterministic=deterministic,
    #     resume_from_checkpoint =
    # )
    # print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    NUM_K_FOLD_TEST = cfg.DATASET.K_FOLD_TEST
    START_TEST_FOLD = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.START_TEST_FOLD
    END_TEST_FOLD = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.END_TEST_FOLD
    TEST_FOLD_PREFIX = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.TEST_FOLD_PREFIX
    NUM_K_FOLD_VALID = cfg.DATASET.K_FOLD
    START_VALID_FOLD = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.START_VALID_FOLD
    END_VALID_FOLD = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.END_VALID_FOLD
    VALID_FOLD_PREFIX = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.VALID_FOLD_PREFIX

    result_folder = cfg.TRAIN_EVAL_PROCEDURE.RESULT_FOLDER
    history_folder = cfg.TRAIN_EVAL_PROCEDURE.HISTORY_FOLDER

    #conduct incremental subject experiments
    INCREMENT_FOLDS = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.NUM_INCREMENT_FOLDS
    INCREMENT_FOLD_PREFIX = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.INCREMENT_FOLD_PREFIX
    START_INCREMENT_FOLD = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.START_INCREMENT_FOLD
    END_INCREMENT_FOLD = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.END_INCREMENT_FOLD
    USE_INCREMENT = cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.START_NUM_TRAIN_SUGJECT > 0



    # if args.eval_only and args.train_k_folds:
    #     generate_detail = True
    #     test_folds_results = []
    #     test_fold_detail_results = []
    #     combine_prefix = dict()
    #
    #     for current_test_fold in range(START_TEST_FOLD,END_TEST_FOLD+1):
    #     # for test_fold in range(1,test_k_folds+1):
    #
    #         # print("current test fold : ",current_test_fold)
    #         cfg.DATASET['VALID_FOLD_TEST'] = current_test_fold
    #         combine_prefix[TEST_FOLD_PREFIX] = TEST_FOLD_PREFIX+"_"+str(current_test_fold)
    #
    #         for current_increment_fold in range(START_INCREMENT_FOLD,END_INCREMENT_FOLD+1):
    #
    #             cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL['CURRENT_FOLD'] = current_increment_fold
    #             # if INCREMENT_FOLDS>1:
    #             if USE_INCREMENT:
    #                 combine_prefix[INCREMENT_FOLD_PREFIX] = INCREMENT_FOLD_PREFIX + "_" + str(current_increment_fold)
    #
    #             # train_valid_fold_results = []
    #             # detail_results = []
    #             for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD+1):
    #                 combine_prefix[VALID_FOLD_PREFIX] = VALID_FOLD_PREFIX+"_"+str(current_valid_fold)
    #                 cfg.DATASET['VALID_FOLD'] = current_valid_fold
    #
    #                 history_dir = cfg.TRAIN_EVAL_PROCEDURE.HISTORY_FOLDER
    #                 history_dir = generate_path_for_multi_sub_model(cfg, history_dir)
    #                 output_dir = cfg.OUTPUT_DIR
    #                 output_dir = generate_path_for_multi_sub_model(cfg, output_dir)
    #
    #                 trainer = build_trainer(cfg)
    #                 model_dir = args.model_dir
    #                 # generate_history_result(trainer, output_dir=args.output_dir,model_dir=model_dir,cfg=cfg)
    #
    #                 if NUM_K_FOLD_TEST>1:
    #                     model_dir = os.path.join(model_dir,TEST_FOLD_PREFIX+"_"+str(current_test_fold))
    #                 if USE_INCREMENT:
    #                     model_dir = os.path.join(model_dir,INCREMENT_FOLD_PREFIX+"_"+str(current_increment_fold))
    #                 fold_model_dir = os.path.join(model_dir,str(current_valid_fold))
    #                 print("fold model dir : ",fold_model_dir)
    #                 trainer.load_model(fold_model_dir, epoch=args.load_epoch)
    #                 test_result = trainer.analyze_result()
    #                 test_result.update(combine_prefix)
    #                 test_folds_results.append(test_result)
    #                 if generate_detail:
    #                     test_detail_result = trainer.detail_test()
    #                     test_fold_detail_results.append(test_detail_result)
    #
    #     print("test results size {}, valid result size {}".format(len(test_folds_results),len(test_folds_results[0])))
    #     generate_excel_report(test_folds_results,args.output_dir,cfg)
    #     # generate_detail_report(test_fold_detail_results,args.output_dir,cfg)
    #     generate_model_info_config(cfg,args.output_dir)
    #     return

    if args.eval_only and not args.train_k_folds:
        trainer = build_trainer(cfg)
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    from dassl.data.data_manager_v1 import DataManagerV1, MultiDomainDataManagerV1
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn')
    # process = []
    import pytorch_lightning as pl
    if args.train_k_folds:
        generate_detail = True
        test_folds_results = []
        test_fold_detail_results = []
        combine_prefix = dict()

        for current_test_fold in range(START_TEST_FOLD, END_TEST_FOLD + 1):
            cfg.DATASET['VALID_FOLD_TEST'] = current_test_fold
            combine_prefix[TEST_FOLD_PREFIX] = TEST_FOLD_PREFIX+"_"+str(current_test_fold)

            for current_increment_fold in range(START_INCREMENT_FOLD,END_INCREMENT_FOLD+1):
                cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL['CURRENT_FOLD'] = current_increment_fold
                if USE_INCREMENT:
                    combine_prefix[INCREMENT_FOLD_PREFIX] = INCREMENT_FOLD_PREFIX + "_" + str(current_increment_fold)
                for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD + 1):
                    combine_prefix[VALID_FOLD_PREFIX] = VALID_FOLD_PREFIX+"_"+str(current_valid_fold)
                    cfg.DATASET['VALID_FOLD'] = current_valid_fold
                    output_dir = cfg.OUTPUT_DIR
                    history_dir = cfg.TRAIN_EVAL_PROCEDURE.HISTORY_FOLDER
                    history_dir = os.path.join(output_dir,history_dir)
                    output_dir = generate_path_for_multi_sub_model(cfg, output_dir)
                    history_dir = generate_path_for_multi_sub_model(cfg, history_dir)
                    cfg.merge_from_list(["history_dir",history_dir])
                    cfg.merge_from_list( ["output_dir",output_dir])
                    print("current output dir : ",output_dir)
                    pl.seed_everything(42)
                    cfg_dict = convert_to_dict(cfg,[])
                    print("cfg dict : ",cfg_dict)
                    print("test LR variable  : ", cfg_dict['OPTIM']['LR'])
                    data_manager = DataManagerV1(cfg)


                    # def train_fuc(cfg,output_dir):
                    print("check multi process")
                    # data_manager = MultiDomainDataManagerV1(cfg)

                    data_manager.prepare_data()
                    data_manager.setup()
                    require_parameter = data_manager.get_require_parameter()
                    trainer_model = build_trainer(cfg,require_parameter=require_parameter)

                    checkpoint_callback = CustomModelCheckPoint(
                    # checkpoint_callback = ModelCheckpoint(
                        verbose=True,
                        monitor = 'val_loss',
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
                        # accelerator='ddp',
                        default_root_dir=output_dir,
                        benchmark=benchmark,
                        deterministic=deterministic,
                        max_epochs=cfg.OPTIM.MAX_EPOCH,
                        # max_epochs=16,
                        resume_from_checkpoint=resume,
                        multiple_trainloader_mode='max_size_cycle',
                        # callbacks=[early_stopping,checkpoint_callback],
                        callbacks=[checkpoint_callback],
                        logger=[csv_logger,tensorboard_logger],
                        progress_bar_refresh_rate=100,
                        profiler='simple',
                        # stochastic_weight_avg=True

                    )
                    #lr finder

                    # Run learning rate finder
                    # lr_finder = trainer_lightning.tuner.lr_find(trainer_model,datamodule=data_manager,max_lr=0.1,min_lr=1e-5)
                    #
                    # # Results can be found in
                    # print("lr finder results : ",lr_finder.results)
                    #
                    # # Plot with
                    # fig = lr_finder.plot(suggest=True)
                    # fig.savefig(os.path.join(output_dir,'lr_finder.png'))
                    # fig.show()
                    #
                    #
                    # # Pick point based on plot, or get suggestion
                    # new_lr = lr_finder.suggestion()
                    #
                    # print("new suggest lr : ",new_lr)
                    #
                    # trainer_model.lr = new_lr

                    if not args.eval_only:
                        trainer_lightning.fit(trainer_model,datamodule=data_manager)

                        # trainer_lightning.fit(trainer_model)

                    # trainer_lightning.tuner.lr_find()
                    # trainer.train()
                    # print("model test result : ",trainer_lightning.test())
                    # print("trainer log dir " ,trainer_lightning.log_dir)
                    else:
                        # trainer_lightning.checkpoin
                        model = torch.load(os.path.join(output_dir,'checkpoint.ckpt'))
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
                        # p = mp.Process(target=train_fuc, args=(cfg,output_dir,))
                        # p.start()
                        # process.append(p)
        # for p in process:
        #     p.join()

        print('Done')

        generate_excel_report(test_folds_results, args.output_dir, cfg)
        generate_model_info_config(cfg, args.output_dir)
        # cross_fold_tuning(cfg)


    if not args.no_train and not args.train_k_folds:
        trainer = build_trainer(cfg)
        save_cfg(cfg)
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
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
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains for DA/DG'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domains for DA/DG'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
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
