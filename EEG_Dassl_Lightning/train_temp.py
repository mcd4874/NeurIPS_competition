import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
import pytorch_lightning as pl
from train_util import (
    setup_cfg,print_args,reset_cfg,convert_to_dict,CustomModelCheckPoint,CustomeCSVLogger,CustomExperimentWriter,generate_excel_report,
    generate_model_info_config,trainer_setup,generate_setup
)

# def train_model(cfg,eval=False,pretrain_dir="",use_best_model_pretrain=False):
#     benchmark = False
#     deterministic = False
#     if torch.cuda.is_available() and cfg.USE_CUDA:
#         print("use determinstic ")
#         benchmark = False
#         deterministic = True #this can help to reproduce the result
#
#     print("Experiment setup ...")
#     # cross test fold setup
#     N_TEST_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.N_TEST_FOLDS
#     START_TEST_FOLD = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.START_TEST_FOLD
#     END_TEST_FOLD = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.END_TEST_FOLD
#     TEST_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_FOLD_PREFIX
#
#     # shuffle fold
#     SHUFFLE_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_FOLD_PREFIX
#     N_SHUFFLE_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS
#     START_SHUFFLE_FOLD = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.START_SHUFFLE_FOLD
#     END_SHUFFLE_FOLD = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.END_SHUFFLE_FOLD
#     USE_SHUFFLE = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS > 1
#
#     # increment fold setup
#     # conduct incremental subject experiments
#     N_INCREMENT_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.N_INCREMENT_FOLDS
#     INCREMENT_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.INCREMENT_FOLD_PREFIX
#     START_INCREMENT_FOLD = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_INCREMENT_FOLD
#     END_INCREMENT_FOLD = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.END_INCREMENT_FOLD
#     USE_INCREMENT = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_NUM_TRAIN_SUGJECT > 0
#
#     # valid fold setup
#     N_VALID_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.N_VALID_FOLDS
#     START_VALID_FOLD = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.START_VALID_FOLD
#     END_VALID_FOLD = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.END_VALID_FOLD
#     VALID_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.VALID_FOLD_PREFIX
#
#     data_manager_type = cfg.DATAMANAGER.MANAGER_TYPE
#
#     generate_detail = True
#     test_folds_results = []
#     test_fold_detail_results = []
#     combine_prefix = dict()
#
#     for current_shuffle_fold in range(START_SHUFFLE_FOLD, END_SHUFFLE_FOLD + 1):
#         cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD['CURRENT_SHUFFLE_FOLD'] = current_shuffle_fold
#         shuffle_fold_prefix = ""
#         if USE_SHUFFLE:
#             shuffle_fold_prefix = SHUFFLE_FOLD_PREFIX + "_" + str(current_shuffle_fold)
#             combine_prefix[SHUFFLE_FOLD_PREFIX] = shuffle_fold_prefix
#         for current_increment_fold in range(START_INCREMENT_FOLD, END_INCREMENT_FOLD + 1):
#             cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD['CURRENT_INCREMENT_FOLD'] = current_increment_fold
#             increment_fold_prefix = ""
#             if USE_INCREMENT:
#                 increment_fold_prefix = INCREMENT_FOLD_PREFIX + "_" + str(current_increment_fold)
#                 combine_prefix[INCREMENT_FOLD_PREFIX] = increment_fold_prefix
#
#             for current_test_fold in range(START_TEST_FOLD, END_TEST_FOLD + 1):
#                 cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD['CURRENT_TEST_FOLD'] = current_test_fold
#                 combine_prefix[TEST_FOLD_PREFIX] = TEST_FOLD_PREFIX + "_" + str(current_test_fold)
#                 for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD + 1):
#                     combine_prefix[VALID_FOLD_PREFIX] = VALID_FOLD_PREFIX + "_" + str(current_valid_fold)
#                     cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD['CURRENT_VALID_FOLD'] = current_valid_fold
#
#                     output_dir = cfg.OUTPUT_DIR
#                     generate_path = generate_path_for_multi_sub_model(cfg,
#                                                                       test_fold_prefix=combine_prefix[
#                                                                           TEST_FOLD_PREFIX],
#                                                                       shuffle_fold_prefix=shuffle_fold_prefix,
#                                                                       increment_fold_prefix=increment_fold_prefix,
#                                                                       valid_fold_prefix=combine_prefix[
#                                                                           VALID_FOLD_PREFIX])
#                     output_dir = os.path.join(output_dir, generate_path)
#                     if not os.path.isdir(output_dir):
#                         os.makedirs(output_dir)
#                     cfg.merge_from_list(["output_dir", output_dir])
#                     print("current output dir : ", output_dir)
#                     pl.seed_everything(42)
#                     cfg_dict = convert_to_dict(cfg, [])
#                     print("cfg dict : ", cfg_dict)
#
#                     if data_manager_type == "single_dataset":
#                         data_manager = DataManagerV1(cfg)
#                     elif data_manager_type == "multi_datasetV2":
#                         print("use data manager for domain adaptation")
#                         data_manager = MultiDomainDataManagerV2(cfg)
#                     else:
#                         print("check multi process")
#                         data_manager = MultiDomainDataManagerV1(cfg)
#
#                     data_manager.prepare_data()
#                     data_manager.setup()
#                     require_parameter = data_manager.get_require_parameter()
#                     trainer_model = build_trainer(cfg, require_parameter=require_parameter)
#                     # monitor = 'val_acc/dataloader_idx_0'
#                     monitor = 'val_loss'
#                     checkpoint_callback = CustomModelCheckPoint(
#                         # checkpoint_callback = ModelCheckpoint(
#                         verbose=True,
#                         monitor=monitor,
#                         dirpath=output_dir,
#                         filename='checkpoint',
#                         save_top_k=1,
#                         save_last=True,
#                         every_n_val_epochs=1,
#                         auto_insert_metric_name=False)
#
#                     early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#
#                     csv_logger = CustomeCSVLogger(
#                         save_dir=output_dir,
#                         version=0,
#                         experiment_writer=CustomExperimentWriter,
#                         # step_key='epoch'
#                     )
#                     tensorboard_logger = TensorBoardLogger(
#                         save_dir=output_dir,
#                         version=1
#                     )
#
#                     resume_dir = os.path.join(output_dir, 'last.ckpt')
#
#                     if os.path.exists(resume_dir):
#                         resume = resume_dir
#                     else:
#                         resume = None
#                     trainer_lightning = Trainer(
#                         gpus=1,
#                         default_root_dir=output_dir,
#                         benchmark=benchmark,
#                         deterministic=deterministic,
#                         max_epochs=cfg.OPTIM.MAX_EPOCH,
#                         resume_from_checkpoint=resume,
#                         multiple_trainloader_mode=cfg.LIGHTNING_TRAINER.multiple_trainloader_mode,
#                         # callbacks=[early_stopping,checkpoint_callback],
#                         callbacks=[checkpoint_callback],
#                         logger=[csv_logger, tensorboard_logger],
#                         progress_bar_refresh_rate=cfg.LIGHTNING_TRAINER.progress_bar_refresh_rate,
#                         profiler=cfg.LIGHTNING_TRAINER.profiler,
#                         num_sanity_val_steps=cfg.LIGHTNING_TRAINER.num_sanity_val_steps,
#                         stochastic_weight_avg=cfg.LIGHTNING_TRAINER.stochastic_weight_avg
#
#                     )
#
#                     if not eval:
#                         if pretrain_dir != "":
#                             pretrain_dir = os.path.join(args.pretrain_dir, generate_path)
#                             if os.path.exists(pretrain_dir):
#                                 if use_best_model_pretrain:
#                                     pretrain_dir = os.path.join(pretrain_dir, 'checkpoint.ckpt')
#                                 else:
#                                     pretrain_dir = os.path.join(pretrain_dir, 'last.ckpt')
#                                 pretrain_model_state = torch.load(pretrain_dir, map_location='cuda:0')
#                                 trainer_model.load_state_dict(pretrain_model_state['state_dict'])
#                                 print("load pretrain model from {}".format(pretrain_dir))
#
#
#                         trainer_lightning.fit(trainer_model, datamodule=data_manager)
#
#                     else:
#                         model = torch.load(os.path.join(output_dir, 'checkpoint.ckpt'), map_location='cuda:0')
#                         print("save checkpoint keys : ", model.keys())
#                         # print("state dict : ", model['state_dict'])
#                         trainer_model.load_state_dict(model['state_dict'])
#                         test_result = trainer_lightning.test(trainer_model, datamodule=data_manager)[0]
#                         if len(test_result) > 1 and isinstance(test_result, list):
#                             test_result = test_result[0]
#                         print("test result : ", test_result)
#                         test_result.update(combine_prefix)
#                         test_folds_results.append(test_result)
#         if eval:
#             generate_excel_report(test_folds_results, args.output_dir, result_folder='result_folder')
#             generate_model_info_config(cfg, args.output_dir, result_folder='result_folder')
#
def train_full_experiment(cfg,experiments_setup,benchmark=False,deterministic=True,eval=False,use_best_model_pretrain=True,pretrain_dir="",seed=42):
    fold_results = list()
    for experiment in experiments_setup:
        sub_exp_path = experiment["generate_sub_exp_path"]
        output_dir = experiment["output_dir"]
        combine_prefix = experiment["sub_exp_prefix"]
        cfg = experiment["cfg"]
        trainer_model,trainer_lightning,data_manager = trainer_setup(output_dir,cfg,benchmark,deterministic,seed=seed)


        if eval:
            model_state = torch.load(os.path.join(output_dir, 'checkpoint.ckpt'), map_location='cuda:0')
            print("save checkpoint keys : ", model_state.keys())
            # print("state dict : ", model['state_dict'])
            trainer_model.load_state_dict(model_state['state_dict'])
            test_result = trainer_lightning.test(trainer_model, datamodule=data_manager)[0]
            if len(test_result) > 1 and isinstance(test_result, list):
                test_result = test_result[0]
            print("test result : ", test_result)
            test_result.update(combine_prefix)
            fold_results.append(test_result)
        else:
            if pretrain_dir != "":
                pretrain_dir = os.path.join(args.pretrain_dir, sub_exp_path)
                if os.path.exists(pretrain_dir):
                    if use_best_model_pretrain:
                        pretrain_dir = os.path.join(pretrain_dir, 'checkpoint.ckpt')
                    else:
                        pretrain_dir = os.path.join(pretrain_dir, 'last.ckpt')
                    pretrain_model_state = torch.load(pretrain_dir, map_location='cuda:0')
                    trainer_model.load_state_dict(pretrain_model_state['state_dict'])
                    # require_parameter = data_manager.get_require_parameter()
                    # trainer_model.load_from_checkpoint(pretrain_dir,cfg=cfg,require_parameter=require_parameter)
                    print("load pretrain model from {}".format(pretrain_dir))
            trainer_lightning.fit(trainer_model, datamodule=data_manager)

    if eval:
        generate_excel_report(fold_results, args.output_dir, result_folder='result_folder')
        generate_model_info_config(cfg, args.output_dir, result_folder='result_folder')

def main(args):
    benchmark = False
    deterministic = False  # this can help to reproduce the result
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)

    eval = args.eval_only
    pretrain_dir = args.pretrain_dir
    use_best_model_pretrain = args.use_pretrain_best
    if torch.cuda.is_available() and cfg.USE_CUDA:
        print("use determinstic ")
        benchmark = False
        deterministic = True #this can help to reproduce the result

    seed = 42
    pl.seed_everything(seed)

    experiments_setup = generate_setup(cfg)
    train_full_experiment(cfg, experiments_setup,benchmark=benchmark ,deterministic=deterministic,eval=eval, use_best_model_pretrain=use_best_model_pretrain, pretrain_dir=pretrain_dir,seed=seed)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--pretrain-dir',
        type=str,
        default='',
        help='load pre-train model from this directory'
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
        '--use_pretrain_best', action='store_true', help='use best record checkpoint from pre-train path'
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
