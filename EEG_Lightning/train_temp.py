import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
import pytorch_lightning as pl
from train_util import (
    setup_cfg,print_args,reset_cfg,convert_to_dict,CustomModelCheckPoint,CustomeCSVLogger,CustomExperimentWriter,generate_excel_report,
    generate_model_info_config,trainer_setup,generate_setup,train_full_experiment
)

def main(args):
    benchmark = False
    deterministic = False  # this can help to reproduce the result
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)

    eval = args.eval_only
    if cfg.LIGHTNING_MODEL.PRETRAIN.DIR != "":
        pretrain_dir = cfg.LIGHTNING_MODEL.PRETRAIN.DIR
        use_best_model_pretrain = cfg.LIGHTNING_MODEL.PRETRAIN.USE_BEST
    else:
        pretrain_dir = args.pretrain_dir
        use_best_model_pretrain = args.use_pretrain_best
        cfg.merge_from_list(["LIGHTNING_MODEL.PRETRAIN.DIR", pretrain_dir])
        cfg.merge_from_list(["LIGHTNING_MODEL.PRETRAIN.USE_BEST", use_best_model_pretrain])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
