import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from NeurIPS_competition.util.support import (
    expand_data_dim, normalization_channels,normalization_time, generate_common_chan_test_data, load_Cho2017, load_Physionet, load_BCI_IV,
    correct_EEG_data_order, relabel, process_target_data, relabel_target, load_dataset_A, load_dataset_B, modify_data,reformat,
    filterBank
)

from train_util import (
    setup_cfg,print_args,reset_cfg,convert_to_dict,CustomModelCheckPoint,CustomeCSVLogger,CustomExperimentWriter,generate_excel_report,
    generate_model_info_config,trainer_setup,generate_setup
)


from dassl.data.datasets.data_util import EuclideanAlignment
from collections import defaultdict
from numpy.random import RandomState

def generate_pred_MI_label(fold_predict_results, output_dir, predict_folder="predict_folder",
                           relabel=False):

    probs = fold_predict_results[0]["probs"]
    preds = fold_predict_results[0]["preds"]

    final_pred = np.zeros(probs.shape)
    final_prob = np.zeros(preds.shape)

    for predict_result in fold_predict_results:
        current_prob = predict_result["probs"]
        current_pred = predict_result["preds"]

        final_pred = final_pred + current_pred
        final_prob = final_prob + current_prob

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
    print("save folder : ",combine_folder)
    np.savetxt(os.path.join(combine_folder, "pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")

def generate_assemble_result(fold_predict_results, output_dir,
                             predict_folder="predict_folder", relabel=False):
    # unique_test_fold = for fold_result in fold_predict_results:
    group_test_folds = defaultdict(list)
    final_fold_result = list()
    for fold_result in fold_predict_results:
        test_fold = fold_result["test_fold"]
        group_test_folds[test_fold].append(fold_result)
    for test_fold,test_fold_result in group_test_folds.items():
        probs = test_fold_result[0]["probs"]
        preds = test_fold_result[0]["preds"]
        final_label = test_fold_result[0]["labels"]
        final_pred = np.zeros(probs.shape)
        final_prob = np.zeros(preds.shape)
        for predict_result in test_fold_result:
            current_prob = predict_result["probs"]
            current_pred = predict_result["preds"]

            final_pred = final_pred + current_pred
            final_prob = final_prob + current_prob

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

        print("test fold {} has acc {} ".format(test_fold, acc))
        # current_test_fold = test_fold_prefix + str(test_fold + 1)
        result = {
            "test_fold": test_fold,
            "test_acc": acc
        }
        final_fold_result.append(result)
    result = pd.DataFrame.from_dict(final_fold_result)
    result_output_dir = os.path.join(output_dir, predict_folder)
    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)
    result_filename = 'ensemble_result.xlsx'
    result.to_excel(os.path.join(result_output_dir, result_filename), index=False)
#

from scipy.io import loadmat


def load_test_data_from_file(provide_path,dataset_type):

    temp = loadmat(provide_path)
    datasets = temp['datasets'][0]
    target_dataset = None
    list_r_op =  None
    if len(datasets) == 1:
        dataset = datasets[0]
        dataset = dataset[0][0]
        target_dataset = dataset
    else:
        for dataset in datasets:
            dataset = dataset[0][0]
            dataset_name = dataset['dataset_name'][0]
            if dataset_name == dataset_type:
                target_dataset = dataset

    # data = target_dataset['data'].astype(np.float32)

    data = target_dataset['data'].astype(np.float32)
    label = np.squeeze(target_dataset['label']).astype(int)
    meta_data = target_dataset['meta_data'][0][0]
    new_meta_data = {}
    new_meta_data['subject'] = meta_data['subject'][0]
    new_meta_data['session'] = [session[0] for session in meta_data['session'][0]]
    new_meta_data['run'] = [run[0] for run in meta_data['run'][0]]
    meta_data = pd.DataFrame.from_dict(new_meta_data)
    test_data, test_label, meta_data = reformat(data, label, meta_data)
    potential_r_op = dataset_type + '_r_op.mat'

    # if dataset_type=="dataset_A":
    #     potential_r_op="dataset_A_r_op.mat"
    # elif dataset_type=="dataset_B":
    #     potential_r_op="dataset_B_r_op.mat"
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
    return test_data,list_r_op

def print_info(source_data,dataset_name):
    print("current dataset {}".format(dataset_name))
    for subject_idx in range(len(source_data)):
        print("source_data subject_idx {} has shape : {}, with range scale ({},{}) ".format(
            subject_idx, source_data[subject_idx].shape,
            np.max(source_data[subject_idx]), np.min(source_data[subject_idx])))

def get_test_data(dataset_type, norm, provide_data_path = None,EA=False):
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
        print("load test data from file {}".format(provide_data_path))
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
    if norm == 'cross_channel_norm':
        print("normalize across channels ")
        test_data = normalization_channels(test_data)
    elif norm == 'time_norm':
        print("normalize across time in each channel ")
        test_data = normalization_time(test_data)
    test_data = expand_data_dim(test_data)
    print("data shape before predict : ",test_data.shape)
    return test_data

def generate_ensemble_predict(cfg,experiments_setup,benchmark=False,deterministic=True,generate_predict=False,use_assemble_test_dataloader=False,relabel=False,seed=42):
    """Apply data transformation/normalization"""
    norm = cfg.INPUT.TRANSFORMS[0]
    EA = cfg.DATAMANAGER.DATASET.USE_Euclidean_Aligment
    print("use cross channel norm : ", norm)
    print("generate predict : ", generate_predict)
    dataset_type = cfg.DATAMANAGER.DATASET.SETUP.TARGET_DATASET_NAME

    test_file_path = args.test_data if args.test_data != '' else None
    if generate_predict and not use_assemble_test_dataloader:
        dataset = get_test_data(dataset_type, norm, EA=EA, provide_data_path=test_file_path)
    fold_predict_results = list()

    for experiment in experiments_setup:
        sub_exp_path = experiment["generate_sub_exp_path"]
        output_dir = experiment["output_dir"]
        combine_prefix = experiment["sub_exp_prefix"]
        cfg = experiment["cfg"]

        predict_folder = "predict_folder"
        combine_predict_folder = os.path.join(cfg.OUTPUT_DIR, predict_folder, sub_exp_path)
        if not os.path.exists(combine_predict_folder):
            os.makedirs(combine_predict_folder)

        if generate_predict:
            trainer_model, trainer_lightning, data_manager = trainer_setup(output_dir, cfg, benchmark, deterministic,
                                                                           seed=seed)
            model_state = torch.load(os.path.join(output_dir, 'checkpoint.ckpt'), map_location='cuda:0')
            print("save checkpoint keys : ", model_state.keys())
            trainer_model.load_state_dict(model_state['state_dict'])
            trainer_model.eval()
            probs_list = []
            preds_list = []
            label_list = []
            if use_assemble_test_dataloader:
                def parser(test_input):
                    input, label, domain = test_input
                    label = label.numpy()
                    return input, label

                test_dataloader = data_manager.predict_dataloader()

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

            label_list = np.concatenate(label_list)
            probs_list = np.concatenate(probs_list)
            probs_list = np.around(probs_list, decimals=4)
            preds_list = np.concatenate(preds_list).astype(int)

            predict_info = {
                "probs": probs_list,
                "preds": preds_list
            }
            if label_list[0] != None:
                predict_info.update({"label": label_list})
            predict_info.update(combine_prefix)
            fold_predict_results.append(predict_info)

            print("combine predict folder : ", combine_predict_folder)
            if use_assemble_test_dataloader:
                np.savetxt(os.path.join(combine_predict_folder, 'ensemble_label.txt'), label_list, delimiter=',',
                           fmt="%d")
                np.savetxt(os.path.join(combine_predict_folder, 'ensemble_pred.txt'), preds_list, delimiter=',',
                           fmt="%d")
                np.savetxt(os.path.join(combine_predict_folder, 'ensemble_prob.txt'), probs_list, delimiter=',',
                           fmt='%1.4f')
            else:
                np.savetxt(os.path.join(combine_predict_folder, 'pred.txt'), preds_list, delimiter=',', fmt="%d")
                np.savetxt(os.path.join(combine_predict_folder, 'prob.txt'), probs_list, delimiter=',', fmt='%1.4f')

        else:
            if use_assemble_test_dataloader:
                pred = np.loadtxt(os.path.join(combine_predict_folder, 'ensemble_pred.txt'), delimiter=',')
                probs = np.loadtxt(os.path.join(combine_predict_folder, 'ensemble_prob.txt'), delimiter=',')
                labels = np.loadtxt(os.path.join(combine_predict_folder, 'ensemble_label.txt'), delimiter=',')
                predict_info = {
                    "labels": labels,
                    "probs": probs,
                    "preds": pred
                }
            else:
                pred = np.loadtxt(os.path.join(combine_predict_folder, 'pred.txt'), delimiter=',')
                probs = np.loadtxt(os.path.join(combine_predict_folder, 'prob.txt'), delimiter=',')
                predict_info = {
                    "probs": probs,
                    "preds": pred
                }
            predict_info.update(combine_prefix)
            fold_predict_results.append(predict_info)
    if not generate_predict:
        if not use_assemble_test_dataloader:
            generate_pred_MI_label(fold_predict_results, output_dir=cfg.OUTPUT_DIR, relabel=relabel)
        else:
            generate_assemble_result(fold_predict_results, output_dir=cfg.OUTPUT_DIR,
                                     relabel=relabel)

def main(args):
    benchmark = False
    deterministic = False  # this can help to reproduce the result
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)
    seed = 42
    pl.seed_everything(seed)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        print("use determinstic ")
        benchmark = False
        deterministic = True #this can help to reproduce the result

    experiments_setup = generate_setup(cfg)

    generate_predict = args.generate_predict
    use_assemble_test_dataloader = args.use_assemble_test_dataloader
    relabel = args.relabel

    generate_ensemble_predict(cfg, experiments_setup,benchmark=benchmark ,deterministic=deterministic, generate_predict=generate_predict,
                              use_assemble_test_dataloader=use_assemble_test_dataloader,
                              relabel=relabel,seed=seed)







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
