import os
import numpy as np
from analyze_experiment.util.util import  generate_data_paths,generate_history_results_path, load_history_data, generate_concat_dataset,filter_history_information,load_experiment_data
from collections import defaultdict
from scipy.io import loadmat
import pandas as pd

def count(label,name=""):
    count_0 = len(np.where(label == 0)[0])
    count_1 = len(np.where(label == 1)[0])
    count_2 = len(np.where(label == 2)[0])
    count_3 = len(np.where(label == 3)[0])

    print("dataset {},  has {} label 0, {} label 1, and {} label 2, {} label 3".format(name,count_0,count_1,count_2,count_3))

def generate_bag_experiment_MI_label(experiment_test_fold_preds, experiment_test_fold_probs, output_dir, predict_folder="predict_folder",only_count_best=False):
    final_pred = np.zeros(experiment_test_fold_preds[0][0][0][0].shape)
    final_prob = np.zeros(experiment_test_fold_probs[0][0][0][0].shape)
    total_sub_exp = 0
    for experiment_idx in range(len(experiment_test_fold_preds)):
        model_preds = experiment_test_fold_preds[experiment_idx]
        model_probs = experiment_test_fold_probs[experiment_idx]
        for model_idx in range(len(model_preds)):
            test_fold_preds=model_preds[model_idx]
            test_fold_probs=model_probs[model_idx]
            for test_fold in range(len(test_fold_preds)):
                current_fold_preds = test_fold_preds[test_fold]
                current_fold_probs = test_fold_probs[test_fold]
                for valid_fold in range(len(current_fold_preds)):
                    current_valid_pred = current_fold_preds[valid_fold]
                    current_valid_prob = current_fold_probs[valid_fold]
                    final_pred = final_pred + current_valid_pred
                    final_prob = final_prob + current_valid_prob
                    total_sub_exp+=1
    count= 0
    pred_output = list()
    subject_id = 0
    subject_trials = 200
    total_best_trial = 0

    subject_info = defaultdict()

    for trial_idx in range(len(final_pred)):
        if trial_idx % subject_trials == 0:
            subject_id += 1
            preds_list_count = [0]*(total_sub_exp+1)
            subject_info[subject_id] = (preds_list_count)
            total_best_trial = 0

        trial_pred = final_pred[trial_idx]
        trial_prob = final_prob[trial_idx]
        best_idx = np.argmax(trial_pred)
        best_pred = trial_pred[best_idx]
        best_prob = trial_prob[best_idx]
        if best_pred == total_sub_exp:
            print("subject {} has trial {} has max predict {}".format(subject_id,trial_idx,total_sub_exp))
            count+=1
            total_best_trial+=1
        for idx in range(len(trial_pred)):
            if idx !=best_idx:
                pred = trial_pred[idx]
                prob = trial_prob[idx]
                if pred > best_pred:
                    best_pred = pred
                    best_idx = idx
                    best_prob = prob
                elif pred == best_pred:
                    print("1 issue")
                    if prob > best_prob:
                        best_idx = idx
                        best_prob = prob
                    print("trial {} has pred {}, with probs {}, final pick {}".format(trial_idx,trial_pred,trial_prob,best_idx))
        temp = subject_info[subject_id]
        temp[int(best_pred)] = temp[int(best_pred)]+1
        subject_info[subject_id] = temp
        if only_count_best:
            if best_pred !=total_sub_exp:
                pred_output.append(-1)
            else:
                pred_output.append(best_idx)
        else:
            pred_output.append(best_idx)
    pred_output = np.array(pred_output)
    combine_folder = os.path.join(output_dir, predict_folder)
    print("pred output : ",pred_output)
    print("total count max prediction : ",count)
    print("subject info : ",subject_info)
    # np.savetxt(os.path.join(combine_folder, "pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")
    return pred_output

def load_prediction(dataset_type,common_path,experiment_type,model_list_prefix,augmentation_list_prefix,norm_list_prefix,target_dataset_list_prefix):
    test_folds=["test_fold_1"]
    increment_folds=["increment_fold_1"]
    valid_folds=["valid_fold_1","valid_fold_2","valid_fold_3","valid_fold_4","valid_fold_5"]
    predict_folder = "predict_folder"
    print("common path before load : ",common_path)
    experiment_preds = list()
    experiment_probs = list()
    for experiment in experiment_type:
        experiment = [experiment]
        print("experiment : ",experiment)
        model_preds = list()
        model_probs = list()
        for model_prefix in model_list_prefix:
            model_prefix = [model_prefix]
            prefix_list = [experiment,augmentation_list_prefix, norm_list_prefix, model_prefix, target_dataset_list_prefix]
            list_full_path = generate_data_paths(common_path, prefix_list, [])
            print("full path : ",list_full_path)
            path = list_full_path[0]

            test_fold_preds=list()
            test_fold_probs=list()
            for test_fold in test_folds:
                for increment_fold in increment_folds:
                    valid_fold_preds=list()
                    valid_fold_probs=list()
                    for valid_fold in valid_folds:
                        generate_path = os.path.join(test_fold,increment_fold,valid_fold)
                        combine_folder = os.path.join(path, predict_folder, generate_path)
                        pred = np.loadtxt(os.path.join(combine_folder, 'pred.txt'), delimiter=',')
                        probs = np.loadtxt(os.path.join(combine_folder, 'prob.txt'), delimiter=',')
                        valid_fold_preds.append(pred)
                        valid_fold_probs.append(probs)
                    test_fold_preds.append(valid_fold_preds)
                    test_fold_probs.append(valid_fold_probs)
            model_preds.append(test_fold_preds)
            model_probs.append(test_fold_probs)
        experiment_preds.append(model_preds)
        experiment_probs.append(model_probs)
        # print(test_fold_preds)
    # generate_pred_MI_label(test_fold_preds, test_fold_probs, "", predict_folder="predict_folder")
    pred = generate_bag_experiment_MI_label(experiment_preds,experiment_probs, "", predict_folder="predict_folder",only_count_best=True)

    print(pred)
    count(pred,name=dataset_type)
    return pred

def load_test_data_from_file(provide_path,dataset_type):
    temp = loadmat(provide_path)
    datasets = temp['datasets'][0]
    target_dataset = None
    for dataset in datasets:
        dataset = dataset[0][0]
        dataset_name = dataset['dataset_name'][0]
        if dataset_name == dataset_type:
            target_dataset = dataset
    data = target_dataset['data'].astype(np.float32)
    label = np.squeeze(target_dataset['label']).astype(int)
    meta_data = target_dataset['meta_data'][0][0]
    new_meta_data = {}
    new_meta_data['subject'] = meta_data['subject'][0]
    new_meta_data['session'] = [session[0] for session in meta_data['session'][0]]
    new_meta_data['run'] = [run[0] for run in meta_data['run'][0]]
    meta_data = pd.DataFrame.from_dict(new_meta_data)
    test_data, test_label, test_meta_data = data, label, meta_data
    return test_data,test_label,test_meta_data




# load_prediction("dataset_A",common_path,experiment_type,model_list_prefix,augmentation_list_prefix,norm_list_prefix,target_dataset_list_prefix)


