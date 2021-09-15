import os
import numpy as np
from analyze_experiment.util.util import  generate_data_paths,generate_history_results_path, load_history_data, generate_concat_dataset,filter_history_information,load_experiment_data
from collections import defaultdict


def count(label,name=""):
    count_0 = len(np.where(label == 0)[0])
    count_1 = len(np.where(label == 1)[0])
    count_2 = len(np.where(label == 2)[0])
    count_3 = len(np.where(label == 3)[0])
    count_4 = len(np.where(label == 4)[0])
    count_5 = len(np.where(label == 5)[0])

    print("dataset {},  has {} label 0, {} label 1, and {} label 2, {} label 3, {} label 4, {} label 5".format(name,count_0,count_1,count_2,count_3,count_4,count_5))

    percent_0 = round(count_0/len(label) *100,2)
    percent_1 = round(count_1/len(label) *100,2)
    percent_2 = round(count_2/len(label) *100,2)
    percent_3 = round(count_3/len(label) *100,2)
    percent_4 = round(count_4/len(label) *100,2)
    percent_5 = round(count_5/len(label) *100,2)

    print("dataset {},  has {}% label 0, {}% label 1, and {}% label 2, {}% label 3, {}% label 4, {}% label 5".format(name,percent_0,percent_1,percent_2,percent_3,percent_4,percent_5))



def generate_bag_experiment_MI_label(experiment_test_fold_preds, experiment_test_fold_probs, output_dir, predict_folder="predict_folder",only_count_best=False,confidence_level=-1):
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


        # valid_fold_pred = test_fold_preds[test_fold]

    count= 0
    pred_output = list()
    subject_id = 0
    subject_trials = 200
    total_best_trial = 0

    subject_info = defaultdict()
    subject_pred_info = defaultdict()
    best_subject_pred_info = defaultdict()

    # print("total sub exp : ",total_sub_exp)
    if confidence_level == -1:
        confidence_level = total_sub_exp


    for trial_idx in range(len(final_pred)):
        if trial_idx % subject_trials == 0:
            subject_id += 1
            preds_list_count = [0]*(total_sub_exp+1)
            subject_info[subject_id] = (preds_list_count)
            total_best_trial = 0
            subject_pred_info[subject_id] = [0]*len(final_pred[0])
            best_subject_pred_info[subject_id] = [0]*len(final_pred[0])

        trial_pred = final_pred[trial_idx]
        trial_prob = final_prob[trial_idx]
        best_idx = np.argmax(trial_pred)
        best_pred = trial_pred[best_idx]
        best_prob = trial_prob[best_idx]
        # if best_pred >= total_sub_exp-1:

        if best_pred >= confidence_level:
            # print("subject {} has trial {} has max predict {}".format(subject_id,trial_idx,best_pred))
            count+=1
            total_best_trial+=1
            temp = best_subject_pred_info[subject_id]
            temp[int(best_idx)] = temp[int(best_idx)] + 1
            best_subject_pred_info[subject_id] = temp
        for idx in range(len(trial_pred)):
            if idx !=best_idx:
                pred = trial_pred[idx]
                prob = trial_prob[idx]
                if pred > best_pred:
                    best_pred = pred
                    best_idx = idx
                    best_prob = prob
                elif pred == best_pred:
                    # print("1 issue")
                    if prob > best_prob:
                        best_idx = idx
                        best_prob = prob
                    # print("trial {} has pred {}, with probs {}, final pick {}".format(trial_idx,trial_pred,trial_prob,best_idx))
        temp = subject_info[subject_id]
        temp[int(best_pred)] = temp[int(best_pred)]+1
        subject_info[subject_id] = temp

        temp = subject_pred_info[subject_id]
        temp[int(best_idx)] = temp[int(best_idx )]+1
        subject_pred_info[subject_id] = temp
        if only_count_best:
            # if best_pred !=total_sub_exp:
            if best_pred < confidence_level:
                pred_output.append(-1)
            else:
                pred_output.append(best_idx)
        else:
            pred_output.append(best_idx)
    pred_output = np.array(pred_output)
    combine_folder = os.path.join(output_dir, predict_folder)
    # print("pred output : ",pred_output)
    print("total count max prediction : ",count)
    print("subject info : ",subject_info)
    print("subject pred info : ",subject_pred_info)
    print("subject best pred info : ",best_subject_pred_info)
    # np.savetxt(os.path.join(combine_folder, "pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")
    return pred_output



def generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix):
    experiment_preds = list()
    experiment_probs = list()
    for experiment in experiment_type:
        experiment = [experiment]
        # print("experiment : ",experiment)
        model_preds = list()
        model_probs = list()
        for model_prefix in model_list_prefix:
            model_prefix = [model_prefix]
            prefix_list = [experiment,augmentation_list_prefix, norm_list_prefix, model_prefix, target_dataset_list_prefix]
            list_full_path = generate_data_paths(common_path, prefix_list, [])
            # print("full path : ",list_full_path)
            path = list_full_path[0]

            test_fold_preds=list()
            test_fold_probs=list()
            for test_fold in test_folds:
                # for increment_fold in increment_folds:
                valid_fold_preds=list()
                valid_fold_probs=list()
                for valid_fold in valid_folds:
                    # generate_path = os.path.join(test_fold,increment_fold,valid_fold)
                    generate_path = os.path.join(test_fold,valid_fold)

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
    only_count_best = False
    # confidence_level=18
    # confidence_level=20
    confidence_level=-1
    # only_count_best = True

    pred = generate_bag_experiment_MI_label(experiment_preds,experiment_probs, "", predict_folder="predict_folder",only_count_best=only_count_best,confidence_level=confidence_level)
    return pred


predict_folder = "predict_folder"

target_dataset_list_prefix = [
    "full_dataset",
]
norm_list_prefix = [
    'no_norm',
    # 'time_norm'
]
augmentation_list_prefix = [
    "no_aug"
]
model_list_prefix = [
    # 'deepsleep_vanilla',
    'deepsleep_share_adaptV1'
    # 'deepsleep_share_mcd'
    # 'vanilla',
    # 'share_adaptV1'
]
# valid_folds=["valid_fold_2","valid_fold_3","valid_fold_4","valid_fold_5"]

valid_folds=["valid_fold_1","valid_fold_2","valid_fold_3","valid_fold_4","valid_fold_5"]
# test_folds=["test_fold_1","test_fold_2","test_fold_3","test_fold_4"]
# test_folds=["test_fold_1","test_fold_3"]

test_folds=["test_fold_1"]
common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\{}\\{}\\{}\\{}\\{}\\model"
# experiment_type=["task_1_exp_2/al_pretrain"]
# experiment_type=["task_1_exp_3\\al_model"]
# experiment_type=["task_1_exp_5"]
# experiment_type=["task_1_exp_5\\al_model"]
# experiment_type=["task_1_exp_5\\al_al_model"]

# experiment_type=["task_1_final_2\\quick_ver"]
experiment_type=["task_1_final_2\\quick_ver_1"]

# experiment_type=["task_1_final_2\\quick_ver"]


print("experiment {}".format(experiment_type[0]))
pred_1 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
count(pred_1,"task_1")
# test_folds=["test_fold_1","test_fold_2","test_fold_3","test_fold_4"]

# model_list_prefix = [
#     'deepsleep_share_adaptV1'
#     # 'deepsleep_vanilla',
# #     'vanilla',
# ]

# experiment_type=["task_1_exp_5\\al_model"]
# print("experiment {}".format(experiment_type[0]))
# pred_2 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
# count(pred_2,"task_2")

#
# experiment_type=["task_1_exp_5\\al_al_pretrain"]
# print("experiment {}".format(experiment_type[0]))
# pred_3 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
# count(pred_3,"task_3")
# #
# #
# experiment_type=["task_1_exp_5\\al_pretrain_2"]
# print("experiment {}".format(experiment_type[0]))
# pred_2_1 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
# count(pred_2_1,"task_2_1")
#
# experiment_type=["task_1_exp_5\\al_al_pretrain_2"]
# print("experiment {}".format(experiment_type[0]))
# pred_3_1 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
# count(pred_3_1,"task_3_1")


# similar_1_2 = np.sum(pred_1 == pred_2)
# print("similar 1-2 : ",similar_1_2)

# similar_1_3 = np.sum(pred_1 == pred_3)
# print("similar 1-3 : ",similar_1_3)
# #
# similar_2_3 = np.sum(pred_2 == pred_3)
# print("similar 2-3 : ",similar_2_3)
# #
# similar_2_2_1 = np.sum(pred_2 == pred_2_1)
# print("similar_2_2_1 : ",similar_2_2_1)
#
# similar_1_2_1 = np.sum(pred_1 == pred_2_1)
# print("similar_1_2_1 : ",similar_1_2_1)
#
# similar_1_3_1 = np.sum(pred_1 == pred_3_1)
# print("similar_1_3_1 : ",similar_1_3_1)


