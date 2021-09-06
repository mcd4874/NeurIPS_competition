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



def generate_bag_experiment_MI_label(experiment_test_fold_preds, experiment_test_fold_probs, output_dir, predict_folder="predict_folder",only_count_best=False,confidence_level=5):
    final_pred = np.zeros(experiment_test_fold_preds[0][0][0][0].shape)
    final_prob = np.zeros(experiment_test_fold_probs[0][0][0][0].shape)
    # print("test fold preds : ",test_fold_preds)
    # print("len test fold : ",len(test_fold_preds))
    # print("val fold size : ",len(test_fold_preds[0]))
    # print("val pred size : ",test_fold_preds[0][0].shape)
    # print("org final pred shape : ",final_pred.shape)
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
                    # print("current valid pred shape : ",current_valid_pred.shape)
                    # print("final pred shape : ",final_pred.shape)
                    final_pred = final_pred + current_valid_pred
                    final_prob = final_prob + current_valid_prob
                    total_sub_exp+=1


        # valid_fold_pred = test_fold_preds[test_fold]

    # print("result current pred : ", current_pred)
    count= 0
    pred_output = list()
    subject_id = 0
    subject_trials = 200
    total_best_trial = 0

    subject_info = defaultdict()
    subject_pred_info = defaultdict()
    best_subject_pred_info = defaultdict()

    if confidence_level > total_sub_exp:
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
        print("temp : ",temp)
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

model_list_prefix = [
    # 'vanilla',
    # 'adaptation',
    # 'adaptationV1',
    # 'share_adaptV1'
    # 'dannV1',
    'mcdV1',
    # 'addaV1',
    # 'SRDA'
#
]
target_dataset_list_prefix = [
    "dataset_A",
    # "dataset_B",
]
augmentation_list_prefix = [
    'no_aug',
    # 'temp_aug',
#     'T_F_aug'
]
norm_list_prefix = [
    'no_norm',
    # 'chan_norm'
]
# common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\{}\\{}\\{}\\{}\\model"

# experiment_type=["final_result_11_4_3_0","final_result_11_4_3_1","final_result_11_4_3_2"]
# experiment_type=["final_result_11_4_1_0","final_result_11_4_1_1","final_result_11_4_1_2"]
# experiment_type=["final_result_11_4_1_0","final_result_11_4_1_2"]

# experiment_type=["final_result_11_0_1","final_result_11_4_1_1"]
# experiment_type=["final_result_11_4_3_1"]

# experiment_type=["final_result_11_0_1","final_result_11_4_1"]
# experiment_type=["final_result_14_0_3_2"]
# experiment_type=["final_result_14_0_1_0","final_result_14_0_1_1","final_result_14_0_1_2"]
# experiment_type=["final_result_14_0_2_0","final_result_14_0_2_1","final_result_14_0_2_2"]
# experiment_type=["final_result_14_0_3_0","final_result_14_0_3_1","final_result_14_0_3_2"]

experiment_type=["final_result_14_3_1"]
# experiment_type=["final_result_12_3_1","final_result_12_3_3"]

common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\al_model_2\\{}\\{}\\{}\\{}\\model"


test_folds=["test_fold_1"]
increment_folds=["increment_fold_1"]
# valid_folds=["valid_fold_1","valid_fold_2","valid_fold_3","valid_fold_4","valid_fold_5"]
valid_folds=["valid_fold_1","valid_fold_2","valid_fold_3","valid_fold_4","valid_fold_5"]

predict_folder = "predict_folder"

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
    confidence_level=5
    # only_count_best = True

    pred = generate_bag_experiment_MI_label(experiment_preds,experiment_probs, "", predict_folder="predict_folder",only_count_best=only_count_best,confidence_level=confidence_level)
    return pred

# common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\main_model_2\\{}\\{}\\{}\\{}\\model"
# experiment_type=["final_result_14_3_1"]
# target_dataset_list_prefix = [
#     "dataset_A",
# ]
# pred_A = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
#
#
# target_dataset_list_prefix = [
#     "dataset_B",
# ]
# pred_B = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
#
# model_list_prefix = [
#     # 'vanilla',
#     # 'adaptationV1',
#     # 'dannV1',
#     # 'mcdV1',
#     # 'SRDA',
#     'm3sda'
# #
# ]
#
# augmentation_list_prefix = [
#     'no_aug',
#     # 'temp_aug',
# #     'T_F_aug'
# ]
#
# # common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\main_model_3\\ratio_tune\\1.0\\{}\\{}\\{}\\{}\\model"
#
# common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\main_model_3\\{}\\{}\\{}\\{}\\model"
# experiment_type=["final_result_15_3_1"]
# target_dataset_list_prefix = [
#     "dataset_A_0",
# ]
# pred_A_0 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
#
# target_dataset_list_prefix = [
#     "dataset_A_1",
# ]
# pred_A_1 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
#
#
# # common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\main_model_3\\{}\\{}\\{}\\{}\\model"
#
# target_dataset_list_prefix = [
#     "dataset_B_0",
# ]
# pred_B_0 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
#
# target_dataset_list_prefix = [
#     "dataset_B_1",
# ]
# pred_B_1 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
# target_dataset_list_prefix = [
#     "dataset_B_2",
# ]
# pred_B_2 = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
#
# count(pred_A,name="A")
# count(pred_B,name="B")
# #
# #
# count(pred_A_0,name="A_0")
# count(pred_A_1,name="A_1")
#
# count(pred_B_0,name="B_0")
# count(pred_B_1,name="B_1")
# count(pred_B_2,name="B_2")


common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\{}\\{}\\{}\\{}\\{}\\model"
experiment_type=["task_1_exp_1"]
target_dataset_list_prefix = [
    "full_dataset",
]
model_list_prefix = [
    'vanilla',
    # 'adaptationV1',
    # 'dannV1',
    # 'mcdV1',
    # 'SRDA',
    # 'm3sda'
#
]
pred = generate_predict(common_path,experiment_type,model_list_prefix,augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
count(pred,"task_1")

# def relabel_target(l):
#     if l == 0: return 0
#     elif l == 1: return 1
#     else: return 2
# relabel=True
# if relabel:
#     pred_output = np.array([relabel_target(l) for l in pred])
#     print("update pred output : ", pred_output)
# count(pred_output,name="dataset_A")
#
# path_A="ensemble_results/case2/dataset_A"
# path_B="ensemble_results/case2/dataset_B"
#
file = "pred_MI_label.txt"
# path_A = os.path.join(path_A,file)
# path_B = os.path.join(path_B,file)
# np.savetxt("util/pred_MI_label.txt",MI_label,delimiter=',',fmt="%d")

# np.savetxt("util/pred_MI_label.txt",MI_label,delimiter=',',fmt="%d")
# pred_A = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
# pred_B = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')
# final_pred = np.concatenate([pred_A,pred_B])
# path = os.path.join("util",case,"answer.txt")
# np.savetxt(path_A,pred_output,delimiter=',',fmt="%d")
# np.savetxt(path_B,pred_output,delimiter=',',fmt="%d")
