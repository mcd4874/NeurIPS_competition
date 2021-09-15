#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate tf-gpu
conda activate tensorflow2

#DIR="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/train_script/common_func_script"
#/home/wduong/tmp/Dassl_pytorch/
DIR="/home/wduong/tmp/EEG_Dassl_Lightning/train_script/common_func_script"

source "${DIR}/common_script.sh"


#EXP_TYPE="NeurIPS_competition/final_result_14_3_1/main_model_0"
#USE_BEST_EXP="True"
#AL_EXP_TYPE="NeurIPS_competition/final_result_14_3_1/al_model_0"
#test_path="$test_case_16_microvolt_path"
#model_update_al="NeurIPS_competition/final_result_14_3_1/main_model_0"
#
#EXP_SETUP=("$EXP_TYPE" "$AL_EXP_TYPE")
#BEST_SETUP=("$USE_BEST_EXP")
#run_active_learning $gpu_device_2 $test_path $model_update_al EXP_SETUP BEST_SETUP group_aug group_norm group_model group_datasets
ROOT="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_1"
test_case_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_1/task_1_final_test_case_1/full_test_sleep.mat"

group_aug=("$no_aug")
group_norm=("$no_norm")
group_datasets=("$full_dataset")
group_model=("$deepsleep_share_mcd_prefix")
#group_model=("$deepsleep_share_adaptV1_prefix")
#group_model=("$share_adaptV1_prefix")
task_exp="NeurIPS_1/task_1_final_2/quick_ver"
#model_update_path="NeurIPS_1/task_1_exp_3"
run_full_multi_gpu $gpu_device_0 $task_exp group_aug group_norm group_model group_datasets
run_predict_task_1 $gpu_device_0 $task_exp $test_case_path group_aug group_norm group_model group_datasets

#run_al_full $gpu_device_0 $task_exp $test_case_path $model_update_path group_aug group_norm group_model group_datasets
#run_predict_task_1 $gpu_device_0 $task_exp $test_case_path group_aug group_norm group_model group_datasets
