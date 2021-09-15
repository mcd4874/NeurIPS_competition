#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate tf-gpu
conda activate tensorflow2

#DIR="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/train_script/common_func_script"
#/home/wduong/tmp/Dassl_pytorch/
DIR="/home/wduong/tmp/EEG_Dassl_Lightning/train_script/common_func_script"

source "${DIR}/common_script.sh"


#experiment_1="NeurIPS_competition/experiment_1"
#temp_aug="temp_aug"
#no_aug="no_aug"
#chan_norm="chan_norm"
#no_norm="no_norm"
#
#adaptation_prefix="adaptation"
#vanilla_prefix="vanilla"
#
#BCI_IV_dataset="BCI_IV"
#Cho2017_dataset="Cho2017"
#Physionet_dataset="Physionet"

#group_aug=("$temp_aug")
group_aug=("$no_aug")

#group_norm=("$chan_norm")
group_norm=("$no_norm")

#group_datasets=("$BCI_IV_dataset")
#group_datasets=("$Cho2017_dataset")
#group_datasets=("$Physionet")
#group_datasets=("$Dataset_A_dataset")
group_datasets=("$Dataset_B_dataset")
#group_model=("$adaptationV1_prefix" "$adaptation_prefix" "$vanilla_prefix")
#group_model=("$FBCNET_adaptV1_prefix")
#group_model=("$shallowcon_adaptV1_prefix")
#run_full_multi_gpu $gpu_device_0 $final_result_9_0_3 group_aug group_norm group_model group_datasets
#run_predict $gpu_device_0 $final_result_9_0_3 $test_case_13_microvolt_path group_aug group_norm group_model group_datasets

#run_full_multi_gpu $gpu_device_0 $final_result_9_0_2 group_aug group_norm group_model group_datasets
#run_ensemble_predict $gpu_device_0 $final_result_9_0_2 $test_case_13_microvolt_path group_aug group_norm group_model group_datasets
#
#run_full_multi_gpu $gpu_device_0 $final_result_9_0_1 group_aug group_norm group_model group_datasets
#run_predict $gpu_device_0 $final_result_9_0_1 $test_case_13_microvolt_path group_aug group_norm group_model group_datasets
#group_model=("$share_adaptV1_prefix")
#LIST_EXP_TYPE=("$private_exp_14_3_1")
#sub_list=("sub" "sub_0" "sub_1" "sub_2")
#group_datasets=("$BCI_IV_dataset")
#group_model=("$addaV1_prefix")
sub_list=("sub_20")
group_model=("$mcdV1_prefix")
#group_model=("$vanilla_prefix")

#LIST_EXP_TYPE=("$final_result_14_0_1" "$final_result_14_0_1_0" "$final_result_14_0_1_1" "$final_result_14_0_1_2" "$final_result_14_0_2" "$final_result_14_0_2_0" "$final_result_14_0_2_1" "$final_result_14_0_2_2" "$final_result_14_0_3" "$final_result_14_0_3_0" "$final_result_14_0_3_1" "$final_result_14_0_3_2")
#for EXP_TYPE in "${LIST_EXP_TYPE[@]}";
#do
#  for sub in "${sub_list[@]}";
#  do
#    NEW_TYPE="${EXP_TYPE}/${sub}"
#    run_al_full $gpu_device_0 $NEW_TYPE $test_case_16_microvolt_path group_aug group_norm group_model group_datasets
#
##    run_full_multi_gpu $gpu_device_0 $NEW_TYPE group_aug group_norm group_model group_datasets
##    run_ensemble_predict $gpu_device_0 $NEW_TYPE group_aug group_norm group_model group_datasets
#  #  run_predict $gpu_device_0 $EXP_TYPE $test_data_path group_aug group_norm group_model group_datasets
##    run_predict_relabel $gpu_device_0 $NEW_TYPE $test_case_16_microvolt_path group_aug group_norm group_model group_datasets
#  done
#done

#PRETRAIN_EXP_TYPE="NeurIPS_competition/final_result_14_3_1/pretrain_0"
#USE_BEST_PRETRAIN="False"
#EXP_TYPE="NeurIPS_competition/final_result_14_3_1/main_model_0"
#USE_BEST_EXP="True"
#AL_EXP_TYPE="NeurIPS_competition/final_result_14_3_1/al_model_0"
#test_path="$test_case_16_microvolt_path"
#model_update_al="NeurIPS_competition/final_result_14_3_1/main_model_0"
#EXP_SETUP=("$PRETRAIN_EXP_TYPE" "$EXP_TYPE" "$AL_EXP_TYPE")
#BEST_SETUP=("$USE_BEST_PRETRAIN" "$USE_BEST_EXP")
#echo ${EXP_SETUP[0]}
#echo ${EXP_SETUP[1]}
#
#run_full_active_learning $gpu_device_2 $test_path $model_update_al EXP_SETUP BEST_SETUP group_aug group_norm group_model group_datasets

#EXP_TYPE="NeurIPS_competition/final_result_14_3_1/main_model_0"
#USE_BEST_EXP="True"
#AL_EXP_TYPE="NeurIPS_competition/final_result_14_3_1/al_model_0"
#test_path="$test_case_16_microvolt_path"
#model_update_al="NeurIPS_competition/final_result_14_3_1/main_model_0"
#
#EXP_SETUP=("$EXP_TYPE" "$AL_EXP_TYPE")
#BEST_SETUP=("$USE_BEST_EXP")
#run_active_learning $gpu_device_2 $test_path $model_update_al EXP_SETUP BEST_SETUP group_aug group_norm group_model group_datasets
#
ROOT="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_1"
test_case_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_1/task_1_test_case_1/full_test_sleep.mat"

group_aug=("$no_aug")
group_norm=("$no_norm")
group_datasets=("$full_dataset")
#group_model=("$deepsleep_vanilla_prefix")
group_model=("$deepsleep_share_adaptV1_prefix")
task_exp="NeurIPS_1/task_1_exp_5/al_pretrain"
model_update_path="NeurIPS_1/task_1_exp_5"
pretrain_dir="NeurIPS_1/task_1_exp_5"
#run_full_multi_gpu $gpu_device_0 $task_exp group_aug group_norm group_model group_datasets
#run_predict_task_1 $gpu_device_0 $task_exp $test_case_path group_aug group_norm group_model group_datasets

run_pretrain_al_full $gpu_device_3 $task_exp $test_case_path $model_update_path $pretrain_dir group_aug group_norm group_model group_datasets
run_predict_task_1 $gpu_device_3 $task_exp $test_case_path group_aug group_norm group_model group_datasets

#task_exp="NeurIPS_1/task_1_exp_5/al_al_pretrain"
#model_update_path="NeurIPS_1/task_1_exp_5/al_pretrain"
#pretrain_dir="NeurIPS_1/task_1_exp_5/al_pretrain"
#run_pretrain_al_full $gpu_device_3 $task_exp $test_case_path $model_update_path $pretrain_dir group_aug group_norm group_model group_datasets
#run_predict_task_1 $gpu_device_3 $task_exp $test_case_path group_aug group_norm group_model group_datasets