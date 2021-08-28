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

#group_aug=("$T_F_aug")
group_aug=("$temp_aug")
#group_aug=("$no_aug")

#group_norm=("$chan_norm")
group_norm=("$no_norm")

#group_datasets=("$BCI_IV_dataset")
#group_datasets=("$Cho2017_dataset")
#group_datasets=("$Physionet")
#group_datasets=("$Dataset_A_dataset")
#group_datasets=("$Dataset_B_dataset")
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
LIST_EXP_TYPE=("$private_exp_14_3_1")
#sub_list=("sub" "sub_0" "sub_1" "sub_2")
group_datasets=("$BCI_IV_dataset")
group_model=("$SRDA_prefix")
sub_list=("sub_5")
#group_model=("$mcdV1_prefix")
#group_model=("$vanilla_prefix")

#LIST_EXP_TYPE=("$final_result_14_0_1" "$final_result_14_0_1_0" "$final_result_14_0_1_1" "$final_result_14_0_1_2" "$final_result_14_0_2" "$final_result_14_0_2_0" "$final_result_14_0_2_1" "$final_result_14_0_2_2" "$final_result_14_0_3" "$final_result_14_0_3_0" "$final_result_14_0_3_1" "$final_result_14_0_3_2")
for EXP_TYPE in "${LIST_EXP_TYPE[@]}";
do
  for sub in "${sub_list[@]}";
  do
    NEW_TYPE="${EXP_TYPE}/${sub}"
    run_full_multi_gpu $gpu_device_0 $NEW_TYPE group_aug group_norm group_model group_datasets
#    run_ensemble_predict $gpu_device_0 $NEW_TYPE group_aug group_norm group_model group_datasets

  #  run_predict $gpu_device_0 $EXP_TYPE $test_data_path group_aug group_norm group_model group_datasets
#    run_predict_relabel $gpu_device_0 $NEW_TYPE $test_case_16_microvolt_path group_aug group_norm group_model group_datasets
  done
done

