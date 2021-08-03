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
#group_aug=("$temp_aug")
group_aug=("$no_aug")

group_norm=("$chan_norm")
#group_norm=("$no_norm")

#group_datasets=("$BCI_IV_dataset")
#group_datasets=("$Cho2017_dataset")
#group_datasets=("$Physionet")
group_datasets=("$Dataset_A_dataset")
#group_datasets=("$Dataset_B_dataset")

#group_model=("$vanilla_prefix" "$adaptation_prefix")
#group_model=("$adaptationV1_prefix" "$vanilla_prefix")
#group_model=("$adaptation_prefix")
group_model=("$FBCNET_adaptV1_prefix")

#group_model=("$component_adapt_prefix" "$vanilla_prefix" "$adaptation_prefix")

#run_full_multi_gpu $gpu_device_0 $experiment_2 group_aug group_norm group_model group_datasets

#run_full_multi_gpu $gpu_device_0 $experiment_5 group_aug group_norm group_model group_datasets
#run_full_multi_gpu $gpu_device_0 $final_result_3 group_aug group_norm group_model group_datasets
run_full_multi_gpu $gpu_device_1 $final_result_4 group_aug group_norm group_model group_datasets

#run_full_multi_gpu $gpu_device_0 $final_result_6 group_aug group_norm group_model group_datasets
