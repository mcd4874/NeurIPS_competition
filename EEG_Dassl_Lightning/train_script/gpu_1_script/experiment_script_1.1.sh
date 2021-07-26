#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate tf-gpu
conda activate tensorflow2

#DIR="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/train_script/common_func_script"
#/home/wduong/tmp/Dassl_pytorch/
DIR="/home/wduong/tmp/EEG_Dassl_Lightning/train_script/common_func_script"

source "${DIR}/common_script.sh"

#norm_none="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"
#norm_none="/home/wduong/tmp/Dassl_pytorch/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"
#norm_none="/data1/wduong_experiment_data/Dassl_pytorch/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"


#group_model=("$adaptation_prefix" "$adapt_equal_label_prefix")
#group_model=("$adapt_dann_prefix" "$adapt_equal_dann_prefix")

#group_model=("$adaptation_DSBN_prefix" "$adapt_equal_label_DSBN_prefix")
#group_model=("$adapt_dann_DSBN_prefix" "$adapt_equal_dann_DSBN_prefix")
#group_model=("$adapt_equal_cdan_prefix" "$adapt_equal_cdan_DSBN_prefix")
#group_model=("$adapt_dan_prefix" "$adapt_equal_dan_prefix")


#group_model=("$adapt_dan_dann_prefix" "$adapt_equal_dan_dann_prefix")
#group_model=("$adapt_dan_dann_DSBN_prefix" "$adapt_equal_dan_dann_DSBN_prefix")

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

group_aug=("$temp_aug")
#group_aug=("$no_aug")

#group_norm=("$chan_norm")
group_norm=("$no_norm")

#group_datasets=("$BCI_IV_dataset")
#group_datasets=("$Cho2017_dataset")
#group_datasets=("$Physionet")
group_datasets=("$Dataset_B_dataset")
group_model=("$vanilla_prefix" "$adaptation_prefix")

#group_model=("$component_adapt_prefix" "$vanilla_prefix" "$adaptation_prefix")
#group_model=("$adaptation_prefix")
#run_full_multi_gpu $gpu_device_1 $experiment_2 group_aug group_norm group_model group_datasets

run_full_multi_gpu $gpu_device_1 $final_result_3 group_aug group_norm group_model group_datasets
