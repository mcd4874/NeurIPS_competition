#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate tf-gpu
conda activate tensorflow2


#DIR="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/train_script/common_func_script"
#/home/wduong/tmp/Dassl_pytorch/
DIR="/home/wduong/tmp/Dassl_pytorch/train_script/common_func_script"

source "${DIR}/common_script.sh"

norm_none="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"
#norm_none="/home/wduong/tmp/Dassl_pytorch/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"
#norm_none="/data1/wduong_experiment_data/Dassl_pytorch/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"


#group_model=("$vanilla_prefix" "$vanilla_equal_prefix")
#group_model=("$adaptation_prefix" "$adapt_equal_label_prefix")
#group_model=("$adapt_dann_prefix" "$adapt_equal_dann_prefix")
#group_model=("$adaptation_DSBN_prefix" "$adapt_equal_label_DSBN_prefix")

#group_model=("$adapt_dann_DSBN_prefix" "$adapt_equal_dann_DSBN_prefix")
#group_model=("$adapt_equal_cdan_prefix" "$adapt_equal_cdan_DSBN_prefix")
#group_model=("$adapt_dan_prefix" "$adapt_equal_dan_prefix")
#group_model=("$adapt_dan_dann_prefix" "$adapt_equal_dan_dann_prefix")
group_model=("$adapt_dan_dann_DSBN_prefix" "$adapt_equal_dan_dann_DSBN_prefix")



group_seed=("seed_v0" "seed_v1" "seed_v2" "seed_v3" "seed_v4" "seed_v5" "seed_v6" "seed_v7" "seed_v8")
group_datasets=("$BCI_IV_MI_prefix" )
run_full $gpu_device_1 $experiment_case_1 $norm_none_prefix group_seed group_model group_datasets