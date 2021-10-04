#!/bin/bash
#eval "$(conda shell.bash hook)"
#conda activate tf-gpu
#conda activate tensorflow2
#conda activate beetl


computer_dir="/home/vernon/software/NeurIPS_competition/EEG_Lightning/"



DIR="${computer_dir}/submission/common_func_script"
source "${DIR}/common_script.sh"

prefix_path="${computer_dir}"
train_script="${computer_dir}"
predict_script="${computer_dir}"
ROOT="${computer_dir}/da_dataset/task_1"
test_case_path="${ROOT}/task_1_final_test_case_1/full_test_sleep.mat"

group_aug=("")
group_norm=("")
group_datasets=("$full_dataset")
group_model=("$deepsleep_vanilla_prefix")
task_exp="submission/task_1"
tune_cla_exp="submission/task_1/tune_cla"

pretrain_best="True"

run_full_multi_gpu $gpu_device_0 $task_exp group_aug group_norm group_model group_datasets
run_predict_task_1 $gpu_device_0 $task_exp $test_case_path group_aug group_norm group_model group_datasets

run_pretrain $gpu_device_0 $tune_cla_exp $task_exp $pretrain_best group_aug group_norm group_model group_datasets
run_predict_task_1 $gpu_device_0 $tune_cla_exp $test_case_path group_aug group_norm group_model group_datasets
