!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tf-gpu
#conda activate tensorflow2


#computer_dir="/home/wduong/tmp/EEG_Lightning"
#data_path_dir="/data1/wduong_experiment_data/EEG_Lightning/"

computer_dir="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Lightning/"
data_path_dir="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Lightning/"


DIR="${computer_dir}/train_script/common_func_script"
source "${DIR}/common_script.sh"

prefix_path="${data_path_dir}"
train_script="${computer_dir}"
predict_script="${computer_dir}"
ROOT="${data_path_dir}/da_dataset/NeurIPS_1"
test_case_path="${data_path_dir}/da_dataset/NeurIPS_1/task_1_final_test_case_1/full_test_sleep.mat"

group_aug=("$no_aug")
group_norm=("$no_norm")
group_datasets=("$full_dataset")
group_model=("$deepsleep_vanilla_prefix")

task_exp="NeurIPS_1/task_1_final_1/quick_ver_1_1"
tune_cla_exp="NeurIPS_1/task_1_final_1/quick_ver_1_1/tune_cla"
pretrain_best="True"

run_full_multi_gpu $gpu_device_0 $task_exp group_aug group_norm group_model group_datasets
run_predict_task_1 $gpu_device_0 $task_exp $test_case_path group_aug group_norm group_model group_datasets

run_pretrain $gpu_device_0 $tune_cla_exp $task_exp $pretrain_best group_aug group_norm group_model group_datasets
run_predict_task_1 $gpu_device_0 $tune_cla_exp $test_case_path group_aug group_norm group_model group_datasets

#run_al_full $gpu_device_0 $task_exp $test_case_path $model_update_path group_aug group_norm group_model group_datasets
#run_predict_task_1 $gpu_device_0 $task_exp $test_case_path group_aug group_norm group_model group_datasets
