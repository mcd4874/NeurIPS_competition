#!/bin/bash
#eval "$(conda shell.bash hook)"
#conda activate tf-gpu
#conda activate tensorflow2

#conda activate beetl

computer_dir="/home/vernon/software/NeurIPS_competition/EEG_Lightning/"


#computer_dir="../"

DIR="${computer_dir}/submission/common_func_script"
source "${DIR}/common_script.sh"

prefix_path="${computer_dir}"
train_script="${computer_dir}"
ROOT="${computer_dir}/da_dataset/task_2"
predict_script="${computer_dir}"

MAIN_EXP_TYPE="submission/task_2"



TRAINER_MODEL_PREFIXS=("$mcdV1_prefix")

DATASETS=("$Dataset_A_0_dataset" "$Dataset_A_1_dataset" "$Dataset_A_2_dataset")

TEST_DATASET_CASE="final_MI_test_A_1"

for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
do
  for DATASET in "${DATASETS[@]}";
  do

    EXP_TYPE="${MAIN_EXP_TYPE}"
    MAIN_DIR="${prefix_path}/${EXP_TYPE}/${TRAINER_MODEL_PREFIX}/${DATASET}"
    test_path="${ROOT}/${TEST_DATASET_CASE}/${DATASET}/NeurIPS_TL.mat"
    echo $MAIN_DIR
    run_simple_train $gpu_device_0 $test_path $MAIN_DIR

  done
done


DATASETS=("$Dataset_B_0_dataset" "$Dataset_B_1_dataset")
TEST_DATASET_CASE="final_MI_test_B_1"
for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
do
  for DATASET in "${DATASETS[@]}";
  do

    EXP_TYPE="${MAIN_EXP_TYPE}"
    MAIN_DIR="${prefix_path}/${EXP_TYPE}/${TRAINER_MODEL_PREFIX}/${DATASET}"
    test_path="${ROOT}/${TEST_DATASET_CASE}/${DATASET}/NeurIPS_TL.mat"
    echo $MAIN_DIR
    run_simple_train $gpu_device_0 $test_path $MAIN_DIR

  done
done
