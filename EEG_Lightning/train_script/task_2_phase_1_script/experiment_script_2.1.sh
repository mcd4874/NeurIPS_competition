#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate tf-gpu
#conda activate tensorflow2

conda activate beetl

#computer_dir="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Lightning/"
#experiment_dir="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Lightning/"
#data_path_dir="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Lightning/"

computer_dir="/home/wduong/tmp/EEG_Lightning/"
experiment_dir="/data1/wduong_experiment_data/EEG_Lightning/"
data_path_dir="/data1/wduong_experiment_data/EEG_Lightning/"



DIR="${computer_dir}/train_script/common_func_script"
source "${DIR}/common_script.sh"

prefix_path="${experiment_dir}"
train_script="${computer_dir}"
ROOT="${data_path_dir}/da_dataset/NeurIPS_4"
predict_script="${computer_dir}"


PRETRAIN="NeurIPS_4/phase_1_best/0"
MODEL_UPDATE="NeurIPS_4/phase_1_best/0"
MAIN_EXP_TYPE="NeurIPS_4/phase_1_al_pretrain"

LIST_EXP_TYPES=("1_0" "1_1")

LIST_AUG_PREFIX=("$no_aug")
LIST_NORMALIZE_PREFIX=("$no_norm")
TRAINER_MODEL_PREFIXS=("$mcdV1_prefix")

DATASETS=("$Dataset_A_0_dataset" "$Dataset_A_1_dataset")
TEST_DATASET_CASE="phase_1_MI_test_A_1"
for SUB_EXP_TYPE in "${LIST_EXP_TYPES[@]}";
do
  for AUG_PREFIX in "${LIST_AUG_PREFIX[@]}";
  do
    for NORMALIZE_PREFIX in "${LIST_NORMALIZE_PREFIX[@]}";
    do
      for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
      do
        for DATASET in "${DATASETS[@]}";
        do

          EXP_TYPE="${MAIN_EXP_TYPE}/${SUB_EXP_TYPE}"
          MAIN_DIR="${prefix_path}/${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
          test_path="${ROOT}/${TEST_DATASET_CASE}/${DATASET}/NeurIPS_TL.mat"

          echo $PRETRAIN_DIR
          echo $MAIN_DIR
#          run_simple_train $gpu_device_1 $test_path $MAIN_DIR

          USE_BEST_PRETRAIN_EXP="True"
          PRETRAIN_DIR="${prefix_path}/${PRETRAIN}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
          MODEL_UPDATE_DIR="${prefix_path}/${MODEL_UPDATE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
#          PRETRAIN_DIR="empty"
          echo $MODEL_UPDATE_DIR
          run_simple_active_learning $gpu_device_1 $test_path $PRETRAIN_DIR $USE_BEST_PRETRAIN_EXP $MAIN_DIR $MODEL_UPDATE_DIR

        done
      done
    done
  done
done
#
DATASETS=("$Dataset_B_0_dataset" "$Dataset_B_1_dataset" "$Dataset_B_2_dataset")
#DATASETS=("$Dataset_B_2_dataset")
PRETRAIN_DATASET="dataset_B"
TEST_DATASET_CASE="phase_1_MI_test_B_1"
for SUB_EXP_TYPE in "${LIST_EXP_TYPES[@]}";
do
  for AUG_PREFIX in "${LIST_AUG_PREFIX[@]}";
  do
    for NORMALIZE_PREFIX in "${LIST_NORMALIZE_PREFIX[@]}";
    do
      for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
      do
        for DATASET in "${DATASETS[@]}";
        do
          EXP_TYPE="${MAIN_EXP_TYPE}/${SUB_EXP_TYPE}"
          MAIN_DIR="${prefix_path}/${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
          test_path="${ROOT}/${TEST_DATASET_CASE}/${DATASET}/NeurIPS_TL.mat"

          echo $PRETRAIN_DIR
          echo $MAIN_DIR
#          run_simple_train $gpu_device_1 $test_path $MAIN_DIR

          USE_BEST_PRETRAIN_EXP="True"
          PRETRAIN_DIR="${prefix_path}/${PRETRAIN}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
          MODEL_UPDATE_DIR="${prefix_path}/${MODEL_UPDATE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
#          PRETRAIN_DIR="empty"
          echo $MODEL_UPDATE_DIR
          run_simple_active_learning $gpu_device_1 $test_path $PRETRAIN_DIR $USE_BEST_PRETRAIN_EXP $MAIN_DIR $MODEL_UPDATE_DIR

        done
      done
    done
  done
done