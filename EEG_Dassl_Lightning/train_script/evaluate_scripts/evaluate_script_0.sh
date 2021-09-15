#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tf-gpu
#conda activate tensorflow2


DIR="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/train_script/common_func_script"
#DIR="/home/wduong/tmp/EEG_Dassl_Lightning/train_script/common_func_script"
source "${DIR}/common_script.sh"

prefix_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
train_script="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
ROOT="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition"
predict_script="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
test_data_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data/NeurIPS_TL.mat"
test_data_microvolt_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data_microvolt/NeurIPS_TL.mat"
test_data_volt_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data_volt/NeurIPS_TL.mat"
test_case_13_microvolt_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_13_microvolt/NeurIPS_TL.mat"

test_case_16_microvolt_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_16_microvolt/NeurIPS_TL.mat"


function run_predict_relabel() {
    #run a group of experiment for mroe efficient
    #treat GPU_device, exp_type, and normalize_prefix as individual variable
    #treat DATASET,train_model_prefix as a list
    local GPU_device=$1
    local EXP_TYPE=$2
    local test_path=$3
    local -n LIST_AUG_PREFIX=$4
    local -n LIST_NORMALIZE_PREFIX=$5
    local -n TRAINER_MODEL_PREFIXS=$6
    local -n DATASETS=$7

    printf '1: %q\n' "${TRAINER_MODEL_PREFIXS[@]}"
    printf '2: %q\n' "${DATASETS[@]}"
    echo $test_path

    echo TRAINER_MODEL_PREFIXS
    for AUG_PREFIX in "${LIST_AUG_PREFIX[@]}";
    do
      for NORMALIZE_PREFIX in "${LIST_NORMALIZE_PREFIX[@]}";
      do
        for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
        do
          for DATASET in "${DATASETS[@]}";
          do
            OUTPUT_DIR="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            MAIN_CONFIG="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"
            echo $OUTPUT_DIR
            echo $MAIN_CONFIG
#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel
          done
        done
      done
    done
}

function run_full_active_learning() {
    #run a group of experiment for mroe efficient
    #treat GPU_device, exp_type, and normalize_prefix as individual variable
    #treat DATASET,train_model_prefix as a list
    local GPU_device=$1
#    local PRETRAIN_EXP_TYPE=$2
#    local USE_BEST_PRETRAIN=$3
#    local EXP_TYPE=$4
#    local USE_BEST_EXP=$5
#    local AL_EXP_TYPE=$6

    local test_path=$2
    local -n EXP_LIST=$3
    local -n USE_BEST_LIST=$4
    local -n LIST_AUG_PREFIX=$5
    local -n LIST_NORMALIZE_PREFIX=$6
    local -n TRAINER_MODEL_PREFIXS=$7
    local -n DATASETS=$8
    printf '1: %q\n' "${TRAINER_MODEL_PREFIXS[@]}"
    printf '2: %q\n' "${DATASETS[@]}"

    local PRETRAIN_EXP_TYPE=${EXP_LIST[0]}
    local USE_BEST_PRETRAIN=${USE_BEST_LIST[0]}
    local EXP_TYPE=${EXP_LIST[1]}
    local USE_BEST_EXP=${USE_BEST_LIST[2]}
    local AL_EXP_TYPE=${EXP_LIST[2]}

    echo $TRAINER_MODEL_PREFIXS
    for AUG_PREFIX in "${LIST_AUG_PREFIX[@]}";
    do
      for NORMALIZE_PREFIX in "${LIST_NORMALIZE_PREFIX[@]}";
      do
        for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
        do
          for DATASET in "${DATASETS[@]}";
          do
            PRETRAIN_DIR="${prefix_path}${PRETRAIN_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            PRETRAIN_CONFIG="${prefix_path}${PRETRAIN_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"


            MAIN_DIR="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            MAIN_CONFIG="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"

            AL_DIR="${prefix_path}${AL_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            AL_CONFIG="${prefix_path}${AL_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"

            echo $PRETRAIN_DIR
            echo $PRETRAIN_CONFIG

            echo $MAIN_DIR
            echo $MAIN_CONFIG

            echo $AL_DIR
            echo $AL_CONFIG

            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$PRETRAIN_DIR" --main-config-file "$PRETRAIN_CONFIG"
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$PRETRAIN_DIR" --main-config-file "$PRETRAIN_CONFIG" --eval-only --model-dir "$PRETRAIN_DIR"

            if [ $USE_BEST_PRETRAIN == "TRUE" ]
            then
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir "$PRETRAIN_DIR" --use_pretrain_best
            else
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir "$PRETRAIN_DIR"
            fi
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --eval-only --model-dir "$MAIN_DIR"
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel

            if [ $USE_BEST_EXP == "TRUE" ]
            then
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$MAIN_DIR" --use_pretrain_best --test-data  "$test_path"
            else
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$MAIN_DIR" --test-data  "$test_path"
            fi
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --eval-only --model-dir "$AL_DIR"
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --relabel
          done
        done
      done
    done
}

#prefix_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/"
#train_script="/home/wduong/tmp/EEG_Dassl_Lightning/"
#ROOT="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition"
#predict_script="/home/wduong/tmp/EEG_Dassl_Lightning/"
##test_data_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data/NeurIPS_TL.mat"
#test_data_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data_volt/NeurIPS_TL.mat"


#/home/wduong/tmp/Dassl_pytorch/
#DIR="/home/wduong/tmp/EEG_Dassl_Lightning/train_script/common_func_script"

group_aug=("")

group_norm=("")

#group_aug=("$temp_aug")
#group_aug=("$no_aug")
#
#group_norm=("$no_norm")

#group_datasets=("$BCI_IV_dataset")
#group_datasets=("$Cho2017_dataset")
#group_datasets=("$Physionet")
group_datasets=("$Dataset_B_dataset")
#group_datasets=("$Dataset_B_dataset")

LIST_EXP_TYPE=("$final_result_14_3_1")
#sub_list=("sub" "sub_0" "sub_1" "sub_2")
#group_model=("$adaptationV1_prefix" "$dannV1_prefix")
sub_list=("sub_30")
#sub_list=("pretrain_0")
#group_model=("$adaptationV1_prefix")
group_model=("$mcdV1_prefix")
#group_model=("$SRDA_prefix")
PRETRAIN_EXP_TYPE="test/final_result_14_3_1/pretrain_model"
USE_BEST_PRETRAIN="False"
EXP_TYPE="test/final_result_14_3_1/adapt_model"
USE_BEST_EXP="True"
AL_EXP_TYPE="test/final_result_14_3_1/al_model"
test_path="$test_case_16_microvolt_path"

EXP_SETUP=("$PRETRAIN_EXP_TYPE" "$EXP_TYPE" "$AL_EXP_TYPE")
BEST_SETUP=("$USE_BEST_PRETRAIN" "$USE_BEST_EXP")
echo ${EXP_SETUP[0]}
echo ${EXP_SETUP[1]}

run_full_active_learning $gpu_device_0 $test_path EXP_SETUP BEST_SETUP group_aug group_norm group_model group_datasets
