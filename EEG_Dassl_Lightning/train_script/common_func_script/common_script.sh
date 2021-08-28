#    prefix_path=""
#prefix_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
#train_script="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
#ROOT="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition"
#predict_script="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
#test_data_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data/NeurIPS_TL.mat"
#test_data_microvolt_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data_microvolt/NeurIPS_TL.mat"
#test_data_volt_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data_volt/NeurIPS_TL.mat"


prefix_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/"
train_script="/home/wduong/tmp/EEG_Dassl_Lightning/"
ROOT="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition"
predict_script="/home/wduong/tmp/EEG_Dassl_Lightning/"
test_data_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data/NeurIPS_TL.mat"
test_data_microvolt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data_microvolt/NeurIPS_TL.mat"
test_data_volt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data_volt/NeurIPS_TL.mat"
test_case_10_microvolt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_10_microvolt/NeurIPS_TL.mat"
test_case_11_microvolt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_11_microvolt/NeurIPS_TL.mat"
test_case_12_microvolt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_12_microvolt/NeurIPS_TL.mat"

test_case_13_microvolt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_13_microvolt/NeurIPS_TL.mat"
test_case_14_microvolt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_14_microvolt/NeurIPS_TL.mat"
test_case_15_microvolt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_15_microvolt/NeurIPS_TL.mat"
test_case_16_microvolt_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_case_16_microvolt/NeurIPS_TL.mat"

function run_train_only() {
    #run a group of experiment for mroe efficient
    #treat GPU_device, exp_type, and normalize_prefix as individual variable
    #treat DATASET,train_model_prefix as a list
    local GPU_device=$1
    local EXP_TYPE=$2
    local -n LIST_AUG_PREFIX=$3
    local -n LIST_NORMALIZE_PREFIX=$4
    local -n TRAINER_MODEL_PREFIXS=$5
    local -n DATASETS=$6
    printf '1: %q\n' "${TRAINER_MODEL_PREFIXS[@]}"
    printf '2: %q\n' "${DATASETS[@]}"

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
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds
          done
        done
      done
    done
}

function run_pretrain() {
    #run a group of experiment for mroe efficient
    #treat GPU_device, exp_type, and normalize_prefix as individual variable
    #treat DATASET,train_model_prefix as a list
    local GPU_device=$1
    local EXP_TYPE=$2
    local PRETRAIN_EXP_TYPE=$3
    local -n LIST_AUG_PREFIX=$4
    local -n LIST_NORMALIZE_PREFIX=$5
    local -n TRAINER_MODEL_PREFIXS=$6
    local -n DATASETS=$7
    printf '1: %q\n' "${TRAINER_MODEL_PREFIXS[@]}"
    printf '2: %q\n' "${DATASETS[@]}"

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
            PRETRAIN_DIR="${prefix_path}${PRETRAIN_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            echo $OUTPUT_DIR
            echo $MAIN_CONFIG
            echo $PRETRAIN_DIR

            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --pretrain-dir $PRETRAIN_DIR
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --eval-only --model-dir $OUTPUT_DIR
          done
        done
      done
    done
}

function run_full_multi_gpu() {
    #run a group of experiment for mroe efficient
    #treat GPU_device, exp_type, and normalize_prefix as individual variable
    #treat DATASET,train_model_prefix as a list
    local GPU_device=$1
    local EXP_TYPE=$2
    local -n LIST_AUG_PREFIX=$3
    local -n LIST_NORMALIZE_PREFIX=$4
    local -n TRAINER_MODEL_PREFIXS=$5
    local -n DATASETS=$6
    printf '1: %q\n' "${TRAINER_MODEL_PREFIXS[@]}"
    printf '2: %q\n' "${DATASETS[@]}"

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

            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --eval-only --model-dir $OUTPUT_DIR
          done
        done
      done
    done
}

function run_al_full() {
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

            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --test-data $test_path
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --eval-only --model-dir $OUTPUT_DIR --test-data $test_path
          done
        done
      done
    done
}

function run_predict() {
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
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path
          done
        done
      done
    done
}

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
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel
          done
        done
      done
    done
}

function run_ensemble_predict() {
    #run a group of experiment for mroe efficient
    #treat GPU_device, exp_type, and normalize_prefix as individual variable
    #treat DATASET,train_model_prefix as a list
    local GPU_device=$1
    local EXP_TYPE=$2
    local -n LIST_AUG_PREFIX=$3
    local -n LIST_NORMALIZE_PREFIX=$4
    local -n TRAINER_MODEL_PREFIXS=$5
    local -n DATASETS=$6
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

            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --generate-predict --use-assemble-test-dataloader
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --use-assemble-test-dataloader

          done
        done
      done
    done
}


#
#
gpu_device_0=0
gpu_device_1=1
gpu_device_2=2
gpu_device_3=3



experiment_1="NeurIPS_competition/experiment_1"
experiment_2="NeurIPS_competition/experiment_2"
experiment_3="NeurIPS_competition/experiment_3"
experiment_4="NeurIPS_competition/experiment_4"
#experiment_4_2="NeurIPS_competition/experiment_4_2"
experiment_4_4="NeurIPS_competition/experiment_4_4"
experiment_4_5="NeurIPS_competition/experiment_4_5"
experiment_5="NeurIPS_competition/experiment_5"
experiment_5_1="NeurIPS_competition/experiment_5_1"

experiment_6="NeurIPS_competition/experiment_6"
experiment_7="NeurIPS_competition/experiment_7"

experiment_9_0_1="NeurIPS_competition/experiment_9_0_1"
experiment_9_0_3="NeurIPS_competition/experiment_9_0_3"

experiment_10_0_1="NeurIPS_competition/experiment_10_0_1"
experiment_10_0_3="NeurIPS_competition/experiment_10_0_3"

experiment_11_0_1="NeurIPS_competition/experiment_11_0_1"
experiment_11_0_2="NeurIPS_competition/experiment_11_0_2"
experiment_11_0_3="NeurIPS_competition/experiment_11_0_3"

final_result_3="NeurIPS_competition/final_result_3"
final_result_4="NeurIPS_competition/final_result_4"
final_result_4_1_1="NeurIPS_competition/final_result_4_1_1"
final_result_4_3="NeurIPS_competition/final_result_4_3"
final_result_4_3_1="NeurIPS_competition/final_result_4_3_1"

final_result_4_5="NeurIPS_competition/final_result_4_5"
final_result_5_1="NeurIPS_competition/final_result_5_1"
final_result_6="NeurIPS_competition/final_result_6"
final_result_7="NeurIPS_competition/final_result_7"
final_result_7_0_1="NeurIPS_competition/final_result_7_0_1"
final_result_7_0_2="NeurIPS_competition/final_result_7_0_2"
final_result_7_1="NeurIPS_competition/final_result_7_1"
final_result_7_1_1="NeurIPS_competition/final_result_7_1_1"
final_result_7_1_2="NeurIPS_competition/final_result_7_1_2"
final_result_7_1_3="NeurIPS_competition/final_result_7_1_3"
final_result_7_1_4="NeurIPS_competition/final_result_7_1_4"

final_result_8="NeurIPS_competition/final_result_8"
final_result_8_0_1="NeurIPS_competition/final_result_8_0_1"
final_result_8_0_2="NeurIPS_competition/final_result_8_0_2"
final_result_8_0_3="NeurIPS_competition/final_result_8_0_3"

final_result_8_1_3="NeurIPS_competition/final_result_8_1_3"

final_result_8_2_1="NeurIPS_competition/final_result_8_2_1"
final_result_8_2_2="NeurIPS_competition/final_result_8_2_2"
final_result_8_2_3="NeurIPS_competition/final_result_8_2_3"

final_result_9_0_1="NeurIPS_competition/final_result_9_0_1"
final_result_9_0_2="NeurIPS_competition/final_result_9_0_2"
final_result_9_0_3="NeurIPS_competition/final_result_9_0_3"

final_result_10_0_1="NeurIPS_competition/final_result_10_0_1"
final_result_10_0_3="NeurIPS_competition/final_result_10_0_3"

final_result_11_0_1="NeurIPS_competition/final_result_11_0_1"
final_result_11_0_2="NeurIPS_competition/final_result_11_0_2"
final_result_11_0_3="NeurIPS_competition/final_result_11_0_3"
final_result_11_1_3="NeurIPS_competition/final_result_11_1_3"
final_result_11_2_3="NeurIPS_competition/final_result_11_2_3"
final_result_11_3_3="NeurIPS_competition/final_result_11_3_3"

final_result_11_4_1="NeurIPS_competition/final_result_11_4_1"
final_result_11_4_3="NeurIPS_competition/final_result_11_4_3"

final_result_11_4_1_0="NeurIPS_competition/final_result_11_4_1_0"
final_result_11_4_1_1="NeurIPS_competition/final_result_11_4_1_1"
final_result_11_4_1_2="NeurIPS_competition/final_result_11_4_1_2"

final_result_11_4_3_0="NeurIPS_competition/final_result_11_4_3_0"
final_result_11_4_3_1="NeurIPS_competition/final_result_11_4_3_1"
final_result_11_4_3_2="NeurIPS_competition/final_result_11_4_3_2"

final_result_12_4_1="NeurIPS_competition/final_result_12_4_1"
final_result_12_4_1_0="NeurIPS_competition/final_result_12_4_1_0"
final_result_12_4_1_1="NeurIPS_competition/final_result_12_4_1_1"
final_result_12_4_1_2="NeurIPS_competition/final_result_12_4_1_2"
final_result_12_4_3="NeurIPS_competition/final_result_12_4_3"
final_result_12_4_3_0="NeurIPS_competition/final_result_12_4_3_0"
final_result_12_4_3_1="NeurIPS_competition/final_result_12_4_3_1"
final_result_12_4_3_2="NeurIPS_competition/final_result_12_4_3_2"

final_result_12_3_1="NeurIPS_competition/final_result_12_3_1"
final_result_12_3_3="NeurIPS_competition/final_result_12_3_3"


final_result_13_4_0="NeurIPS_competition/final_result_13_4_0"
final_result_13_4_1="NeurIPS_competition/final_result_13_4_1"
final_result_13_4_2="NeurIPS_competition/final_result_13_4_2"
final_result_13_4_3="NeurIPS_competition/final_result_13_4_3"

final_result_14_0_1="NeurIPS_competition/final_result_14_0_1"
final_result_14_0_1_0="NeurIPS_competition/final_result_14_0_1_0"
final_result_14_0_1_1="NeurIPS_competition/final_result_14_0_1_1"
final_result_14_0_1_2="NeurIPS_competition/final_result_14_0_1_2"

final_result_14_0_2="NeurIPS_competition/final_result_14_0_2"
final_result_14_0_2_0="NeurIPS_competition/final_result_14_0_2_0"
final_result_14_0_2_1="NeurIPS_competition/final_result_14_0_2_1"
final_result_14_0_2_2="NeurIPS_competition/final_result_14_0_2_2"

final_result_14_0_3="NeurIPS_competition/final_result_14_0_3"
final_result_14_0_3_0="NeurIPS_competition/final_result_14_0_3_0"
final_result_14_0_3_1="NeurIPS_competition/final_result_14_0_3_1"
final_result_14_0_3_2="NeurIPS_competition/final_result_14_0_3_2"

final_result_14_3_1="NeurIPS_competition/final_result_14_3_1"

final_result_14_3_3="NeurIPS_competition/final_result_14_3_3"

private_exp_14_3_1="NeurIPS_competition/private_exp_14_3_1"
private_exp_14_3_3="NeurIPS_competition/private_exp_14_3_3"


T_F_aug="T_F_aug"
temp_aug="temp_aug"
no_aug="no_aug"
chan_norm="chan_norm"
no_norm="no_norm"

adaptation_prefix="adaptation"
adaptationV1_prefix="adaptationV1"
share_adaptV1_prefix="share_adaptV1"

dannV1_prefix="dannV1"
mcdV1_prefix="mcdV1"
addaV1_prefix="addaV1"
SRDA_prefix="SRDA"

shallowcon_adaptV1_prefix="shallowcon_adaptV1"
FBCNET_adaptV1_prefix="FBCNET_adaptV1"
vanilla_prefix="vanilla"
component_adapt_prefix="component_adapt"

BCI_IV_dataset="BCI_IV"
Cho2017_dataset="Cho2017"
Physionet_dataset="Physionet"

Dataset_A_dataset="dataset_A"
Dataset_B_dataset="dataset_B"