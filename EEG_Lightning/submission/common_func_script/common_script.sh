


function run_pretrain() {
    #run a group of experiment for mroe efficient
    #treat GPU_device, exp_type, and normalize_prefix as individual variable
    #treat DATASET,train_model_prefix as a list
    local GPU_device=$1
    local EXP_TYPE=$2
    local PRETRAIN_EXP_TYPE=$3
    local USE_BEST_PRETRAIN=$4
    local -n LIST_AUG_PREFIX=$5
    local -n LIST_NORMALIZE_PREFIX=$6
    local -n TRAINER_MODEL_PREFIXS=$7
    local -n DATASETS=$8
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
            if [ "$USE_BEST_PRETRAIN" == "True" ];
            then
              CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir $PRETRAIN_DIR --use_pretrain_best
            else
              CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir $PRETRAIN_DIR
            fi
            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --eval-only --model-dir $OUTPUT_DIR
          done
        done
      done
    done
}


function run_simple_train(){
    local GPU_device=$1
    local test_path=$2
    local MAIN=$3

    MAIN_CONFIG="${MAIN}/main_config/transfer_adaptation.yaml"
    MAIN_DIR="${MAIN}/model"

    echo "start pretrain"

    echo $MAIN_DIR

#    CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG"
#    CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
#    CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
#    CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel

#    CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG"
#    CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
#    CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
#    CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel
}

function run_simple_active_learning(){
    local GPU_device=$1
    local test_path=$2
    local PRETRAIN_DIR=$3
    local USE_BEST_EXP=$4
    local AL_DIR=$5
    local MODEL_UPDATE_AL_DIR=$6
    AL_CONFIG="${AL_DIR}/main_config/transfer_adaptation.yaml"
    AL_DIR="${AL_DIR}/model"
    MODEL_UPDATE_AL_DIR="${MODEL_UPDATE_AL_DIR}/model"
    if [ "$PRETRAIN_DIR" != "empty" ];
    then
      PRETRAIN_PATH="${PRETRAIN_DIR}/model"
      echo "conduct al with pretrain model"
      echo $PRETRAIN_PATH
      echo $USE_BEST_EXP
      if [ "$USE_BEST_EXP" == "True" ];
      then
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$PRETRAIN_PATH" --use_pretrain_best --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
      else
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$PRETRAIN_PATH" --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
      fi
    else
      echo "run only al learning strategy"
      echo $AL_DIR
      echo $MODEL_UPDATE_AL_DIR
      CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
    fi
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --eval-only
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --generate-predict
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --relabel
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

            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG"
            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
          done
        done
      done
    done
}


function run_pretrain_al_full() {
    #run a group of experiment for mroe efficient
    #treat GPU_device, exp_type, and normalize_prefix as individual variable
    #treat DATASET,train_model_prefix as a list
    local GPU_device=$1
    local EXP_TYPE=$2
    local test_path=$3
    local model_update_al=$4
    local pretrain_exp=$5
    local -n LIST_AUG_PREFIX=$6
    local -n LIST_NORMALIZE_PREFIX=$7
    local -n TRAINER_MODEL_PREFIXS=$8
    local -n DATASETS=$9
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
            PRETRAIN_DIR="${prefix_path}${pretrain_exp}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            MODEL_UPDATE_AL_DIR="${prefix_path}${model_update_al}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            OUTPUT_DIR="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            MAIN_CONFIG="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"

            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --test-data $test_path --model-update-dir "$MODEL_UPDATE_AL_DIR" --use_pretrain_best --pretrain-dir "$PRETRAIN_DIR"
            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
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
    local model_update_al=$4
    local -n LIST_AUG_PREFIX=$5
    local -n LIST_NORMALIZE_PREFIX=$6
    local -n TRAINER_MODEL_PREFIXS=$7
    local -n DATASETS=$8
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
            MODEL_UPDATE_AL_DIR="${prefix_path}${model_update_al}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            OUTPUT_DIR="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            MAIN_CONFIG="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"

            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --test-data $test_path --model-update-dir "$MODEL_UPDATE_AL_DIR"
            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
          done
        done
      done
    done
}

function run_predict_task_1() {
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
            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict_task_1.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
            CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict_task_1.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path
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

temp_aug="temp_aug"
no_aug="no_aug"
chan_norm="chan_norm"
no_norm="no_norm"
time_norm="time_norm"

adaptation_prefix="adaptation"
adaptationV1_prefix="adaptationV1"


deepsleep_share_adaptV1_prefix="deepsleep_share_adaptV1"
deepsleep_share_mcd_prefix="deepsleep_share_mcd"


dannV1_prefix="dannV1"
dann_prefix="dann"
deepsleep_dann_prefix="deepsleep_dann"

dan_prefix="dan"

mcdV1_prefix="mcdV1"
mcd_prefix="mcd"
deepsleep_mcd_prefix="deepsleep_mcd"
addaV1_prefix="addaV1"
SRDA_prefix="SRDA"
m3sda_prefix="m3sda"



shallowcon_adaptV1_prefix="shallowcon_adaptV1"
FBCNET_adaptV1_prefix="FBCNET_adaptV1"
vanilla_prefix="vanilla"
deepsleep_vanilla_prefix="deepsleep_vanilla"
eegnetsleep_vanilla_prefix="eegnetsleep_vanilla"


component_adapt_prefix="component_adapt"

BCI_IV_dataset="BCI_IV"
Cho2017_dataset="Cho2017"
Physionet_dataset="Physionet"

Dataset_A_dataset="dataset_A"
Dataset_B_dataset="dataset_B"

full_dataset="full_dataset"

Dataset_A_0_dataset="dataset_A_0"
Dataset_A_1_dataset="dataset_A_1"
Dataset_A_2_dataset="dataset_A_2"

Dataset_B_0_dataset="dataset_B_0"
Dataset_B_1_dataset="dataset_B_1"
Dataset_B_2_dataset="dataset_B_2"