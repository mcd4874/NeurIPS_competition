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

task_2_final_test_case="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/final_MI_test/NeurIPS_TL.mat"


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
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir $PRETRAIN_DIR --use_pretrain_best
            else
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir $PRETRAIN_DIR
            fi
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --eval-only --model-dir $OUTPUT_DIR
          done
        done
      done
    done
}


function run_simple_pretrain(){
    local GPU_device=$1
    local test_path=$2
    local PRETRAIN=$3
    local USE_BEST=$4
    local MAIN=$5

    PRETRAIN="${PRETRAIN}/model"
    MAIN_CONFIG="${MAIN}/main_config/transfer_adaptation.yaml"
    MAIN_DIR="${MAIN}/model"

    echo "start pretrain"
    echo $PRETRAIN
    echo $USE_BEST
    echo $MAIN_DIR
    if [ "$USE_BEST" == "True" ];
    then
      CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir "$PRETRAIN" --use_pretrain_best
    else
      CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir "$PRETRAIN"
    fi
    CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
    CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
    CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel
}

function run_simple_train(){
    local GPU_device=$1
    local test_path=$2
    local MAIN=$3

    MAIN_CONFIG="${MAIN}/main_config/transfer_adaptation.yaml"
    MAIN_DIR="${MAIN}/model"

    echo "start pretrain"

    echo $MAIN_DIR

    CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG"
    CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
    CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
    CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel
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
        CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$PRETRAIN_PATH" --use_pretrain_best --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
      else
        CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$PRETRAIN_PATH" --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
      fi
    else
      echo "run only al learning strategy"
      echo $AL_DIR
      echo $MODEL_UPDATE_AL_DIR
      CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
    fi
    CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --eval-only
    CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --generate-predict
    CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --relabel
}

function run_active_learning(){
    local GPU_device=$1
    local test_path=$2
    local model_update_al=$3
    local -n EXP_LIST=$4
    local -n USE_BEST_LIST=$5
    local -n LIST_AUG_PREFIX=$6
    local -n LIST_NORMALIZE_PREFIX=$7
    local -n TRAINER_MODEL_PREFIXS=$8
    local -n DATASETS=$9
    printf '1: %q\n' "${TRAINER_MODEL_PREFIXS[@]}"
    printf '2: %q\n' "${DATASETS[@]}"

    local EXP_TYPE=${EXP_LIST[0]}
    local USE_BEST_EXP=${USE_BEST_LIST[0]}
    local AL_EXP_TYPE=${EXP_LIST[1]}
    for AUG_PREFIX in "${LIST_AUG_PREFIX[@]}";
    do
      for NORMALIZE_PREFIX in "${LIST_NORMALIZE_PREFIX[@]}";
      do
        for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
        do
          for DATASET in "${DATASETS[@]}";
          do
            MAIN_DIR="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            AL_DIR="${prefix_path}${AL_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            AL_CONFIG="${prefix_path}${AL_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"
            MODEL_UPDATE_AL_DIR="${prefix_path}${model_update_al}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            if [ "$USE_BEST_EXP" == "True" ];
            then
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$MAIN_DIR" --use_pretrain_best --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
            else
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$MAIN_DIR" --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
            fi
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --eval-only
#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --generate-predict
#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --relabel
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
    local test_path=$2
    local model_update_al=$3
    local -n EXP_LIST=$4
    local -n USE_BEST_LIST=$5
    local -n LIST_AUG_PREFIX=$6
    local -n LIST_NORMALIZE_PREFIX=$7
    local -n TRAINER_MODEL_PREFIXS=$8
    local -n DATASETS=$9
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
            if [ "$PRETRAIN_EXP_TYPE" == "" ];
            then
              PRETRAIN_DIR=""
              PRETRAIN_CONFIG=""
              echo "no pretrain setup"
            else
              PRETRAIN_DIR="${prefix_path}${PRETRAIN_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
              PRETRAIN_CONFIG="${prefix_path}${PRETRAIN_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"
              echo $PRETRAIN_DIR
              echo $PRETRAIN_CONFIG
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$PRETRAIN_DIR" --main-config-file "$PRETRAIN_CONFIG"
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$PRETRAIN_DIR" --main-config-file "$PRETRAIN_CONFIG" --eval-only --model-dir "$PRETRAIN_DIR"
            fi

            MAIN_DIR="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            MAIN_CONFIG="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"
            echo $MAIN_DIR
            echo $MAIN_CONFIG
            if [ "$USE_BEST_PRETRAIN" == "True" ];
            then
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir "$PRETRAIN_DIR" --use_pretrain_best
            else
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --pretrain-dir "$PRETRAIN_DIR"
            fi
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --eval-only --model-dir "$MAIN_DIR"
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$MAIN_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --relabel


            AL_DIR="${prefix_path}${AL_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"
            AL_CONFIG="${prefix_path}${AL_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/main_config/transfer_adaptation.yaml"
            MODEL_UPDATE_AL_DIR="${prefix_path}${model_update_al}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}/model"

            echo $AL_DIR

            if [ "$USE_BEST_EXP" == "True" ];
            then
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$MAIN_DIR" --use_pretrain_best --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
            else
              CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --pretrain-dir "$MAIN_DIR" --test-data  "$test_path" --model-update-dir "$MODEL_UPDATE_AL_DIR"
            fi
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --eval-only --model-dir "$AL_DIR"
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$AL_DIR" --main-config-file "$AL_CONFIG" --test-data $test_path --relabel
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

            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG"
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
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

            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --test-data $test_path --model-update-dir "$MODEL_UPDATE_AL_DIR" --use_pretrain_best --pretrain-dir "$PRETRAIN_DIR"
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
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

            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_active_learning.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --test-data $test_path --model-update-dir "$MODEL_UPDATE_AL_DIR"
            CUDA_VISIBLE_DEVICES=$GPU_device python ${train_script}train_temp.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --eval-only
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
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict_task_1.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict_task_1.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data $test_path
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

task_1_exp_1="NeurIPS_1/task_1_exp_1"
task_1_exp_2="NeurIPS_1/task_1_exp_2"
task_1_exp_3="NeurIPS_1/task_1_exp_3"
task_1_exp_4="NeurIPS_1/task_1_exp_4"



T_F_aug="T_F_aug"
temp_aug="temp_aug"
no_aug="no_aug"
chan_norm="chan_norm"
no_norm="no_norm"
time_norm="time_norm"

adaptation_prefix="adaptation"
adaptationV1_prefix="adaptationV1"
adaptationV2_prefix="adaptationV2"
share_adaptV1_prefix="share_adaptV1"

deepsleep_share_adaptV1_prefix="deepsleep_share_adaptV1"
deepsleep_share_mcd_prefix="deepsleep_share_mcd"


dannV1_prefix="dannV1"
dann_prefix="dann"
deepsleep_dann_prefix="deepsleep_dann"

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