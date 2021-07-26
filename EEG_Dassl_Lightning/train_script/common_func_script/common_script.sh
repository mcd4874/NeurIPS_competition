#    prefix_path=""
#prefix_path="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/"
#train_script="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/"
#ROOT="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"
#    prefix_path="/home/wduong/tmp/Dassl_pytorch/"

prefix_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/"
train_script="/home/wduong/tmp/EEG_Dassl_Lightning/"
ROOT="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition"
#ROOT="/data1/wduong_experiment_data/Dassl_pytorch/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"


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
#    printf '1: %q\n' "${TRAINER_MODEL_PREFIXS[@]}"
#    printf '2: %q\n' "${DATASETS[@]}"

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


#
#
gpu_device_0=0
gpu_device_1=1
gpu_device_2=2
gpu_device_3=3



experiment_1="NeurIPS_competition/experiment_1"
experiment_2="NeurIPS_competition/experiment_2"
experiment_3="NeurIPS_competition/experiment_3"
final_result_3="NeurIPS_competition/final_result_3"


temp_aug="temp_aug"
no_aug="no_aug"
chan_norm="chan_norm"
no_norm="no_norm"

adaptation_prefix="adaptation"
vanilla_prefix="vanilla"
component_adapt_prefix="component_adapt"

BCI_IV_dataset="BCI_IV"
Cho2017_dataset="Cho2017"
Physionet_dataset="Physionet"

Dataset_A_dataset="dataset_A"
Dataset_B_dataset="dataset_B"