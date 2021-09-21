#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate tf-gpu
conda activate tensorflow2


#DIR="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/train_script/common_func_script"
DIR="/home/wduong/tmp/EEG_Dassl_Lightning/train_script/common_func_script"
source "${DIR}/common_script.sh"

#prefix_path="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
#train_script="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
#ROOT="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition"
#predict_script="C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/"
#test_data_path="C:\wduong_folder\Dassl.pytorch-master\NeurIPS_competition\EEG_Dassl_Lightning\da_dataset\NeurIPS_competition\test_data\NeurIPS_TL.mat"

prefix_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/"
train_script="/home/wduong/tmp/EEG_Dassl_Lightning/"
ROOT="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition"
predict_script="/home/wduong/tmp/EEG_Dassl_Lightning/"
test_data_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/test_data/NeurIPS_TL.mat"

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

#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --generate-predict --use-assemble-test-dataloader --relabel
#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --use-assemble-test-dataloader --relabel

#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --generate-predict --relabel
#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --relabel

#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --generate-predict --use-assemble-test-dataloader
#            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --use-assemble-test-dataloader

            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data "$test_data_path" --generate-predict
            CUDA_VISIBLE_DEVICES=$GPU_device python ${predict_script}predict.py --gpu-id $GPU_device --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --test-data "$test_data_path"
          done
        done
      done
    done
}



#/home/wduong/tmp/Dassl_pytorch/
#DIR="/home/wduong/tmp/EEG_Dassl_Lightning/train_script/common_func_script"




#experiment_1="NeurIPS_competition/experiment_1"
#temp_aug="temp_aug"
#no_aug="no_aug"
#chan_norm="chan_norm"
#no_norm="no_norm"
#
#adaptation_prefix="adaptation"
#vanilla_prefix="vanilla"
#
#BCI_IV_dataset="BCI_IV"
#Cho2017_dataset="Cho2017"
#Physionet_dataset="Physionet"

#group_aug=("$T_F_aug")
group_aug=("$temp_aug" "$no_aug")
#group_aug=("$no_aug")

#group_norm=("$chan_norm")
#group_norm=("$no_norm")

group_norm=("$no_norm" "$chan_norm")


#group_datasets=("$BCI_IV_dataset")
#group_datasets=("$Cho2017_dataset")
#group_datasets=("$Physionet")
#group_datasets=("$Dataset_A_dataset")
group_datasets=("$Dataset_A_dataset" "$Dataset_B_dataset")

#group_model=("$vanilla_prefix" "$adaptation_prefix")
#group_model=("$adaptationV1_prefix" "$adaptation_prefix" "$vanilla_prefix")
#group_model=("$FBCNET_adaptV1_prefix")
group_model=("$adaptationV1_prefix")

#group_model=("$component_adapt_prefix" "$vanilla_prefix" "$adaptation_prefix")

#run_full_multi_gpu $gpu_device_0 $experiment_2 group_aug group_norm group_model group_datasets

#run_full_multi_gpu $gpu_device_0 $experiment_5 group_aug group_norm group_model group_datasets
#run_full_multi_gpu $gpu_device_0 $final_result_3 group_aug group_norm group_model group_datasets
#run_full_multi_gpu $gpu_device_0 $final_result_4 group_aug group_norm group_model group_datasets

#run_full_multi_gpu $gpu_device_0 $experiment_6 group_aug group_norm group_model group_datasets

run_ensemble_predict $gpu_device_0 $final_result_4_5 group_aug group_norm group_model group_datasets
#run_ensemble_predict $gpu_device_0 $final_result_6 group_aug group_norm group_model group_datasets