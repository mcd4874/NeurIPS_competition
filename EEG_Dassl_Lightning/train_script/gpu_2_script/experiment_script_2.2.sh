#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate tf-gpu
conda activate tensorflow2

#DIR="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/train_script/common_func_script"
#/home/wduong/tmp/Dassl_pytorch/
DIR="/home/wduong/tmp/EEG_Dassl_Lightning/train_script/common_func_script"

source "${DIR}/common_script.sh"


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

#group_aug=("$temp_aug")
#group_aug=("$no_aug")

#group_norm=("$chan_norm")
#group_norm=("$no_norm")

#group_datasets=("$BCI_IV_dataset")
#group_datasets=("$Cho2017_dataset")
#group_datasets=("$Physionet")
#group_datasets=("$Dataset_A_dataset")
#group_datasets=("$Dataset_B_dataset")
#group_model=("$adaptationV1_prefix" "$adaptation_prefix" "$vanilla_prefix")
#group_model=("$FBCNET_adaptV1_prefix")
#group_model=("$shallowcon_adaptV1_prefix")
#run_full_multi_gpu $gpu_device_0 $final_result_9_0_3 group_aug group_norm group_model group_datasets
#run_predict $gpu_device_0 $final_result_9_0_3 $test_case_13_microvolt_path group_aug group_norm group_model group_datasets

#run_full_multi_gpu $gpu_device_0 $final_result_9_0_2 group_aug group_norm group_model group_datasets
#run_ensemble_predict $gpu_device_0 $final_result_9_0_2 $test_case_13_microvolt_path group_aug group_norm group_model group_datasets
#
#run_full_multi_gpu $gpu_device_0 $final_result_9_0_1 group_aug group_norm group_model group_datasets
#run_predict $gpu_device_0 $final_result_9_0_1 $test_case_13_microvolt_path group_aug group_norm group_model group_datasets
#group_model=("$share_adaptV1_prefix")
#LIST_EXP_TYPE=("$private_exp_14_3_1")
#sub_list=("sub" "sub_0" "sub_1" "sub_2")
#group_datasets=("$BCI_IV_dataset")
#group_model=("$addaV1_prefix")
#sub_list=("tune_batch/main_model_2")
#group_model=("$mcdV1_prefix")
##group_model=("$vanilla_prefix")
#group_aug=("$temp_aug")
##group_aug=("$no_aug")
#group_norm=("$no_norm")
#group_datasets=("$Dataset_A_dataset" "$Dataset_B_dataset")
#LIST_EXP_TYPE=("$final_result_14_3_1")
#for EXP_TYPE in "${LIST_EXP_TYPE[@]}";
#do
#  for sub in "${sub_list[@]}";
#  do
#    NEW_TYPE="${EXP_TYPE}/${sub}"
##    run_al_full $gpu_device_0 $NEW_TYPE $test_case_16_microvolt_path group_aug group_norm group_model group_datasets
#
#    run_full_multi_gpu $gpu_device_1 $NEW_TYPE group_aug group_norm group_model group_datasets
#    run_predict_relabel $gpu_device_1 $NEW_TYPE $test_case_16_microvolt_path group_aug group_norm group_model group_datasets
#  done
#done

#PRETRAIN_EXP_TYPE=""
#USE_BEST_PRETRAIN="False"
#EXP_TYPE="NeurIPS_competition/final_result_14_3_1/main_model_2"
#USE_BEST_EXP="True"
#AL_EXP_TYPE="NeurIPS_competition/final_result_14_3_1/al_model_2"
#test_path="$test_case_16_microvolt_path"
#model_update_al="NeurIPS_competition/final_result_14_3_1/main_model_2"
#
#EXP_SETUP=("$PRETRAIN_EXP_TYPE" "$EXP_TYPE" "$AL_EXP_TYPE")
#BEST_SETUP=("$USE_BEST_PRETRAIN" "$USE_BEST_EXP")
#echo ${EXP_SETUP[0]}
#echo ${EXP_SETUP[1]}
#
#run_full_active_learning $gpu_device_1 $test_path $model_update_al EXP_SETUP BEST_SETUP group_aug group_norm group_model group_datasets


#EXP_TYPE="NeurIPS_competition/final_result_14_3_1/main_model_2"
#USE_BEST_EXP="True"
#AL_EXP_TYPE="NeurIPS_competition/final_result_14_3_1/al_model_2_1"
#test_path="$test_case_16_microvolt_path"
#model_update_al="NeurIPS_competition/final_result_14_3_1/main_model_2"
#
#EXP_SETUP=("$EXP_TYPE" "$AL_EXP_TYPE")
#BEST_SETUP=("$USE_BEST_EXP")
#run_active_learning $gpu_device_1 $test_path $model_update_al EXP_SETUP BEST_SETUP group_aug group_norm group_model group_datasets
#
#run_predict_relabel $gpu_device_1 $AL_EXP_TYPE $test_path group_aug group_norm group_model group_datasets


#group_datasets=("$Dataset_A_0_dataset" "$Dataset_A_1_dataset" "$Dataset_B_0_dataset" "$Dataset_B_1_dataset" "$Dataset_B_2_dataset")



#EXP_TYPE="NeurIPS_competition/final_result_14_3_1/main_model_2"
#USE_BEST_EXP="True"
#AL_EXP_TYPE="NeurIPS_competition/final_result_15_3_1/al_model_2_1"
#model_update_al="NeurIPS_competition/final_result_14_3_1/main_model_2"
#LIST_AUG_PREFIX=("$no_aug")
#LIST_NORMALIZE_PREFIX=("$no_norm")
#TRAINER_MODEL_PREFIXS=("$mcdV1_prefix")
#
#
#DATASETS=("$Dataset_A_0_dataset" "$Dataset_A_1_dataset")
#PRETRAIN_DATASET="dataset_A"
#TEST_DATASET_CASE="test_case_17_A"
#for AUG_PREFIX in "${LIST_AUG_PREFIX[@]}";
#do
#  for NORMALIZE_PREFIX in "${LIST_NORMALIZE_PREFIX[@]}";
#  do
#    for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
#    do
#      for DATASET in "${DATASETS[@]}";
#      do
#        PRETRAIN_DIR="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${PRETRAIN_DATASET}"
#        MODEL_UPDATE_AL_DIR="${prefix_path}${model_update_al}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${PRETRAIN_DATASET}"
#        AL_DIR="${prefix_path}${AL_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
#        test_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/${TEST_DATASET_CASE}/${DATASET}/NeurIPS_TL.mat"
#
#        echo $PRETRAIN_DIR
#        echo $MODEL_UPDATE_AL_DIR
#        echo $AL_DIR
#        echo $test_path
#        run_simple_active_learning $gpu_device_1 $test_path $PRETRAIN_DIR $USE_BEST_EXP $AL_DIR $MODEL_UPDATE_AL_DIR
#
#      done
#    done
#  done
#done
#
##DATASETS=("$Dataset_B_2_dataset")
#DATASETS=("$Dataset_B_0_dataset" "$Dataset_B_1_dataset" "$Dataset_B_2_dataset")
#PRETRAIN_DATASET="dataset_B"
#TEST_DATASET_CASE="test_case_17_B"
#for AUG_PREFIX in "${LIST_AUG_PREFIX[@]}";
#do
#  for NORMALIZE_PREFIX in "${LIST_NORMALIZE_PREFIX[@]}";
#  do
#    for TRAINER_MODEL_PREFIX in "${TRAINER_MODEL_PREFIXS[@]}";
#    do
#      for DATASET in "${DATASETS[@]}";
#      do
#        PRETRAIN_DIR="${prefix_path}${EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${PRETRAIN_DATASET}"
#        MODEL_UPDATE_AL_DIR="${prefix_path}${model_update_al}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${PRETRAIN_DATASET}"
#        AL_DIR="${prefix_path}${AL_EXP_TYPE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
#        test_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/${TEST_DATASET_CASE}/${DATASET}/NeurIPS_TL.mat"
#
#        echo $PRETRAIN_DIR
#        echo $MODEL_UPDATE_AL_DIR
#        echo $AL_DIR
#        echo $test_path
#        run_simple_active_learning $gpu_device_1 $test_path $PRETRAIN_DIR $USE_BEST_EXP $AL_DIR $MODEL_UPDATE_AL_DIR
#
#      done
#    done
#  done
#done


#PRETRAIN="NeurIPS_competition/final_result_15_3_1/main_model_3/weight_sampler"
#MODEL_UPDATE="NeurIPS_competition/final_result_15_3_1/main_model_3/weight_sampler"
#EXP_TYPE="NeurIPS_competition/final_result_15_3_1/main_model_3/tune_opt"

#EXP_TYPE="NeurIPS_competition/final_result_15_3_1/main_model_3/weight_sampler"
MAIN_EXP_TYPE="NeurIPS_competition/final_result_15_3_1/main_model_3/tune_opt"

LIST_EXP_TYPES=("7" "8" "9")
LIST_AUG_PREFIX=("$no_aug")
LIST_NORMALIZE_PREFIX=("$no_norm")
TRAINER_MODEL_PREFIXS=("$mcdV1_prefix")

DATASETS=("$Dataset_A_0_dataset" "$Dataset_A_1_dataset")
PRETRAIN_DATASET="dataset_A"
TEST_DATASET_CASE="test_case_17_A"
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
          test_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/${TEST_DATASET_CASE}/${DATASET}/NeurIPS_TL.mat"

  #        echo $PRETRAIN_DIR
          echo $MAIN_DIR
          run_simple_train $gpu_device_2 $test_path $MAIN_DIR
  #        USE_BEST_PRETRAIN_EXP="True"
  #        PRETRAIN_DIR="${prefix_path}/${PRETRAIN}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
  #        PRETRAIN_DIR="empty"
  #        MODEL_UPDATE_DIR="${prefix_path}/${MODEL_UPDATE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
  #        echo $MODEL_UPDATE_DIR
  #        run_simple_active_learning $gpu_device_0 $test_path $PRETRAIN_DIR $USE_BEST_PRETRAIN_EXP $MAIN_DIR $MODEL_UPDATE_DIR

        done
      done
    done
  done
done

DATASETS=("$Dataset_B_0_dataset" "$Dataset_B_1_dataset" "$Dataset_B_2_dataset")
PRETRAIN_DATASET="dataset_B"
TEST_DATASET_CASE="test_case_17_B"
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
          test_path="/data1/wduong_experiment_data/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/${TEST_DATASET_CASE}/${DATASET}/NeurIPS_TL.mat"

  #        echo $PRETRAIN_DIR
          echo $MAIN_DIR
          run_simple_train $gpu_device_2 $test_path $MAIN_DIR
  #        USE_BEST_PRETRAIN_EXP="True"
  #        PRETRAIN_DIR="${prefix_path}/${PRETRAIN}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
  #        PRETRAIN_DIR="empty"
  #        MODEL_UPDATE_DIR="${prefix_path}/${MODEL_UPDATE}/${AUG_PREFIX}/${NORMALIZE_PREFIX}/${TRAINER_MODEL_PREFIX}/${DATASET}"
  #        echo $MODEL_UPDATE_DIR
  #        run_simple_active_learning $gpu_device_0 $test_path $PRETRAIN_DIR $USE_BEST_PRETRAIN_EXP $MAIN_DIR $MODEL_UPDATE_DIR

        done
      done
    done
  done
done