#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate tf-gpu
conda activate tensorflow2

function run() {
    ROOT=$2
    EXP_TYPE=$3
    SEED=$4
    TRAINER_MODEL_PREFIX=$5
    NORMALIZE_PREFIX=$6
    DATASET=$7
    GPU_device=$1
#    prefix_path=""
#    prefix_path="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/"
    prefix_path="/home/wduong/tmp/Dassl_pytorch/"
    OUTPUT_DIR="${prefix_path}${EXP_TYPE}/${SEED}/${TRAINER_MODEL_PREFIX}/${NORMALIZE_PREFIX}/${DATASET}_adaptation/transfer_adaptation"
    MAIN_CONFIG="${prefix_path}${EXP_TYPE}/${SEED}/${TRAINER_MODEL_PREFIX}/${NORMALIZE_PREFIX}/${DATASET}_adaptation/main_config/transfer_adaptation.yaml"
    echo $OUTPUT_DIR
    echo $MAIN_CONFIG
    echo $ROOT
    CUDA_VISIBLE_DEVICES=$GPU_device python ${prefix_path}train_temp.py --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds
    CUDA_VISIBLE_DEVICES=$GPU_device python ${prefix_path}train_temp.py --root "$ROOT" --output-dir "$OUTPUT_DIR" --main-config-file "$MAIN_CONFIG" --train-k-folds --eval-only --model-dir $OUTPUT_DIR
}



#norm_none="C:/wduong_folder/Dassl.pytorch-master/Dassl.pytorch-master/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"
norm_none="/home/wduong/tmp/Dassl_pytorch/da_dataset/HETEROGENEOUS_ADAPTATION_V1/norm_none"


experiment_type_1="heterogeneous_adaptation"
experiment_type_2="heterogeneous_adaptation_v1"
experiment_type_3="heterogeneous_adaptation_v2"
experiment_type_4="heterogeneous_adaptation_v3"
experiment_type_5="heterogeneous_adaptation_v1.1"
experiment_type_6="heterogeneous_adaptation_v1.2"
experiment_type_7="heterogeneous_adaptation_v2.2"
experiment_type_8="heterogeneous_adaptation_v1.3"
experiment_type_9="heterogeneous_adaptation_v1.4"







vanilla_prefix="vanilla"
vanilla_equal_prefix="vanilla_equal_label"

adaptation_prefix="adaptation"
adaptation_DSBN_prefix="adaptation_DSBN"
adapt_equal_label_prefix="adapt_equal_label"
adapt_equal_label_DSBN_prefix="adapt_equal_label_DSBN"



vanilla_aug_prefix="vanilla_aug"
vanilla_aug_1_prefix="vanilla_aug_1"
vanilla_aug_2_prefix="vanilla_aug_2"
vanilla_equal_aug_prefix="vanilla_equal_aug"
vanilla_equal_aug_1_prefix="vanilla_equal_aug_1"
vanilla_equal_aug_2_prefix="vanilla_equal_aug_2"



adaptation_aug_prefix="adapt_aug"
adaptation_equal_aug_prefix="adapt_equal_aug"
adaptation_aug_DSBN_prefix="adapt_aug_DSBN"
adaptation_equal_aug_DSBN_prefix="adapt_equal_aug_DSBN"






adapt_dann_prefix="adapt_dann"
adapt_dann_aug_prefix="adapt_dann_aug"
adapt_dann_DSBN_prefix="adapt_dann_DSBN"
adapt_dann_aug_DSBN_prefix="adapt_dann_aug_DSBN"


adapt_equal_dann_DSBN_prefix="adapt_equal_dann_DSBN"
adapt_equal_dann_prefix="adapt_equal_dann"
adapt_equal_dann_aug_prefix="adapt_equal_dann_aug"
adapt_equal_dann_aug_DSBN_prefix="adapt_equal_dann_aug_DSBN"


adapt_cdan_prefix="adapt_cdan"
adapt_equal_cdan_prefix="adapt_equal_cdan"
adapt_equal_cdan_aug_prefix="adapt_equal_cdan_aug"


SEED_V0="seed_v0"
SEED_V1="seed_v1"
SEED_V2="seed_v2"


norm_zscore_prefix="norm_zscore"
norm_zscore_1_prefix="norm_zscore_1"
norm_min_max_prefix="norm_min_max"

norm_none_prefix="norm_none"
#norm_max_prefix="norm_max"

BCI_IV_prefix="BCI_IV"
BCI_IV_MI_prefix="BCI_IV_MI"

BCI_IV_MI_1_prefix="BCI_IV_MI_1"
BCI_IV_MI_V2_prefix="BCI_IV_MI_V2"
BCI_IV_MI_V3_prefix="BCI_IV_MI_V3"

BCI_IV_GIGA_MI_V1_prefix="BCI_IV_GIGA_MI_V1"


BCI_IV_GIGA_MI_prefix="BCI_IV_GIGA_MI"

DSBN_BCI_IV_MI_prefix="DSBN_BCI_IV_MI"


GIGA_prefix="GIGA"



gpu_device_0=0
gpu_device_1=1
gpu_device_2=2
gpu_device_3=3


#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_prefix $norm_zscore_prefix $GIGA_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_prefix $norm_zscore_prefix $GIGA_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_prefix $norm_zscore_prefix $GIGA_prefix

#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_prefix

#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_MI_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix

#run $norm_none $experiment_type_2 $SEED_V0 $adapt_dann_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_dann_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_dann_prefix $norm_none_prefix $BCI_IV_MI_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_dann_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_dann_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_dann_prefix $norm_none_prefix $BCI_IV_MI_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_dann_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_dann_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_dann_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_dann_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_dann_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_dann_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix

#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_prefix $norm_none_prefix $BCI_IV_GIGA_MI_V1_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_prefix $norm_none_prefix $BCI_IV_GIGA_MI_V1_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_prefix $norm_none_prefix $BCI_IV_GIGA_MI_V1_prefix

#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_GIGA_MI_V1_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_GIGA_MI_V1_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_label_prefix $norm_none_prefix $BCI_IV_GIGA_MI_V1_prefix


#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix

#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_label_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_label_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_label_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix

#run $norm_none $experiment_type_2 $SEED_V0 $adapt_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_label_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_label_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_label_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_dann_DSBN_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix

run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V0 $adaptation_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_V3_prefix
run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V1 $adaptation_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_V3_prefix
run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V2 $adaptation_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_V3_prefix

run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V0 $adapt_dann_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_V3_prefix
run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V1 $adapt_dann_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_V3_prefix
run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V2 $adapt_dann_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_V3_prefix

run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V0 $adaptation_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix
run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V1 $adaptation_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix
run $gpu_device_1 $norm_none $experiment_type_4 $SEED_V2 $adaptation_aug_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix




#run $norm_none $experiment_type_2 $SEED_V0 $adapt_dann_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_dann_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_dann_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_dann_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_dann_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_dann_DSBN_prefix $norm_zscore_1_prefix $BCI_IV_MI_prefix




#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_label_prefix $norm_zscore_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_label_prefix $norm_zscore_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_label_prefix $norm_zscore_prefix $BCI_IV_MI_prefix
#

#
#run $norm_none $experiment_type_2 $SEED_V0 $adapt_equal_label_prefix $norm_min_max_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adapt_equal_label_prefix $norm_min_max_prefix $BCI_IV_MI_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adapt_equal_label_prefix $norm_min_max_prefix $BCI_IV_MI_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_prefix $norm_none_prefix $BCI_IV_MI_V3_prefix

#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_aug_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_aug_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_aug_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#
#run $norm_none $experiment_type_2 $SEED_V0 $adaptation_equal_aug_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V1 $adaptation_equal_aug_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
#run $norm_none $experiment_type_2 $SEED_V2 $adaptation_equal_aug_prefix $norm_none_prefix $BCI_IV_MI_V2_prefix
