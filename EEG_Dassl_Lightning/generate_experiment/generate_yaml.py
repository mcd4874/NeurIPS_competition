import yaml
from yacs.config import CfgNode as CN
from ruamel.yaml import YAML
import os




def convert_to_dict(cfg_node, key_list):
    def _valid_type(value, allow_cfg_node=False):
        return (type(value) in _VALID_TYPES) or (
                allow_cfg_node and isinstance(value, CN)
        )
    def _assert_with_logging(cond, msg):
        if not cond:
            logger.debug(msg)
        assert cond, msg
    import logging
    logger = logging.getLogger(__name__)
    _VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
    if not isinstance(cfg_node, CN):
        _assert_with_logging(
            _valid_type(cfg_node),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES
            ),
        )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def generate_sub_experiment_config(config_path, trainer_conditions, dataset_conditions):
    import itertools
    def findsubsets(s, n):
        return list(itertools.combinations(s, n))

    # Driver Code
    s = ["cho2017", "physionet", "BCI_IV"]
    n = 2
    out_paths = list()
    subsets = findsubsets(s, n)
    index = 0
    for subset in subsets:
        new_config_path = config_path + "_{}".format(index)
        index += 1
        for trainer in trainer_conditions:
            for dataset in dataset_conditions:
                update_config_path = os.path.join(new_config_path, trainer, dataset)
                match_pairs = [
                    ["DATAMANAGER.DATASET.SETUP.SOURCE_DATASET_NAMES", list(subset)],
                    ["EXTRA_FIELDS.source_dataset", list(subset)],
                    ["DATAMANAGER.DATALOADER.LIST_TRAIN_U.SAMPLERS", ['RandomSampler', 'RandomSampler']],
                    ["DATAMANAGER.DATALOADER.LIST_TRAIN_U.BATCH_SIZES", [32, 32]]
                ]

                current_config_path = os.path.join(config_path, trainer, dataset, 'main_config',
                                                   'transfer_adaptation.yaml')
                config_f = open(current_config_path)
                config_file = CN(new_allowed=True).load_cfg(config_f)
                for match_pair in match_pairs:
                    config_file.merge_from_list(match_pair)
                generate_config_file([], config_file, current_folder_path=update_config_path,
                                     output_file="transfer_adaptation.yaml")
        out_paths.append(new_config_path)
    return out_paths

def generate_config_file(conditions,config_file,current_folder_path="",output_file="transfer_adaptation.yaml"):
    print("list conditions : ",conditions)
    if len(conditions) == 0:
        if current_folder_path:
            current_folder_path = os.path.join(current_folder_path,"main_config")
            if not os.path.exists(current_folder_path):
                os.makedirs(current_folder_path)
            output_file = os.path.join(current_folder_path, output_file)
            f = open(output_file, 'w')
            new_yaml = YAML()
            new_yaml.indent(sequence=4, offset=2)
            new_yaml.default_flow_style = False
            new_yaml.preserve_quotes = True
            dict_config_file = convert_to_dict(config_file,[])
            new_yaml.dump(data=dict_config_file, stream=f)
    else:
        current_conditions = conditions[0]
        remain_conditions = conditions[1:]
        print("current conditions : ",current_conditions)
        for current_condition_info in current_conditions:

            print("current condition info : ",current_condition_info)
            current_config_file =  config_file.clone()
            folder_name = current_condition_info["dir_name"]
            update_folder_name = os.path.join(current_folder_path,folder_name)
            if not os.path.exists(update_folder_name):
                os.makedirs(update_folder_name)

            pair_lists = current_condition_info["match_pair"]
            for pair_list in pair_lists:
                current_config_file.merge_from_list(pair_list)
            generate_config_file(remain_conditions,current_config_file,update_folder_name)

# def check_type_info(type_name,type_param):
#     if type_name == "aug_type":
#         aug_type = type_param['aug']
#         if aug_type == "temp_aug":
#             match_pair = [
#                 ["DATAMANAGER.DATASET.AUGMENTATION.NAME", "temporal_segment"],
#                 ["EXTRA_FIELDS.aug", "temporal_aug"]
#             ]
#         elif aug_type == "T_F_aug":
#             match_pair = [
#                 ["DATAMANAGER.DATASET.AUGMENTATION.NAME", "temporal_segment_T_F"],
#                 ["EXTRA_FIELDS.aug", "T_F_aug"]
#             ]
#         else:
#             match_pair = [
#                 ["DATAMANAGER.DATASET.AUGMENTATION.NAME", ""],
#                 ["EXTRA_FIELDS.aug", "no_aug"]
#             ]
#         aug_info = [
#             dict(dir_name=aug_type, match_pair=match_pair)
#         ]
#         return aug_info
#     elif type_name == "norm_type":
#         norm_type = type_param['norm']
#         if norm_type == "chan_norm":
#             match_pair = [
#                 ["EXTRA_FIELDS.normalize", "chan_norm"],
#                 ["INPUT.TRANSFORMS", ["cross_channel_norm"]],
#                 ["INPUT.NO_TRANSFORM", False]
#             ]
#         else:
#             match_pair = [
#                 ["EXTRA_FIELDS.normalize", "no_norm"],
#                 ["INPUT.TRANSFORMS", []],
#                 ["INPUT.NO_TRANSFORM", True]
#             ]
#         norm_info = [
#             dict(dir_name=norm_type, match_pair=match_pair)
#         ]
#         return norm_info
#     elif type_name == "trainer_type":
#         trainer_type = type_param['trainer']
#         if trainer_type == "adaptation":
#             match_pair = [
#                 ["EXTRA_FIELDS.model", "MultiDatasetAdaptation"],
#                 ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetAdaptation"],
#                 ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
#             ]
#         elif trainer_type == "adaptationV1":
#             match_pair = [
#                 ["EXTRA_FIELDS.model", "MultiDatasetAdaptationV1"],
#                 ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetAdaptationV1"],
#                 ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
#             ]
#         elif trainer_type == "FBCNET_adaptV1":
#             match_pair = [
#                 ["EXTRA_FIELDS.model", "MultiDatasetAdaptationV1"],
#                 ["EXTRA_FIELDS.backbone", "fbcnet"],
#                 ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetAdaptationV1"],
#                 ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
#             ]
#         elif trainer_type == "component_adapt":
#             match_pair = [
#                 ["EXTRA_FIELDS.model", "ComponentAdaptation"],
#                 ["LIGHTNING_MODEL.TRAINER.NAME", "ComponentAdaptation"],
#                 ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
#             ]
#         else:
#             match_pair = [
#                 ["EXTRA_FIELDS.model", "BaseModel"],
#                 ["LIGHTNING_MODEL.TRAINER.NAME", "BaseModel"],
#                 ["DATAMANAGER.MANAGER_TYPE", 'single_dataset']
#             ]
#         trainer_info = [
#             dict(dir_name=trainer_type, match_pair=match_pair)
#         ]
#         return trainer_info
#     elif type_name == "dataset_type":
#         dataset_type = type_param['dataset']
#         if dataset_type == "BCI_IV":
#             match_pair = [
#                 ["EXTRA_FIELDS.target_dataset", "BCI_IV"],
#             ]
#         elif dataset_type == "Cho2017":
#             match_pair = [
#                 ["EXTRA_FIELDS.target_dataset", "cho2017"],
#             ]
#         elif dataset_type == "dataset_A":
#             match_pair = [
#                 ["EXTRA_FIELDS.target_dataset", "dataset_A"],
#             ]
#         elif dataset_type == "dataset_B":
#             match_pair = [
#                 ["EXTRA_FIELDS.target_dataset", "dataset_B"],
#             ]
#         else:
#             match_pair = [
#                 ["EXTRA_FIELDS.target_dataset", "physionet"],
#             ]
#         dataset_info = [
#             dict(dir_name=dataset_type, match_pair=match_pair)
#         ]
#         return dataset_info
#     elif type_name == "lr":
#         lr = type_param['lr']
#
#
#
# def generate_tuning_config(main_path,config_path,list_type):
#     for info_type in list_type:
#         type_name = info_type["name"]
#         type_param =
#

def generate_transfer_learning_config(main_path,config_path,aug_type,norm_type,trainer_type,dataset_type,extra_merge=None):
    config_f = open(config_path)
    config_file = CN(new_allowed=True).load_cfg(config_f)
    if extra_merge:
        for pair_list in extra_merge:
            config_file.merge_from_list(pair_list)

    # aug_type = "temp_aug"
    if aug_type == "temp_aug":
        match_pair =[
            ["DATAMANAGER.DATASET.AUGMENTATION.NAME","temporal_segment"],
            ["EXTRA_FIELDS.aug","temporal_aug"]
        ]
    elif aug_type == "T_F_aug":
        match_pair =[
            ["DATAMANAGER.DATASET.AUGMENTATION.NAME","temporal_segment_T_F"],
            ["EXTRA_FIELDS.aug","T_F_aug"]
        ]
    else:
        match_pair = [
            ["DATAMANAGER.DATASET.AUGMENTATION.NAME", ""],
            ["EXTRA_FIELDS.aug", "no_aug"]
        ]
    aug_info = [
        dict(dir_name=aug_type, match_pair=match_pair)
    ]

    # norm_type = "chan_norm"
    if norm_type == "chan_norm":
        match_pair = [
            ["EXTRA_FIELDS.normalize", "chan_norm"],
             ["INPUT.TRANSFORMS", ["cross_channel_norm"]],
             ["INPUT.NO_TRANSFORM", False]
        ]
    elif norm_type == "time_norm":
        match_pair = [
            ["EXTRA_FIELDS.normalize", "time_norm"],
            ["INPUT.TRANSFORMS", ["time_norm"]],
            ["INPUT.NO_TRANSFORM", False]
        ]
    else:
        match_pair = [
            ["EXTRA_FIELDS.normalize", "no_norm"],
            ["INPUT.TRANSFORMS", []],
            ["INPUT.NO_TRANSFORM", True]
        ]
    norm_info = [
        dict(dir_name=norm_type, match_pair=match_pair)
    ]


    # trainer_type = "adaptation"
    if trainer_type == "adaptation":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetAdaptation"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetAdaptation"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
        ]
    elif trainer_type == "share_adaptV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiShareAdaptationV1"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiShareAdaptationV1"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
        ]
    elif trainer_type == "adaptationV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetAdaptationV1"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetAdaptationV1"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
        ]
    elif trainer_type == "adaptationV2":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetAdaptationV2"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetAdaptationV2"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
        ]
    elif trainer_type == "shallowcon_adaptV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetAdaptationV1"],
            ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetAdaptationV1"],
            ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
        ]
    elif trainer_type == "dannV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetDannV1"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetDannV1"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_datasetV2']
        ]
    elif trainer_type == "mcdV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetMCDV1"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetMCDV1"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_datasetV2']
        ]
    elif trainer_type == "m3sda":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetM3SDA"],
            ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetM3SDA"],
            ["DATAMANAGER.MANAGER_TYPE", 'multi_datasetV2']
        ]

    elif trainer_type == "SRDA":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetSRDA"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetSRDA"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_datasetV2']
        ]
    elif trainer_type == "addaV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetADDAV1"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetADDAV1"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_datasetV2']
        ]
    elif trainer_type == "FBCNET_adaptV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetAdaptationV1"],
            ["EXTRA_FIELDS.backbone", "fbcnet"],
            ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetAdaptationV1"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
        ]
    elif trainer_type == "component_adapt":
        match_pair = [
            ["EXTRA_FIELDS.model", "ComponentAdaptation"],
            ["LIGHTNING_MODEL.TRAINER.NAME", "ComponentAdaptation"],
            ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
        ]
    else:
        match_pair = [
            ["EXTRA_FIELDS.model", "BaseModel"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "BaseModel"],
             ["DATAMANAGER.MANAGER_TYPE", 'single_dataset']
        ]
    trainer_info = [
        dict(dir_name=trainer_type, match_pair=match_pair)
    ]

    # dataset_type = "BCI_IV"
    if dataset_type == "BCI_IV":
        match_pair = [
            ["EXTRA_FIELDS.target_dataset", "BCI_IV"],
        ]
    elif dataset_type == "Cho2017":
        match_pair = [
            ["EXTRA_FIELDS.target_dataset", "cho2017"],
        ]
    elif dataset_type == "dataset_A":
        match_pair = [
            ["EXTRA_FIELDS.target_dataset", "dataset_A"],
        ]
    elif dataset_type == "dataset_B":
        match_pair = [
            ["EXTRA_FIELDS.target_dataset", "dataset_B"],
        ]
    elif dataset_type == "physionet":
        match_pair = [
            ["EXTRA_FIELDS.target_dataset", "physionet"],
        ]
    else:
        match_pair = [
            ["EXTRA_FIELDS.target_dataset", dataset_type],
        ]
    dataset_info = [
        dict(dir_name=dataset_type, match_pair=match_pair)
    ]



    generate_config_file([aug_info,norm_info, trainer_info, dataset_info], config_file,main_path)


def setup_experiments(main_path, config_path, aug_conditions, norm_conditions, trainer_conditions, dataset_conditions,extra_merge=None):
    # config_path = "main_config/data_augmentation/{}/{}/transfer_adaptation.yaml"
    # main_path = "test_mdd/eegnet_1_1/{}"

    for trainer in trainer_conditions:
        for dataset in dataset_conditions:
            current_config_path = os.path.join(config_path, trainer, dataset, 'main_config', 'transfer_adaptation.yaml')
            for aug in aug_conditions:
                for norm in norm_conditions:
                    generate_transfer_learning_config(main_path, current_config_path, aug_type=aug, norm_type=norm,
                                                      trainer_type=trainer, dataset_type=dataset,extra_merge=extra_merge)
# import os
# os.path.exists()
# os.makedirs()

#generate heterogeneous adaptation and vanilla for update datasets
# main_path = "experiment_1"
# config_path = "main_config/experiment_1"
# main_path = "experiment_2"
# config_path = "main_config/experiment_2"
# main_path = "experiment_3"
# config_path = "main_config/experiment_3"
# main_path = "experiment_4"
# config_path = "main_config/experiment_4"
# main_path = "experiment_4_2"
# config_path = "main_config/experiment_4_2"
# main_path = "experiment_4_3"
# config_path = "main_config/experiment_4_3"
# main_path = "experiment_4_4"
# config_path = "main_config/experiment_4_4"
# main_path = "experiment_4_5"
# config_path = "main_config/experiment_4_5"
# main_path = "experiment_5"
# config_path = "main_config/experiment_5"
# main_path = "experiment_5_1"
# config_path = "main_config/experiment_5_1"
# main_path = "experiment_6"
# config_path = "main_config/experiment_6"
# main_path = "experiment_7"
# config_path = "main_config/experiment_7"
# main_path = "experiment_9_0_1"
# config_path = "main_config/experiment_9_0_1"
# main_path = "experiment_9_0_3"
# config_path = "main_config/experiment_9_0_3"

# main_path = "experiment_10_0_1"
# config_path = "main_config/experiment_10_0_1"
# main_path = "experiment_10_0_3"
# config_path = "main_config/experiment_10_0_3"

# main_path = "experiment_11_0_1"
# config_path = "main_config/experiment_11_0_1"
# main_path = "experiment_11_0_2"
# config_path = "main_config/experiment_11_0_2"
# main_path = "experiment_11_0_3"
# config_path = "main_config/experiment_11_0_3"


# main_path = "final_result_3"
# config_path = "main_config/final_result_3"
# main_path = "final_result_4"
# config_path = "main_config/final_result_4"
# main_path = "final_result_4_1_1"
# config_path = "main_config/final_result_4_1_1"
# main_path = "final_result_4_3"
# config_path = "main_config/final_result_4_3"
# main_path = "final_result_4_3_1"
# config_path = "main_config/final_result_4_3_1"
# main_path = "final_result_4_5"
# config_path = "main_config/final_result_4_5"
# main_path = "final_result_5_1"
# config_path = "main_config/final_result_5_1"
# main_path = "final_result_6"
# config_path = "main_config/final_result_6"
# main_path = "final_result_7"
# config_path = "main_config/final_result_7"
# main_path = "final_result_7_0_1"
# config_path = "main_config/final_result_7_0_1"
# main_path = "final_result_7_0_2"
# config_path = "main_config/final_result_7_0_2"
# main_path = "final_result_7_1"
# config_path = "main_config/final_result_7_1"
# main_path = "final_result_7_1_1"
# config_path = "main_config/final_result_7_1_1"
# main_path = "final_result_7_1_2"
# config_path = "main_config/final_result_7_1_2"
# main_path = "final_result_7_1_3"
# config_path = "main_config/final_result_7_1_3"
# main_path = "final_result_7_1_4"
# config_path = "main_config/final_result_7_1_4"

# main_path = "final_result_8"
# config_path = "main_config/final_result_8"
# main_path = "final_result_8_0_1"
# config_path = "main_config/final_result_8_0_1"
# main_path = "final_result_8_0_2"
# config_path = "main_config/final_result_8_0_2"
# main_path = "final_result_8_0_3"
# config_path = "main_config/final_result_8_0_3"
# main_path = "final_result_8_1_3"
# config_path = "main_config/final_result_8_1_3"
# main_path = "final_result_8_2_1"
# config_path = "main_config/final_result_8_2_1"
# main_path = "final_result_8_2_2"
# config_path = "main_config/final_result_8_2_2"
# main_path = "final_result_8_2_3"
# config_path = "main_config/final_result_8_2_3"
# main_path = "final_result_9"
# config_path = "main_config/final_result_9"
# main_path = "final_result_9_0_1"
# config_path = "main_config/final_result_9_0_1"
# main_path = "final_result_9_0_2"
# config_path = "main_config/final_result_9_0_2"
# main_path = "final_result_9_0_3"
# config_path = "main_config/final_result_9_0_3"
# main_path = "final_result_10_0_1"
# config_path = "main_config/final_result_10_0_1"
# main_path = "final_result_10_0_3"
# config_path = "main_config/final_result_10_0_3"

# main_path = "final_result_11_0_1"
# config_path = "main_config/final_result_11_0_1"
# main_path = "final_result_11_0_2"
# config_path = "main_config/final_result_11_0_2"
# main_path = "final_result_11_0_3"
# config_path = "main_config/final_result_11_0_3"

# main_path = "final_result_11_1_3"
# config_path = "main_config/final_result_11_1_3"
# main_path = "final_result_11_2_3"
# config_path = "main_config/final_result_11_2_3"
# main_path = "final_result_11_3_3"
# config_path = "main_config/final_result_11_3_3"

# main_path = "final_result_11_4_1"
# config_path = "main_config/final_result_11_4_1"
# main_path = "final_result_11_4_3"
# config_path = "main_config/final_result_11_4_3"
# main_path = "final_result_11_4_1_0"
# config_path = "main_config/final_result_11_4_1_0"
# main_path = "final_result_11_4_1_1"
# config_path = "main_config/final_result_11_4_1_1"
# main_path = "final_result_11_4_1_2"
# config_path = "main_config/final_result_11_4_1_2"
# main_path = "final_result_11_4_3_0"
# config_path = "main_config/final_result_11_4_3_0"
# main_path = "final_result_11_4_3_1"
# config_path = "main_config/final_result_11_4_3_1"
# main_path = "final_result_11_4_3_2"
# config_path = "main_config/final_result_11_4_3_2"

# main_path = "final_result_12_4_1"
# config_path = "main_config/final_result_12_4_1"
# main_path = "final_result_12_4_1_0"
# config_path = "main_config/final_result_12_4_1_0"
# main_path = "final_result_12_4_1_1"
# config_path = "main_config/final_result_12_4_1_1"
# main_path = "final_result_12_4_1_2"
# config_path = "main_config/final_result_12_4_1_2"

# main_path = "final_result_12_4_3"
# config_path = "main_config/final_result_12_4_3"
# main_path = "final_result_12_4_3_0"
# config_path = "main_config/final_result_12_4_3_0"
# main_path = "final_result_12_4_3_1"
# config_path = "main_config/final_result_12_4_3_1"
# main_path = "final_result_12_4_3_2"
# config_path = "main_config/final_result_12_4_3_2"

# main_path = "final_result_12_4_0"
# config_path = "main_config/final_result_12_4_0"
# main_path = "final_result_12_4_2"
# config_path = "main_config/final_result_12_4_2"
# main_path = "final_result_13_4_0"
# config_path = "main_config/final_result_13_4_0"
# main_path = "final_result_13_4_1"
# config_path = "main_config/final_result_13_4_1"
# main_path = "final_result_13_4_2"
# config_path = "main_config/final_result_13_4_2"
# main_path = "final_result_13_4_3"
# config_path = "main_config/final_result_13_4_3"


# main_path = "final_result_12_3_1/sub"
# config_path = "main_config/final_result_12_3_1/sub"
# main_path = "final_result_12_3_3/sub"
# config_path = "main_config/final_result_12_3_3/sub"

# main_path = "final_result_14_0_3"
# config_path = "main_config/final_result_14_0_3"
# main_path = "final_result_14_0_2"
# config_path = "main_config/final_result_14_0_2"
# main_path = "final_result_14_3_1/sub_20"
# config_path = "main_config/final_result_14_3_1/sub_20"
# main_path = "final_result_14_3_1/sub_30"
# config_path = "main_config/final_result_14_3_1/sub_30"
# main_path = "final_result_14_3_1/pretrain_0"
# config_path = "main_config/final_result_14_3_1/pretrain_0"
# main_path = "final_result_14_3_1/main_model_0"
# config_path = "main_config/final_result_14_3_1/main_model_0"
# main_path = "final_result_14_3_1/al_model_0"
# config_path = "main_config/final_result_14_3_1/al_model_0"
# main_path = "final_result_14_3_1/main_model_1"
# config_path = "main_config/final_result_14_3_1/main_model_1"
# main_path = "final_result_14_3_1/al_model_1"
# config_path = "main_config/final_result_14_3_1/al_model_1"
# main_path = "final_result_14_3_1/al_model_1_1"
# config_path = "main_config/final_result_14_3_1/al_model_1_1"
# main_path = "final_result_14_3_3/sub"
# config_path = "main_config/final_result_14_3_3/sub"

# main_path = "final_result_15_3_1/al_model_2_1"
# config_path = "main_config/final_result_15_3_1/al_model_2_1"
# main_path = "final_result_15_3_1/al_model_2_2"
# config_path = "main_config/final_result_15_3_1/al_model_2_2"

# main_path = "final_result_15_3_1/main_model_3"
# config_path = "main_config/final_result_15_3_1/main_model_3"
# main_path = "final_result_15_3_1/main_model_3_2"
# config_path = "main_config/final_result_15_3_1/main_model_3_2"
# main_path = "final_result_15_3_1/main_model_4"
# config_path = "main_config/final_result_15_3_1/main_model_4"
# main_path = "private_exp_14_3_1/sub_5"
# config_path = "main_config/private_exp_14_3_1/sub_5"
# main_path = "private_exp_14_3_3/sub"
# config_path = "main_config/private_exp_14_3_3/sub"
# dataset_conditions = ["BCI_IV"]
# main_path = "final_result_14_3_1/tune_batch/main_model_2"
# config_path = "main_config/final_result_14_3_1/main_model_2"
# main_path = "task_1_exp_1"
# config_path = "main_config/task_1_exp_1"
# main_path = "task_1_exp_2"
# config_path = "main_config/task_1_exp_2"
main_path = "task_1_exp_3"
config_path = "main_config/task_1_exp_3"
# aug_conditions=["temp_aug","no_aug"]
aug_conditions=["no_aug"]
# aug_conditions=["no_aug"]
# aug_conditions=["T_F_aug"]
# norm_conditions = ["chan_norm","no_norm"]
norm_conditions = ["no_norm","time_norm"]
# dataset_conditions = ["BCI_IV", "Cho2017", "Physionet"]
# trainer_conditions = ["vanilla", "adaptation", "component_adapt"]
# trainer_conditions = ["adaptationV1","dannV1","mcdV1","vanilla"]
# trainer_conditions = ["adaptationV1","dannV1"]
# trainer_conditions = ["adaptationV1"]
# trainer_conditions = ["mcdV1","adaptationV1"]
# trainer_conditions = ["adaptationV2"]
trainer_conditions = ["vanilla","adaptationV1","mcdV1","dannV1"]
# trainer_conditions = ["m3sda"]

# trainer_conditions = ["addaV1"]
# trainer_conditions = ["SRDA","addaV1"]


# trainer_conditions = ["shallowcon_adaptV1"]
# trainer_conditions = ["dannV1"]
# trainer_conditions = ["share_adaptV1"]
# trainer_conditions = ["FBCNET_adaptV1"]
# dataset_conditions = ["dataset_A", "dataset_B"]
# dataset_conditions = ["dataset_A"]
# dataset_conditions = ["dataset_A_0","dataset_A_1", "dataset_B_0","dataset_B_1","dataset_B_2"]
# dataset_conditions = ["full_dataset"]
dataset_conditions = ["full_dataset"]



setup_experiments(main_path,config_path,aug_conditions,norm_conditions,trainer_conditions,dataset_conditions)
# config_path = "main_config/final_result_15_3_1/main_model_3"
# dataset_conditions = ["dataset_A_0","dataset_A_1", "dataset_B_0","dataset_B_1","dataset_B_2"]

#deal with different pretrain setup
# main_path = "final_result_15_3_1/main_model_3/pretrain_tune/6"
# extra_merge= list()
# extra_merge.append([
#     "LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_PRE_TRAIN_EPOCHS",6,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO", 1.0,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO", 0.2,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_TARGET_LOSS_RATIO", 1.0,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_SOURCE_LOSS_RATIO", 1.0
#
# ])
# setup_experiments(main_path,config_path,aug_conditions,norm_conditions,trainer_conditions,dataset_conditions,extra_merge=extra_merge)
#
# main_path = "final_result_15_3_1/main_model_3/pretrain_tune/8"
# extra_merge= list()
# extra_merge.append([
#     "LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_PRE_TRAIN_EPOCHS",8,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO", 1.0,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO", 0.4,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_TARGET_LOSS_RATIO", 1.0,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_SOURCE_LOSS_RATIO", 1.0
#
# ])
# setup_experiments(main_path,config_path,aug_conditions,norm_conditions,trainer_conditions,dataset_conditions,extra_merge=extra_merge)
#
# main_path = "final_result_15_3_1/main_model_3/pretrain_tune/10"
# extra_merge= list()
# extra_merge.append([
#     "LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_PRE_TRAIN_EPOCHS",10,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO", 1.0,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO", 1.0,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_TARGET_LOSS_RATIO", 1.0,
#     # "LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_SOURCE_LOSS_RATIO", 1.0
#
# ])
# setup_experiments(main_path,config_path,aug_conditions,norm_conditions,trainer_conditions,dataset_conditions,extra_merge=extra_merge)


#
# all_configs = list()
# list_config_path = ["final_result_14_3_1/sub","final_result_14_3_3/sub"]
# for path in list_config_path:
#     config = os.path.join("main_config/",path)
#     list_config_out = generate_sub_experiment_config(config, trainer_conditions, dataset_conditions)
#     list_config_out.append(config)
#     all_configs.extend(list_config_out)
#
# print("all configs : ",all_configs)
# for config_path in all_configs:
#     main_path = config_path.split("/")
#     main_path = '/'.join(main_path[1:])
#     print("main path : ",main_path)
#     setup_experiments(main_path,config_path,aug_conditions,norm_conditions,trainer_conditions,dataset_conditions)

# Python Program to Print
# all subsets of given size of a set

# main_path = "final_result_14_3_1/main_model_1"
# config_path = "main_config/final_result_14_3_1/main_model_1"