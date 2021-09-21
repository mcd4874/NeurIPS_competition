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
    elif trainer_type == "share_adaptV1" or trainer_type == "deepsleep_share_adaptV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiShareAdaptationV1"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiShareAdaptationV1"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_dataset']
        ]
    elif trainer_type == "deepsleep_share_mcd":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiShareMCD"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiShareMCD"],
             ["DATAMANAGER.MANAGER_TYPE", 'multi_datasetV2']
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

    elif trainer_type == "dan":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetDan"],
             ["LIGHTNING_MODEL.TRAINER.NAME", "MultiDatasetDan"],
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
    elif trainer_type == "dann" or trainer_type == "deepsleep_dann":
        match_pair = [
            ["EXTRA_FIELDS.model", "DANN"],
            ["LIGHTNING_MODEL.TRAINER.NAME", "DANN"],
            ["DATAMANAGER.MANAGER_TYPE", 'single_datasetV2']
        ]
    elif trainer_type == "mcd" or trainer_type == "deepsleep_mcd":
        match_pair = [
            ["EXTRA_FIELDS.model", "MCD"],
            ["LIGHTNING_MODEL.TRAINER.NAME", "MCD"],
            ["DATAMANAGER.MANAGER_TYPE", 'single_datasetV2']
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
    for trainer in trainer_conditions:
        for dataset in dataset_conditions:
            current_config_path = os.path.join(config_path, trainer, dataset, 'main_config', 'transfer_adaptation.yaml')
            for aug in aug_conditions:
                for norm in norm_conditions:
                    generate_transfer_learning_config(main_path, current_config_path, aug_type=aug, norm_type=norm,
                                                      trainer_type=trainer, dataset_type=dataset,extra_merge=extra_merge)


# main_path = "task_2_final_2"
# config_path = "main_config/task_2_final_2"

main_path = "../NeurIPS_2/task_2_final_2/LA_EA"
config_path = "main_config/task_2_final_2/LA_EA"


# aug_conditions=["temp_aug","no_aug"]
aug_conditions=["no_aug"]

# norm_conditions = ["chan_norm","no_norm"]
norm_conditions = ["no_norm"]
# norm_conditions = ["chan_norm"]


# trainer_conditions = ["adaptationV1"]
# trainer_conditions = ["mcdV1","adaptationV1"]
# trainer_conditions = ["vanilla"]
trainer_conditions = ["mcdV1"]
# trainer_conditions = ["dan"]

# trainer_conditions = ["deepsleep_vanilla"]


# trainer_conditions = ["dannV1"]

# dataset_conditions = ["dataset_A", "dataset_B"]
dataset_conditions = ["dataset_A_0","dataset_A_1","dataset_A_2", "dataset_B_0","dataset_B_1"]

setup_experiments(main_path,config_path,aug_conditions,norm_conditions,trainer_conditions,dataset_conditions)


