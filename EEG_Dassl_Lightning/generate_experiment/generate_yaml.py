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

def generate_transfer_learning_config(main_path,config_path,aug_type,norm_type,trainer_type,dataset_type):
    config_f = open(config_path)
    config_file = CN(new_allowed=True).load_cfg(config_f)

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
    elif trainer_type == "adaptationV1":
        match_pair = [
            ["EXTRA_FIELDS.model", "MultiDatasetAdaptationV1"],
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
    else:
        match_pair = [
            ["EXTRA_FIELDS.target_dataset", "physionet"],
        ]
    dataset_info = [
        dict(dir_name=dataset_type, match_pair=match_pair)
    ]

    generate_config_file([aug_info,norm_info, trainer_info, dataset_info], config_file,main_path)


def setup_experiments(main_path, config_path, aug_conditions, norm_conditions, trainer_conditions, dataset_conditions):
    # config_path = "main_config/data_augmentation/{}/{}/transfer_adaptation.yaml"
    # main_path = "test_mdd/eegnet_1_1/{}"

    for trainer in trainer_conditions:
        for dataset in dataset_conditions:
            current_config_path = os.path.join(config_path, trainer, dataset, 'main_config', 'transfer_adaptation.yaml')
            for aug in aug_conditions:
                for norm in norm_conditions:
                    generate_transfer_learning_config(main_path, current_config_path, aug_type=aug, norm_type=norm,
                                                      trainer_type=trainer, dataset_type=dataset)
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
main_path = "experiment_4"
config_path = "main_config/experiment_4"
# main_path = "final_result_3"
# config_path = "main_config/final_result_3"
# main_path = "final_result_4"
# config_path = "main_config/final_result_4"

aug_conditions=["temp_aug","no_aug"]
# aug_conditions=["T_F_aug"]

norm_conditions = ["chan_norm","no_norm"]
# dataset_conditions = ["BCI_IV", "Cho2017", "Physionet"]
# trainer_conditions = ["vanilla", "adaptation", "component_adapt"]
trainer_conditions = ["vanilla","adaptationV1","adaptation"]
dataset_conditions = ["dataset_A", "dataset_B"]


setup_experiments(main_path,config_path,aug_conditions,norm_conditions,trainer_conditions,dataset_conditions)
