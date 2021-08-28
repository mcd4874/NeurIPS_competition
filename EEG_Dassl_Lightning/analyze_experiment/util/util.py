import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def generate_data_paths(common_path, prefix_lists, append_dir):
    if (len(prefix_lists) == 0):
        # print("append dir : ",append_dir)
        new_path = common_path.format(*append_dir)
        if not os.path.exists(new_path):
            return []
        return [new_path]
    else:
        results = []
        current_prefixs = prefix_lists[0]
        remain_prefix_lists = prefix_lists[1:]
        for prefix in current_prefixs:
            dirs = append_dir.copy()
            dirs.append(prefix)
            new_paths = generate_data_paths(common_path, remain_prefix_lists, dirs)
            results = results + new_paths
        return results



def generate_history_results_path(row, full_result_path):
    # history_folder = 'history'
    # history_file = 'history.csv'
    remain='default/version_0/metrics.csv'
    test_fold = row['test_fold']
    shuffle_fold = row['shuffle_fold']
    increment_fold = row['increment_fold']
    valid_fold = row['valid_fold']
    # provided_valid_fold = valid_fold.split("_")[-1]
    history_path = os.path.join(full_result_path,test_fold, shuffle_fold,increment_fold, valid_fold,
                                remain)


def load_data(data_paths, result_folder, result_file_name, info_file_name, load_history=False):
    list_data = []
    if len(data_paths) ==0:
        print("no data path ")
    for data_path in data_paths:
        result_folder_path = os.path.join(data_path, result_folder)
        result_data_path = os.path.join(result_folder_path, result_file_name)
        # check if file result exists
        if os.path.exists(result_data_path):
            data = pd.read_excel(result_data_path)

            info_data_path = os.path.join(result_folder_path, info_file_name)
            if os.path.exists(info_data_path):
                with open(info_data_path) as f:
                    info_data = json.load(f)
                    extra_fields = info_data["EXTRA_FIELDS"]
                    field_names = list(extra_fields.keys())
                    for field_name in field_names:
                        data[field_name] = extra_fields[field_name]
                list_data.append(data)
            else:
                print("no data info for {} ".format(result_data_path))

            if load_history:
                data['history_path'] = data.apply(lambda row: generate_history_results_path(row, data_path), axis=1)
        #                 print(data['history_path'])

        else:
            print("the current data path {} does not exist ".format(result_data_path))

    final_data = pd.concat(list_data).reset_index(drop=True)
    return final_data


def load_history_data(data_table, pick_cols, data_path_col='history_path'):
    if data_path_col not in data_table.columns:
        print("there are no history path to load history data")
        return
    history_information_table = []
    temp = data_table[pick_cols]
    history_cols = temp[data_path_col]
    for path in history_cols.values:
        history_data = pd.read_csv(path)
        if 'val_loss_x' in history_data.columns:
            history_data = history_data.rename(columns={"val_loss_x": "val_loss"})
        history_data[data_path_col] = [path] * len(history_data)
        history_data["epoch"] = np.arange(len(history_data))
        history_information_table.append(history_data)
    history_information_table = pd.concat(history_information_table)
    merge_table = pd.merge(temp, history_information_table, on=[data_path_col])
    return merge_table


def summarize_history(data_table, pick_cols, col_pick_model="val_loss", pick_min=True, max_epochs=100,
                      col_pick_max="test_accuracy", data_path_col='history_path'):
    if data_path_col not in data_table.columns:
        print("there are no history path to load history data")
        return
    history_information_table = []
    temp = data_table[pick_cols]
    history_cols = temp[data_path_col]
    for path in history_cols.values:
        fix_col_pick_model = col_pick_model
        history_data = pd.read_csv(path)
        available_cols = history_data.columns
        # modify val_loss for target model across experiment
        if 'val_loss_x' in available_cols:
            history_data = history_data.rename(columns={'val_loss_x': 'val_loss'}, inplace=False)

        # check if col_pick_model exist
        if not col_pick_model in history_data.columns:
            print("col {} isn't in the history data ".format(col_pick_model))
            print("use default val_loss as pick col")
            fix_col_pick_model = "val_loss"

        # limit total epoch to max epoch
        max_history_epoch = len(history_data)
        if max_history_epoch > max_epochs:
            history_data = history_data[:max_epochs]

        # deal with how to use a metric to pick best model
        if pick_min:
            pick_row_idx = history_data[fix_col_pick_model].argmin()
        else:
            pick_row_idx = history_data[fix_col_pick_model].argmax()

        #

        # val_loss_name = 'val_loss' if 'val_loss' in history_data.columns else 'val_loss_x'
        metric_pick_model = ['']
        # get max possible test_auc score information
        best_row_idx = history_data[col_pick_max].argmax()

        # best_col_pick_model = history_data.loc[best_row_idx, col_pick_max]
        # best_col_pick_max = history_data.loc[best_row_idx, col_pick_max]
        #
        # best_test_auc = history_data.loc[best_row_idx, col_pick_max]
        test_class_col = [col for col in pick_cols if "test_class_" in col]

        history_info_dict = {
            "model_choice": ["best_possible_epoch", "picked_epoch"],
            "epoch": [best_row_idx, pick_row_idx],
            col_pick_max: [history_data.loc[best_row_idx, col_pick_max], history_data.loc[pick_row_idx, col_pick_max]],
            fix_col_pick_model: [history_data.loc[best_row_idx, fix_col_pick_model],
                                 history_data.loc[pick_row_idx, fix_col_pick_model]],
            "history_path": [path, path]
        }
        for test_class in test_class_col:
            best_test_class_acc = history_data.loc[best_row_idx, test_class]
            pick_test_class_acc = history_data.loc[pick_row_idx, test_class]
            history_info_dict[test_class] = [best_test_class_acc, pick_test_class_acc]

        history_information = pd.DataFrame(history_info_dict)
        history_information_table.append(history_information)
    history_information_table = pd.concat(history_information_table)
    merge_table = pd.merge(temp, history_information_table, on=[data_path_col])
    return merge_table




def filter_history_information(data, condition_list):
    temp = data.copy()
    for condition in condition_list:
        col = condition[0]
        match_values = condition[1]
        if len(match_values) == 1:
            temp = temp[temp[col] == match_values[0]]
        else:
            temp = temp[temp[col].isin(match_values)]
    return temp


def generate_concat_dataset(list_dataset, extra_field_name):
    dataset_field_names = list(list_dataset.keys())
    concat_dataset = []
    for dataset_field_name in dataset_field_names:
        update_dataset = list_dataset[dataset_field_name]
        update_dataset[extra_field_name] = len(update_dataset) * [dataset_field_name]
        concat_dataset.append(update_dataset)
    concat_dataset = pd.concat(concat_dataset)
    concat_dataset = concat_dataset.reset_index(drop=True)
    return concat_dataset


def load_experiment_data(common_path, model_list, seed_list=None, norm_list=None, model_data_prefix=None,
                         new_col_generate=None):
    #             model_list = [
    #     'vanilla',
    #     'adaptation',
    #     'adapt_dann',
    #     'cdan',

    #     'adapt_equal_dann',
    #     'adapt_equal_label',
    #     'vanilla_equal_label'
    #     'adapt_share_label'
    #     'vanilla_aug',
    #     'vanilla_aug_1',
    #         'vanilla_aug_2',

    #             'adapt_aug'
    #     'vanilla_equal_aug',
    #     'adapt_equal_aug'

    #         ]
    if seed_list is None:
        seed_list = [
            "seed_v0",
            "seed_v1",
            "seed_v2"
        ]
    if norm_list is None:
        norm_list = [
            'norm_none',
            #     'norm_zscore'

        ]

    if model_data_prefix is None:
        model_data_prefix = [
                "BCI_IV",
            # "GIGA"
        ]

    result_folder = 'results_v1'
    file_name = 'model_result..xlsx'
    info_file_name = 'model_info.json'
    prefix_lists = [seed_list, model_list, norm_list, model_data_prefix]
    list_full_path = generate_data_paths(common_path, prefix_lists, [])
    data_result = load_data(list_full_path, result_folder, file_name, info_file_name, load_history=True)
    data_cols = data_result.columns

    #get test_class_{}_acc col
    test_class_col = [col for col in data_cols if "test_class_" in col]
    pick_cols = ["seed", "normalize", "dataset", "test_fold", "increment_fold", "valid_fold", "history_path","model"]
    pick_cols = pick_cols+test_class_col

    if new_col_generate is not None:
        new_col_name = new_col_generate[0]
        func = new_col_generate[1]
        data_result[new_col_name] = data_result.apply(lambda row: func(row,data_cols), axis=1)
        pick_cols.append(new_col_name)

    summary_history = summarize_history(data_result, pick_cols)
    history_data = load_history_data(data_result, pick_cols)

    # modify the increment_fold name manually
    data_result['increment_fold'] = data_result['increment_fold'].replace(
        ['increment_fold_1', 'increment_fold_2', 'increment_fold_3', 'increment_fold_4'], ['1', '2', '3', '4'])
    summary_history['increment_fold'] = summary_history['increment_fold'].replace(
        ['increment_fold_1', 'increment_fold_2', 'increment_fold_3', 'increment_fold_4'], ['1', '2', '3', '4'])
    history_data['increment_fold'] = history_data['increment_fold'].replace(
        ['increment_fold_1', 'increment_fold_2', 'increment_fold_3', 'increment_fold_4'], ['1', '2', '3', '4'])


    return [data_result, summary_history, history_data]

def generate_model_types(row):
    model = row['model']
    source_label_spaces = row['source_label_space']
    target_label_spaces = row['target_label_space']
    if model == 'BaseModel':
        return 'vanilla'
    elif model == 'HeterogeneousModelAdaptation' and (source_label_spaces == target_label_spaces):
        return 'AdaptationV1'
    elif model == 'HeterogeneousModelAdaptation':
        return 'Adaptation'
    elif model == 'ShareLabelModelAdaptation' and (source_label_spaces == target_label_spaces):
        return 'AdaptationV2'
    elif model == 'HeterogeneousDANN' and (source_label_spaces == target_label_spaces):
        return 'DannV1'
    elif  model == 'HeterogeneousDANN':
        return 'Dann'
    elif  model == 'HeterogeneousCDAN':
        return 'CDAN'
    else:
        return 'NA'