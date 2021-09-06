from moabb.datasets import BNCI2014001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np
from numpy.random import RandomState
import pickle
import time
import torch
import os
import pandas as pd
import mne

from NeurIPS_1.util.support import (
    reformat,load_source_sleep,load_target_sleep,load_test_sleep,print_info,print_dataset_info,convert_volt_to_micro,generate_data_file,
    combine,load_test_sleep_combine
)

from NeurIPS_competition.util.setup_dataset import (
setup_datasets,setup_specific_subject_dataset
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cuda = torch.cuda.is_available()
print('gpu: ', cuda)
device = 'cuda' if cuda else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
rng = RandomState(seed)

#get common channel between dataset A and dataset B


import matplotlib
matplotlib.use('TkAgg')

source_data,source_label,source_meta = load_source_sleep()
source_data = convert_volt_to_micro(source_data)

def split_source_data(source_data,source_label,source_meta,n_subset=5):
    source_data, source_label, source_meta = reformat(source_data,source_label,source_meta)
    subject_per_subset = len(source_data)//n_subset
    start_idx = 0
    subset_data,subset_label,subset_meta = list(),list(),list()
    for subset in range(n_subset-1):
        end_idx = start_idx + subject_per_subset
        current_data = [source_data[i] for i in range(start_idx,end_idx)]
        current_label = [source_label[i] for i in range(start_idx,end_idx)]
        current_meta = [source_meta[i] for i in range(start_idx,end_idx)]
        current_data,current_label,current_meta = combine(current_data,current_label,current_meta)
        subset_data.append(current_data)
        subset_label.append(current_label)
        subset_meta.append(current_meta)
        start_idx = end_idx

    current_data = [source_data[i] for i in range(start_idx,len(source_data))]
    current_label = [source_label[i] for i in range(start_idx,len(source_data))]
    current_meta = [source_meta[i] for i in range(start_idx,len(source_data))]
    current_data, current_label, current_meta = combine(current_data, current_label, current_meta)
    subset_data.append(current_data)
    subset_label.append(current_label)
    subset_meta.append(current_meta)
    return subset_data,subset_label,subset_meta

subset_data,subset_label,subset_meta = split_source_data(source_data,source_label,source_meta,n_subset=5)
# for subset_idx in range(len(subset_data)):
#     print("current subset : ",subset_idx)
#     print_dataset_info(subset_data[subset_idx], "source subset sleep")
#     temp_data, temp_label, temp_meta = subset_data[subset_idx], subset_label[subset_idx], subset_meta[subset_idx]
#     temp_data, temp_label, temp_meta = reformat(temp_data, temp_label, temp_meta)
#     print_info(temp_data, "subject source data info sleep ")

target_data_1,target_label_1,train_meta_1,target_data_2,target_label_2,train_meta_2 = load_target_sleep()
target_data_1 = convert_volt_to_micro(target_data_1)
target_data_2 = convert_volt_to_micro(target_data_2)
target_test_data_1, target_test_label_1, test_meta_1, target_test_data_2, target_test_label_2, test_meta_2 = load_test_sleep()
target_test_data_1 = convert_volt_to_micro(target_test_data_1)
target_test_data_2 = convert_volt_to_micro(target_test_data_2)

combine_test_data,combine_test_label,combine_test_meta = load_test_sleep_combine()
combine_test_data = convert_volt_to_micro(combine_test_data)


print_dataset_info(source_data,"source sleep")
print_dataset_info(target_data_1,"target data 1")
print_dataset_info(target_data_2,"target data 2")
print_dataset_info(target_test_data_1,"target test data 1")
print_dataset_info(target_test_data_2,"target test data 2")

# print("source meta : ",source_meta)
# source_data,source_label,source_meta = reformat(source_data,source_label,source_meta)
# target_data_1,target_label_1,train_meta_1 = reformat(target_data_1,target_label_1,train_meta_1)
# target_data_2,target_label_2,train_meta_2 = reformat(target_data_2,target_label_2,train_meta_2)
# target_test_data_1, target_test_label_1,test_meta_1 = reformat(target_test_data_1, target_test_label_1,test_meta_1)
# target_test_data_2, target_test_label_2, test_meta_2 = reformat(target_test_data_2, target_test_label_2, test_meta_2)
# print_info(source_data,"subject source data info sleep ")
# print_info(target_data_1,"subject target data 1 info sleep ")
# print_info(target_data_2,"subject target data 2 info sleep ")
# print_info(target_test_data_1,"subject test data 1 info sleep ")
# print_info(target_test_data_2,"subject test data 2 info sleep ")



# source_meta = {name: col.values for name, col in source_meta.items()}
train_meta_1 = {name: col.values for name, col in train_meta_1.items() if name in ["subject","session","run"]}
print("current train meta data : ", train_meta_1)

train_meta_2 = {name: col.values for name, col in train_meta_2.items() if name in ["subject","session","run"]}
test_meta_1 = {name: col.values for name, col in test_meta_1.items() if name in ["subject","session","run"]}
test_meta_2 = {name: col.values for name, col in test_meta_2.items() if name in ["subject","session","run"]}
combine_test_meta = {name: col.values for name, col in combine_test_meta.items() if name in ["subject","session","run"]}


save_folder = "task_1_case_1"

for subset_idx in range(len(subset_data)):
    print("current subset : ",subset_idx)
    print_dataset_info(subset_data[subset_idx], "source subset sleep")
    temp_data, temp_label, temp_meta = subset_data[subset_idx], subset_label[subset_idx], subset_meta[subset_idx]
    temp_meta = {name: col.values for name, col in temp_meta.items() if name in ["subject","session","run"]}
    # print("current temp data : ",temp_meta)

    # temp_data, temp_label, temp_meta = reformat(temp_data, temp_label, temp_meta)
    # print_info(temp_data, "subject source data info sleep ")
    dataset_name = "source_sleep_{}".format(subset_idx)
    source_dataset = {
        'data': temp_data,
        'label': temp_label,
        'meta_data': temp_meta,
        'dataset_name': dataset_name,
    }
    file_name = dataset_name
    generate_data_file([source_dataset], folder_name=save_folder, file_name=file_name)


target_dataset_1 = {
    'data': target_data_1,
    'label': target_label_1,
    'meta_data': train_meta_1,
    'dataset_name': "target_sleep_1",
}

target_dataset_2 = {
    'data': target_data_2,
    'label': target_label_2,
    'meta_data': train_meta_2,
    'dataset_name': "target_sleep_2",
}

test_dataset_1 = {
    'data': target_test_data_1,
    'label': target_test_label_1,
    'meta_data': test_meta_1,
    'dataset_name': "test_sleep_1",
}

test_dataset_2 = {
    'data': target_test_data_2,
    'label': target_test_label_2,
    'meta_data': test_meta_2,
    'dataset_name': "test_sleep_2",
}

combine_test_dataset = {
    'data': combine_test_data,
    'label': combine_test_label,
    'meta_data': combine_test_meta,
    'dataset_name': "full_test_sleep",
}

save_folder = "task_1_case_1"
generate_data_file([target_dataset_1], folder_name=save_folder, file_name='target_sleep_1')
generate_data_file([target_dataset_2], folder_name=save_folder, file_name='target_sleep_2')
#
save_folder = "task_1_test_case_1"
generate_data_file([test_dataset_1], folder_name=save_folder, file_name='test_sleep_1')
generate_data_file([test_dataset_2], folder_name=save_folder, file_name='test_sleep_2')
generate_data_file([combine_test_dataset], folder_name=save_folder, file_name='full_test_sleep')
