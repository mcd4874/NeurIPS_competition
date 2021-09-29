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
import matplotlib
matplotlib.use('TkAgg')
from NeurIPS_1.util.support import (
    reformat,load_source_sleep,load_target_sleep,load_test_sleep,print_info,print_dataset_info,convert_volt_to_micro,generate_data_file,
    combine,load_test_sleep_combine,load_full_target_sleep
)



def generate_class_weight(label):
    """
    generate the weight ratio based on total labels of every subjects
    label : [subject_1,subject_2,..] and subject = (trials)
    """

    total = len(label)
    print("unique label : ",np.unique(label))
    class_sample_count = np.array([len(np.where(label == t)[0]) for t in np.unique(label)])
    weight = total / class_sample_count
    return weight



def print_label_info(subject_labels,dataset_name=""):
    print("dataset {}".format(dataset_name))
    for subject_id in range(len(subject_labels)):
        subject_label = subject_labels[subject_id]
        # print("subject label shape : ",subject_label.shape)
        subject_class_weight = generate_class_weight(subject_label)
        print("subject {} has class weight {}".format(subject_id,subject_class_weight))



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

##provide path to where you store the source data, target sleep data and test sleep data
# source_path = "C:/Users/wduong/mne_data/SleepSource/SleepSource"
# final_target_sleep_data = "C:\\Users\\wduong\\mne_data\\finalSleep\\sleep_target\\"
# final_test_sleep_data = "C:\\Users\\wduong\\mne_data\\finalSleep\\testing\\"


source_path = "../da_dataset/SleepSource/SleepSource"
final_target_sleep_data = "../da_dataset/finalSleep/sleep_target"
final_test_sleep_data = "../da_dataset/finalSleep/testing"
#
#

source_data,source_label,source_meta = load_source_sleep(path=source_path)
source_data = convert_volt_to_micro(source_data)

subset_data,subset_label,subset_meta = split_source_data(source_data,source_label,source_meta,n_subset=5)


format = "testing"
target_data_1,target_label_1,train_meta_1,target_data_2,target_label_2,train_meta_2 = load_target_sleep(path=final_target_sleep_data,file_format=format,start_id=0,end_id=5)
target_data_1 = convert_volt_to_micro(target_data_1)
target_data_2 = convert_volt_to_micro(target_data_2)
format = "testing"
target_test_data_1, target_test_label_1, test_meta_1, target_test_data_2, target_test_label_2, test_meta_2 = load_test_sleep(path=final_test_sleep_data,file_format=format,start_id=5,end_id=14)
target_test_data_1 = convert_volt_to_micro(target_test_data_1)
target_test_data_2 = convert_volt_to_micro(target_test_data_2)

full_target_data,full_target_label,full_target_meta = load_full_target_sleep(path=final_target_sleep_data,file_format=format,start_id=0,end_id=5)
full_target_data = convert_volt_to_micro(full_target_data)


combine_test_data,combine_test_label,combine_test_meta = load_test_sleep_combine(path=final_test_sleep_data,file_format=format,start_id=5,end_id=14)
combine_test_data = convert_volt_to_micro(combine_test_data)


temp_target_data_1,temp_target_label_1,temp_train_meta_1 = reformat(target_data_1,target_label_1,train_meta_1)
temp_target_data_2,temp_target_label_2,temp_train_meta_2 = reformat(target_data_2,target_label_2,train_meta_2)

temp_full_target_data,temp_ful_target_label,temp_full_target_meta=reformat(full_target_data,full_target_label,full_target_meta)


print_label_info(temp_target_label_1,"session_1_data")
print_label_info(temp_target_label_2,"session_2_data")

print_label_info(temp_ful_target_label,"full_target_data")

print("full target label ",generate_class_weight(full_target_label))


# source_meta = {name: col.values for name, col in source_meta.items()}
train_meta_1 = {name: col.values for name, col in train_meta_1.items() if name in ["subject","session","run"]}
print("current train meta data : ", train_meta_1)

train_meta_2 = {name: col.values for name, col in train_meta_2.items() if name in ["subject","session","run"]}

combine_train_meta = {name: col.values for name, col in full_target_meta.items() if name in ["subject","session","run"]}

test_meta_1 = {name: col.values for name, col in test_meta_1.items() if name in ["subject","session","run"]}
test_meta_2 = {name: col.values for name, col in test_meta_2.items() if name in ["subject","session","run"]}
combine_test_meta = {name: col.values for name, col in combine_test_meta.items() if name in ["subject","session","run"]}


save_folder = "../da_dataset/task_1/task_1_final_case_1"

for subset_idx in range(len(subset_data)):
    print("current subset : ",subset_idx)
    print_dataset_info(subset_data[subset_idx], "source subset sleep")
    temp_data, temp_label, temp_meta = subset_data[subset_idx], subset_label[subset_idx], subset_meta[subset_idx]
    temp_meta = {name: col.values for name, col in temp_meta.items() if name in ["subject","session","run"]}

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

combine_target_dataset = {
    'data': full_target_data,
    'label': full_target_label,
    'meta_data': combine_train_meta,
    'dataset_name': "full_target_sleep",
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

save_folder = "../da_dataset/task_1/task_1_final_case_1"
generate_data_file([combine_target_dataset], folder_name=save_folder, file_name='full_target_sleep')

#
save_folder = "../da_dataset/task_1/task_1_final_test_case_1"
generate_data_file([combine_test_dataset], folder_name=save_folder, file_name='full_test_sleep')
