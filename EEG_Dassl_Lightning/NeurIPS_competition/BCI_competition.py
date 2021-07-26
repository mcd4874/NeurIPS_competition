from braindecode.util import set_random_seeds, np_to_var, var_to_np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np
from numpy.random import RandomState
import pickle
import time
import torch
import os
from torch import nn
from torch import optim
import torch.nn.functional as F

import mne

import NeurIPS_competition.util.shallow_net
from NeurIPS_competition.util.utilfunc import get_balanced_batches
from NeurIPS_competition.util.preproc import plot_confusion_matrix

cuda = torch.cuda.is_available()
print('gpu: ', cuda)
device = 'cuda' if cuda else 'cpu'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
rng = RandomState(seed)

# subj = 1
#
# for dataset in [BNCI2014001(), PhysionetMI(), Cho2017()]:
#     data = dataset.get_data(subjects=[subj])
#     ds_name = dataset.code
#     ds_type = dataset.paradigm
#     sess = 'session_T' if ds_name == "001-2014" else 'session_0'
#     run = sorted(data[subj][sess])[0]
#     ds_ch_names = data[subj][sess][run].info['ch_names']  # [0:22]
#     ds_sfreq = data[subj][sess][run].info['sfreq']
#     current_data_run = data[subj][sess][run]
#     tmp_data = current_data_run.get_data()
#     print("data info : ",current_data_run.get_data().shape)
#     print("max {} , min {} : ".format(np.max(tmp_data),np.min(tmp_data)))
#     print("{} is an {} dataset, acquired at {} Hz, with {} electrodes\nElectrodes names: ".format(ds_name, ds_type, ds_sfreq, len(ds_ch_names)))
#     print(ds_ch_names)
#     print()
def modify_data(data,time=256):
    return data[:, :, :time]
ds_src1 = Cho2017()
ds_src2 = PhysionetMI()
ds_tgt = BNCI2014001()

fmin, fmax = 4, 36
raw = ds_tgt.get_data(subjects=[1])[1]['session_T']['run_1']
tgt_channels = raw.pick_types(eeg=True).ch_names
sfreq = 128
# prgm_2classes = MotorImagery(n_classes=2, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax)
tmin=0.5
tmax=3.5
max_time_length = int((tmax-tmin)*sfreq)
tgt_prgm = MotorImagery(n_classes=4, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
X_tgt, label_tgt, m_tgt = tgt_prgm.get_data(dataset=ds_tgt)
# X_tgt, label_tgt, m_tgt = tgt_prgm.get_data(dataset=ds_tgt,subjects=[1, 2])
X_tgt = modify_data(X_tgt,time=max_time_length)

raw_src_1 = ds_src1.get_data(subjects=[1])[1]['session_0']['run_0']
raw_src_1_channels = raw_src_1.pick_types(eeg=True).ch_names
sfreq = 128
tmin=0
tmax=3
max_time_length = int((tmax-tmin)*sfreq)
src_1_prgm =MotorImagery(n_classes=2, channels=raw_src_1_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
X_src1, label_src1, m_src1 = src_1_prgm.get_data(dataset=ds_src1)
# X_src1, label_src1, m_src1 = src_1_prgm.get_data(dataset=ds_src1,subjects=[1, 2, 3])
X_src1 = modify_data(X_src1,time=max_time_length)

raw_src_2 = ds_src2.get_data(subjects=[1])[1]['session_0']['run_4']
raw_src_2_channels = raw_src_2.pick_types(eeg=True).ch_names
sfreq = 128
tmin=0
tmax=3
max_time_length = int((tmax-tmin)*sfreq)
src_2_prgm =MotorImagery(events=dict(left_hand=2, right_hand=3, feet=5),n_classes=3, channels=raw_src_2_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
X_src2, label_src2, m_src2 = src_2_prgm.get_data(dataset=ds_src2)
# X_src2, label_src2, m_src2 = src_2_prgm.get_data(dataset=ds_src2,subjects=[1, 2, 3,4])

# subjects=[1, 2, 3,4]
X_src2 = modify_data(X_src2,time=max_time_length)

# X_src1, label_src1, m_src1 = prgm_2classes.get_data(dataset=ds_src1,subjects=[1, 2, 3])
# X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2,subjects=[1, 2, 3,4])



print("First source dataset has {} trials with {} electrodes and {} time samples".format(*X_src1.shape))
print("Second source dataset has {} trials with {} electrodes and {} time samples".format(*X_src2.shape))
print("Target dataset has {} trials with {} electrodes and {} time samples".format(*X_tgt.shape))

print("First source dataset max : {}, min : {} ".format(np.max(X_src1),np.min(X_src1)))
print("Second source dataset max : {}, min : {} ".format(np.max(X_src2),np.min(X_src2)))
print("Target dataset max : {}, min : {} ".format(np.max(X_tgt),np.min(X_tgt)))

# print("target dataset : ",X_tgt.shape)
print ("\nSource dataset 1 include labels: {}".format(np.unique(label_src1)))
print ("Source dataset 2 include labels: {}".format(np.unique(label_src2)))
print ("Target dataset 1 include labels: {}".format(np.unique(label_tgt)))



def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    else: return 2

y_src1 = np.array([relabel(l) for l in label_src1])
y_src2 = np.array([relabel(l) for l in label_src2])
y_tgt = np.array([relabel(l) for l in label_tgt])


# class TrainObject(object):
#     def __init__(self, X, y):
#         assert len(X) == len(y)
#         # Normalised, you could choose other normalisation strategy
#         mean = np.mean(X,axis=1,keepdims=True)
#         # here normalise across channels as an example, unlike the in the sleep kit
#         std = np.std(X, axis=1, keepdims=True)
#         X = (X - mean) / std
#         # we scale it to 1000 as a better training scale of the shallow CNN
#         # according to the orignal work of the paper referenced above
#         self.X = X.astype(np.float32)
#         self.y = y.astype(np.int64)

# train_set = TrainObject(X_train, y=y_train)
# valid_set = TrainObject(X_val, y=y_val)
# test_set = TrainObject(X_test, y=y_test)

#normalize data
# src_1_set = TrainObject(X_src1,y_src1)
# src_2_set = TrainObject(X_src2,y_src2)
# tgt_set = TrainObject(X_tgt,y_tgt)
# print("First source dataset max : {}, min : {} ".format(np.max(src_1_set.X),np.min(src_1_set.X)))
# print("Second source dataset max : {}, min : {} ".format(np.max(src_2_set.X),np.min(src_2_set.X)))
# print("Target dataset max : {}, min : {} ".format(np.max(tgt_set.X),np.min(tgt_set.X)))

# def generate_matlab_files(target_dataset,source_datasets,save_path,file_name):
#     target_dataset_name = list(target_dataset.keys())[0]
#     target_dataset_data = target_dataset[target_dataset_name]
#
#
#     source_list = list()
#     for source_dataset_name,source_dataset_data in source_datasets.items():
#         source = {
#             "source_domain_data":source_dataset_data[0],
#             "source_domain_label":source_dataset_data[1],
#             "source_label_name_map":source_dataset_data[3],
#             "dataset_name":source_dataset_name,
#             "subject_id": source_dataset_data[2]
#         }
#         source_list.append(source)
#
#     matlab_data = {
#         "source_domain": source_list,
#         "target_domain": {
#             "target_domain_data": target_dataset_data[0],
#             "target_domain_label": target_dataset_data[1],
#             "target_label_name_map": target_dataset_data[3],
#             "dataset_name":target_dataset_name,
#             "subject_id":target_dataset_data[2]
#         }
#     }
#
#
#     if not os.path.isdir(save_path):
#         os.makedirs(save_path)
#
#     data_file = '{}_transfer_learning.mat'.format(file_name)
#     data_file = join(save_path,data_file)
#     text_file = 'target_source_data_record.json'
#     text_file = join(save_path,text_file)
#
#
#     import json
#
#
#     dictionary = {'target_dataet': target_dataset_name, 'source_datasets': list(source_datasets.keys())}
#     with open(text_file, "w") as outfile:
#         json.dump(dictionary, outfile)
#     from scipy.io import savemat
#     savemat(data_file, matlab_data)

# print("meta data dict : ",m_src1.to_dict(orient='list'))
# m_src1 = m_src1.to_dict(orient='list')
m_src1 = {name: col.values for name, col in m_src1.items()}
m_src2 = {name: col.values for name, col in m_src2.items()}
m_tgt = {name: col.values for name, col in m_tgt.items()}

for k,v in m_src1.items():
    print(" k : ",k)
    print(" val : ",v)
list_dataset = list()
dataset_1 = {
    'data':X_src1,
    'label':y_src1,
    'meta_data':m_src1,
    'dataset_name':'cho2017'
}

dataset_2 = {
    'data':X_src2,
    'label':y_src2,
    'meta_data':m_src2,
    'dataset_name':'physionet'
}

dataset_3 = {
    'data':X_tgt,
    'label':y_tgt,
    'meta_data':m_tgt,
    'dataset_name':'BCI_IV'
}
list_dataset.append(dataset_1)
list_dataset.append(dataset_2)
list_dataset.append(dataset_3)
file_name = 'NeurIPS_TL'
# data_file = '{}.mat'.format(file_name)
data_file = '{}.mat'.format(file_name)
folder = 'case_2'
if not os.path.isdir(folder):
    os.makedirs(folder)
data_file = os.path.join(folder,data_file)

from scipy.io import savemat
savemat(data_file, {'datasets':list_dataset})
