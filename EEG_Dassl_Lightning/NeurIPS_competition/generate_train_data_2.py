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

from NeurIPS_competition.util.support import (
    expand_data_dim,normalization,generate_common_chan_test_data,load_Cho2017,load_Physionet,load_BCI_IV,
    correct_EEG_data_order,process_target_data,load_dataset_A,load_dataset_B,modify_data,
    generate_data_file
)

def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    elif l =='feet': return 2
    else: return 3
def relabel_target(l):
    if l == 0: return 0
    elif l == 1: return 1
    elif l == 2: return 2
    else: return 3
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
target_channels = generate_common_chan_test_data()
print("common target chans : ",target_channels)
fmin=4
fmax=36
tmax=3
tmin=0
sfreq=128
max_time_length = int((tmax - tmin) * sfreq)

epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels)
# epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1,2,3])

print("cho2017 current chans : ",epoch_X_src1.ch_names)
print("size : ",len(epoch_X_src1.ch_names))
montage = epoch_X_src1.get_montage()
target_channels = epoch_X_src1.ch_names

epoch_X_src2, label_src2, m_src2 = load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels)
# epoch_X_src2, label_src2, m_src2 = load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1,2,3])

print("physionet current chans : ",epoch_X_src2.ch_names)
print("size : ",len(epoch_X_src2.ch_names))

epoch_X_src3, label_src3, m_src3 = load_BCI_IV(fmin=fmin,fmax=fmax,selected_chans=target_channels,montage=montage)
# epoch_X_src3, label_src3, m_src3 = load_BCI_IV(fmin=fmin,fmax=fmax,selected_chans=target_channels,montage=montage,subjects=[1,2])


src3 = epoch_X_src3.get_data()
src2 = correct_EEG_data_order(epoch_X_src2,target_channels)
src1 = epoch_X_src1.get_data()


X_src1 = modify_data(src1,time=max_time_length)
X_src2 = modify_data(src2,time=max_time_length)
X_src3 = modify_data(src3,time=max_time_length)

y_src1 = np.array([relabel(l) for l in label_src1])
y_src2 = np.array([relabel(l) for l in label_src2])
y_src3 = np.array([relabel(l) for l in label_src3])

m_src1 = {name: col.values for name, col in m_src1.items()}
m_src2 = {name: col.values for name, col in m_src2.items()}
m_src3 = {name: col.values for name, col in m_src3.items()}

dataset_1 = {
    'data': X_src1,
    'label': y_src1,
    'meta_data': m_src1,
    'dataset_name': 'cho2017'
}

dataset_2 = {
    'data': X_src2,
    'label': y_src2,
    'meta_data': m_src2,
    'dataset_name': 'physionet'
}

dataset_3 = {
    'data': X_src3,
    'label': y_src3,
    'meta_data': m_src3,
    'dataset_name': 'BCI_IV'
}

X_MIA_train_data,X_MIA_train_label,m_tgt_A = load_dataset_A(train=True,selected_chans=target_channels)
X_MIB_train_data,X_MIB_train_label,m_tgt_B = load_dataset_B(train=True,selected_chans=target_channels)

X_MIA_train_label = np.array([relabel_target(l) for l in X_MIA_train_label])
X_MIB_train_label = np.array([relabel_target(l) for l in X_MIB_train_label])

target_dataset_A = {
    'data': X_MIA_train_data,
    'label': X_MIA_train_label,
    'meta_data': m_tgt_A,
    'dataset_name': 'dataset_A'
}

target_dataset_B = {
    'data': X_MIB_train_data,
    'label': X_MIB_train_label,
    'meta_data': m_tgt_B,
    'dataset_name': 'dataset_B'
}



generate_data_file([dataset_1,dataset_2,dataset_3,target_dataset_A],folder_name='case_6_A')
generate_data_file([dataset_1,dataset_2,dataset_3,target_dataset_B],folder_name='case_6_B')
