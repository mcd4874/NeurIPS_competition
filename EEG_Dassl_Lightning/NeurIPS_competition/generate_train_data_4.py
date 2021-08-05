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
    correct_EEG_data_order,relabel,process_target_data,relabel_target,load_dataset_A,load_dataset_B,modify_data,
    generate_data_file,print_dataset_info
)


from scipy.linalg import sqrtm, inv
#
# def euclidean_alignment(x):
#     """
#     convert trials in data with EA technique
#     """
#
#     assert len(x.shape) == 3
#
#     r = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
#     if np.iscomplexobj(r):
#         print("covariance matrix problem")
#     if np.iscomplexobj(sqrtm(r)):
#         print("covariance matrix problem sqrt")
#
#     r_op = inv(sqrtm(r))
#     # print("r_op shape : ", r_op.shape)
#     # print("data shape : ",x.shape)
#     # print("r_op : ", r_op)
#     if np.iscomplexobj(r_op):
#         print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
#               "not, check data!!")
#         r_op = np.real(r_op).astype(np.float32)
#     elif not np.any(np.isfinite(r_op)):
#         print("WARNING! Not finite values in R Matrix")
#
#     results = np.matmul(r_op, x)
#     # print("r_op shape : ",r_op.shape)
#     # print("data shape : ",x.shape)
#     # print("r_op : ",r_op)
#     # print("result shape : ",results.shape)
#     # print("a trial before convert : ",x[0,:,:])
#     # print("a trial after convert : ",results[0,:,:])
#     return results
class EuclideanAlignment:
    """
    convert trials of each subject to a new format with Euclidean Alignment technique
    https://arxiv.org/pdf/1808.05464.pdf
    """
    def __init__(self,list_r_op=None):
        self.list_r_op = list_r_op
    def calculate_r_op(self,data):
        assert len(data.shape) == 3
        r = np.matmul(data, data.transpose((0, 2, 1))).mean(0)
        if np.iscomplexobj(r):
            print("covariance matrix problem")
        if np.iscomplexobj(sqrtm(r)):
            print("covariance matrix problem sqrt")

        r_op = inv(sqrtm(r))
        # print("r_op shape : ", r_op.shape)
        # print("data shape : ",x.shape)
        # print("r_op : ", r_op)
        if np.iscomplexobj(r_op):
            print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
                  "not, check data!!")
            r_op = np.real(r_op).astype(np.float32)
        elif not np.any(np.isfinite(r_op)):
            print("WARNING! Not finite values in R Matrix")
        return r_op
    def convert_trials(self,data,r_op):
        results = np.matmul(r_op, data)
        return results
    def generate_list_r_op(self,subjects_data):
        list_r_op = list()
        for subject_idx in range(len(subjects_data)):
            subject_data = subjects_data[subject_idx]
            r_op = self.calculate_r_op(subject_data)
            list_r_op.append(r_op)
        self.list_r_op = list_r_op
    def convert_subjects_data_with_EA(self,subjects_data):
        #calculate r_op for each subject
        if self.list_r_op:
            assert len(self.list_r_op) == len(subjects_data)
            print("use exist r_op")
        else:
            self.list_r_op = self.generate_list_r_op(subjects_data)
        new_data = list()
        for subject_idx in range(len(subjects_data)):
            subject_data = subjects_data[subject_idx]
            r_op = self.list_r_op[subject_idx]
            subject_data = self.convert_trials(subject_data,r_op)
            new_data.append(subject_data)
        return new_data


# def convert_subjects_data_with_EA(subjects_data):
#     new_data = list()
#     for subject_idx in range(len(subjects_data)):
#         subject_data = subjects_data[subject_idx]
#         subject_data = euclidean_alignment(subject_data)
#         new_data.append(subject_data)
#     return new_data



def reformat(data,label,meta_data):
    n_subjects = len(np.unique(meta_data['subject']))
    new_data = []
    new_label = []
    new_meta_data = []
    start=0
    unique_subject_ids = np.unique(meta_data['subject'])
    for i in range(n_subjects):
        current_subject = unique_subject_ids[i]
        subject_meta_data = meta_data[meta_data['subject']==current_subject]
        if len(subject_meta_data) > 0:
            trials = len(subject_meta_data)
            end = start+trials
            subject_data = data[start:end]
            subject_label = label[start:end]
            new_data.append(subject_data)
            new_label.append(subject_label)
            new_meta_data.append(subject_meta_data)
            # print("current meta : ",subject_meta_data)
            # print("len meta : ",len(subject_meta_data))
            # print("len subject size : ",len(subject_data))
            # print("len label size : ",len(subject_label))
            start = end
    return new_data,new_label,new_meta_data

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

# epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels)
epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1,2,3])

print("cho2017 current chans : ",epoch_X_src1.ch_names)
print("size : ",len(epoch_X_src1.ch_names))
montage = epoch_X_src1.get_montage()
target_channels = epoch_X_src1.ch_names

# epoch_X_src2, label_src2, m_src2 = load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels)
epoch_X_src2, label_src2, m_src2 = load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1,2,3])

print("physionet current chans : ",epoch_X_src2.ch_names)
print("size : ",len(epoch_X_src2.ch_names))

# epoch_X_src3, label_src3, m_src3 = load_BCI_IV(fmin=fmin,fmax=fmax,selected_chans=target_channels,montage=montage)
# epoch_X_src3, label_src3, m_src3 = load_BCI_IV(fmin=fmin,fmax=fmax,selected_chans=target_channels,montage=montage,subjects=[1,2])
epoch_X_src3, label_src3, m_src3 = load_BCI_IV(fmin=fmin,fmax=fmax,selected_chans=target_channels,montage=montage,subjects=[1,2])



src3 = epoch_X_src3.get_data()
src2 = correct_EEG_data_order(epoch_X_src2,target_channels)
src1 = epoch_X_src1.get_data()

def convert_volt_to_micro(data):
    data = data * 1e6
    return data


X_src1 = modify_data(src1,time=max_time_length)
X_src2 = modify_data(src2,time=max_time_length)
X_src3 = modify_data(src3,time=max_time_length)

X_src1 = convert_volt_to_micro(X_src1)
X_src2 = convert_volt_to_micro(X_src2)
X_src3 = convert_volt_to_micro(X_src3)

# tmp_X_src3, tmp_label_src3,_ = reformat(X_src3,label_src3, m_src3)
# tmp_X_src3 = convert_subjects_data_with_EA(tmp_X_src3)

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

X_MIA_test_data = load_dataset_A(train=False, norm=False, selected_chans=target_channels)
X_MIB_test_data = load_dataset_B(train=False, norm=False, selected_chans=target_channels)


X_MIA_train_data = convert_volt_to_micro(X_MIA_train_data)
X_MIB_train_data = convert_volt_to_micro(X_MIB_train_data)

X_MIA_test_data = convert_volt_to_micro(X_MIA_test_data)
X_MIB_test_data = convert_volt_to_micro(X_MIB_test_data)

temp_X_MIA_test_data = np.split(X_MIA_test_data,2)
temp_X_MIB_test_data = np.split(X_MIB_test_data,3)
temp_X_MIA_train_data = np.split(X_MIA_train_data,2)
temp_X_MIB_train_data = np.split(X_MIB_train_data,3)
# print("test data shape {}.{}: ".format(len(temp_X_MIA_test_data),temp_X_MIA_test_data[0].shape))
# print("train data shape {}.{}: ".format(len(temp_X_MIA_train_data),temp_X_MIA_train_data[0].shape))
temp_X_MIA = np.concatenate([temp_X_MIA_train_data,temp_X_MIA_test_data],axis=1)
temp_X_MIB = np.concatenate([temp_X_MIB_train_data,temp_X_MIB_test_data],axis=1)

# print("temp X _MIA shape : ",temp_X_MIA.shape)
# print("temp X _MIA shape {}.{}: ".format(len(temp_X_MIA),temp_X_MIA[0].shape))

EA_A = EuclideanAlignment()
EA_A.generate_list_r_op(temp_X_MIA)
# print("len of list r op : ",len(EA_A.list_r_op))
# print(" r op shape : ",EA_A.list_r_op[0].shape)
# print("list_r_op : ",EA_A.list_r_op)
temp_X_MIA_train_data = EA_A.convert_subjects_data_with_EA(temp_X_MIA_train_data)
# print("convert trials : ",)
print("new trial : ",temp_X_MIA_train_data[0])
EA_B = EuclideanAlignment()
EA_B.generate_list_r_op(temp_X_MIB)
X_MIA_train_label = np.array([relabel_target(l) for l in X_MIA_train_label])
X_MIB_train_label = np.array([relabel_target(l) for l in X_MIB_train_label])

target_dataset_A = {
    'data': X_MIA_train_data,
    'label': X_MIA_train_label,
    'meta_data': m_tgt_A,
    'dataset_name': 'dataset_A',
    'r_op_list':EA_A.list_r_op
}

target_dataset_B = {
    'data': X_MIB_train_data,
    'label': X_MIB_train_label,
    'meta_data': m_tgt_B,
    'dataset_name': 'dataset_B',
    'r_op_list': EA_B.list_r_op
}

test_dataset_A = {
    'data': temp_X_MIA_test_data,
    'dataset_name': 'dataset_A',
    'r_op_list':EA_A.list_r_op
}

test_dataset_B = {
    'data': temp_X_MIB_test_data,
    'dataset_name': 'dataset_B',
    'r_op_list': EA_B.list_r_op
}

print_dataset_info(X_MIA_train_data,"train dataset A")
print_dataset_info(X_MIB_train_data,"train dataset B")

print_dataset_info(X_MIA_test_data,"test dataset A")
print_dataset_info(X_MIB_test_data,"test dataset B")

print_dataset_info(X_src1,"source 1 ")
print_dataset_info(X_src2,"source 2 ")
print_dataset_info(X_src3,"source 3 ")


generate_data_file([dataset_1,dataset_2,dataset_3,target_dataset_A],folder_name='case_8_A')
generate_data_file([dataset_1,dataset_2,dataset_3,target_dataset_B],folder_name='case_8_B')

generate_data_file([test_dataset_A,test_dataset_B],folder_name='test_data')

