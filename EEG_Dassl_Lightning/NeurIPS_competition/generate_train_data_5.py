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
from scipy import signal
from collections import defaultdict

class LabelAlignment:
    def __init__(self,target_dataset):
        """
        assume target_data is (trials,channels,samples)
        target_label is (trials)
        """
        self.target_data,self.target_label = target_dataset
        self.target_r_op = self.generate_class_cov(self.target_data,self.target_label)

    def convert_source_data_with_LA(self, source_data,source_label):
        """

        Args:
            source_data: (n_subject,(trials,channels,samples))
            source_label: (n_subject,(trials))
        Returns:

        """
        new_source_data = list()
        for subject in range(len(source_data)):
            subject_data = source_data[subject]
            subject_label = source_label[subject]
            category_A_m = dict()
            new_subject_data = list()
            subject_category_r_op = self.generate_class_cov(subject_data,subject_label)
            for subject_label in list(subject_category_r_op.keys()):
                if subject_label not in list(self.target_r_op.keys()):
                    print("current label {} is not in target dataset ".format(subject_label))
                    return
                source_r_op = subject_category_r_op[subject_label]
                target_r_op = self.target_r_op[subject_label]
                A_m = np.matmul(target_r_op, source_r_op)
                category_A_m[subject_label] = A_m
            for trial in range(len(subject_data)):
                trial_data = subject_data[trial]
                trial_label = subject_label[trial]
                trial_A_m = category_A_m[trial_label]
                convert_trial_data = np.matmul(trial_A_m, trial_data)
                new_subject_data.append(convert_trial_data)
            new_subject_data = np.concatenate(new_subject_data)
            new_source_data.append(new_subject_data)
        # new_source_data = np.concatenate(new_source_data)
        return new_source_data,source_label

    def generate_class_cov(self,target_data,target_label):
        """
        Use the target data to generate an inverse Covariance for each class category.
        Args:
            target_data: (trials,channels,samples)
            target_label: (trials)

        Returns:

        """
        category_data = defaultdict(list)
        category_r_op = dict()
        for data,label in enumerate(zip(target_data,target_label)):
            category_data[label].append(data)
        for label,data in category_data.items():
            data= np.concatenate(data)
            r_op = self.calculate_r_op(data)
            category_r_op[label] = r_op
        return category_r_op

    def calculate_r_op(self, data):
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
        return list_r_op
    def convert_subjects_data_with_EA(self,subjects_data):
        #calculate r_op for each subject
        if self.list_r_op is not None:
            assert len(self.list_r_op) == len(subjects_data)
            print("use exist r_op")
        else:
            print("generate new r_op")
            self.list_r_op = self.generate_list_r_op(subjects_data)
        new_data = list()
        # print("size list r : ",len(self.list_r_op))
        # print("subject dat size : ",len(subjects_data))
        for subject_idx in range(len(subjects_data)):
            subject_data = subjects_data[subject_idx]
            r_op = self.list_r_op[subject_idx]
            subject_data = self.convert_trials(subject_data,r_op)
            new_data.append(subject_data)
        return new_data
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
def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    else: return 2
def relabel_target(l):
    if l == 0: return 0
    elif l == 1: return 1
    elif l ==2 : return 2
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

# epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels)
epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1])

print("cho2017 current chans : ",epoch_X_src1.ch_names)
print("size : ",len(epoch_X_src1.ch_names))
montage = epoch_X_src1.get_montage()
target_channels = epoch_X_src1.ch_names
print("target channels : ------",target_channels)


events=dict(left_hand=2, right_hand=3, feet=5, rest=1)
# epoch_X_src2, label_src2, m_src2 = load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels,events=events)
epoch_X_src2, label_src2, m_src2 = load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1,2,3],events=events)

print("physionet current chans : ",epoch_X_src2.ch_names)
print("size : ",len(epoch_X_src2.ch_names))

# epoch_X_src3, label_src3, m_src3 = load_BCI_IV(fmin=fmin,fmax=fmax,selected_chans=target_channels,montage=montage)
epoch_X_src3, label_src3, m_src3 = load_BCI_IV(fmin=fmin,fmax=fmax,selected_chans=target_channels,montage=montage,subjects=[1,2])



src3 = epoch_X_src3.get_data()
src2 = correct_EEG_data_order(epoch_X_src2,target_channels)
src1 = epoch_X_src1.get_data()
def create_epoch_array(data,label,channel_name,sampling_freq = 128,event_id=None):
    total_trials = len(label)
    ch_types = ['eeg'] * len(channel_name)
    info = mne.create_info(channel_name, ch_types=ch_types, sfreq=sampling_freq)
    if event_id is None:
        event_id = dict(left_hand=0, right_hand=1, feet=2, rest=3)
    events = np.column_stack((np.arange(0, sampling_freq * total_trials, sampling_freq),
                              np.zeros(total_trials, dtype=int),
                              label))

    mne_data = mne.EpochsArray(data, info, event_id=event_id, events=events, tmin=0)
    return mne_data
def plot(data,label,channel_name,event_id=None):
    print("plot")
    print_dataset_info(data[0],"default")
    epoch_array_1 = create_epoch_array(data[0]*1e-6,label[0],channel_name,event_id=event_id)
    # epoch_array_1['left_hand'][0].plot()
    epoch_array_1['right_hand'][0].plot()
    # epoch_array_1['rest'][0].plot()




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

event_id = dict(left_hand=0, right_hand=1, feet=2)
tmp_x3,tmp_lb_3,tmp_mt3 = reformat(X_src3, y_src3, m_src3)
plot(tmp_x3,tmp_lb_3,target_channels,event_id=event_id)

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


# LA = LabelAlignment(target_dataset=(X_MIA_train_data,X_MIA_train_label))
# tmp_X_src2,tmp_y_src2,_ = reformat(X_src2,y_src2,m_src2)
# update_X_src2, update_y_src2 = LA.convert_source_data_with_LA(tmp_X_src2,tmp_y_src2)
# X_MIA_train_label = np.array([relabel_target(l) for l in X_MIA_train_label])
# X_MIB_train_label = np.array([relabel_target(l) for l in X_MIB_train_label])

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


temp_X_MIA_test_data = np.split(X_MIA_test_data,2)
temp_X_MIB_test_data = np.split(X_MIB_test_data,3)
test_dataset_A = {
    'data': temp_X_MIA_test_data,
    'dataset_name': 'dataset_A'
}

test_dataset_B = {
    'data': temp_X_MIB_test_data,
    'dataset_name': 'dataset_B'
}

print_dataset_info(X_MIA_train_data,"train dataset A")
print_dataset_info(X_MIB_train_data,"train dataset B")

print_dataset_info(X_MIA_test_data,"test dataset A")
print_dataset_info(X_MIB_test_data,"test dataset B")

print_dataset_info(X_src1,"source 1 ")
print_dataset_info(X_src2,"source 2 ")
print_dataset_info(X_src3,"source 3 ")

# generate_data_file([dataset_1,dataset_2,dataset_3,target_dataset_A],folder_name='case_9_A')
# generate_data_file([dataset_1,dataset_2,dataset_3,target_dataset_B],folder_name='case_9_B')

# generate_data_file([test_dataset_A,test_dataset_B],folder_name='test_data_microvolt')
