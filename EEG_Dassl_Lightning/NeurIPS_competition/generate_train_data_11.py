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
    generate_data_file,print_dataset_info,get_dataset_A_ch,get_dataset_B_ch,shuffle_data
)
from scipy.linalg import sqrtm, inv
from scipy import signal
from collections import defaultdict

def print_info(source_data,dataset_name):
    print("current dataset {}".format(dataset_name))
    for subject_idx in range(len(source_data)):
        print("source_data subject_idx {} has shape : {}, with range scale ({},{}) ".format(
            subject_idx, source_data[subject_idx].shape,
            np.max(source_data[subject_idx]), np.min(source_data[subject_idx])))

def reduce_category_data(subject_data,subject_label,reduce_label=3):
    category_data = defaultdict(list)
    new_subject_data = list()
    new_subject_label = list()
    for data, label in zip(subject_data, subject_label):
        # print("current label : ",label)
        category_data[label].append(data)
    reduce_label_size = len(category_data[reduce_label])
    other_label_size = len(category_data[0])
    if reduce_label_size>other_label_size:
        reduce_data = np.array(category_data[reduce_label])
        reduce_data = reduce_data[:other_label_size]
        category_data[reduce_label] = reduce_data
    for label,data in category_data.items():
        data = np.array(data)
        label = np.array([label]*len(data))
        # print("label {} has data shape {} ".format(label,data.shape))
        new_subject_data.append(data)
        new_subject_label.append(label)
    new_subject_data = np.concatenate(new_subject_data)
    new_subject_label = np.concatenate(new_subject_label)

    new_subject_data,new_subject_label = shuffle_data(new_subject_data,new_subject_label)

    return new_subject_data,new_subject_label

def reduce_dataset(data,label,meta_data):

    update_data = list()
    update_label = list()
    update_ids = list()
    for subject in range(len(data)):
        subject_data = data[subject]
        subject_label = label[subject]
        subject_meta_data = meta_data[subject]
        subject_data,subject_label = reduce_category_data(subject_data,subject_label)
        # print("subject data after reduce : ",subject_data.shape)
        subject_id = [subject+1]*len(subject_data)

        update_data.append(subject_data)
        update_label.append(subject_label)
        update_ids.extend(subject_id)
    update_data=np.concatenate(update_data)
    update_label=np.concatenate(update_label)
    dataset_meta = pd.DataFrame({"subject":update_ids,"session":["session_0"]*len(update_ids),"run":["run_0"]*len(update_ids)})
    # print("update data shape : ",update_data.shape)
    # print("update label shape : ",update_label.shape)
    return update_data,update_label,dataset_meta

class LabelAlignment:
    def __init__(self,target_dataset):
        """
        assume target_data is (trials,channels,samples)
        target_label is (trials)
        """
        self.target_data,self.target_label = target_dataset
        self.target_r_op = self.generate_class_cov(self.target_data,self.target_label,invert=False)
        # for k,v in self.target_r_op.items():
        #     print("target label {} has r_op : {}".format(k,v))

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
            # print(" subject {} is converted : ".format(subject))

            category_A_m = dict()
            new_subject_data = list()
            subject_category_r_op = self.generate_class_cov(subject_data,subject_label,invert=True)
            for label in sorted(list(subject_category_r_op.keys())):
                if label not in list(self.target_r_op.keys()):
                    print("current label {} is not in target dataset ".format(label))
                    return
                source_r_op = subject_category_r_op[label]
                target_r_op = self.target_r_op[label]
                # print("target label {}".format(label))
                # print("source label {}".format(label))
                # print("target r op shape : ",target_r_op.shape)
                # print("source r op shape : ",source_r_op.shape)
                A_m = np.matmul(target_r_op, source_r_op)
                category_A_m[label] = A_m
                # for k, v in self.target_r_op.items():

                # print("label {} has A_m : {}".format(label, A_m))


            for trial in range(len(subject_data)):
                trial_data = subject_data[trial]
                # print("trials {} with max {}".format(trial,len(subject_data)))
                # print("subject label : ",subject_label.shape)
                trial_label = subject_label[trial]
                trial_A_m = category_A_m[trial_label]
                convert_trial_data = np.matmul(trial_A_m, trial_data)
                new_subject_data.append(convert_trial_data)
            new_subject_data = np.array(new_subject_data)
            new_source_data.append(new_subject_data)
        # new_source_data = np.concatenate(new_source_data)
        return new_source_data,source_label
        # return source_data,source_label

    def generate_class_cov(self,target_data,target_label,invert=True):
        """
        Use the target data to generate an inverse Covariance for each class category.
        Args:
            target_data: (trials,channels,samples)
            target_label: (trials)

        Returns:

        """
        category_data = defaultdict(list)
        category_r_op = dict()
        for data,label in zip(target_data,target_label):
            # print("current label : ",label)
            category_data[label].append(data)
        for label,data in category_data.items():
            data= np.array(data)
            # print("data shape : ",data.shape)
            if invert:
                # print("calculate inv sqrt cov")
                r_op = self.calculate_inv_sqrt_cov(data)
            else:
                # print("calculate sqrt cov")
                r_op = self.calcualte_sqrt_cov(data)

            category_r_op[label] = r_op
        return category_r_op

    def calculate_inv_sqrt_cov(self,data):
        assert len(data.shape) == 3
        r = np.matmul(data, data.transpose((0, 2, 1))).mean(0)
        # print("origin cov : ", r)
        if np.iscomplexobj(r):
            print("covariance matrix problem")
        # print("sqrt cov : ", sqrtm(r))
        if np.iscomplexobj(sqrtm(r)):
            print("covariance matrix problem sqrt")
            # print("sqrt cov : ",sqrtm(r))

        r_op = inv(sqrtm(r))
        # print("r_op shape : ", r_op.shape)
        # print("data shape : ",x.shape)
        # print("r_op : ", r_op)
        if np.iscomplexobj(r_op):
            print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
                  "not, check data!!")
            # print("r op : ",r_op)
            r_op = np.real(r_op).astype(np.float32)
        elif not np.any(np.isfinite(r_op)):
            print("WARNING! Not finite values in R Matrix")
        return r_op

    def calcualte_sqrt_cov(self,data):
        assert len(data.shape) == 3
        r = np.matmul(data, data.transpose((0, 2, 1))).mean(0)
        if np.iscomplexobj(r):
            print("covariance matrix problem")
        if np.iscomplexobj(sqrtm(r)):
            print("covariance matrix problem sqrt")

        r_op = sqrtm(r)
        return r_op

def reformat(data,label,meta_data):
    """
    assume the meta_data['subject'] is a lsit of order ids. EX: 1,1,1,2,2,3,3,3,3,6,6,6
    Args:
        data:
        label:
        meta_data:

    Returns:

    """
    n_subjects = len(np.unique(meta_data['subject']))
    new_data = []
    new_label = []
    new_meta_data = []
    start=0
    unique_subject_ids = np.unique(meta_data['subject'])
    for i in range(n_subjects):
        current_subject = unique_subject_ids[i]
        subject_meta_data = meta_data[meta_data['subject']==current_subject]
        # print("current subject : ",current_subject)
        # print("current meta dta : ",subject_meta_data)
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
def combine(subjects_data,subjects_label,subjects_meta_data):
    tmp_data = np.concatenate(subjects_data)
    tmp_label = np.concatenate(subjects_label)
    tmp_meta_data = pd.concat(subjects_meta_data).reset_index()
    return tmp_data,tmp_label,tmp_meta_data
def convert_volt_to_micro(data):
    data = data * 1e6
    return data

def generate_common_target_chans(target_chan,source_chans):
    target_channels = target_chan
    for source_chan in source_chans:
        target_channels = generate_common_chan_test_data(target_channels, source_chan)
    return target_channels


def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    elif l == 'feet': return 2
    else: return 3
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
dataset_A_channels = get_dataset_A_ch()
dataset_B_channels = get_dataset_B_ch()
X_src1,_,_ = load_Cho2017(subjects=[1])
X_src2, _, _ = load_Physionet(subjects=[1])
X_src3, _, _ = load_BCI_IV(subjects=[1])

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# X_src1.get_montage().plot(show_names=True,show=False)
# plt.savefig('plots/ch2017_montage.png')
# plt.show()
# X_src2.get_montage().plot(show_names=True,show=False)
# plt.savefig('plots/physionet_montage.png')
# plt.show()
# X_src3.get_montage().plot(show_names=True,show=False)
# plt.savefig('plots/BCI_IV_montage.png')
# plt.show()
Cho2017_channels = X_src1.ch_names
Physionet_channels = X_src2.ch_names
BCI_IV_channels = X_src3.ch_names
# print("Cho2017 chans : ",Cho2017_channels)
# print("Cho2017 chans : ",Physionet_channels)
montage = X_src1.get_montage()
# montage = X_src2.get_montage()

common_channel_A_B = generate_common_target_chans(target_chan=BCI_IV_channels,source_chans=[dataset_A_channels,dataset_B_channels,Cho2017_channels,Physionet_channels])
montage = None
# common_channel_A_B = generate_common_target_chans(target_chan=dataset_A_channels,source_chans=[dataset_B_channels,Cho2017_channels,Physionet_channels])
print("common chan A_B size : ",len(common_channel_A_B))
print("common chan A_B : ",common_channel_A_B)

# common_channel_A_B = generate_common_target_chans(target_chan=Cho2017_channels,source_chans=[dataset_A_channels,dataset_B_channels,Physionet_channels])
# common_channel_A_B = generate_common_target_chans(target_chan=Physionet_channels,source_chans=[dataset_A_channels,dataset_B_channels,Cho2017_channels])

print("common chan A_B size : ",len(common_channel_A_B))
print("common chan A_B : ",common_channel_A_B)
# target_channels_A = generate_common_target_chans(target_chan=dataset_A_channels,source_chans=[Cho2017_channels,Physionet_channels,BCI_IV_channels])
# target_channels_B = generate_common_target_chans(target_chan=dataset_B_channels,source_chans=[Cho2017_channels,Physionet_channels,BCI_IV_channels])


def load_source_data(target_channels,dataset_name="cho2017",montage=None):
    print("common target chans : ",target_channels)
    print("target size : ",len(target_channels))
    fmin=4
    fmax=36
    tmax=3
    tmin=0
    sfreq=128
    max_time_length = int((tmax - tmin) * sfreq)
    if dataset_name == "cho2017":
        # epoch_X_src, label_src, m_src = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels)
        epoch_X_src, label_src, m_src = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1,2,3])
        print("cho2017 current chans : ",epoch_X_src.ch_names)
        reorder_epoch_X_src = epoch_X_src.copy().reorder_channels(target_channels)
        print("reorder cho2017 chans : ",reorder_epoch_X_src.ch_names)
        # print("total chans : ",len(epoch_X_src.ch_names))

    elif dataset_name == "physionet":
        events=dict(left_hand=2, right_hand=3, feet=5, rest=1)
        # epoch_X_src, label_src, m_src= load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels,events=events)
        epoch_X_src, label_src, m_src = load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1,2,3],events=events)
        print("physionet current chans : ",epoch_X_src.ch_names)
        reorder_epoch_X_src = epoch_X_src.copy().reorder_channels(target_channels)
        print("reorder physionet chans : ",reorder_epoch_X_src.ch_names)
        print("total chans : ",len(epoch_X_src.ch_names))
        # src2 = correct_EEG_data_order(epoch_X_src2, target_channels)
        # X_src2 = modify_data(src2, time=max_time_length)
        # X_src2 = convert_volt_to_micro(X_src2)
    elif dataset_name == "BCI_IV":
        # events = dict(left_hand=1, right_hand=2,feet=3)
        epoch_X_src, label_src, m_src = load_BCI_IV(fmin=fmin, fmax=fmax, selected_chans=target_channels,
                                                       montage=montage, subjects=[1, 2])
        print("BCI_IV current chans : ",epoch_X_src.ch_names)
        reorder_epoch_X_src = epoch_X_src.copy().reorder_channels(target_channels)
        print("reorder BCI_IV chans : ",reorder_epoch_X_src.ch_names)
        # epoch_X_src, label_src, m_src = load_BCI_IV(fmin=fmin, fmax=fmax, selected_chans=target_channels,montage=montage,events=events)
    # src1 = reorder_epoch_X_src.get
    # src1 = correct_EEG_data_order(epoch_X_src, target_channels)
    # print("src : ",src1[0,0,:5])
    # print("reorder : ",reorder_epoch_X_src.get_data()[0,0,:5])

    src = reorder_epoch_X_src.get_data()
    X_src = modify_data(src, time=max_time_length)
    X_src = convert_volt_to_micro(X_src)
    y_src = np.array([relabel(l) for l in label_src])
    return X_src,y_src,m_src
def load_target_data(target_channels,dataset_name="dataset_B"):
    if dataset_name == "dataset_A":
        X_train_data,X_train_label,m_tgt = load_dataset_A(train=True,selected_chans=target_channels)
        X_test_data = load_dataset_A(train=False, norm=False, selected_chans=target_channels)
    else:
        X_train_data,X_train_label,m_tgt = load_dataset_B(train=True,selected_chans=target_channels)
        X_test_data = load_dataset_B(train=False, norm=False, selected_chans=target_channels)

    X_train_data = convert_volt_to_micro(X_train_data)
    X_test_data = convert_volt_to_micro(X_test_data)

    return X_train_data,X_train_label,m_tgt,X_test_data

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

import matplotlib.pyplot as plt
def plot(subject_data,subject_label,channel_name,event_id=None,trials_select=None,specific_category=None,save_folder="default",subject_id=-1,plot_multi_trials=False):
    print("plot")
    epoch_array_1 = create_epoch_array(subject_data*1e-6,subject_label,channel_name,event_id=event_id)
    if specific_category is None:
        specific_category = 'left_hand'
        if trials_select is None:
            trials_select = [0]
    save_folder = os.path.join("plots",save_folder,specific_category)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    if plot_multi_trials:
        save_eeg_plot = os.path.join(save_folder, "{}_{}_multi_trials.png".format(subject_id, specific_category))
        epoch_array_1[specific_category][trials_select].plot(show=False).savefig(save_eeg_plot)
    else:
        for trial in trials_select:
            save_eeg_plot = os.path.join(save_folder,"{}_{}_{}.png".format(subject_id,specific_category,trial))
            epoch_array_1[specific_category][trial].plot(show=False).savefig(save_eeg_plot)


def plot_subject_data(dataset,channel_name,event_id=None,trials_select=None,specific_category=None,save_folder="default",multi_trials=False):
    if len(dataset) == 3:
        data,label,meta = dataset
        data,label,meta = reformat(data,label,meta)
    else:
        data = dataset
        label = [np.zeros(len(data[subject])) for subject in range(len(data))]
    for subject_id in range(len(data)):
        subject_data = data[subject_id]
        subject_label = label[subject_id]
        print("subject data shape : ",subject_data.shape)
        plot(subject_data,subject_label,channel_name,event_id=event_id,trials_select=trials_select,specific_category=specific_category,save_folder=save_folder,subject_id=subject_id,plot_multi_trials=multi_trials)


X_src1,y_src1,m_src1 = load_source_data(target_channels=common_channel_A_B,dataset_name="cho2017")
X_src2,y_src2,m_src2 = load_source_data(target_channels=common_channel_A_B,dataset_name="physionet")
X_src3,y_src3,m_src3 = load_source_data(target_channels=common_channel_A_B,dataset_name="BCI_IV",montage=montage)

print("before update meta data : ",m_src2)
X_src2,y_src2,m_src2 = reformat(X_src2,y_src2,m_src2)
X_src2,y_src2,m_src2 = reduce_dataset(X_src2,y_src2,m_src2)

print("update meta data : ",m_src2)

# X_MIA_train_data,X_MIA_train_label,m_tgt_A,X_MIA_test_data = load_target_data(target_channels=target_channels_A,dataset_name="dataset_A")
X_MIA_train_data,X_MIA_train_label,m_tgt_A,X_MIA_test_data = load_target_data(target_channels=common_channel_A_B,dataset_name="dataset_A")

print("dataset a meta data : ",m_tgt_A)

#conduct label alignment
tmp_X_src1,tmp_y_src1, tmp_m_src1 = reformat(X_src1,y_src1,m_src1)
tmp_X_src2,tmp_y_src2,tmp_m_src2 = reformat(X_src2,y_src2,m_src2)
tmp_X_src3,tmp_y_src3,tmp_m_src3 = reformat(X_src3,y_src3,m_src3)

LA_A = LabelAlignment(target_dataset=(X_MIA_train_data,X_MIA_train_label))

update_X_src1, update_y_src1 = LA_A.convert_source_data_with_LA(tmp_X_src1,tmp_y_src1)
update_X_src2, update_y_src2 = LA_A.convert_source_data_with_LA(tmp_X_src2,tmp_y_src2)
update_X_src3, update_y_src3 = LA_A.convert_source_data_with_LA(tmp_X_src3,tmp_y_src3)

print_info(update_X_src1,dataset_name="Cho2017")
print_info(update_X_src2,dataset_name="Physionet")
print_info(update_X_src3,dataset_name="BCI_IV")

LA_A_X_src1,LA_A_y_src1,LA_A_m_src1 = combine(update_X_src1, update_y_src1,tmp_m_src1)
LA_A_X_src2,LA_A_y_src2,LA_A_m_src2 = combine(update_X_src2, update_y_src2,tmp_m_src2)
LA_A_X_src3,LA_A_y_src3,LA_A_m_src3 = combine(update_X_src3, update_y_src3,tmp_m_src3)


m_src1 = {name: col.values for name, col in m_src1.items()}
m_src2 = {name: col.values for name, col in m_src2.items()}
m_src3 = {name: col.values for name, col in m_src3.items()}


# LA_A_m_src1 = {name: col.values for name, col in LA_A_m_src1.items()}
# LA_A_m_src2 = {name: col.values for name, col in LA_A_m_src2.items()}
# LA_A_m_src3 = {name: col.values for name, col in LA_A_m_src3.items()}
#
# m_tgt_A = {name: col.values for name, col in m_tgt_A.items()}


# dataset_1 = {
#     'data': X_src1,
#     'label': y_src1,
#     'meta_data': m_src1,
#     'dataset_name': 'cho2017'
# }
#
# dataset_2 = {
#     'data': X_src2,
#     'label': y_src2,
#     'meta_data': m_src2,
#     'dataset_name': 'physionet'
# }
#
# dataset_3 = {
#     'data': X_src3,
#     'label': y_src3,
#     'meta_data': m_src3,
#     'dataset_name': 'BCI_IV'
# }
#
# LA_dataset_1 = {
#     'data': LA_X_src1,
#     'label': LA_y_src1,
#     'meta_data': LA_m_src1,
#     'dataset_name': 'cho2017'
# }
#
# LA_dataset_2 = {
#     'data': LA_X_src2,
#     'label': LA_y_src2,
#     'meta_data': LA_m_src2,
#     'dataset_name': 'physionet'
# }
#
# LA_dataset_3 = {
#     'data': LA_X_src3,
#     'label': LA_y_src3,
#     'meta_data': LA_m_src3,
#     'dataset_name': 'BCI_IV'
# }
#
# target_dataset_A = {
#     'data': X_MIA_train_data,
#     'label': X_MIA_train_label,
#     'meta_data': m_tgt_A,
#     'dataset_name': 'dataset_A'
# }
#
# temp_X_MIA_test_data = np.split(X_MIA_test_data,2)
# test_dataset_A = {
#     'data': temp_X_MIA_test_data,
#     'dataset_name': 'dataset_A'
# }
#
# print_dataset_info(X_MIA_train_data,"train dataset A")
# # print_dataset_info(X_MIA_test_data,"test dataset A")
# # print_dataset_info(X_src1,"source 1 ")
# # print_dataset_info(X_src2,"source 2 ")
# #
# # print_dataset_info(np.concatenate(update_X_src1),"update source 1 ")
# # print_dataset_info(np.concatenate(update_X_src2),"update source 2 ")
#
# generate_data_file([target_dataset_A],folder_name='case_15_A',file_name = 'NeurIPS_TL')
# generate_data_file([dataset_1],folder_name='case_15_A',file_name = 'dataset_1')
# generate_data_file([dataset_2],folder_name='case_15_A',file_name = 'dataset_2')
# generate_data_file([dataset_3],folder_name='case_15_A',file_name = 'dataset_3')
#
#
# generate_data_file([LA_dataset_1],folder_name='case_15_A/LA',file_name = 'dataset_1')
# generate_data_file([LA_dataset_2],folder_name='case_15_A/LA',file_name = 'dataset_2')
# generate_data_file([LA_dataset_3],folder_name='case_15_A/LA',file_name = 'dataset_3')
#
#
#
# #set up dataset B
X_src1,y_src1,m_src1 = load_source_data(target_channels=common_channel_A_B,dataset_name="cho2017")
X_src2,y_src2,m_src2 = load_source_data(target_channels=common_channel_A_B,dataset_name="physionet")
X_src3,y_src3,m_src3 = load_source_data(target_channels=common_channel_A_B,dataset_name="BCI_IV",montage=montage)



X_MIB_train_data,X_MIB_train_label,m_tgt_B,X_MIB_test_data = load_target_data(target_channels=common_channel_A_B,dataset_name="dataset_B")
#
print("before update meta data : ",m_src2)
X_src2,y_src2,m_src2 = reformat(X_src2,y_src2,m_src2)
X_src2,y_src2,m_src2 = reduce_dataset(X_src2,y_src2,m_src2)

print("update meta data : ",m_src2)


tmp_X_src1,tmp_y_src1,tmp_m_src1 = reformat(X_src1,y_src1,m_src1)
tmp_X_src2,tmp_y_src2,tmp_m_src2 = reformat(X_src2,y_src2,m_src2)
tmp_X_src3,tmp_y_src3,tmp_m_src3 = reformat(X_src3,y_src3,m_src3)

LA_B = LabelAlignment(target_dataset=(X_MIB_train_data,X_MIB_train_label))

print("LA convert Cho2017")
update_X_src1, update_y_src1 = LA_B.convert_source_data_with_LA(tmp_X_src1,tmp_y_src1)
print("LA convert Physionet")
update_X_src2, update_y_src2 = LA_B.convert_source_data_with_LA(tmp_X_src2,tmp_y_src2)
print("LA convert BCI_IV")
update_X_src3, update_y_src3 = LA_B.convert_source_data_with_LA(tmp_X_src3,tmp_y_src3)

train_B = np.split(X_MIB_train_data,3)


print_info(update_X_src1,dataset_name="Cho2017")
print_info(update_X_src2,dataset_name="Physionet")
print_info(update_X_src3,dataset_name="BCI_IV")

print_info(train_B,dataset_name="dataset B")

LA_B_X_src1,LA_B_y_src1,LA_B_m_src1 = combine(update_X_src1, update_y_src1,tmp_m_src1)
LA_B_X_src2,LA_B_y_src2,LA_B_m_src2 = combine(update_X_src2, update_y_src2,tmp_m_src2)
LA_B_X_src3,LA_B_y_src3,LA_B_m_src3 = combine(update_X_src3, update_y_src3,tmp_m_src3)


def plot_figures(folder="",multi_trials_plot=False):
    ###deal with cho2017 dataset
    event_id = dict(left_hand=0, right_hand=1)
    categories = ["left_hand", "right_hand"]
    dataset = "cho2017"
    save_folder = os.path.join(folder,dataset)
    for category in categories:
        plot_subject_data((X_src1, y_src1, m_src1), channel_name=common_channel_A_B, event_id=event_id,
                          trials_select=[0, 1, 2], specific_category=category, save_folder=save_folder,multi_trials=multi_trials_plot)

    event_id = dict(left_hand=0, right_hand=1)
    categories = ["left_hand", "right_hand"]
    dataset = "LA_B_cho2017"
    save_folder = os.path.join(folder,dataset)
    for category in categories:
        # specific_category="left_hand"
        plot_subject_data((LA_B_X_src1, LA_B_y_src1, LA_B_m_src1), channel_name=common_channel_A_B, event_id=event_id,
                          trials_select=[0, 1, 2], specific_category=category, save_folder=save_folder,multi_trials=multi_trials_plot)


    ###deal with Physionet dataset
    event_id = dict(left_hand=0, right_hand=1, feet=2, rest=3)
    # event_id=dict(left_hand=0,right_hand=1)
    categories = ["left_hand", "right_hand", "feet", "rest"]
    dataset = "physionet"
    save_folder = os.path.join(folder,dataset)
    for category in categories:
        plot_subject_data((X_src2, y_src2, m_src2), channel_name=common_channel_A_B, event_id=event_id,
                          trials_select=[0, 1, 2], specific_category=category, save_folder=save_folder,multi_trials=multi_trials_plot)

    event_id = dict(left_hand=0, right_hand=1, feet=2, rest=3)
    # event_id=dict(left_hand=0,right_hand=1)
    categories = ["left_hand", "right_hand", "feet", "rest"]
    dataset = "LA_B_physionet"
    save_folder = os.path.join(folder,dataset)

    for category in categories:
        # specific_category="left_hand"
        plot_subject_data((LA_B_X_src2, LA_B_y_src2, LA_B_m_src2), channel_name=common_channel_A_B, event_id=event_id,
                          trials_select=[0, 1, 2], specific_category=category, save_folder=save_folder,multi_trials=multi_trials_plot)

    ###deal with BCI_IV dataset
    event_id=dict(left_hand=0,right_hand=1,feet=2,tongue=3)
    # event_id=dict(left_hand=0,right_hand=1)
    categories = ["left_hand","right_hand","feet","tongue"]
    dataset = "BCI_IV"
    save_folder = os.path.join(folder,dataset)

    for category in categories:
        plot_subject_data((X_src3,y_src3,m_src3),channel_name=common_channel_A_B,event_id=event_id,trials_select=[1,2,3],specific_category=category,save_folder=save_folder,multi_trials=multi_trials_plot)

    event_id=dict(left_hand=0,right_hand=1,feet=2,rest=3)
    # event_id=dict(left_hand=0,right_hand=1)
    categories = ["left_hand","right_hand","feet","rest"]
    dataset = "LA_B_BCI_IV"
    save_folder = os.path.join(folder,dataset)
    for category in categories:
    # specific_category="left_hand"
        plot_subject_data((LA_B_X_src3,LA_B_y_src3,LA_B_m_src3),channel_name=common_channel_A_B,event_id=event_id,trials_select=[1,2,3],specific_category=category,save_folder=save_folder,multi_trials=multi_trials_plot)

    ###deal with dataset_B
    event_id = dict(left_hand=0, right_hand=1, feet=2, rest=3)
    # event_id=dict(left_hand=0,right_hand=1)
    categories = ["left_hand", "right_hand", "feet", "rest"]
    dataset = "dataset_B"
    save_folder = os.path.join(folder,dataset)
    for category in categories:
        plot_subject_data((X_MIB_train_data,X_MIB_train_label,m_tgt_B), channel_name=common_channel_A_B, event_id=event_id,
                          trials_select=[0,1,2], specific_category=category, save_folder=save_folder,multi_trials=multi_trials_plot)

    ###deal with cho2017 dataset LA_A

    event_id = dict(left_hand=0, right_hand=1)
    categories = ["left_hand", "right_hand"]
    dataset = "LA_A_cho2017"
    save_folder = os.path.join(folder, dataset)
    for category in categories:
        # specific_category="left_hand"
        plot_subject_data((LA_A_X_src1, LA_A_y_src1, LA_A_m_src1), channel_name=common_channel_A_B, event_id=event_id,
                          trials_select=[0, 1, 2], specific_category=category, save_folder=save_folder,multi_trials=multi_trials_plot)

    ###deal with Physionet dataset LA_A
    event_id = dict(left_hand=0, right_hand=1, feet=2, rest=3)
    categories = ["left_hand", "right_hand", "feet", "rest"]
    dataset = "LA_A_physionet"
    save_folder = os.path.join(folder,dataset)

    for category in categories:
        # specific_category="left_hand"
        plot_subject_data((LA_A_X_src2, LA_A_y_src2, LA_A_m_src2), channel_name=common_channel_A_B, event_id=event_id,
                          trials_select=[0, 1, 2], specific_category=category, save_folder=save_folder,multi_trials=multi_trials_plot)

    ###deal with BCI_IV dataset LA_A
    event_id=dict(left_hand=0,right_hand=1,feet=2,rest=3)
    categories = ["left_hand","right_hand","feet","rest"]
    dataset = "LA_A_BCI_IV"
    save_folder = os.path.join(folder,dataset)

    for category in categories:
    # specific_category="left_hand"
        plot_subject_data((LA_A_X_src3,LA_A_y_src3,LA_A_m_src3),channel_name=common_channel_A_B,event_id=event_id,trials_select=[1,2,3],specific_category=category,save_folder=save_folder,multi_trials=multi_trials_plot)


    ###deal with dataset_A
    event_id = dict(left_hand=0, right_hand=1, feet=2, rest=3)
    # event_id=dict(left_hand=0,right_hand=1)
    categories = ["left_hand", "right_hand", "feet", "rest"]
    dataset = "dataset_A"
    save_folder = os.path.join(folder,dataset)
    for category in categories:
        plot_subject_data((X_MIA_train_data,X_MIA_train_label,m_tgt_A), channel_name=common_channel_A_B, event_id=event_id,
                          trials_select=[0, 1], specific_category=category, save_folder=save_folder,multi_trials=multi_trials_plot)

# plot_figures(folder="31_common_chans",multi_trials_plot=True)

plot_figures(folder="17_common_chans",multi_trials_plot=False)
plot_figures(folder="17_common_chans",multi_trials_plot=True)


m_src1 = {name: col.values for name, col in m_src1.items()}
m_src2 = {name: col.values for name, col in m_src2.items()}
m_src3 = {name: col.values for name, col in m_src3.items()}


LA_m_src1 = {name: col.values for name, col in LA_m_src1.items()}
LA_m_src2 = {name: col.values for name, col in LA_m_src2.items()}
LA_m_src3 = {name: col.values for name, col in LA_m_src3.items()}

m_tgt_B = {name: col.values for name, col in m_tgt_B.items()}


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

LA_dataset_1 = {
    'data': LA_B_X_src1,
    'label': LA_B_y_src1,
    'meta_data': LA_B_m_src1,
    'dataset_name': 'cho2017'
}

LA_dataset_2 = {
    'data': LA_B_X_src2,
    'label': LA_B_y_src2,
    'meta_data': LA_B_m_src2,
    'dataset_name': 'physionet'
}

LA_dataset_3 = {
    'data': LA_B_X_src3,
    'label': LA_B_y_src3,
    'meta_data': LA_B_m_src3,
    'dataset_name': 'BCI_IV'
}

target_dataset_B = {
    'data': X_MIB_train_data,
    'label': X_MIB_train_label,
    'meta_data': m_tgt_B,
    'dataset_name': 'dataset_B'
}

temp_X_MIB_test_data = np.split(X_MIB_test_data,3)
test_dataset_B = {
    'data': temp_X_MIB_test_data,
    'dataset_name': 'dataset_B'
}

print_dataset_info(X_MIB_train_data,"train dataset B")

print_dataset_info(X_MIB_test_data,"test dataset B")

print_dataset_info(X_src1,"source 1 ")
print_dataset_info(X_src2,"source 2 ")
print_dataset_info(X_src3,"source 3 ")

# #
# generate_data_file([target_dataset_B],folder_name='case_15_B',file_name = 'NeurIPS_TL')
# generate_data_file([dataset_1],folder_name='case_15_B',file_name = 'dataset_1')
# generate_data_file([dataset_2],folder_name='case_15_B',file_name = 'dataset_2')
# generate_data_file([dataset_3],folder_name='case_15_B',file_name = 'dataset_3')
#
#
# generate_data_file([LA_dataset_1],folder_name='case_15_B/LA',file_name = 'dataset_1')
# generate_data_file([LA_dataset_2],folder_name='case_15_B/LA',file_name = 'dataset_2')
# generate_data_file([LA_dataset_3],folder_name='case_15_B/LA',file_name = 'dataset_3')

# generate_data_file([test_dataset_A,test_dataset_B],folder_name='test_case_15_microvolt')

