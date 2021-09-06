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

import scipy.signal as signal
import copy
from scipy.linalg import sqrtm, inv
from collections import defaultdict


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

def print_info(source_data,dataset_name):
    print("current dataset {}".format(dataset_name))
    for subject_idx in range(len(source_data)):
        print("source_data subject_idx {} has shape : {}, with range scale ({},{}) ".format(
            subject_idx, source_data[subject_idx].shape,
            np.max(source_data[subject_idx]), np.min(source_data[subject_idx])))
def print_dataset_info(data,dataset_name="train dataset A"):
    print(dataset_name)
    # for subject_idx in range(len(data)):
    print("Train subject has shape : {}, with range scale ({},{}) ".format(
        data.shape,
        np.max(data), np.min(data)))

class LabelAlignment:
    """
    Label Alignment technique
    https://arxiv.org/pdf/1912.01166.pdf
    """
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

            category_A_m = dict()
            new_subject_data = list()
            subject_category_r_op = self.generate_class_cov(subject_data,subject_label,invert=True)
            for label in sorted(list(subject_category_r_op.keys())):
                if label not in list(self.target_r_op.keys()):
                    print("current label {} is not in target dataset ".format(label))
                    return
                source_r_op = subject_category_r_op[label]
                target_r_op = self.target_r_op[label]
                A_m = np.matmul(target_r_op, source_r_op)
                category_A_m[label] = A_m


            for trial in range(len(subject_data)):
                trial_data = subject_data[trial]
                trial_label = subject_label[trial]
                trial_A_m = category_A_m[trial_label]
                convert_trial_data = np.matmul(trial_A_m, trial_data)
                new_subject_data.append(convert_trial_data)
            new_subject_data = np.array(new_subject_data)
            new_source_data.append(new_subject_data)
        return new_source_data,source_label

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
        if np.iscomplexobj(sqrtm(r)):
            print("covariance matrix problem sqrt")

        r_op = inv(sqrtm(r))
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
def expand_data_dim(data):
    if isinstance(data, list):
        for idx in range(len(data)):
            if len(data[idx].shape) == 3:
                new_data = np.expand_dims(data[idx], axis=1)
            else:
                new_data = data[idx]
            data[idx] = new_data
        return data
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 3:
            return np.expand_dims(data, axis=1)
        else:
            return data
    else:
        raise ValueError("the data format during the process section is not correct")




def shuffle_data(subject_data,subject_label):
    available_index = np.arange(subject_data.shape[0])
    shuffle_index = np.random.permutation(available_index)
    shuffle_subject_data = subject_data[shuffle_index,]
    shuffle_subject_label = subject_label[shuffle_index,]
    return [shuffle_subject_data,shuffle_subject_label]


def modify_data(data,time=256):
    return data[:, :, :time]


def load_source_sleep(path=None):
    if path is None:
        path =  "C:/Users/wduong/mne_data/SleepSource/SleepSource"
    train_data_1 = []
    train_label_1 = []
    train_data_2 = []
    train_label_2 = []
    subject_ids = []
    for subj in range(39):
        with open(os.path.join(path, "training_s{}r1X.npy".format(subj)), 'rb') as f:
            x = pickle.load(f)
            train_data_1.append(x)
        with open(os.path.join(path, "training_s{}r1y.npy".format(subj)), 'rb') as f:
            y = pickle.load(f)
            train_label_1.append(y)
        subject_ids.extend([subj]*len(y))

    for subj in range(39):
        with open(os.path.join(path, "training_s{}r2X.npy".format(subj)), 'rb') as f:
            x = pickle.load(f)
            train_data_1.append(x)
        with open(os.path.join(path, "training_s{}r2y.npy".format(subj)), 'rb') as f:
            y = pickle.load(f)
            train_label_1.append(y)
        subject_ids.extend([subj+39]*len(y))

        # with open(os.path.join(path, "training_s{}r2X.npy".format(subj)), 'rb') as f:
        #     train_data_2.append(pickle.load(f))
        # with open(os.path.join(path, "training_s{}r2y.npy".format(subj)), 'rb') as f:
        #     train_label_2.append(pickle.load(f))

    dataset_meta = pd.DataFrame({"subject":subject_ids,"session":["session_0"]*len(subject_ids),"run":["run_0"]*len(subject_ids)})
    train_data = np.concatenate(train_data_1)
    train_label = np.concatenate(train_label_1)
    return train_data,train_label,dataset_meta

def load_target_sleep(path=None):
    if path is None:
        path =  "C:/Users/wduong/mne_data/MNE-beetlsleepleaderboard-data/sleep_target"
    target_train_data_1 = []
    target_train_label_1 = []
    subject_ids_1 = []

    target_train_data_2 = []
    target_train_label_2 = []
    subject_ids_2 = []
    for subj in range(5):
        with open(os.path.join(path, "leaderboard_s{}r1X.npy".format(subj)), 'rb') as f:
            x = pickle.load(f)
            target_train_data_1.append(x)
        with open(os.path.join(path, "leaderboard_s{}r1y.npy".format(subj)), 'rb') as f:
            y = pickle.load(f)
            target_train_label_1.append(y)

        subject_ids_1.extend([subj]*len(y))


        with open(os.path.join(path, "leaderboard_s{}r2X.npy".format(subj)), 'rb') as f:
            x = pickle.load(f)
            target_train_data_2.append(x)
        with open(os.path.join(path, "leaderboard_s{}r2y.npy".format(subj)), 'rb') as f:
            y = pickle.load(f)
            target_train_label_2.append(y)
        subject_ids_2.extend([subj+5]*len(y))

    dataset_meta_1 = pd.DataFrame({"subject":subject_ids_1,"session":["session_0"]*len(subject_ids_1),"run":["run_0"]*len(subject_ids_1)})
    target_train_data_1 = np.concatenate(target_train_data_1)
    target_train_label_1 = np.concatenate(target_train_label_1)

    dataset_meta_2 = pd.DataFrame({"subject":subject_ids_2,"session":["session_0"]*len(subject_ids_2),"run":["run_0"]*len(subject_ids_2)})
    target_train_data_2 = np.concatenate(target_train_data_2)
    target_train_label_2 = np.concatenate(target_train_label_2)

    return target_train_data_1,target_train_label_1,dataset_meta_1,target_train_data_2,target_train_label_2,dataset_meta_2

def load_test_sleep(path=None):
    if path is None:
        path = "C:/Users/wduong/mne_data/MNE-beetlsleepleaderboard-data/testing"
    target_test_data_1 = []
    target_test_label_1 = []
    subject_ids_1 = []

    target_test_data_2 = []
    target_test_label_2 = []
    subject_ids_2 = []
    start_idx = 6
    for subj in range(6,18):
        with open(os.path.join(path, "leaderboard_s{}r1X.npy".format(subj)), 'rb') as f:
            x = pickle.load(f)
            target_test_data_1.append(x)
        test_label = np.array([-1] * len(x))
        target_test_label_1.append(test_label)
        subject_ids_1.extend([start_idx]*len(x))
        start_idx+=1
    for subj in range(6,18):
        with open(os.path.join(path, "leaderboard_s{}r2X.npy".format(subj)), 'rb') as f:
            x = pickle.load(f)
            target_test_data_2.append(x)
        test_label = np.array([-1] * len(x))
        target_test_label_2.append(test_label)
        subject_ids_2.extend([start_idx] * len(x))
        start_idx += 1

    dataset_meta_1 = pd.DataFrame({"subject": subject_ids_1, "session": ["session_0"] * len(subject_ids_1),
                                   "run": ["run_0"] * len(subject_ids_1)})
    target_test_data_1 = np.concatenate(target_test_data_1)
    target_test_label_1 = np.concatenate(target_test_label_1)

    dataset_meta_2 = pd.DataFrame({"subject": subject_ids_2, "session": ["session_0"] * len(subject_ids_2),
                                   "run": ["run_0"] * len(subject_ids_2)})
    target_test_data_2 = np.concatenate(target_test_data_2)
    target_test_label_2 = np.concatenate(target_test_label_2)

    return target_test_data_1, target_test_label_1, dataset_meta_1, target_test_data_2, target_test_label_2, dataset_meta_2

def load_test_sleep_combine(path=None):

    if path is None:
        path = "C:/Users/wduong/mne_data/MNE-beetlsleepleaderboard-data/testing/"
    target_test_data = []
    target_test_label = []
    subject_ids = []
    start_idx = 0
    for subj in range(6, 18):
        for session in range(1, 3):
            with open(path + "leaderboard_s{}r{}X.npy".format(subj, session), 'rb') as f:
                x = pickle.load(f)
                target_test_data.append(x)
            test_label = np.array([-1] * len(x))
            target_test_label.append(test_label)
            subject_ids.extend([start_idx] * len(x))
            start_idx += 1
    dataset_meta = pd.DataFrame({"subject": subject_ids, "session": ["session_0"] * len(subject_ids),
                                   "run": ["run_0"] * len(subject_ids)})
    target_test_data = np.concatenate(target_test_data)
    target_test_label = np.concatenate(target_test_label)
    return target_test_data,target_test_label,dataset_meta
def generate_data_file(list_dataset_info,folder_name='case_0',file_name = 'NeurIPS_TL'):

    list_dataset = list()
    for dataset in list_dataset_info:
        list_dataset.append(dataset)
    data_file = '{}.mat'.format(file_name)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    data_file = os.path.join(folder_name,data_file)
    from scipy.io import savemat
    savemat(data_file, {'datasets':list_dataset})

class EuclideanAlignment:
    """
    convert trials of each subject to a new format with Euclidean Alignment technique
    https://arxiv.org/pdf/1808.05464.pdf
    """
    def __init__(self,list_r_op=None,subject_ids=None):
        self.list_r_op = list_r_op
        if subject_ids is not None:
            update_list_r_op = [self.list_r_op[subject_id] for subject_id in subject_ids]
            print("only use r-op for subjects {}".format(subject_ids))
            self.list_r_op = update_list_r_op
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

def load_source_data(target_channels,dataset_name="cho2017",montage=None,subject_ids=None,events=None,relabel_func=None):
    if relabel_func is None:
        print("use default relabel function")
        print("left_hand ->0 , right_hand ->1, all other -> 2")
        def relabel(l):
            if l == 'left_hand':
                return 0
            elif l == 'right_hand':
                return 1
            else:
                return 2
        relabel_func = relabel


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
        epoch_X_src, label_src, m_src = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=subject_ids)
        print("cho2017 current chans : ",epoch_X_src.ch_names)
        reorder_epoch_X_src = epoch_X_src.copy().reorder_channels(target_channels)
        print("reorder cho2017 chans : ",reorder_epoch_X_src.ch_names)

    elif dataset_name == "physionet":
        if events is None:
            events=dict(left_hand=2, right_hand=3, feet=5, rest=1)
        epoch_X_src, label_src, m_src = load_Physionet(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=subject_ids,events=events)
        print("physionet current chans : ",epoch_X_src.ch_names)
        reorder_epoch_X_src = epoch_X_src.copy().reorder_channels(target_channels)
        print("reorder physionet chans : ",reorder_epoch_X_src.ch_names)
        print("total chans : ",len(epoch_X_src.ch_names))
    elif dataset_name == "BCI_IV":
        epoch_X_src, label_src, m_src = load_BCI_IV(fmin=fmin, fmax=fmax, selected_chans=target_channels,montage=montage,subjects=subject_ids)

        print("BCI_IV current chans : ",epoch_X_src.ch_names)
        reorder_epoch_X_src = epoch_X_src.copy().reorder_channels(target_channels)
        print("reorder BCI_IV chans : ",reorder_epoch_X_src.ch_names)

    src = reorder_epoch_X_src.get_data()
    X_src = modify_data(src, time=max_time_length)
    X_src = convert_volt_to_micro(X_src)
    y_src = np.array([relabel_func(l) for l in label_src])
    return X_src,y_src,m_src
def load_target_data(target_channels,dataset_name="dataset_B"):
    if dataset_name == "dataset_A":
        X_train_data,X_train_label,train_meta = load_dataset_A(train=True,selected_chans=target_channels)
        X_test_data,X_test_label,test_meta = load_dataset_A(train=False, norm=False, selected_chans=target_channels)
    else:
        X_train_data,X_train_label,train_meta = load_dataset_B(train=True,selected_chans=target_channels)
        X_test_data,X_test_label,test_meta = load_dataset_B(train=False, norm=False, selected_chans=target_channels)

    X_train_data = convert_volt_to_micro(X_train_data)
    X_test_data = convert_volt_to_micro(X_test_data)

    return X_train_data,X_train_label,train_meta,X_test_data,X_test_label,test_meta

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

def reformat(data,label,meta_data):
    """
    assume the meta_data['subject'] is a lsit of order ids. EX: 1,1,1,2,2,3,3,3,3,6,6,6
    convert data from (total_trials,channels,samples) -> (subjects,trials,channels,samples)
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
        if len(subject_meta_data) > 0:
            trials = len(subject_meta_data)
            end = start+trials
            subject_data = data[start:end]
            subject_label = label[start:end]
            new_data.append(subject_data)
            new_label.append(subject_label)
            new_meta_data.append(subject_meta_data)
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
