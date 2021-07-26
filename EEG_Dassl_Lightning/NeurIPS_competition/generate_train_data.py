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


def shuffle_data(subject_data,subject_label):
    available_index = np.arange(subject_data.shape[0])
    shuffle_index = np.random.permutation(available_index)
    shuffle_subject_data = subject_data[shuffle_index,]
    shuffle_subject_label = subject_label[shuffle_index,]
    return [shuffle_subject_data,shuffle_subject_label]


def modify_data(data,time=256):
    return data[:, :, :time]

ds_src1 = Cho2017()
ds_src2 = PhysionetMI()
ds_src3 = BNCI2014001()
#
fmin, fmax = 4, 36

# #process Cho2017 dataset
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

# process physionet dataset
raw_src_2 = ds_src2.get_data(subjects=[1])[1]['session_0']['run_4']
raw_src_2_channels = raw_src_2.pick_types(eeg=True).ch_names
sfreq = 128
tmin=0
tmax=3
max_time_length = int((tmax-tmin)*sfreq)
# src_2_prgm =MotorImagery(events=dict(left_hand=2, right_hand=3, feet=5),n_classes=3, channels=raw_src_2_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
src_2_prgm =MotorImagery(events=dict(left_hand=2, right_hand=3, feet=5),n_classes=3, channels=raw_src_2_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
# src_2_prgm =MotorImagery(events=dict(rest=1,left_hand=2, right_hand=3, feet=5),n_classes=4, channels=raw_src_2_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)

X_src2, label_src2, m_src2 = src_2_prgm.get_data(dataset=ds_src2)
# X_src2, label_src2, m_src2 = src_2_prgm.get_data(dataset=ds_src2,subjects=[1, 2, 3, 4])
# subjects=[1, 2, 3,4]
X_src2 = modify_data(X_src2,time=max_time_length)

# #process BCI_IV dataset
raw = ds_src3.get_data(subjects=[1])[1]['session_T']['run_1']
raw_src_3_channels = raw.pick_types(eeg=True).ch_names
sfreq = 128
# prgm_2classes = MotorImagery(n_classes=2, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax)
tmin=0.5
tmax=3.5
max_time_length = int((tmax-tmin)*sfreq)
src_3_prgm = MotorImagery(n_classes=4, channels=raw_src_3_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
X_src3, label_src3, m_src3 = src_3_prgm.get_data(dataset=ds_src3)
# X_src3, label_src3, m_src3 = src_3_prgm.get_data(dataset=ds_src3,subjects=[1, 2])
X_src3 = modify_data(X_src3,time=max_time_length)

def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    else: return 2

y_src1 = np.array([relabel(l) for l in label_src1])
y_src2 = np.array([relabel(l) for l in label_src2])
y_src3 = np.array([relabel(l) for l in label_src3])

def process_target_data(epoch_data,f_min=4,f_max=36,resample=128,t_min=0,t_max=3):
    epoch_f = epoch_data.copy().filter(
        f_min, f_max, method="iir")
    # if bmin < tmin or bmax > tmax:
    epoch_f=     epoch_f.crop(tmin=t_min, tmax=t_max)
    # if self.resample is not None:
    epoch_f = epoch_f.resample(resample)
    return epoch_f

def relabel_target(l):
    if l == 0: return 0
    elif l == 1: return 1
    else: return 2

# process Dataset A (S1, S2)
# get train target data
path = '/Users/wduong/mne_data/MNE-beetlmileaderboard-data/'
import matplotlib.pyplot as plt
sfreq = 128
tmin=0
tmax=3
max_time_length = int((tmax-tmin)*sfreq)
X_MIA_train_data = []
X_MIA_train_label = []
subject_ids = []
for subj in range(1, 3):
    savebase = os.path.join(path, "S{}".format(subj), "training")
    subject_train_data = []
    subject_train_label = []
    for i in range(1, 6):
        with open(os.path.join(savebase, "race{}_padsData.npy".format(i)), 'rb') as f:
            subject_train_data.append(pickle.load(f))
        with open(os.path.join(savebase, "race{}_padsLabel.npy".format(i)), 'rb') as f:
            subject_train_label.append(pickle.load(f))

    subject_train_data = np.concatenate(subject_train_data)
    subject_train_label = np.concatenate(subject_train_label)
    total_trials = len(subject_train_data)
    n_channels = 63
    sampling_freq = 500  # in Hertz
    # info = mne.create_info(n_channels, sfreq=sampling_freq)
    ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7',
                'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
                'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8',
                'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
                'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5',
                'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
                'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
                'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
    ch_types = ['eeg']*63
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
    event_id = dict(left_hand=0, right_hand=1,feet=2,rest=3)
    # events = None

    events = np.column_stack((np.arange(0, sampling_freq*total_trials, sampling_freq),
                              np.zeros(total_trials, dtype=int),
                              subject_train_label))
    # print(events)
    # event_dict = dict(left_hand=0, right_hand=1,feet=2,rest=3)
    mne_data = mne.EpochsArray(subject_train_data,info,event_id=event_id,events=events,tmin=0)

    new_mne_data= process_target_data(mne_data,f_min=fmin,f_max=fmax,resample=sfreq,t_min=tmin,t_max=tmax)

    # print(new_mne_data.get_data().shape)

    subject_train_data = new_mne_data.get_data()

    subject_train_data,subject_train_label = shuffle_data(subject_train_data,subject_train_label)
    print("dataset A label : ",subject_train_label)
    X_MIA_train_data.append(subject_train_data)
    X_MIA_train_label.append(subject_train_label)
    subject_id = [subj]*len(subject_train_data)
    subject_ids.extend(subject_id)

for subj in range(len(X_MIA_train_data)):
    print("subject {}".format(subj + 1))
    subject_train_data = X_MIA_train_data[subj]
    subject_train_label = X_MIA_train_label[subj]
    print("There are {} trials with {} electrodes and {} time samples".format(*subject_train_data.shape))
    print("label shape : ", subject_train_label.shape)
# print(subject_ids)
dataset_A_meta = pd.DataFrame({"subject":subject_ids,"session":["session_0"]*len(subject_ids),"run":["run_0"]*len(subject_ids)})
# print("A meta : ",dataset_A_meta)
X_MIA_train_data = np.concatenate(X_MIA_train_data)
X_MIA_train_label = np.concatenate(X_MIA_train_label)
# print(X_MIA_train_data.shape)
# print(X_MIA_train_label.shape)
# print("original dataset A label : ",X_MIA_train_label)
X_MIA_train_data = modify_data(X_MIA_train_data,time=max_time_length)

X_MIA_train_label = np.array([relabel_target(l) for l in X_MIA_train_label])


# process Dataset B (S1, S2, S3)
# get train target data
path = '/Users/wduong/mne_data/MNE-beetlmileaderboard-data/'
sfreq = 128
tmin=0
tmax=3
max_time_length = int((tmax-tmin)*sfreq)
X_MIB_train_data = []
X_MIB_train_label = []
subject_ids = []

for subj in range(3, 6):
    savebase = os.path.join(path, "S{}".format(subj), "training")
    subject_train_data = []
    subject_train_label = []
    with open(os.path.join(savebase, "training_s{}X.npy".format(subj)), 'rb') as f:
        subject_train_data.append(pickle.load(f))
    with open(os.path.join(savebase, "training_s{}Y.npy".format(subj)), 'rb') as f:
        subject_train_label.append(pickle.load(f))
    subject_train_data = np.concatenate(subject_train_data)
    subject_train_label = np.concatenate(subject_train_label)

    # print("subject data shape : ",subject_train_data.shape)
    # print("subject label shape : ",subject_train_label.shape)

    # print("subject label : ",subject_train_label[:10])

    subject_train_label = subject_train_label.astype(int)

    total_trials = len(subject_train_data)
    n_channels = 32
    sampling_freq = 200  # in Hertz
    # info = mne.create_info(n_channels, sfreq=sampling_freq)
    ch_names = ['Fp1', 'Fp2', 'F3',
                'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz',
                'P2', 'P4', 'P6', 'P8']
    ch_types = ['eeg']*32

    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
    event_id = dict(left_hand=0, right_hand=1,feet=2,rest=3)
    # events = None

    events = np.column_stack((np.arange(0, sampling_freq*total_trials, sampling_freq),
                              np.zeros(total_trials, dtype=int),
                              subject_train_label))

    # print("events : ",events)
    # event_dict = dict(left_hand=0, right_hand=1,feet=2,rest=3)
    mne_data = mne.EpochsArray(subject_train_data,info,event_id=event_id,events=events,tmin=0)

    new_mne_data= process_target_data(mne_data,f_min=fmin,f_max=fmax,resample=sfreq,t_min=tmin,t_max=tmax)

    # print(new_mne_data.get_data().shape)

    subject_train_data = new_mne_data.get_data()

    subject_train_data,subject_train_label = shuffle_data(subject_train_data,subject_train_label)
    print("dataset B label : ",subject_train_label)
    X_MIB_train_data.append(subject_train_data)
    X_MIB_train_label.append(subject_train_label)
    subject_id = [subj]*len(subject_train_data)
    subject_ids.extend(subject_id)

for subj in range(len(X_MIB_train_data)):
    # print("subject {}".format(subj + 1))
    subject_train_data = X_MIB_train_data[subj]
    subject_train_label = X_MIB_train_label[subj]
    print("There are {} trials with {} electrodes and {} time samples".format(*subject_train_data.shape))
    # print("label shape : ", subject_train_label.shape)
# print(subject_ids)
dataset_B_meta = pd.DataFrame({"subject":subject_ids,"session":["session_0"]*len(subject_ids),"run":["run_0"]*len(subject_ids)})
# print("A meta : ",dataset_B_meta)
X_MIB_train_data = np.concatenate(X_MIB_train_data)
X_MIB_train_label = np.concatenate(X_MIB_train_label)

X_MIB_train_data = modify_data(X_MIB_train_data,time=max_time_length)
X_MIB_train_label = np.array([relabel_target(l) for l in X_MIB_train_label])

m_src1 = {name: col.values for name, col in m_src1.items()}
m_src2 = {name: col.values for name, col in m_src2.items()}
m_src3 = {name: col.values for name, col in m_src3.items()}
m_tgt_A = {name: col.values for name, col in dataset_A_meta.items()}

m_tgt_B = {name: col.values for name, col in dataset_B_meta.items()}

for k, v in m_src1.items():
    print(" k : ", k)
    print(" val : ", v)

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

# print("dataset A label : ",X_MIA_train_label)
target_dataset_A = {
    'data':X_MIA_train_data,
    'label':X_MIA_train_label,
    'meta_data':m_tgt_A,
    'dataset_name':'dataset_A'
}

# print("dataset B label : ",X_MIB_train_label)

target_dataset_B = {
    'data': X_MIB_train_data,
    'label': X_MIB_train_label,
    'meta_data': m_tgt_B,
    'dataset_name': 'dataset_B'
}
def generate_data_file(list_dataset_info,folder_name='case_0'):
    list_dataset = list()
    for dataset in list_dataset_info:
        list_dataset.append(dataset)
    # list_dataset.append(dataset_2)
    # list_dataset.append(dataset_3)
    # list_dataset.append(target_dataset)
    file_name = 'NeurIPS_TL'
    # data_file = '{}.mat'.format(file_name)
    data_file = '{}.mat'.format(file_name)
    # folder = 'case_3'
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    data_file = os.path.join(folder_name,data_file)
    from scipy.io import savemat
    savemat(data_file, {'datasets':list_dataset})

# generate_data_file([dataset_1,dataset_2,dataset_3,target_dataset_A],folder_name='case_3')


generate_data_file([dataset_1,dataset_2,dataset_3,target_dataset_B],folder_name='case_4')

