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



def process_target_data(epoch_data,f_min=4,f_max=36,resample=128,t_min=0,t_max=3):
    epoch_f = epoch_data.copy().filter(
        f_min, f_max, method="iir")
    # if bmin < tmin or bmax > tmax:
    epoch_f=     epoch_f.crop(tmin=t_min, tmax=t_max)
    # if self.resample is not None:
    epoch_f = epoch_f.resample(resample)
    return epoch_f
def modify_data(data,time=256):
    return data[:, :, :time]


# process Dataset B (S1, S2, S3)
# get train target data
path = '/Users/wduong/mne_data/MNE-beetlmileaderboard-data/'
sfreq = 128
tmin=0
tmax=3
fmin, fmax = 4, 36

max_time_length = int((tmax-tmin)*sfreq)
X_MIB_test_data = []
subject_ids = []

# for subj in range(3, 6):
#     savebase = os.path.join(path, "S{}".format(subj), "testing")
#     subject_test_data = []
#     with open(os.path.join(savebase, "testing_s{}X.npy".format(subj)), 'rb') as f:
#         subject_test_data.append(pickle.load(f))
#     subject_test_data = np.concatenate(subject_test_data)
#
#     total_trials = len(subject_test_data)
#     n_channels = 32
#     sampling_freq = 200  # in Hertz
#     # info = mne.create_info(n_channels, sfreq=sampling_freq)
#     ch_names = ['Fp1', 'Fp2', 'F3',
#                 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C5', 'C3',
#                 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
#                 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz',
#                 'P2', 'P4', 'P6', 'P8']
#     ch_types = ['eeg']*32
#
#     info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
#     # print("events : ",events)
#     # event_dict = dict(left_hand=0, right_hand=1,feet=2,rest=3)
#     mne_data = mne.EpochsArray(subject_test_data,info,tmin=0)
#
#     new_mne_data= process_target_data(mne_data,f_min=fmin,f_max=fmax,resample=sfreq,t_min=tmin,t_max=tmax)
#
#     # print(new_mne_data.get_data().shape)
#     subject_test_data = new_mne_data.get_data()
#
#     print("dataset B label : ",subject_test_data)
#     X_MIB_test_data.append(subject_test_data)
#     subject_id = [subj]*len(subject_test_data)
#     subject_ids.extend(subject_id)
#
# for subj in range(len(X_MIB_test_data)):
#     # print("subject {}".format(subj + 1))
#     subject_train_data = X_MIB_test_data[subj]
#     print("There are {} trials with {} electrodes and {} time samples".format(*subject_train_data.shape))
#     # print("label shape : ", subject_train_label.shape)
# # print(subject_ids)
# dataset_B_meta = pd.DataFrame({"subject":subject_ids,"session":["session_0"]*len(subject_ids),"run":["run_0"]*len(subject_ids)})
# # print("A meta : ",dataset_B_meta)
# X_MIB_test_data = np.concatenate(X_MIB_test_data)
# X_MIB_test_data = modify_data(X_MIB_test_data,time=max_time_length)
#
# print(X_MIB_test_data.shape)

#process dataset A
# X_MIA_test_data = []
# subject_ids = []
# for subj in range(1, 3):
#     savebase = os.path.join(path, "S{}".format(subj), "testing")
#     subject_test_data = []
#     for i in range(6, 16):
#         with open(os.path.join(savebase, "race{}_padsData.npy".format(i)), 'rb') as f:
#             subject_test_data.append(pickle.load(f))
#     subject_test_data = np.concatenate(subject_test_data)
#
#
#     total_trials = len(subject_test_data)
#     n_channels = 63
#     sampling_freq = 500  # in Hertz
#     # info = mne.create_info(n_channels, sfreq=sampling_freq)
#     ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7',
#                 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
#                 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8',
#                 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
#                 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5',
#                 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
#                 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
#                 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
#     ch_types = ['eeg'] * 63
#     info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
#     mne_data = mne.EpochsArray(subject_test_data,info,tmin=0)
#
#     new_mne_data= process_target_data(mne_data,f_min=fmin,f_max=fmax,resample=sfreq,t_min=tmin,t_max=tmax)
#
#     # print(new_mne_data.get_data().shape)
#     subject_test_data = new_mne_data.get_data()
#
#     print("dataset A label : ",subject_test_data)
#     X_MIA_test_data.append(subject_test_data)
#     subject_id = [subj]*len(subject_test_data)
#     subject_ids.extend(subject_id)
#
# for subj in range(len(X_MIA_test_data)):
#     subject_test_data = X_MIA_test_data[subj]
#     print("There are {} trials with {} electrodes and {} time samples".format(*subject_test_data.shape))
#
# X_MIA_test_data = np.concatenate(X_MIA_test_data)
# X_MIA_test_data = modify_data(X_MIA_test_data,time=max_time_length)
#
# print(X_MIA_test_data.shape)

def generate_common_chan_test_data():
    ch_names_B = ['Fp1', 'Fp2', 'F3',
                    'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C5', 'C3',
                    'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                    'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz',
                    'P2', 'P4', 'P6', 'P8']

    ch_names_A = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7',
                'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
                'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8',
                'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
                'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5',
                'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
                'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
                'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
    commonList = []
    for chan_A in ch_names_A:
        for chan_B in ch_names_B:
            if chan_A == chan_B:
                commonList.append(chan_A)
    return commonList
# print(commonList)
# print(len(commonList))
