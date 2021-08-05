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
    generate_data_file
)
import scipy.signal as signal
import copy

class filterBank(object):
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank, fs, filtAllowance=2, axis=-1, filtType='filter'):
        self.filtBank = filtBank
        self.fs = fs
        self.filtAllowance = filtAllowance
        self.axis = axis
        self.filtType = filtType

    def bandpassFilter(self, data, bandFiltCutF, fs, filtAllowance=2, axis=-1, filtType='filter'):
        """
         Filter a signal using cheby2 iir filtering.
        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'
        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30  # stopband attenuation
        aPass = 3  # passband attenuation
        nFreq = fs / 2  # Nyquist frequency

        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
                bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data

        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass = bandFiltCutF[1] / nFreq
            fStop = (bandFiltCutF[1] + filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')

        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass = bandFiltCutF[0] / nFreq
            fStop = (bandFiltCutF[0] - filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')

        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass = (np.array(bandFiltCutF) / nFreq).tolist()
            fStop = [(bandFiltCutF[0] - filtAllowance) / nFreq, (bandFiltCutF[1] + filtAllowance) / nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut

    def __call__(self, data1):

        data = copy.deepcopy(data1)
        d = data
        # d = data['data']

        # initialize output
        out = np.zeros([*d.shape, len(self.filtBank)])
        # print("out shape : ",out.shape)
        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            filter = self.bandpassFilter(d, filtBand, self.fs, self.filtAllowance,
                                               self.axis, self.filtType)
            # print("filter shape : ",filter.shape)
            out[:,:, :, i] =filter


        # remove any redundant 3rd dimension
        if len(self.filtBank) <= 1:
            out = np.squeeze(out, axis=2)

        # data['data'] = torch.from_numpy(out).float()
        return out
        # return data


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

# #get common channel between dataset A and dataset B
# target_channels = generate_common_chan_test_data()
# print("common target chans : ",target_channels)
# fmin=0
# fmax=80
# tmax=3
# tmin=0
# sfreq=128
# max_time_length = int((tmax - tmin) * sfreq)
#
# # epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels)
# epoch_X_src1, label_src1, m_src1 = load_Cho2017(fmin=fmin,fmax=fmax,selected_chans=target_channels,subjects=[1,2,3])
# X_src1 = epoch_X_src1.get_data()
# diff = 4
# filter_bands = []
# # axis = 2
# for i in range(1,9):
#     filter_bands.append([i*diff,(i+1)*diff])
# print("build filter band : ",filter_bands)
# filter = filterBank(
#     filtBank= filter_bands,
#     fs=sfreq
#     # axis=axis
# )
#
# filter_data = filter(X_src1)
# print("filter data : ",filter_data.shape)
# # filter_data = filter_data.permute((0, 3, 1, 2))
# source = [0,1,2,3]
# destination = [0,2,3,1]
#
# # destination = [0,3,1,2]
# update_filter = np.moveaxis(filter_data, source, destination)
# print("update filter data : ",update_filter.shape)



# process Dataset B (S1, S2, S3)
# get train target data
# path = '/Users/wduong/mne_data/MNE-beetlmileaderboard-data/'
# sfreq = 128
# tmin=0
# tmax=3
# fmin, fmax = 4, 36
#
# max_time_length = int((tmax-tmin)*sfreq)
# X_MIB_test_data = []
# subject_ids = []

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

# print(commonList)
# print(len(commonList))

def load_dataset(data_path,target_dataset_name):
    from scipy.io import loadmat
    import numpy as np
    def reformat(data, label, meta_data):
        n_subjects = len(np.unique(meta_data['subject']))
        new_data = []
        new_label = []
        new_meta_data = []
        start = 0
        unique_subject_ids = np.unique(meta_data['subject'])
        for i in range(n_subjects):
            current_subject = unique_subject_ids[i]
            subject_meta_data = meta_data[meta_data['subject'] == current_subject]
            if len(subject_meta_data) > 0:
                trials = len(subject_meta_data)
                end = start + trials
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
        return new_data, new_label, new_meta_data
    temp = loadmat(data_path)

    datasets = temp['datasets'][0]

    # target_dataset_name = 'BCI_IV'
    print("target datast : ", target_dataset_name)
    target_data = None
    target_label = None
    target_meta_data = None
    list_source_data = []
    list_source_label = []
    list_source_meta_data = []
    for dataset in datasets:
        dataset = dataset[0][0]
        # print("dataset field name : ",dataset.dtype.names)
        # print(" frist field name : ",('r_op_list' in list(dataset.dtype.names)))
        dataset_name = dataset['dataset_name']
        data = dataset['data'].astype(np.float32)
        label = np.squeeze(dataset['label']).astype(int)
        meta_data = dataset['meta_data'][0][0]
        new_meta_data = {}
        new_meta_data['subject'] = meta_data['subject'][0]
        new_meta_data['session'] = [session[0] for session in meta_data['session'][0]]
        new_meta_data['run'] = [run[0] for run in meta_data['run'][0]]
        meta_data = pd.DataFrame.from_dict(new_meta_data)
        # print("dataset name : ",dataset_name)
        # print("original data size : ",len(data))
        # print("new meta : ",meta_data)
        # if dataset_name == "dataset_B":
        #     print("new meta : ",meta_data)
        data, label, meta_data = reformat(data, label, meta_data)
        if dataset_name == target_dataset_name:
            target_data = data
            target_label = label
            # print(" original target label : ", target_label)
            # print("len of target data : ",len(target_label))
            target_meta_data = meta_data
            # print("reformat meta data : ",target_meta_data)
        else:
            list_source_data.append(data)
            list_source_label.append(label)
            list_source_meta_data.append(meta_data)
    return target_data, target_label,list_source_data,list_source_label

#voltage dataset
path_5_A = "C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/case_5_A/NeurIPS_TL.mat"
path_5_B = "C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/case_5_B/NeurIPS_TL.mat"

#microvolt dataset
path_5_1_A = "C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/case_5_1_A/NeurIPS_TL.mat"
# path_5_1_B = "C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/case_5_1_B/NeurIPS_TL.mat"

path_5_1_B = "C:/wduong_folder/Dassl.pytorch-master/NeurIPS_competition/EEG_Dassl_Lightning/da_dataset/NeurIPS_competition/case_4/NeurIPS_TL.mat"

# data_A, label_A ,list_source_data,list_source_label = load_dataset(path_5_A,"dataset_A")
# data_A_1, label_A_1 ,list_source_data_1,list_source_label_1 = load_dataset(path_5_1_A,"dataset_A")

data_A, label_A ,list_source_data,list_source_label = load_dataset(path_5_B,"dataset_B")
data_A_1, label_A_1 ,list_source_data_1,list_source_label_1 = load_dataset(path_5_1_B,"dataset_B")

for subject in range(len(data_A)):
    label_subject = label_A[subject]
    print("dataset A subject {} ,   has label {}".format(subject,label_subject))

    label_subject_1 = label_A_1[subject]
    print("dataset A_1 subject {} , has label {}".format(subject,label_subject_1))

    data_subject = data_A[subject][0,:3,:6]
    print(data_subject)
    data_subject_1 = data_A_1[subject][0,:3,:6]
    print(data_subject_1)
