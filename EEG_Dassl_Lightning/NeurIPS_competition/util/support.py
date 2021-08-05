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

def print_dataset_info(data,dataset_name="train dataset A"):
    print(dataset_name)
    # for subject_idx in range(len(data)):
    print("Train subject has shape : {}, with range scale ({},{}) ".format(
        data.shape,
        np.max(data), np.min(data)))

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

def normalization(X):

    # assert len(X) == len(y)
    # Normalised, you could choose other normalisation strategy
    if len(X.shape) == 3:
        # assume the data in format (trials,channels,samples)
        axis = 1
    elif len(X.shape) == 4:
        # assume the data in format (trials,filter,channels,samples)
        axis = 2
    else:
        axis = -1
        raise ValueError("there is problem with data format")
    print("normalize along axis {} for the data ".format(axis))
    # assert len(X) == len(y)
    # Normalised, you could choose other normalisation strategy
    mean = np.mean(X, axis=axis, keepdims=True)
    # here normalise across channels as an example, unlike the in the sleep kit
    std = np.std(X, axis=axis, keepdims=True)
    X = (X - mean) / std
    return X

# def dataset_norm(data):
#     new_data = list()
#     for subject_idx in range(len(data)):
#         subject_data = data[subject_idx]
#         subject_data = normalization(subject_data)
#         new_data.append(subject_data)
#     return new_data

def shuffle_data(subject_data,subject_label):
    available_index = np.arange(subject_data.shape[0])
    shuffle_index = np.random.permutation(available_index)
    shuffle_subject_data = subject_data[shuffle_index,]
    shuffle_subject_label = subject_label[shuffle_index,]
    return [shuffle_subject_data,shuffle_subject_label]


def modify_data(data,time=256):
    return data[:, :, :time]


def generate_common_chan_test_data(ch_names_A=None,ch_names_B=None):
    if not ch_names_B:
        ch_names_B = ['Fp1', 'Fp2', 'F3',
                  'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C5', 'C3',
                  'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                  'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz',
                  'P2', 'P4', 'P6', 'P8']
    if not ch_names_A:
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

def correct_EEG_data(data,channels,correct_chans_order):
    new_eeg_data = np.zeros((data.shape[0],len(correct_chans_order),data.shape[2]))
    # print("current BCI chans : ", data_chans)
    # print("correct chans order : ",correct_chans_order)
    for target_idx in range(len(correct_chans_order)):
        target_chans = correct_chans_order[target_idx]
        check = False
        for current_idx in range(len(channels)):
            ccurrent_hans = channels[current_idx]
            if target_chans == ccurrent_hans:
                new_eeg_data[:, target_idx:target_idx + 1, :] = data[:, current_idx:current_idx + 1, :]
                check = True
        if not check:
            print("current chans not used : ", target_chans)
    print("old chans : ",channels)
    print("new chans : ",correct_chans_order)
    return new_eeg_data

def correct_EEG_data_order(epoch_data,correct_chans_order):
    data_chans = epoch_data.ch_names
    data = epoch_data.get_data()
    return correct_EEG_data(data,data_chans,correct_chans_order)

def interpolate_BCI_IV_dataset(epoch_data,common_target_chans,montage,sfreq=128):
    print("common chans : ",common_target_chans)
    current_chans = epoch_data.ch_names
    current_eeg_data = epoch_data.get_data()
    new_eeg_data = np.zeros((current_eeg_data.shape[0],len(common_target_chans),current_eeg_data.shape[2]))
    print("current BCI chans : ",current_chans)
    bad_chans = []
    for target_idx in range(len(common_target_chans)):
        target_chans = common_target_chans[target_idx]
        check = False
        for current_idx in range(len(current_chans)):
            chans = current_chans[current_idx]
            if target_chans == chans:
                new_eeg_data[:,target_idx:target_idx+1,:] = current_eeg_data[:,current_idx:current_idx+1,:]
                check=True
        if not check:
            bad_chans.append(target_chans)

    new_info = mne.create_info(ch_names=common_target_chans,sfreq=sfreq,ch_types='eeg')
    eeg_data = mne.EpochsArray(new_eeg_data,info=new_info)
    eeg_data.set_montage(montage)
    print("bad chans : ",bad_chans)
    eeg_data.info['bads'] = bad_chans

    eeg_data_interp = eeg_data.copy().interpolate_bads()

    print("old chans : ",current_chans)
    correct_chans_order = eeg_data_interp.ch_names
    print("new chans : ",correct_chans_order)
    return eeg_data_interp


def load_Cho2017(sfreq = 128,fmin=4,fmax=36,tmin=0,tmax=3,selected_chans = None,subjects=None):
    # #process Cho2017 dataset
    ds_src1 = Cho2017()
    raw_src_1 = ds_src1.get_data(subjects=[1])[1]['session_0']['run_0']
    if not selected_chans:
        selected_chans = raw_src_1.pick_types(eeg=True).ch_names
    # src_1_prgm =MotorImagery(n_classes=2, channels=raw_src_1_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
    src_1_prgm = MotorImagery(n_classes=2, channels=selected_chans, resample=sfreq, fmin=fmin, fmax=fmax, tmin=tmin,
                              tmax=tmax)
    # epoch_X_src1, label_src1, m_src1 = src_1_prgm.get_data(dataset=ds_src1,subjects=[1, 2],return_epochs=True)
    if not subjects:
        epoch_X_src1, label_src1, m_src1 = src_1_prgm.get_data(dataset=ds_src1, return_epochs=True)
    else:
        epoch_X_src1, label_src1, m_src1 = src_1_prgm.get_data(dataset=ds_src1,subjects=subjects, return_epochs=True)

    return epoch_X_src1,label_src1,m_src1

def load_Physionet(sfreq = 128,fmin=4,fmax=36,tmin=0,tmax=3,selected_chans = None,subjects=None,events=None):
    ds_src2 = PhysionetMI()
    # process physionet dataset
    raw_src_2 = ds_src2.get_data(subjects=[1])[1]['session_0']['run_4']
    if not selected_chans:
        selected_chans = raw_src_2.pick_types(eeg=True).ch_names
    if events is None:
        events = dict(left_hand=2, right_hand=3, hands=4, feet=5)
    n_classes = len(list(events.keys()))
    print("physionet event dict : ",events)
    print("n classes : ",n_classes)
    # src_2_prgm =MotorImagery(events=dict(left_hand=2, right_hand=3, feet=5),n_classes=3, channels=raw_src_2_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
    # src_2_prgm =MotorImagery(events=dict(left_hand=2, right_hand=3, feet=5),n_classes=3, channels=raw_src_2_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
    # src_2_prgm =MotorImagery(events=dict(rest=1,left_hand=2, right_hand=3, hands=4,feet=5),n_classes=4, channels=target_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)

    src_2_prgm = MotorImagery(events=events, n_classes=n_classes,
                              channels=selected_chans, resample=sfreq, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    if subjects is None:
        epoch_X_src2, label_src2, m_src2 = src_2_prgm.get_data(dataset=ds_src2, return_epochs=True)
    else:
        epoch_X_src2, label_src2, m_src2 = src_2_prgm.get_data(dataset=ds_src2,subjects=subjects,return_epochs=True)

    return epoch_X_src2, label_src2, m_src2

def load_BCI_IV(sfreq = 128,fmin=4,fmax=36,tmin=0.5,tmax=3.5,selected_chans = None,montage=None,subjects=None,events=None):
    ds_src3 = BNCI2014001()
    # #process BCI_IV dataset
    raw = ds_src3.get_data(subjects=[1])[1]['session_T']['run_1']
    raw_src_3_channels = raw.pick_types(eeg=True).ch_names
    if not selected_chans:
        selected_chans = raw_src_3_channels
    common_BCI_chans_A_B = generate_common_chan_test_data(ch_names_A=raw_src_3_channels, ch_names_B=selected_chans)
    max_time_length = int((tmax - tmin) * sfreq)
    if events is None:
        # "left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4
        events = dict(left_hand=1, right_hand=2,feet=3,tongue=4)
    n_classes = len(list(events.keys()))
    print("BCI_IV event dict : ",events)
    print("n classes : ",n_classes)
    src_3_prgm = MotorImagery(n_classes=4, channels=common_BCI_chans_A_B, resample=sfreq, fmin=fmin, fmax=fmax,
                              tmin=tmin, tmax=tmax)
    # src_3_prgm = MotorImagery(n_classes=4, channels=raw_src_3_channels, resample=sfreq, fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax)
    if subjects is None:
        epoch_X_src3, label_src3, m_src3 = src_3_prgm.get_data(dataset=ds_src3, return_epochs=True)
    else:
        epoch_X_src3, label_src3, m_src3 = src_3_prgm.get_data(dataset=ds_src3,subjects=subjects,return_epochs=True)
    # set up the interpolation and get all the data in the same correct order
    if montage:
        update_epoch_src3 = interpolate_BCI_IV_dataset(epoch_X_src3, selected_chans, montage,sfreq=sfreq)
        return update_epoch_src3,label_src3,m_src3
    return epoch_X_src3, label_src3, m_src3


def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    else: return 2
def relabel_target(l):
    if l == 0: return 0
    elif l == 1: return 1
    else: return 2

def process_target_data(epoch_data,f_min=4,f_max=36,resample=128,t_min=0,t_max=3):
    epoch_f = epoch_data.copy().filter(
        f_min, f_max, method="iir")
    epoch_f=     epoch_f.crop(tmin=t_min, tmax=t_max)
    epoch_f = epoch_f.resample(resample)
    return epoch_f




def load_train_A(path,ch_names,ch_types,sfreq = 128,fmin=4,fmax=36,tmin=0,tmax=3):
    """dataset A is in voltage foramt"""

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
        sampling_freq = 500  # in Hertz
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
        event_id = dict(left_hand=0, right_hand=1, feet=2, rest=3)
        events = np.column_stack((np.arange(0, sampling_freq * total_trials, sampling_freq),
                                  np.zeros(total_trials, dtype=int),
                                  subject_train_label))

        mne_data = mne.EpochsArray(subject_train_data, info, event_id=event_id, events=events, tmin=0)

        new_mne_data = process_target_data(mne_data, f_min=fmin, f_max=fmax, resample=sfreq, t_min=tmin, t_max=tmax)
        subject_train_data = new_mne_data.get_data()

        subject_train_data, subject_train_label = shuffle_data(subject_train_data, subject_train_label)
        X_MIA_train_data.append(subject_train_data)
        X_MIA_train_label.append(subject_train_label)
        subject_id = [subj] * len(subject_train_data)
        subject_ids.extend(subject_id)
    dataset_A_meta = pd.DataFrame({"subject":subject_ids,"session":["session_0"]*len(subject_ids),"run":["run_0"]*len(subject_ids)})
    X_MIA_train_data = np.concatenate(X_MIA_train_data)
    X_MIA_train_label = np.concatenate(X_MIA_train_label)
    return X_MIA_train_data,X_MIA_train_label,dataset_A_meta

def load_test_A(path,ch_names,ch_types,sfreq = 128,fmin=4,fmax=36,tmin=0,tmax=3):
    """dataset A is in voltage foramt"""

    X_MIA_test_data = []
    subject_ids = []
    for subj in range(1, 3):
        savebase = os.path.join(path, "S{}".format(subj), "testing")
        subject_test_data = []
        for i in range(6, 16):
            with open(os.path.join(savebase, "race{}_padsData.npy".format(i)), 'rb') as f:
                subject_test_data.append(pickle.load(f))
        subject_test_data = np.concatenate(subject_test_data)
        sampling_freq = 500  # in Hertz
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
        mne_data = mne.EpochsArray(subject_test_data, info, tmin=0)
        new_mne_data = process_target_data(mne_data, f_min=fmin, f_max=fmax, resample=sfreq, t_min=tmin, t_max=tmax)
        subject_test_data = new_mne_data.get_data()

        X_MIA_test_data.append(subject_test_data)
        subject_id = [subj] * len(subject_test_data)
        subject_ids.extend(subject_id)
    # for subj in range(len(X_MIA_test_data)):
        # print("subject {}".format(subj + 1))
        # subject_train_data = X_MIA_test_data[subj]
        # print("There are {} trials with {} electrodes and {} time samples".format(*subject_train_data.shape))
        # print("label shape : ", subject_train_label.shape)
    # format into trials,channels,sample
    X_MIA_test_data = np.concatenate(X_MIA_test_data)
    return X_MIA_test_data


def load_dataset_A(path=None,train=True,norm=False,selected_chans = None,sfreq = 128,fmin=4,fmax=36,tmin=0,tmax=3):
    """dataset A is in voltage foramt"""
    # process Dataset A (S1, S2)
    # get train target data
    if path is None:
        path = '/Users/wduong/mne_data/MNE-beetlmileaderboard-data/'
    # sfreq = 128
    # tmin=0
    # tmax=3
    max_time_length = int((tmax-tmin)*sfreq)
    ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7',
                'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
                'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8',
                'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
                'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5',
                'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
                'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
                'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
    ch_types = ['eeg'] * 63
    if not selected_chans:
        selected_chans = ch_names
    if train:
        X_MIA_train_data, X_MIA_train_label, dataset_A_meta = load_train_A(path,ch_names,ch_types,sfreq = sfreq,fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax)
        X_MIA_train_data = correct_EEG_data(X_MIA_train_data, ch_names, selected_chans)
        X_MIA_train_data = modify_data(X_MIA_train_data, time=max_time_length)
        m_tgt_A = {name: col.values for name, col in dataset_A_meta.items()}
        return X_MIA_train_data,X_MIA_train_label,m_tgt_A
    else:
        X_MIA_test_data = load_test_A(path,ch_names,ch_types,sfreq = sfreq,fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax)
        X_MIA_test_data = modify_data(X_MIA_test_data, time=max_time_length)
        X_MIA_test_data = correct_EEG_data(X_MIA_test_data, ch_names, selected_chans)
        return X_MIA_test_data

def load_train_B(path,ch_names,ch_types,sfreq = 128,fmin=4,fmax=36,tmin=0,tmax=3):
    """problem with dataset B. Dataset B is in the format of microvoltage. Need to convert to voltage """
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

        #convert to voltage
        subject_train_data = subject_train_data*1e-6
        subject_train_label = subject_train_label.astype(int)
        total_trials = len(subject_train_data)
        sampling_freq = 200  # in Hertz
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
        event_id = dict(left_hand=0, right_hand=1, feet=2, rest=3)
        events = np.column_stack((np.arange(0, sampling_freq * total_trials, sampling_freq),
                                  np.zeros(total_trials, dtype=int),
                                  subject_train_label))

        mne_data = mne.EpochsArray(subject_train_data, info, event_id=event_id, events=events, tmin=0)

        new_mne_data = process_target_data(mne_data, f_min=fmin, f_max=fmax, resample=sfreq, t_min=tmin, t_max=tmax)

        subject_train_data = new_mne_data.get_data()

        subject_train_data, subject_train_label = shuffle_data(subject_train_data, subject_train_label)
        X_MIB_train_data.append(subject_train_data)
        X_MIB_train_label.append(subject_train_label)
        subject_id = [subj] * len(subject_train_data)
        subject_ids.extend(subject_id)
    dataset_B_meta = pd.DataFrame(
        {"subject": subject_ids, "session": ["session_0"] * len(subject_ids), "run": ["run_0"] * len(subject_ids)})
    X_MIB_train_data = np.concatenate(X_MIB_train_data)
    X_MIB_train_label = np.concatenate(X_MIB_train_label)
    return X_MIB_train_data,X_MIB_train_label,dataset_B_meta

def load_test_B(path,ch_names,ch_types,sfreq = 128,fmin=4,fmax=36,tmin=0,tmax=3):
    """problem with dataset B. Dataset B is in the format of microvoltage. Need to convert to voltage """

    X_MIB_test_data = []
    subject_ids = []
    for subj in range(3, 6):
        savebase = os.path.join(path, "S{}".format(subj), "testing")
        subject_test_data = []
        with open(os.path.join(savebase, "testing_s{}X.npy".format(subj)), 'rb') as f:
            subject_test_data.append(pickle.load(f))
        subject_test_data = np.concatenate(subject_test_data)
        #convert to voltage
        subject_test_data = subject_test_data*1e-6
        sampling_freq = 200  # in Hertz
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
        mne_data = mne.EpochsArray(subject_test_data, info, tmin=0)
        new_mne_data = process_target_data(mne_data, f_min=fmin, f_max=fmax, resample=sfreq, t_min=tmin, t_max=tmax)
        subject_test_data = new_mne_data.get_data()

        X_MIB_test_data.append(subject_test_data)
        subject_id = [subj] * len(subject_test_data)
        subject_ids.extend(subject_id)

    X_MIB_test_data = np.concatenate(X_MIB_test_data)
    return X_MIB_test_data

def load_dataset_B(path=None,train=True,norm=True,selected_chans = None,sfreq = 128,fmin=4,fmax=36,tmin=0,tmax=3):
    """problem with dataset B. Dataset B is in the format of microvoltage. Need to convert to voltage """

    # process Dataset B (S3, S4,S5)
    # get train target data
    if path is None:
        path = '/Users/wduong/mne_data/MNE-beetlmileaderboard-data/'

    max_time_length = int((tmax-tmin)*sfreq)
    ch_names = ['Fp1', 'Fp2', 'F3',
                'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz',
                'P2', 'P4', 'P6', 'P8']
    ch_types = ['eeg'] * 32
    if not selected_chans:
        selected_chans = ch_names
    if train:
        X_MIB_train_data, X_MIB_train_label, dataset_B_meta = load_train_B(path,ch_names,ch_types,sfreq = sfreq,fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax)
        X_MIB_train_data = correct_EEG_data(X_MIB_train_data, ch_names, selected_chans)
        X_MIB_train_data = modify_data(X_MIB_train_data, time=max_time_length)
        m_tgt_B = {name: col.values for name, col in dataset_B_meta.items()}
        return X_MIB_train_data,X_MIB_train_label,m_tgt_B
    else:
        X_MIB_test_data = load_test_B(path,ch_names,ch_types,sfreq = sfreq,fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax)
        X_MIB_test_data = modify_data(X_MIB_test_data, time=max_time_length)
        X_MIB_test_data = correct_EEG_data(X_MIB_test_data, ch_names, selected_chans)
        return X_MIB_test_data
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