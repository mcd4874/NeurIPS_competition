from scipy.linalg import sqrtm, inv
from scipy import signal
import numpy as np
from dassl.data.datasets.base_dataset import EEGDatum

# class EuclideanAlignment:
#     def __init__(self):
#
#     def

def euclidean_alignment(x):
        """
        convert trials in data with EA technique
        """

        assert len(x.shape) == 3

        r = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
        if np.iscomplexobj(r):
            print("covariance matrix problem")
        if np.iscomplexobj(sqrtm(r)):
            print("covariance matrix problem sqrt")

        r_op = inv(sqrtm(r))
        # print("r_op shape : ",r_op.shape)
        # print("data shape : ",x.shape)
        # print("r_op : ",r_op)
        if np.iscomplexobj(r_op):
            print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
                  "not, check data!!")
            r_op = np.real(r_op).astype(np.float32)
        elif not np.any(np.isfinite(r_op)):
            print("WARNING! Not finite values in R Matrix")

        results = np.matmul(r_op, x)
        # print("r_op shape : ",r_op.shape)
        # print("data shape : ",x.shape)
        # print("r_op : ",r_op)
        # print("result shape : ",results.shape)
        # print("a trial before convert : ",x[0,:,:])
        # print("a trial after convert : ",results[0,:,:])
        return results
def convert_subjects_data_with_EA(subjects_data):
    new_data = list()
    for subject_idx in range(len(subjects_data)):
        subject_data = subjects_data[subject_idx]
        subject_data = euclidean_alignment(subject_data)
        new_data.append(subject_data)
    return new_data
# def generate_datasource(data, label, test_data=False,label_name_map= None):
#     total_subjects = 1
#     if not test_data:
#         total_subjects = len(data)
#
#     subjects_item = []
#     for subject in range(total_subjects):
#         items = []
#         current_subject_data = data[subject]
#         current_subject_label = label[subject]
#         domain = subject
#         for i in range(current_subject_data.shape[0]):
#             trial_data = current_subject_data[i]
#             trial_label = int(current_subject_label[i])
#             label_name = ''
#             if label_name_map is not None and trial_label in label_name_map.keys():
#                 label_name = label_name_map[trial_label]
#             item = EEGDatum(eeg_data= trial_data, label= trial_label, domain=domain,classname=label_name)
#             items.append(item)
#         subjects_item.append(items)
#     return subjects_item
#
# def get_num_classes(data_source):
#     label_set = set()
#     for subject_item in data_source:
#         for item in subject_item:
#             label_set.add(item.label)
#     return max(label_set) + 1
# def get_label_classname_mapping(data_source):
#     tmp = set()
#     for subject_item in data_source:
#         for item in subject_item:
#             tmp.add((item.label, item.classname))
#     mapping = {label: classname for label, classname in tmp}
#     return mapping

def generate_datasource(data, label, test_data=False,label_name_map= None):
    items = []
    total_subjects = 1
    if not test_data:
        total_subjects = len(data)
    for subject in range(total_subjects):
        current_subject_data = data[subject]
        current_subject_label = label[subject]
        domain = subject
        for i in range(current_subject_data.shape[0]):
            trial_data = current_subject_data[i]
            trial_label = int(current_subject_label[i])
            label_name = ''
            if label_name_map is not None and trial_label in label_name_map.keys():
                label_name = label_name_map[trial_label]
            item = EEGDatum(eeg_data= trial_data, label= trial_label, domain=domain,classname=label_name)
            items.append(item)
    return items



def get_num_classes(data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

def get_label_classname_mapping(data_source):
    tmp = set()
    for item in data_source:
        tmp.add((item.label, item.classname))
    mapping = {label: classname for label, classname in tmp}
    return mapping

class DataAugmentation:
    def __init__(self,data,label,max_trials_mul = 3,total_fix_trials = -1,spatial_dataset_name = "BCI_IV"):
        self.data = data
        self.label = label
        self.max_trials_mul = max_trials_mul
        self.total_fix_trials = total_fix_trials
        self.spatial_dataset_name = spatial_dataset_name

    def shuffle_data(self,subject_data, subject_label):
        available_index = np.arange(subject_data.shape[0])
        shuffle_index = np.random.permutation(available_index)
        shuffle_subject_data = subject_data[shuffle_index,]
        shuffle_subject_label = subject_label[shuffle_index,]
        return [shuffle_subject_data, shuffle_subject_label]

    def groupby(self,a, b):
        # Get argsort indices, to be used to sort a and b in the next steps
        sidx = b.argsort(kind='mergesort')
        a_sorted = a[sidx]
        b_sorted = b[sidx]

        # Get the group limit indices (start, stop of groups)
        cut_idx = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1], True])

        # Split input array with those start, stop ones
        out = [a_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])]
        label = [b_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])]
        return [out, label]

    def data_augment_spatial(self,EEG_data, label, dataset_name="BCI_IV", fix_trials=-1):
        max_trials_mul = self.max_trials_mul

        if dataset_name == "BCI_IV":
            print("apply spatial for BCI_IV")
            left_side = [1, 2, 3, 7, 8, 9, 10, 14, 15, 19, 20]
            right_side = [4, 5, 6, 11, 12, 13, 16, 17, 18, 21, 22]
        else:
            print("apply for giga")
            left_side = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                         27, 29, 31
                , 33, 38, 48]
            right_side = [28, 30, 32, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55,
                          56, 57, 58, 59, 60, 61
                , 62, 63, 64]

        left_side = [i - 1 for i in left_side]
        right_side = [i - 1 for i in right_side]

        # max trial for each class
        # max_augment_trials = EEG_data.shape[0]*max_trials_mul

        n_unique_categories = len(np.unique(label))
        unique_labels = np.unique(label)
        # print("unique classes : ", n_unique_categories)
        # print("unique labels : ", unique_labels)
        # print("label : ", label)
        n_channels = EEG_data.shape[1]
        n_samples = EEG_data.shape[2]
        # categories_eeg = [None]*num_unique_categories

        # seperate trials into group for each categories
        group_EEG_data, group_label = self.groupby(EEG_data, label)
        final_data = []
        final_label = []
        # for label_category in range(n_unique_categories):
        for idx in range(n_unique_categories):
            label_category = unique_labels[idx]
            new_artificial_data = []

            category_EEG_data = group_EEG_data[idx]
            track_current_combination = list()

            category_n_trial = category_EEG_data.shape[0]
            if fix_trials == -1:
                print("generate max trials ")
                max_augment_trials = category_n_trial * max_trials_mul
            else:
                max_augment_trials = fix_trials
            print("max augment trials each class : ", max_augment_trials)
            # print("ca")

            for i in range(max_augment_trials):
                artificial_EEG = np.zeros((n_channels, n_samples))
                temp_record = list()

                pick_idx_left = np.random.randint(low=0, high=category_EEG_data.shape[0])
                pick_idx_right = np.random.randint(low=0, high=category_EEG_data.shape[0])
                while (pick_idx_left == pick_idx_right):
                    # print("pick same idx, need to repick")
                    pick_idx_left = np.random.randint(low=0, high=category_EEG_data.shape[0])
                    pick_idx_right = np.random.randint(low=0, high=category_EEG_data.shape[0])
                temp_record.append(pick_idx_left)
                temp_record.append(pick_idx_right)
                left_EEG_data = category_EEG_data[pick_idx_left]
                right_EEG_data = category_EEG_data[pick_idx_right]

                artificial_EEG[left_side, :] = left_EEG_data[left_side, :]
                artificial_EEG[right_side, :] = right_EEG_data[right_side, :]
                new_artificial_data.append(artificial_EEG)
                # print("temp record {} for trial {} : ".format(temp_record, i))

            new_artificial_data = np.stack(new_artificial_data)
            new_label = np.ones(max_augment_trials) * label_category
            final_data.append(new_artificial_data)
            final_label.append(new_label)
        final_data = np.concatenate(final_data)
        final_label = np.concatenate(final_label)

        final_data, final_label = self.shuffle_data(final_data, final_label)
        return final_data, final_label

    def data_augmentation_temporal_STFT(self,EEG_data, label, n_segment=4, fix_trials=-1,
                                        window_size=1 / 8, overlap=0.5, sampling_rate=128):
        max_trials_mul = self.max_trials_mul
        n_unique_categories = len(np.unique(label))
        unique_labels = np.unique(label)
        n_channels = EEG_data.shape[1]
        n_samples = EEG_data.shape[2]

        # seperate trials into group for each categories
        group_EEG_data, group_label = self.groupby(EEG_data, label)

        test_trial = EEG_data[1]
        first_chan = test_trial[0]
        fs = sampling_rate
        nperseg = int(window_size * n_samples)
        noverlap = int(nperseg * overlap)
        f, t, Zxx = signal.stft(first_chan, fs=fs, nperseg=nperseg, noverlap=noverlap)
        f_size = len(f)
        t_size = len(t)
        segment_size = t_size // n_segment

        final_data = []
        final_label = []
        for idx in range(n_unique_categories):
            label_category = unique_labels[idx]
            new_artificial_data = []
            category_EEG_data = group_EEG_data[idx]
            category_n_trial = category_EEG_data.shape[0]
            if fix_trials == -1:
                max_augment_trials = category_n_trial * max_trials_mul
            else:
                max_augment_trials = fix_trials

            for i in range(max_augment_trials):
                temp_record = list()
                artificial_EEG_T_F = np.zeros((n_channels, f_size, t_size), dtype=complex)
                artificial_EEG = np.zeros((n_channels, n_samples))
                for seg_idx in range(n_segment):
                    # randomly pick a trial
                    pick_idx = np.random.randint(low=0, high=category_EEG_data.shape[0])
                    temp_record.append(pick_idx)
                    current_EEG_trial = category_EEG_data[pick_idx]
                    # convert the EEG data trial to frequency-time domain
                    T_F_EEG_trial = np.zeros((n_channels, f_size, t_size), dtype=complex)
                    for c in range(n_channels):
                        channel_data = current_EEG_trial[c]
                        _, _, Zxx = signal.stft(channel_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
                        T_F_EEG_trial[c, :, :] = Zxx


                    if seg_idx < n_segment - 1:
                        artificial_EEG_T_F[:, :, seg_idx * segment_size:(seg_idx + 1) * segment_size] = T_F_EEG_trial[:,:,seg_idx * segment_size:(seg_idx + 1) * segment_size]
                    else:
                        # the last segment has an extra time point due to odd time ppint length
                        artificial_EEG_T_F[:, :, seg_idx * segment_size:] = T_F_EEG_trial[:, :, seg_idx * segment_size:]

                # convert the artificial EEG data back to time domain
                for c in range(artificial_EEG_T_F.shape[0]):
                    channel_data_T_F = artificial_EEG_T_F[c]
                    _, chanel_data = signal.istft(channel_data_T_F, fs=fs, nperseg=nperseg, noverlap=noverlap)
                    artificial_EEG[c, :] = chanel_data

                new_artificial_data.append(artificial_EEG)

            new_artificial_data = np.stack(new_artificial_data)
            new_label = np.ones(max_augment_trials) * label_category
            final_data.append(new_artificial_data)
            final_label.append(new_label)
        final_data = np.concatenate(final_data)
        final_label = np.concatenate(final_label)
        final_data, final_label = self.shuffle_data(final_data, final_label)
        return final_data, final_label

    def data_augment_temporal(self,EEG_data, label, n_segment=4,fix_trials=-1):
        """
        EEG_data = (n_trials,n_channels,n_samples)
        label = (n_trials). Assume label start with 0.
        n_segment: number of segment to cut the temporal samples. Assume that n_samples % segment = 0
        """
        # max trial for each class
        max_trials_mul = self.max_trials_mul
        n_unique_categories = len(np.unique(label))
        unique_labels = np.unique(label)

        # print("unique classes : ", n_unique_categories)
        # print("label : ", label)
        n_channels = EEG_data.shape[1]
        n_samples = EEG_data.shape[2]
        segment_size = n_samples // n_segment

        # seperate trials into group for each categories
        group_EEG_data, group_label = self.groupby(EEG_data, label)
        final_data = []
        final_label = []
        # for label_category in range(n_unique_categories):
        for idx in range(n_unique_categories):
            label_category = unique_labels[idx]
            new_artificial_data = []

            category_EEG_data = group_EEG_data[idx]

            category_n_trial = category_EEG_data.shape[0]
            if fix_trials == -1:
                max_augment_trials = category_n_trial * max_trials_mul
            else:
                max_augment_trials = fix_trials

            for i in range(max_augment_trials):
                artificial_EEG = np.zeros((n_channels, n_samples))
                temp_record = list()
                for seg_idx in range(n_segment):
                    # randomly pick a trial
                    pick_idx = np.random.randint(low=0, high=category_EEG_data.shape[0])
                    # if pick_idx not in temp_record
                    temp_record.append(pick_idx)
                    artificial_EEG[:, seg_idx * segment_size:(seg_idx + 1) * segment_size] = category_EEG_data[pick_idx,
                                                                                             :, seg_idx * segment_size:(
                                                                                                                                   seg_idx + 1) * segment_size]
                new_artificial_data.append(artificial_EEG)

            new_artificial_data = np.stack(new_artificial_data)
            new_label = np.ones(max_augment_trials) * label_category
            final_data.append(new_artificial_data)
            final_label.append(new_label)
        final_data = np.concatenate(final_data)
        final_label = np.concatenate(final_label)
        final_data, final_label = self.shuffle_data(final_data, final_label)
        return final_data, final_label

    def generate_artificial_data(self,method = "temporal_segment",n_segment = 4):
        #augment data for each subject
        data = self.data
        label = self.label

        n_subjects = len(data)
        n_classes = len(np.unique(label[0]))

        n_extra_trials = -1
        if self.total_fix_trials > -1:
            #assume that every suject has very similar number of trials. Generate extra data such that total trials are close to total_fix_trials
            total_trial = sum([subject_data.shape[0] for subject_data in self.data])
            n_extra_trials = (self.total_fix_trials-total_trial)//(n_subjects*n_classes)
        print("n extra trials for each subject : ",n_extra_trials)
        print(" total n classes : ",n_classes)
        update_data = []
        update_label = []

        for subject in range(n_subjects):
            subject_data = data[subject]
            subject_label = label[subject]
            if method == "temporal_segment":
                print("apply temporal segment data augmentation")
                artificial_data, artificial_label = self.data_augment_temporal(subject_data,subject_label,fix_trials=n_extra_trials,n_segment=n_segment)
            elif method =="temporal_segment_T_F":
                print("apply temporal segment T_F data augmentation")
                artificial_data, artificial_label = self.data_augmentation_temporal_STFT(subject_data,subject_label,fix_trials=n_extra_trials,n_segment=n_segment)
            else:
                print("apply spatial segment data augmentation for dataset {}".format(self.spatial_dataset_name))
                artificial_data, artificial_label = self.data_augment_spatial(subject_data,subject_label,fix_trials=n_extra_trials,dataset_name=self.spatial_dataset_name)


            new_subject_data = np.concatenate([subject_data,artificial_data])
            new_subject_label = np.concatenate([subject_label,artificial_label])

            new_subject_data = new_subject_data.astype(np.float32)
            # print("subject {} has new data size {}".format(subject,new_subject_data.shape))
            update_data.append(new_subject_data)
            update_label.append(new_subject_label)
        return update_data,update_label

def normalization(X):
    # assert len(X) == len(y)
    # Normalised, you could choose other normalisation strategy
    if len(X.shape)==3:
        #assume the data in format (trials,channels,samples)
        axis=1
    elif len(X.shape)==4:
        # assume the data in format (trials,filter,channels,samples)
        axis=2
    else:
        axis=-1
        raise ValueError("there is problem with data format")

    mean = np.mean(X,axis=axis,keepdims=True)
    # here normalise across channels as an example, unlike the in the sleep kit
    std = np.std(X, axis=axis, keepdims=True)
    X = (X - mean) / std
    return X
def dataset_norm(data,label):
    new_data = list()
    new_label = list()
    for subject_idx in range(len(data)):
        subject_data = data[subject_idx]
        subject_label = label[subject_idx]
        subject_data = normalization(subject_data)
        new_data.append(subject_data)
        new_label.append(subject_label)
    return new_data,new_label


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
def subjects_filterbank(data,filter,source,destination):
    new_data = list()
    for subject_idx in range(len(data)):
        subject_data = data[subject_idx]
        filter_data = filter(subject_data)
        subject_data = np.moveaxis(filter_data, source, destination)
        subject_data=subject_data.astype(np.float32)
        new_data.append(subject_data)
    return new_data