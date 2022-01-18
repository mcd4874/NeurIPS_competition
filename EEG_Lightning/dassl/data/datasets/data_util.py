from scipy.linalg import sqrtm, inv
from scipy import signal
import numpy as np
from collections import defaultdict
from dassl.data.datasets.base_dataset import EEGDatum

# scipy.seed

def relabel_target(l):
    if l == 0: return 0
    elif l == 1: return 1
    else: return 2
#
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
        #r = np.matmul(data, data.transpose((0, 2, 1))).mean(0)
        #calculate covariance matrix of each trial
        r = 0
        for trial in data:
            cov = np.cov(trial, rowvar=True)
            r += cov

        r = r/data.shape[0]
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
        print("r_op : ", r_op)
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
        # r = np.matmul(data, data.transpose((0, 2, 1))).mean(0)
        #calculate covariance matrix of each trial
        r = 0
        for trial in data:
            cov = np.cov(trial, rowvar=True)
            r += cov

        r = r/data.shape[0]
        if np.iscomplexobj(r):
            print("covariance matrix problem")
        if np.iscomplexobj(sqrtm(r)):
            print("covariance matrix problem sqrt")

        r_op = sqrtm(r)
        return r_op


# class EuclideanAlignment:
#     """
#     convert trials of each subject to a new format with Euclidean Alignment technique
#     https://arxiv.org/pdf/1808.05464.pdf
#     """
#     def __init__(self,list_r_op=None,subject_ids=None):
#         self.list_r_op = list_r_op
#         if subject_ids is not None:
#             update_list_r_op = [self.list_r_op[subject_id] for subject_id in subject_ids]
#             print("only use r-op for subjects {}".format(subject_ids))
#             self.list_r_op = update_list_r_op
#     def calculate_r_op(self,data):
#         ##data shape (trials,channels, sample)
#         # if
#         assert len(data.shape) == 3
#         r = np.matmul(data, data.transpose((0, 2, 1))).mean(0)
#         if np.iscomplexobj(r):
#             print("covariance matrix problem")
#         if np.iscomplexobj(sqrtm(r)):
#             print("covariance matrix problem sqrt")
#         r_op = inv(sqrtm(r))
#         # print("r_op shape : ", r_op.shape)
#         # print("data shape : ",x.shape)
#         # print("r_op : ", r_op)
#         if np.iscomplexobj(r_op):
#             print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
#                   "not, check data!!")
#             r_op = np.real(r_op).astype(np.float32)
#         elif not np.any(np.isfinite(r_op)):
#             print("WARNING! Not finite values in R Matrix")
#         return r_op
#     def convert_trials(self,data,r_op):
#         results = np.matmul(r_op, data)
#         return results
#     def generate_list_r_op(self,subjects_data):
#         list_r_op = list()
#         for subject_idx in range(len(subjects_data)):
#             subject_data = subjects_data[subject_idx]
#             if len(subjects_data.shape) == 3:
#                 r_op = self.calculate_r_op(subject_data)
#             elif len(subjects_data.shape) == 4:
#                 for filter_idx in range(len())
#             list_r_op.append(r_op)
#         return list_r_op
#     def convert_subjects_data_with_EA(self,subjects_data):
#         #calculate r_op for each subject
#         if self.list_r_op is not None:
#             assert len(self.list_r_op) == len(subjects_data)
#             ##check if list r_op has same shape as data
#             assert len(self.list_r_op[0].shape) == len(subjects_data.shape)
#             print("use exist r_op")
#         else:
#             print("generate new r_op")
#             self.list_r_op = self.generate_list_r_op(subjects_data)
#         new_data = list()
#         # print("size list r : ",len(self.list_r_op))
#         # print("subject dat size : ",len(subjects_data))
#         for subject_idx in range(len(subjects_data)):
#             subject_data = subjects_data[subject_idx]
#             r_op = self.list_r_op[subject_idx]
#             subject_data = self.convert_trials(subject_data,r_op)
#             new_data.append(subject_data)
#         return new_data



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
        # r = np.matmul(data, data.transpose((0, 2, 1))).mean(0)

        #calculate covariance matrix of each trial
        # list_cov = list()
        r = 0
        for trial in data:
            cov = np.cov(trial, rowvar=True)
            r += cov

        r = r/data.shape[0]

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
            r_op = np.real(r_op).astype(np.float64)
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
            # print("current trial : ",current_subject_label[i].shape)
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

def normalization_channels(X):
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
    # print("axis to norm : ",axis)
    # print("data shape : ",X.shape)
    mean = np.mean(X,axis=axis,keepdims=True)
    # here normalise across channels as an example, unlike the in the sleep kit
    std = np.std(X, axis=axis, keepdims=True)
    # print("current mean {} and std {}".format(mean,std))
    # print("X-mean : ",(X-mean))
    X = (X - mean) / std
    # print("final norm : ",X)
    return X

def normalization_time(X):
    # assert len(X) == len(y)
    # Normalised, you could choose other normalisation strategy
    if len(X.shape)==3:
        #assume the data in format (trials,channels,samples)
        axis=2
    elif len(X.shape)==4:
        # assume the data in format (trials,filter,channels,samples)
        axis=3
    else:
        axis=-1
        raise ValueError("there is problem with data format")

    mean = np.mean(X,axis=axis,keepdims=True)
    # here normalise across channels as an example, unlike the in the sleep kit
    std = np.std(X, axis=axis, keepdims=True)
    X = (X - mean) / std
    return X

def dataset_norm(data,norm_channels = True):
    new_data = list()
    # new_label = list()
    for subject_idx in range(len(data)):
        subject_data = data[subject_idx]
        if norm_channels:
            subject_data = normalization_channels(subject_data)
        else:
            subject_data =  normalization_time(subject_data)
        new_data.append(subject_data)
    return new_data


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