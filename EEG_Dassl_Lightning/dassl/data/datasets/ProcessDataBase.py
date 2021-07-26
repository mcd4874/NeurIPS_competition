"""
William Duong
"""

import os.path as osp
import os
import errno
from .build import DATASET_REGISTRY
from .base_dataset import  DatasetBase,EEGDatum
from scipy.io import loadmat
from scipy.linalg import sqrtm, inv
from scipy import signal
import numpy as np
from collections import defaultdict


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






class ProcessDataBase(DatasetBase):
    pick_train_subjects = None
    pick_test_subjects = None
    pick_valid_subjects = None


    def __init__(self, cfg):
        # self.check_dataInfo()
        self._n_domain = 0
        self.domain_class_weight = None
        self.whole_class_weight = None
        print("original root : ",cfg.DATASET.ROOT)
        self.root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        # self.root = ''
        print("data root : ",self.root)
        self.dataset_dir = self.dataset_dir if not cfg.DATASET.DIR else cfg.DATASET.DIR
        self.file_name = self.file_name if not cfg.DATASET.FILENAME else cfg.DATASET.FILENAME
        self.cfg = cfg
        self._label_name_map = None
        # self.dataset_dir = osp.join(root, self.dataset_dir)

        data_path = osp.join(self.root,self.dataset_dir, self.file_name)

        if not osp.isfile(data_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), data_path)

        self.check_dataInfo()


        # total_data,total_label,test_data,test_lbl = self._read_data(data_path)
        # train, train_target, val, test = self.process_data_format((total_data, total_label), (test_data, test_lbl), cfg)
        read_data = self._read_data(data_path)
        train, train_target, val, test = self.process_data_format(read_data, cfg)

        if self.cfg.DISPLAY_INFO.DATASET:
            print("target domain : ", cfg.DATASET.TARGET_DOMAINS)



        super().__init__(train_x=train, val=val, test=test, train_u=train_target)



    @property
    def data_domains(self):
        return self._n_domain

    def _read_data(self,data_path):
        raise NotImplementedError

    def check_dataInfo(self):
        return

    def euclidean_alignment(self, x):
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
        if np.iscomplexobj(r_op):
            print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
                  "not, check data!!")
            r_op = np.real(r_op).astype(np.float32)
        elif not np.any(np.isfinite(r_op)):
            print("WARNING! Not finite values in R Matrix")

        results = np.matmul(r_op, x)
        return results

    def expand_data_dim(self,data):
        if isinstance(data,list):
            for idx in range(len(data)):
                new_data = np.expand_dims(data[idx], axis=1)
                data[idx] = new_data
            return data
        elif isinstance(data,np.ndarray):
            return np.expand_dims(data, axis=2)
        else:
            raise ValueError("the data format during the process section is not correct")

    def setup_within_subject_experiment(self, total_data, total_label, test_data, test_lbl, cfg):
        """
        Split the total data set  into k_folds. Each fold contains data from every subjects
        pick 1 fold to be valid data

        """
        folds = cfg.DATASET.K_FOLD
        valid_fold = cfg.DATASET.VALID_FOLD
        train_data, train_label, valid_data, valid_label = self._pick_train_valid_same_set(total_data, total_label,
                                                                                           folds=folds,
                                                                                           valid_fold=valid_fold)
        self.pick_valid_subjects = self.pick_train_subjects.copy()
        return train_data, train_label, valid_data, valid_label, test_data, test_lbl

    def setup_cross_subject_experiment(self, total_data, total_label, test_data, test_lbl, cfg):
        """
        Split the total dataset into k folds. Each fold contains some subjects
        Pick 1 folds to be valid data
        """
        folds = cfg.DATASET.K_FOLD
        valid_fold = cfg.DATASET.VALID_FOLD
        train_data, train_label, pick_train_subjects_idx, valid_data, valid_label, pick_valid_subjects_idx = self._pick_train_valid_cross_set(
            total_data, total_label,
            folds=folds,
            valid_fold=valid_fold)

        if self.pick_train_subjects is not None and len(self.pick_train_subjects) == (
                len(pick_train_subjects_idx) + len(pick_valid_subjects_idx)):
            self.pick_valid_subjects = [self.pick_train_subjects[idx] for idx in pick_valid_subjects_idx]
            self.pick_train_subjects = [self.pick_train_subjects[idx] for idx in pick_train_subjects_idx]

        return train_data, train_label, valid_data, valid_label, test_data, test_lbl

    def _pick_train_valid_cross_set(self, total_data, total_label, folds, valid_fold):
        if valid_fold > folds:
            raise ValueError("can not assign fold identity outside of total cv folds")

        total_subjects = np.arange(len(total_data))
        split_folds = [list(x) for x in np.array_split(total_subjects, folds)]
        pick_test_subjects_idx = split_folds[valid_fold - 1]
        pick_train_subjects_idx = []
        for i in range(folds):
            if i != valid_fold - 1:
                for subject in split_folds[i]:
                    pick_train_subjects_idx.append(subject)

        train_data = [total_data[train_subject] for train_subject in pick_train_subjects_idx]
        train_label = [total_label[train_subject] for train_subject in pick_train_subjects_idx]
        test_data = [total_data[test_subject] for test_subject in pick_test_subjects_idx]
        test_label = [total_label[test_subject] for test_subject in pick_test_subjects_idx]

        return train_data, train_label, pick_train_subjects_idx, test_data, test_label, pick_test_subjects_idx

    def generate_class_weight(self, label):
        """
        generate the weight ratio based on total labels of every subjects
        label : [subject_1,subject_2,..] and subject = (trials)
        """

        if isinstance(label, list):
            new_label = np.empty(0)
            for current_label in label:
                new_label = np.concatenate([new_label, current_label])
            total = new_label.shape[0]
            labels = np.unique(new_label)
            list_ratio = []
            # print("new form label ",)
            for current_label in labels:
                current_ratio = total / len(np.where(new_label == current_label)[0])
                list_ratio.append(current_ratio)
            return list_ratio

        elif isinstance(label, np.ndarray):
            if len(label.shape) == 2:
                # label shall have shape (subjects,trials)
                label = label.reshape(label.shape[0] * label.shape[1])
            # data need to be shape (trials)
            total = label.shape[0]
            labels = np.unique(label)
            list_ratio = []
            for current_label in labels:
                current_ratio = total / len(np.where(label == current_label)[0])
                list_ratio.append(current_ratio)
            return list_ratio
        else:
            raise ValueError("the data format during the process section is not correct")

    def generate_domain_class_weight(self, label):

        """
        assume the label has shape (subjects,trials)
        """
        domain_class_weight = defaultdict()
        for domain in range(len(label)):
            current_domain_class_weight = self.generate_class_weight(label[domain])
            domain_class_weight[domain] = current_domain_class_weight
        return domain_class_weight

    def process_data_format(self, input_data, cfg):

        # data,test = input_data

        CROSS_SUBJECTS = cfg.DATASET.CROSS_SUBJECTS
        WITHIN_SUBJECTS = cfg.DATASET.WITHIN_SUBJECTS

        total_data, total_label,test_data, test_lbl = input_data


        if WITHIN_SUBJECTS:
            train_data, train_label, valid_data, valid_label, test_data, test_lbl = self.setup_within_subject_experiment(
                total_data, total_label, test_data, test_lbl, cfg)
        elif CROSS_SUBJECTS:
            train_data, train_label, valid_data, valid_label, test_data, test_lbl = self.setup_cross_subject_experiment(
                total_data, total_label, test_data, test_lbl, cfg)
        else:
            raise ValueError("need to specify to create train/valid for cross subjects or within subject experiments")

        """Data Augmentation"""
        data_augmentation = self.cfg.DATASET.AUGMENTATION.NAME
        if data_augmentation != "":
            print("apply augmentation")
            MAX_TRIAL_MUL = self.cfg.DATASET.AUGMENTATION.PARAMS.MAX_TRIAL_MUL
            MAX_FIX_TRIAL = self.cfg.DATASET.AUGMENTATION.PARAMS.MAX_FIX_TRIAL
            N_SEGMENT = self.cfg.DATASET.AUGMENTATION.PARAMS.N_SEGMENT
            spatial_dataset_name = self.cfg.DATASET.AUGMENTATION.PARAMS.DATASET_NAME
            augmentation = DataAugmentation(train_data,train_label,max_trials_mul=MAX_TRIAL_MUL,total_fix_trials=MAX_FIX_TRIAL,spatial_dataset_name=spatial_dataset_name)
            train_data,train_label = augmentation.generate_artificial_data(data_augmentation,N_SEGMENT)


        """Create class weight for dataset"""
        if cfg.DATASET.DOMAIN_CLASS_WEIGHT:
            self.domain_class_weight = self.generate_domain_class_weight(train_label)
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            self.whole_class_weight = self.generate_class_weight(train_label)

        """hardcode data normalization"""
        no_transform = self.cfg.INPUT.NO_TRANSFORM
        transforms = self.cfg.INPUT.TRANSFORMS

        if not no_transform and len(transforms)>0:
            transform = transforms[0]
            print("apply {} for train/valid/test data".format(transform))
            train_data = self.transform_subjects(train_data,transform=transform)
            valid_data = self.transform_subjects(valid_data,transform=transform)
            test_data = self.transform_subjects(test_data,transform=transform)





        # assume the number of subjects represent number of domains
        self._n_domain = len(train_data)

        train_data = self.expand_data_dim(train_data)
        valid_data = self.expand_data_dim(valid_data)
        test_data = self.expand_data_dim(test_data)

        if self.cfg.DISPLAY_INFO.DATASET:
            self.print_dataset_info(train_data, train_label, valid_data, valid_label, test_data, test_lbl)

        train_items = self._generate_datasource(train_data, train_label,label_name_map=self._label_name_map)
        valid_items = self._generate_datasource(valid_data, valid_label,label_name_map=self._label_name_map)
        test_items = self._generate_datasource(test_data, test_lbl,label_name_map=self._label_name_map)
        train_target_items = test_items.copy()




        self.raw_test_data = test_data
        self.raw_test_label = test_lbl

        self._list_subject_test_items = [self._generate_datasource([test_data[subject_test_idx]], [test_lbl[subject_test_idx]],test_data=True,label_name_map=self._label_name_map) for subject_test_idx in range(len(test_data))]
        return train_items, train_target_items, valid_items, test_items

    def transform_subjects(self, subjects_data, transform = "z_transform",transform_func=None):
        def Z_normalize(EEG_data, axis=-1, eps=1e-8):
            """
            assume EEG_data has shape (trials,channels,samples)
            perform z_score normalize for each channel
            """
            mean = EEG_data.mean(axis, keepdims=True)
            std = EEG_data.std(axis, keepdims=True)
            return (EEG_data - mean) / (std + eps)
        def Z_normalize_1(EEG_data,eps=1e-8):
            """
             assume EEG_data has shape (trials,channels,samples)
             perform z_score normalize for each trial. Use one mean and one std
             """
            mean = EEG_data.mean((-2,-1), keepdims=True)
            std = EEG_data.std((-2,-1), keepdims=True)
            return (EEG_data - mean) / (std + eps)
        def min_max_normalize(EEG_data,eps=1e-8):
            min = EEG_data.min((-2,-1), keepdims=True)
            max = EEG_data.max((-2,-1), keepdims=True)
            return (EEG_data-min)/(max-min+eps)

        new_subjects_data = list()
        for idx in range(len(subjects_data)):
            subject_data = subjects_data[idx]
            if transform_func is None:
                print("apply {} to transform trial ".format(transform))
                if transform == "z_transform":
                    new_subject_data = Z_normalize(subject_data)
                elif transform =="min_max":
                    new_subject_data = min_max_normalize(subject_data)
                    print("some data : ",new_subject_data[:10])
                else:
                    new_subject_data = Z_normalize_1(subject_data)
            else:
                new_subject_data = transform_func(subject_data)
            new_subjects_data.append(new_subject_data)
        return new_subjects_data
    def get_raw_test_data(self):
        data = {
            "raw_test_data":self.raw_test_data,
            "raw_test_label":self.raw_test_label,
            "raw_subject_ids":self.pick_test_subjects
        }
        return data
    @property
    def list_subject_test(self):
        return self._list_subject_test_items

    @classmethod
    def _pick_train_valid_same_set(self, data, label, folds=4, valid_fold=1):
        if valid_fold > folds:
            raise ValueError("can not assign fold identity outside of total cv folds")

        train_data = list()
        train_label = list()
        valid_data = list()
        valid_label = list()
        for subject in range(len(data)):
            current_subject_data = data[subject]
            current_subject_label = label[subject]

            total_trials = len(current_subject_data)
            fold_trial = int(total_trials / folds)

            valid_mark_start = (valid_fold - 1) * fold_trial
            valid_mark_end = valid_fold * fold_trial

            current_train_data = np.concatenate(
                (current_subject_data[:valid_mark_start, :, :], current_subject_data[valid_mark_end:, :, :]))
            current_train_label = np.concatenate(
                (current_subject_label[:valid_mark_start], current_subject_label[valid_mark_end:]))

            current_valid_data = current_subject_data[valid_mark_start:valid_mark_end, :, :]
            current_valid_label = current_subject_label[valid_mark_start:valid_mark_end]
            # print("current subject id : ",subject)
            # print("pick train data label : ",current_train_label)
            # print("pick valid data label : ",current_valid_label)
            train_data.append(current_train_data)
            train_label.append(current_train_label)
            valid_data.append(current_valid_data)
            valid_label.append(current_valid_label)

        return train_data, train_label, valid_data, valid_label

    @classmethod
    def _leave_N_out(self, data, label, seed=None, num_subjects=1, given_subject_idx=None):

        """PICK valid num subjects out"""
        pick_valid_subjects_idx, pick_train_subjects_idx = self._pick_leave_N_out_ids(len(data), seed,
                                                                                      given_subject_idx, num_subjects)
        subjects = np.arange(data.shape[0])
        pick_train_subjects = subjects[pick_train_subjects_idx]
        pick_valid_subjects = subjects[pick_valid_subjects_idx]
        train_data = [data[train_subject] for train_subject in pick_train_subjects]
        train_label = [label[train_subject] for train_subject in pick_train_subjects]
        valid_data = [data[test_subject] for test_subject in pick_valid_subjects]
        valid_label = [label[test_subject] for test_subject in pick_valid_subjects]
        return train_data, train_label, pick_train_subjects, valid_data, valid_label, pick_valid_subjects

    @classmethod
    def _pick_leave_N_out_ids(self, total_subject, seed=None, given_subject_idx=None, num_subjects=1):
        if seed is None:
            np.random.choice(1)
        else:
            np.random.choice(seed)
        subjects_idx = np.arange(total_subject) if given_subject_idx is None else given_subject_idx
        pick_subjects_idx = np.random.choice(subjects_idx, num_subjects, replace=False)
        pick_subjects_idx = np.sort(pick_subjects_idx)
        remain_subjects_idx = subjects_idx[~np.isin(subjects_idx, pick_subjects_idx)]
        return pick_subjects_idx, remain_subjects_idx

    @classmethod
    def _generate_datasource(self,data, label, test_data=False,label_name_map= None):
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

    def print_dataset_info(self,train_data, train_label, valid_data, valid_label,test_data,test_label):
        # print("Train data info: ")
        print("train subjects : ",self.pick_train_subjects)
        for subject_idx in range(len(train_data)):
            print("Train subject {} has shape : {}, with range scale ({},{}) ".format(self.pick_train_subjects[subject_idx],train_data[subject_idx].shape,np.max(train_data[subject_idx]),np.min(train_data[subject_idx])))
        print("test subjects : ",self.pick_test_subjects)
        for subject_idx in range(len(test_data)):
            print("test subject {} has shape : {}, with range scale ({},{})  ".format(self.pick_test_subjects[subject_idx],test_data[subject_idx].shape,np.max(test_data[subject_idx]),np.min(test_data[subject_idx])))
        print("valid subjects : ",self.pick_valid_subjects)
        for subject_idx in range(len(valid_data)):
            print("valid subject {} has shape : {}, with range scale ({},{})  ".format(self.pick_valid_subjects[subject_idx],valid_data[subject_idx].shape,np.max(valid_data[subject_idx]),np.min(valid_data[subject_idx])))

        for test_subject_idx in range(len(test_data)):
            print("test subject idx : ",test_subject_idx)
            print("pick subject id : ",self.pick_test_subjects[test_subject_idx])
            print("curent test subject data shape : ",test_data[test_subject_idx].shape)
            print("curent test subject label shape : ",test_label[test_subject_idx].shape)

        if self.domain_class_weight is not None:
            print("Train data labels ratio info : ")
            for subject_idx in range(len(train_data)):
                current_subject_id = self.pick_train_subjects[subject_idx]
                subject_ratio = self.domain_class_weight[subject_idx]
                print("subject {} has labels ratio : {}".format(current_subject_id,subject_ratio))

        if self.whole_class_weight is not None:
            print("the labels ratio of whole dataset : {}".format(self.whole_class_weight))





