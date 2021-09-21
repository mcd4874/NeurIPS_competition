"""
William Duong
"""

import os.path as osp
import os
import errno
from .base_dataset import  DatasetBase,EEGDatum
from scipy.linalg import sqrtm, inv
from scipy import signal
import numpy as np
from collections import defaultdict



class MainDataBase(DatasetBase):
    pick_train_subjects = None
    pick_test_subjects = None
    pick_valid_subjects = None

    def __init__(self, cfg):
        self._n_domain = 0
        self.root = osp.abspath(osp.expanduser(cfg.DATAMANAGER.DATASET.ROOT))
        print("data root : ",self.root)
        self.dataset_dir = self.dataset_dir if not cfg.DATAMANAGER.DATASET.DIR else cfg.DATAMANAGER.DATASET.DIR
        self.file_name = self.file_name if not cfg.DATAMANAGER.DATASET.FILENAME else cfg.DATAMANAGER.DATASET.FILENAME
        self.cfg = cfg
        self._label_name_map = None
        print("root : ",self.root)
        print("dataset dir : ",self.dataset_dir)
        print("file name : ",self.file_name)
        data_path = osp.join(self.root,self.dataset_dir, self.file_name)
        if not osp.isfile(data_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), data_path)

        self.check_dataInfo()

        read_data = self._read_data(data_path)
        train, val, test = self.process_data_format(read_data)

        super().__init__(train_x=train, val=val, test=test)



    @property
    def data_domains(self):
        return self._n_domain

    @property
    def label_name_map(self):
        return self._label_name_map

    def _read_data(self,data_path):
        raise NotImplementedError

    def check_dataInfo(self):
        return


    def setup_within_subject_experiment(self, total_data, total_label, test_data, test_lbl):
        """
        Split the total data set  into k_folds. Each fold contains data from every subjects
        pick 1 fold to be valid data

        """
        folds = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.N_VALID_FOLDS
        valid_fold = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.CURRENT_VALID_FOLD
        train_data, train_label, valid_data, valid_label = self._pick_train_valid_same_set(total_data, total_label,
                                                                                           folds=folds,
                                                                                           valid_fold=valid_fold)
        self.pick_valid_subjects = self.pick_train_subjects.copy()
        return train_data, train_label, valid_data, valid_label, test_data, test_lbl

    def setup_cross_subject_experiment(self, total_data, total_label, test_data, test_lbl):
        """
        Split the total dataset into k folds. Each fold contains some subjects
        Pick 1 folds to be valid data
        """
        folds = self.cfg.DATASET.K_FOLD
        valid_fold = self.cfg.DATASET.VALID_FOLD
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


    def process_data_format(self, input_data):

        # data,test = input_data

        CROSS_SUBJECTS = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.CROSS_SUBJECTS
        WITHIN_SUBJECTS = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.WITHIN_SUBJECTS

        total_data, total_label,test_data, test_lbl = input_data


        if WITHIN_SUBJECTS:
            train_data, train_label, valid_data, valid_label, test_data, test_lbl = self.setup_within_subject_experiment(
                total_data, total_label, test_data, test_lbl)
        elif CROSS_SUBJECTS:
            train_data, train_label, valid_data, valid_label, test_data, test_lbl = self.setup_cross_subject_experiment(
                total_data, total_label, test_data, test_lbl)
        else:
            raise ValueError("need to specify to create train/valid for cross subjects or within subject experiments")

        """Data Augmentation"""

        """Create class weight for dataset"""

        """hardcode data normalization"""



        # assume the number of subjects represent number of domains
        self._n_domain = len(train_data)

        train_items = (train_data,train_label)
        valid_items = (valid_data,valid_label)
        test_items = (test_data,test_lbl)


        return train_items, valid_items, test_items


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


    def get_num_classes(self, data_source):
        "temporary fix"
        return None

    def get_label_classname_mapping(self, data_source):
        "temporary fix"
        return None


