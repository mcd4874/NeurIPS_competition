"""
William DUong
"""

import os.path as osp
import os
import errno
from .build import DATASET_REGISTRY
from .base_dataset import Datum, DatasetBase,EEGDatum
from scipy.io import loadmat
import numpy as np
from collections import defaultdict




class ProcessDataBase(DatasetBase):
    dataset_dir = None
    file_name = None
    def __init__(self, cfg):
        # self.check_dataInfo()
        self._n_domain = 0
        self.domain_class_weight = None
        self.whole_class_weight = None
        self.root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = self.dataset_dir if not cfg.DATASET.DIR else cfg.DATASET.DIR
        self.file_name = self.file_name if not cfg.DATASET.FILENAME else cfg.DATASET.FILENAME
        self.cfg = cfg
        # self.dataset_dir = osp.join(root, self.dataset_dir)

        data_path = osp.join(self.root,self.dataset_dir, self.file_name)

        if not osp.isfile(data_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), data_path)

        self.check_dataInfo()


        total_data,total_label,test_data,test_lbl = self._read_data(data_path)
        train, train_target, val, test = self.process_data_format((total_data,total_label),(test_data,test_lbl),cfg)



        print("target domain : ", cfg.DATASET.TARGET_DOMAINS)



        super().__init__(train_x=train, val=val, test=test, train_u=train_target)



    @property
    def data_domains(self):
        return self._n_domain

    def _read_data(self,data_path):
        raise NotImplementedError

    def check_dataInfo(self):
        return

    # def _read_data(self,data_path):
    #     """
    #     Process data from .mat file
    #     Re-implement this function to process new dataset
    #     Generate train data and test data with shape (subjects,trials,channels,frequency)
    #     """
    #     temp = loadmat(data_path)
    #     total_data = []
    #     total_label = []
    #     print(temp.keys())
    #     for idx in range(len(temp['data'][0])):
    #         total_data.append(temp['data'][0][idx])
    #         total_label.append(temp['labels'][0][idx])
    #     total_data = np.array(total_data)  # (subjects,trials,channels,frequency)
    #     total_label = np.array(total_label)
    #     total_label = np.squeeze(total_label)
    #     total_label = total_label.astype(int)
    #
    #     total_test_data = []
    #     total_test_label = []
    #     for idx in range(len(temp['testdata'][0])):
    #         total_test_data.append(temp['testdata'][0][idx])
    #         total_test_label.append(temp['testlabels'][0][idx])
    #
    #     # test_data = np.array(temp['testdata'])  # shape (trials,channels,frequency
    #     # test_lbl = np.array(temp['testlabels'])  # trial,)
    #     # test_lbl = np.squeeze(test_lbl)
    #
    #     test_data = np.array(total_test_data)  # (subjects,trials,channels,frequency)
    #     test_lbl = np.array(total_test_label)
    #     test_lbl = np.squeeze(test_lbl)
    #     test_lbl = test_lbl.astype(int)
    #
    #     print("BCI_IV data shape : ", total_data.shape)
    #     print("BCI_IV test shape : ", test_data.shape)
    #
    #     return [total_data,total_label,test_data,test_lbl]

    # def setup_within_subject_experiment(self,total_data,total_label,test_data,test_lbl,cfg):
    #     folds = cfg.DATASET.K_FOLD
    #     valid_fold = cfg.DATASET.VALID_FOLD
    #     train_data, train_label, valid_data, valid_label = self._pick_train_valid_same_set(total_data, total_label,
    #                                                                                  folds=folds,
    #                                                                                  valid_fold=valid_fold)
    #     return train_data, train_label, valid_data, valid_label,test_data,test_lbl

    # def setup_cross_subject_experiment(self,total_data,total_label,test_data,test_lbl,cfg):
    #     cross_subject_seed = cfg.DATASET.RANDOM_SEED
    #
    #     pick_data_subject_ids, pick_test_subject_ids = self._pick_leave_N_out_ids(total_subject=total_data.shape[0],
    #                                                                         seed=cross_subject_seed,
    #                                                                         given_subject_idx=None, num_subjects=3)
    #     # use the provided train subjects and target subjects id
    #     if len(cfg.DATASET.SOURCE_DOMAINS) > 0 and len(cfg.DATASET.TARGET_DOMAINS) > 0:
    #         pick_data_subject_ids = cfg.DATASET.SOURCE_DOMAINS
    #         pick_test_subject_ids = cfg.DATASET.TARGET_DOMAINS
    #
    #     train_data = total_data[pick_data_subject_ids,]
    #     train_label = total_label[pick_data_subject_ids,]
    #     valid_data = test_data[pick_data_subject_ids,]
    #     valid_label = test_lbl[pick_data_subject_ids,]
    #     test_data = np.concatenate((total_data[pick_test_subject_ids,], test_data[pick_test_subject_ids,]), axis=1)
    #     test_lbl = np.concatenate((total_label[pick_test_subject_ids,], test_lbl[pick_test_subject_ids,]), axis=1)
    #     print("Pick subject to trian/valid : ", pick_data_subject_ids)
    #     print("Pick subject to test : ", pick_test_subject_ids)
    #     print("Train data, valid data, test data shape : ", (train_data.shape, valid_data.shape, test_data.shape))
    #     print("Train label, valid label, test label shape : ", (train_label.shape, valid_label.shape, test_lbl.shape))
    #     return train_data, train_label, valid_data, valid_label,test_data,test_lbl

    def setup_within_subject_experiment(self,total_data,total_label,test_data,test_lbl,cfg):
        """
        Split the total data set  into k_folds. Each fold contains data from every subjects
        pick 1 fold to be valid data

        """
        folds = cfg.DATASET.K_FOLD
        valid_fold = cfg.DATASET.VALID_FOLD
        train_data, train_label, valid_data, valid_label = self._pick_train_valid_same_set(total_data, total_label,
                                                                                     folds=folds,
                                                                                     valid_fold=valid_fold)
        print("train data within subjects shape : {} from k={} split".format(train_data.shape, folds))
        print("valid data within subjects shape : {} from k={} split".format(valid_data.shape, folds))
        return train_data, train_label, valid_data, valid_label,test_data,test_lbl

    def setup_cross_subject_experiment(self,total_data,total_label,test_data,test_lbl,cfg):
        """
        Split the total dataset into k folds. Each fold contains some subjects
        Pick 1 folds to be valid data
        """
        folds = cfg.DATASET.K_FOLD
        valid_fold = cfg.DATASET.VALID_FOLD
        train_data, train_label, valid_data, valid_label = self._pick_train_valid_cross_set(total_data,total_label,folds=folds,valid_fold=valid_fold)
        return train_data, train_label, valid_data, valid_label, test_data, test_lbl

    def _pick_train_valid_cross_set(self, total_data, total_label, folds, valid_fold):
        if valid_fold > folds:
            raise ValueError("can not assign fold identity outside of total cv folds")

        total_subjects = np.arange(total_data.shape[0])
        # total_subjects = [i for i in range(total_data.shape[0])]

        split_folds = [list(x) for x in np.array_split(total_subjects, folds)]
        pick_valid_subjects = split_folds[valid_fold - 1]
        pick_train_subjects = []
        for i in range(folds):
            if i != valid_fold - 1:
                for subject in split_folds[i]:
                    pick_train_subjects.append(subject)
        # subject_train_folds = for subject in [ ]

        # print("train subjects : {} from k={} split".format(pick_valid_subjects, folds))
        # print("valid subjects : {} from k={} split".format(pick_valid_subjects, folds))

        valid_data = total_data[pick_valid_subjects,]
        valid_label = total_label[pick_valid_subjects,]
        train_data = total_data[pick_train_subjects,]
        train_label = total_label[pick_train_subjects,]

        return train_data, train_label,pick_train_subjects, valid_data, valid_label,pick_valid_subjects

        # valid_mark_start = (valid_fold - 1) * fold_trial
        # valid_mark_end = valid_fold * fold_trial
        #
        # train_data = np.concatenate((data[:, :valid_mark_start, :, :], data[:, valid_mark_end:, :, :]), axis=1)
        # train_label = np.concatenate((label[:, :valid_mark_start], label[:, valid_mark_end:]), axis=1)
        #
        # valid_data = data[:, valid_mark_start:valid_mark_end, :, :]
        # valid_label = label[:, valid_mark_start:valid_mark_end]
        #
        #
        # # if len(total_subjects)%folds == 0:
        #     train_folds = [i for i in range(1,folds+1) if i !=valid_fold]
        #     subject_split_folds = np.split(total_subjects,folds)
        #     print("subject splits : ",subject_split_folds)
        #
        #     validation_subject_fold = subject_split_folds[valid_fold-1]
        #     train_subject_fold = np.concatenate(subject_split_folds[train_folds])
        #
        #
        #
        # "still need to implement"
        # return None,None,None,None

    def generate_class_weight(self,label):
        if len(label.shape) == 2:
            #label shall have shape (subjects,trials)
            label = label.reshape(label.shape[0] * label.shape[1])
        #data need to be shape (trials)
        total = label.shape[0]
        labels = np.unique(label)
        list_ratio = []
        for current_label in labels:
            current_ratio = total / len(np.where(label == current_label)[0])
            list_ratio.append(current_ratio)
        return list_ratio

    def generate_domain_class_weight(self,label):

        """
        assume the label has shape (subjects,trials)
        """
        if len(label.shape) != 2:
            raise ValueError("domain labels does not have correct data format")
        domain_class_weight = defaultdict()
        for domain in range(label.shape[0]):
            current_domain_class_weight = self.generate_class_weight(label[domain])
            domain_class_weight[domain] = current_domain_class_weight
        return domain_class_weight

    # def _expand_data_dim(self,data):
    #     i

    def process_data_format(self, data,test,cfg):

        CROSS_SUBJECTS = cfg.DATASET.CROSS_SUBJECTS
        WITHIN_SUBJECTS = cfg.DATASET.WITHIN_SUBJECTS

        total_data,total_label = data
        test_data,test_lbl = test

        if WITHIN_SUBJECTS:
            train_data, train_label, valid_data, valid_label,test_data,test_lbl = self.setup_within_subject_experiment(total_data,total_label,test_data,test_lbl,cfg)
        elif CROSS_SUBJECTS:
            train_data, train_label, valid_data, valid_label, test_data, test_lbl = self.setup_cross_subject_experiment(total_data,total_label,test_data,test_lbl,cfg)
        else:
            raise ValueError("need to specify to create train/valid for cross subjects or within subject experiments")
        """Create class weight for dataset"""

        if cfg.DATASET.DOMAIN_CLASS_WEIGHT:
            self.domain_class_weight =self.generate_domain_class_weight(train_label)
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            self.whole_class_weight = self.generate_class_weight(train_label)

        #assume the number of subjects represent number of domains
        self._n_domain = train_data.shape[0]

        train_data = np.expand_dims(train_data,axis=2)
        valid_data = np.expand_dims(valid_data,axis=2)
        test_data = np.expand_dims(test_data, axis=2)

        train_items = self._generate_datasource(train_data,train_label)
        valid_items = self._generate_datasource(valid_data,valid_label)
        test_items = self._generate_datasource(test_data,test_lbl)
        train_target_items = test_items.copy()
        return train_items,train_target_items,valid_items,test_items

    @classmethod
    def _pick_train_valid_same_set(self,data, label, folds=4, valid_fold=1):
        if valid_fold > folds:
            print("can not assign fold identity outside of total cv folds")
            return
        total_trials = data.shape[1]
        fold_trial = int(total_trials / folds)

        valid_mark_start = (valid_fold - 1) * fold_trial
        valid_mark_end = valid_fold * fold_trial
        # print("valid mark start : ", valid_mark_start)
        # print("valid_mark_end : ", valid_mark_end)

        train_data = np.concatenate((data[:, :valid_mark_start, :, :], data[:, valid_mark_end:, :, :]), axis=1)
        train_label = np.concatenate((label[:, :valid_mark_start], label[:, valid_mark_end:]), axis=1)

        valid_data = data[:, valid_mark_start:valid_mark_end, :, :]
        valid_label = label[:, valid_mark_start:valid_mark_end]

        # print("train data shape : ", train_data.shape)
        # print("valid data shape : ", valid_data.shape)
        return train_data, train_label, valid_data, valid_label

    # @classmethod
    def _leave_N_out(self,data, label, seed=None, num_subjects=1, given_subject_idx=None):

        """PICK valid num subjects out"""

        pick_valid_subjects_idx,pick_train_subjects_idx = self._pick_leave_N_out_ids(data.shape[0], seed, given_subject_idx,num_subjects)
        subjects = np.arange(data.shape[0])
        pick_train_subjects = subjects[pick_train_subjects_idx]
        pick_valid_subjects = subjects[pick_valid_subjects_idx]
        train_data = data[pick_train_subjects_idx,]
        valid_data = data[pick_valid_subjects_idx,]
        train_label = label[pick_train_subjects_idx,]
        valid_label = label[pick_valid_subjects_idx,]

        return train_data, train_label, pick_train_subjects, valid_data, valid_label, pick_valid_subjects

    @classmethod
    def _pick_leave_N_out_ids(self,total_subject, seed=None, given_subject_idx=None, num_subjects=1):
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
    def _generate_datasource(self,data, label, test_data=False):
        items = []
        total_subjects = 1
        if not test_data:
            total_subjects = len(data)
        for subject in range(total_subjects):
            current_subject_data = data[subject]
            current_subject_label = label[subject]
            domain = subject
            for i in range(current_subject_data.shape[0]):
                item = EEGDatum(eeg_data=current_subject_data[i], label=int(current_subject_label[i]), domain=domain)
                items.append(item)
        return items

