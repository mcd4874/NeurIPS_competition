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
from scipy.io import loadmat
import pandas as pd
from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.utils.tools import set_random_seed
from collections import defaultdict
from scipy.io import loadmat
import numpy as np
def reformat(data,label,meta_data):
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
            # print("current meta : ",subject_meta_data)
            # print("len meta : ",len(subject_meta_data))
            # print("len subject size : ",len(subject_data))
            # print("len label size : ",len(subject_label))
            start = end
    return new_data,new_label,new_meta_data


@DATASET_REGISTRY.register()
class MultiDataset(DatasetBase):
    pick_train_subjects = None
    pick_test_subjects = None
    pick_valid_subjects = None

    def __init__(self, cfg):
        self.cfg = cfg

        self._n_domain = 0
        self.root = osp.abspath(osp.expanduser(cfg.DATAMANAGER.DATASET.ROOT))
        print("data root : ",self.root)
        self.dataset_dir = self.dataset_dir if not cfg.DATAMANAGER.DATASET.DIR else cfg.DATAMANAGER.DATASET.DIR
        self.file_name = self.file_name if not cfg.DATAMANAGER.DATASET.FILENAME else cfg.DATAMANAGER.DATASET.FILENAME
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
        multi_dataset_u = (self.source_data_list,self.source_label_list)
        super().__init__(train_x=train, val=val, test=test,multi_dataset_u=multi_dataset_u)

    # def set_up(self):

    @property
    def data_domains(self):
        return self._n_domain

    @property
    def label_name_map(self):
        return self._label_name_map
    #
    # def _read_data(self,data_path):
    #     raise NotImplementedError

    def check_dataInfo(self):
        return

    def _read_data(self,data_path):
        """
        Process data from .mat file
        Re-implement this function to process new dataset
        Given file with whole data without specify test data and test label.
        Generate train data and test data with shape (1,subjects) and subject = (trials,channels,frequency)
        .mat data format shall be

        "dataset": [
            {
                "data":[trials,channels,times]
                "label": [1,trials],
                "meta_data": dataframe [subject,session,run],
                "dataset_name": "name"
            },

        ]


        target_data has format [(subject_trials, channels, frequency), (subject_trials, channels, frequency) , ..]. A list of numpy array
        target_label has format [(1,subject_trials) , (1,subject trials)]. A list of numpy array
        target_label_name_map has format
        """
        temp = loadmat(data_path)

        datasets = temp['datasets'][0]

        # target_dataset_name = 'BCI_IV'
        target_dataset_name=self.cfg.DATAMANAGER.DATASET.SETUP.TARGET_DATASET_NAME
        print("target datast : ",target_dataset_name)
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
            data,label,meta_data = reformat(data,label,meta_data)



            if dataset_name == target_dataset_name:
                if 'r_op_list' in list(dataset.dtype.names):
                    self.r_op_list = np.array(dataset['r_op_list']).astype(np.float32)
                else:
                    self.r_op_list = None
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


        self.source_data_list = list_source_data
        self.source_label_list = list_source_label
        self.source_meta_data_list = list_source_meta_data

        return target_data,target_label,target_meta_data

    def generate_test_data(self,data,label,subject_idx,current_test_fold,n_test_folds,shuffle_test_seed=-1,within_subject_split=True,train_valid_ratio=0.3):
        if within_subject_split:
            train_valid_data,train_valid_label, test_data, test_label= self.split_dataset_same_subjects(data,label,current_fold=current_test_fold,n_folds=n_test_folds,split_ratio=train_valid_ratio,shuffle_trial_seed=shuffle_test_seed)
            train_valid_subject_idx = subject_idx
            test_subject_idx = subject_idx
        return train_valid_data,train_valid_label,train_valid_subject_idx, test_data, test_label,test_subject_idx

    def generate_valid_data(self,data,label,subject_idx,current_valid_fold,n_valid_folds,shuffle_valid_seed=-1,within_subject_split=True,train_ratio=0.8):
        if within_subject_split:
            train_data,train_label, valid_data, valid_label= self.split_dataset_same_subjects(data,label,current_fold=current_valid_fold,n_folds=n_valid_folds,split_ratio=train_ratio,shuffle_trial_seed=shuffle_valid_seed)
            train_subject_idx = subject_idx
            valid_subject_idx = subject_idx
        return train_data,train_label,train_subject_idx, valid_data, valid_label, valid_subject_idx

    def process_data_format(self,target_dataset):
        target_data, target_label, target_meta_data = target_dataset

        # print("size of target dataset : ",len(target_data))

        #set up shuffle data fold
        current_shuffle_fold = self.cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.CURRENT_SHUFFLE_FOLD
        shuffle_seeds = self.cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_SEED
        if len(shuffle_seeds) == 0:
            # no shuffle
            # shuffle_seed = -1
            available_subject_ids = np.arange(len(target_data))
        else:
            #shuffle order of subjects
            shuffle_seed = shuffle_seeds[current_shuffle_fold - 1]
            np.random.seed(shuffle_seed)
            available_subject_ids = np.random.permutation(len(target_data))

        # #set up increment experiment subject available
        START_NUM_TRAIN_SUGJECT = self.cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_NUM_TRAIN_SUGJECT  # number of start out subjects
        assert START_NUM_TRAIN_SUGJECT > 0
        INCREMENT_UPDATE = self.cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.INCREMENT_TRAIN_SUGJECT  # subjects increment
        CURRENT_INCREMENT_FOLD = self.cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.CURRENT_INCREMENT_FOLD
        NUM_TRAIN_SUBJECT = START_NUM_TRAIN_SUGJECT + INCREMENT_UPDATE * (CURRENT_INCREMENT_FOLD - 1)

        # print("availbale subject : ",available_subject_ids)
        # print("num train subject : ",NUM_TRAIN_SUBJECT)
        assert len(available_subject_ids) >= NUM_TRAIN_SUBJECT
        # print("total number of available target subject : ", len(available_subject_ids))
        # print("Num train subject : ", NUM_TRAIN_SUBJECT)
        use_target_subjects = available_subject_ids[:NUM_TRAIN_SUBJECT]

        use_target_data = [target_data[subject] for subject in use_target_subjects]
        use_target_label = [target_label[subject] for subject in use_target_subjects]
        # print("total subject meta data : ",len(target_meta_data))
        use_target_meta_data = [target_meta_data[subject] for subject in use_target_subjects]

        # train_valid_ratio = 0.3
        # data,test = input_data

        #generate test data
        train_valid_ratio= self.cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TRAIN_VALID_DATA_RATIO
        within_subject_test_split = self.cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.WITHIN_SUBJECTS
        n_test_folds = self.cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.N_TEST_FOLDS
        current_test_fold = self.cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.CURRENT_TEST_FOLD

        test_same_as_valid = self.cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.SAME_AS_VALID

        test_fold_seeds = np.random.choice(100, n_test_folds, replace=False)
        test_fold_seed = test_fold_seeds[current_test_fold - 1]

        if not test_same_as_valid:
            train_valid_data,train_valid_label, train_valid_subject_idx, test_data, test_label, test_subject_idx = self.generate_test_data(use_target_data,use_target_label,use_target_subjects,current_test_fold,n_test_folds,shuffle_test_seed=test_fold_seed,within_subject_split=within_subject_test_split,train_valid_ratio=train_valid_ratio)
        else:
            train_valid_data = use_target_data
            train_valid_label = use_target_label
            train_valid_subject_idx = use_target_subjects



        # generate train and valid data
        train_ratio = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.TRAIN_DATA_RATIO
        within_subject_valid_split = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.WITHIN_SUBJECTS
        n_valid_folds = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.N_VALID_FOLDS
        current_valid_fold = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.CURRENT_VALID_FOLD

        train_data, train_label, train_subject_idx, valid_data, valid_label, valid_subject_idx = self.generate_valid_data(train_valid_data,train_valid_label,train_valid_subject_idx,current_valid_fold,n_valid_folds,shuffle_valid_seed=-1,within_subject_split=within_subject_valid_split,train_ratio=train_ratio)

        if test_same_as_valid:
            test_data = valid_data.copy()
            test_label = valid_label.copy()
            test_subject_idx = valid_subject_idx.copy()

        self.pick_train_subjects = train_subject_idx
        self.pick_valid_subjects = valid_subject_idx
        self.pick_test_subjects = test_subject_idx

        # assume the number of subjects represent number of domains
        self._n_domain = len(train_data)

        train_items = (train_data, train_label)
        valid_items = (valid_data, valid_label)
        test_items = (test_data, test_label)

        print("label before split data subject 1 : ",use_target_label[0])
        print("label train split data subject 1 : ",train_label[0])
        print("label valid split data subject 1 : ",valid_label[0])
        print("label test split data subject 1 : ",test_label[0])

        return train_items, valid_items, test_items


    @property
    def list_subject_test(self):
        return self._list_subject_test_items

    def split_dataset_same_subjects(self,data,label,current_fold,n_folds,split_ratio=0.3,shuffle_trial_seed=-1):
        if split_ratio < 0.0:
            train_data, train_label, test_data, test_label = self.fold_split_data(data,label,n_folds=n_folds,current_fold=current_fold)
        else:
            train_data, train_label, test_data, test_label = self.shuffle_split_data(data,labeo,split_ratio=split_ratio,shuffle_trial_seed=shuffle_trial_seed)
        return train_data, train_label, test_data, test_label
    def shuffle_split_data(self,data,label,split_ratio=0.3,shuffle_trial_seed=-1):
        train_data = list()
        train_label = list()

        test_data = list()
        test_label = list()

        for subject in range(len(data)):
            current_subject_data = data[subject]
            current_subject_label = label[subject]

            total_trials = len(current_subject_data)
            trials_index = np.arange(total_trials)

            # shuffle trial
            if shuffle_trial_seed > -1:
                np.random.seed(shuffle_trial_seed)
                np.random.shuffle(trials_index)

            current_subject_data = current_subject_data[trials_index]
            current_subject_label = current_subject_label[trials_index]

            split_mark = int(split_ratio * total_trials)
            subject_train_data = current_subject_data[:split_mark, :, :]
            subject_train_label = current_subject_label[:split_mark, ]

            subject_test_data = current_subject_data[split_mark:, :, :]
            subject_test_label = current_subject_label[split_mark:, ]

            train_data.append(subject_train_data)
            train_label.append(subject_train_label)

            test_data.append(subject_test_data)
            test_label.append(subject_test_label)

        return train_data, train_label, test_data, test_label

    def fold_split_data(self, data, label, n_folds=4, current_fold=1):
        if current_fold > n_folds:
            raise ValueError("can not assign fold identity outside of total cv folds")
        train_data = list()
        train_label = list()
        valid_data = list()
        valid_label = list()
        for subject in range(len(data)):
            current_subject_data = data[subject]
            current_subject_label = label[subject]

            total_trials = len(current_subject_data)
            fold_trial = int(total_trials / n_folds)

            valid_mark_start = (current_fold - 1) * fold_trial
            valid_mark_end = current_fold * fold_trial

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

    def get_num_classes(self, data_source):
        "temporary fix"
        return None

    def get_label_classname_mapping(self, data_source):
        "temporary fix"
        return None


