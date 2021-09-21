import os.path as osp

from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase,EEGDatum
from dassl.data.datasets.ProcessDataBase import ProcessDataBase

from scipy.io import loadmat
import numpy as np




@DATASET_REGISTRY.register()
class GENERAL_WHOLE_DATASET(ProcessDataBase):
    # dataset_dir = 'KAGGLE_BCI'
    # file_name = 'KaggleBCI.mat'
    # domains = [0,3,4,5,6,7,8]
    def __init__(self, cfg):
        super().__init__(cfg)

        # assum that number of subjects represent the domain

    # def check_dataInfo(self):
    #     assert  self.cfg.DATASET.TEST_K_FOLDS and

    def _read_data(self,data_path):
        """
        Process data from .mat file
        Re-implement this function to process new dataset
        Given file with whole data without specify test data and test label.
        Generate train data and test data with shape (subjects,trials,channels,frequency)
        .mat data format shall be

        "total_data":total_data,
        "total_label":total_label,

        """
        temp = loadmat(data_path)

        total_data = temp['total_data']
        total_label = temp['total_label']
        total_label = np.array(total_label)
        total_label = np.squeeze(total_label)
        total_label = total_label.astype(int)

        # available_subject_ids = np.arange(total_label.shape[0])

        available_subject_ids = [ i for i in range(total_label.shape[0])]
        K_FOLD_TEST = self.cfg.DATASET.K_FOLD_TEST
        VALID_FOLD_TEST = self.cfg.DATASET.VALID_FOLD_TEST

        NUM_TEST_SUBJECTS = self.cfg.DATASET.TEST_NUM_SUBJECTS


        if self.cfg.DATASET.TEST_K_FOLDS and self.cfg.DATASET.K_FOLD_TEST > 1 and len(self.cfg.DATASET.TEST_RANDOM_SEEDS) == 0:
            print("case 1")
            train_data, train_label, pick_train_subjects,test_data, test_label,pick_test_subjects = self._pick_train_valid_cross_set(total_data,total_label,folds=K_FOLD_TEST,valid_fold=VALID_FOLD_TEST)
        elif self.cfg.DATASET.TEST_NUM_SUBJECTS > 0 and len(self.cfg.DATASET.TEST_RANDOM_SEEDS) == self.cfg.DATASET.K_FOLD_TEST:
            print("case 2")
            #Randomly select N subjects to be test subjects and define constant random with seed.
            #make sure that the number of random seeds equal K_FOLD_TEST

            CURRENT_TEST_RANDOM_SEED = self.cfg.DATASET.TEST_RANDOM_SEEDS[VALID_FOLD_TEST]
            print("current seed : ",CURRENT_TEST_RANDOM_SEED)
            train_data, train_label, pick_train_subjects,test_data, test_label,pick_test_subjects = self._leave_N_out(total_data,total_label,seed=CURRENT_TEST_RANDOM_SEED,num_subjects=NUM_TEST_SUBJECTS)

        elif len(self.cfg.DATASET.TARGET_DOMAINS) > 0:
            print("case 3")
            # Provide a list of target subjects for test set
            # pick_data_subject_ids = cfg.DATASET.SOURCE_DOMAINS
            pick_test_subjects = list(self.cfg.DATASET.TARGET_DOMAINS)
            # pick_test_subject_ids = np.sort(np.array(pick_test_subject_ids))
            # if np.isin(available_subject_ids,pick_test_subject_ids):
            # print("test subjects : ",pick_test_subjects)
            # print("available : ",available_subject_ids)
            if (set(available_subject_ids) & set(pick_test_subjects))== set(pick_test_subjects):
            # if pick_test_subject_ids in available_subject_ids:
                test_data = total_data[pick_test_subjects,]
                test_label = total_label[pick_test_subjects,]
                pick_train_subjects = [i for i in available_subject_ids if i not in pick_test_subjects]
                train_data = total_data[pick_train_subjects,]
                train_label = total_label[pick_train_subjects,]
            else:
                raise ValueError("given subject index not available in the dataset")
        elif NUM_TEST_SUBJECTS> 0:
            print("case 4")
            # random select target subjects for test set
            train_data, train_label, pick_train_subjects, test_data, test_label, pick_test_subjects = self._leave_N_out(total_data,total_label,num_subjects=NUM_TEST_SUBJECTS)
        else:
            raise ValueError("Need to check the .yaml configuration for how to split the train/test data")

        print("available subjects {} ".format(available_subject_ids))
        print("train subjects : {} ".format(pick_train_subjects))
        print("test subjects : {} ".format(pick_test_subjects))
        print("train data shape : {}  | train label shape : {}".format(train_data.shape,train_label.shape))
        print("test data shape : {}  | test label shape : {}".format(test_data.shape, test_label.shape))

        return [train_data,train_label,test_data,test_label]




