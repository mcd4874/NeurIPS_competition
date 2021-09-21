import os.path as osp

from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.ProcessDataBase import ProcessDataBase
from collections import defaultdict
from scipy.io import loadmat
import numpy as np



@DATASET_REGISTRY.register()
class GENERAL_WHOLE_DATASET(ProcessDataBase):
    def __init__(self, cfg):
        super().__init__(cfg)



    def _read_data(self,data_path):
        """
        Process data from .mat file
        Re-implement this function to process new dataset
        Given file with whole data without specify test data and test label.
        Generate train data and test data with shape (1,subjects) and subject = (trials,channels,frequency)
        .mat data format shall be

        "total_data":total_data,
        "total_label":total_label,

        """
        temp = loadmat(data_path)

        total_data = temp['total_data']
        total_label = temp['total_label']

        data_list = []
        label_list = []
        # case of shape (1,num_subject,trials,chans,samples)
        if len(total_data) == 1 and len(total_label) == 1:
            total_data = total_data[0]
            total_label = total_label[0]

        for subject in range(len(total_data)):
            data = np.array(total_data[subject]).astype(np.float32)

            print("current lab : ",np.squeeze(np.array(total_label[subject])).shape)
            label = np.squeeze(np.array(total_label[subject])).astype(int)

            if self.cfg.DATASET.EA:
                print("run custom EA")
                # print("shape of")
                data = self.euclidean_alignment(data)

            data_list.append(data)
            label_list.append(label)

            # total_label[subject] = label
            # total_data[subject] =
            # total_label[subject] = np.squeeze(np.array(total_label[subject]))
        total_data = data_list
        total_label = label_list

        available_subject_ids = [ i for i in range(len(total_data))]
        K_FOLD_TEST = self.cfg.DATASET.K_FOLD_TEST
        VALID_FOLD_TEST = self.cfg.DATASET.VALID_FOLD_TEST
        NUM_TEST_SUBJECTS = self.cfg.DATASET.TEST_NUM_SUBJECTS
        NUM_TRAIN_VALID_SUBJECTS = self.cfg.DATASET.NUM_TRAIN_VALID_SUBJECTS

        if self.cfg.DATASET.TEST_K_FOLDS and self.cfg.DATASET.K_FOLD_TEST > 1 and len(self.cfg.DATASET.TEST_RANDOM_SEEDS) == 0:
            train_data, train_label, pick_train_subjects,test_data, test_label,pick_test_subjects = self._pick_train_valid_cross_set(total_data,total_label,folds=K_FOLD_TEST,valid_fold=VALID_FOLD_TEST)
        elif self.cfg.DATASET.TEST_NUM_SUBJECTS > 0 and len(self.cfg.DATASET.TEST_RANDOM_SEEDS) == self.cfg.DATASET.K_FOLD_TEST:
            #Randomly select N subjects to be test subjects and define constant random with seed.
            #make sure that the number of random seeds equal K_FOLD_TEST

            CURRENT_TEST_RANDOM_SEED = self.cfg.DATASET.TEST_RANDOM_SEEDS[VALID_FOLD_TEST]
            train_data, train_label, pick_train_subjects,test_data, test_label,pick_test_subjects = self._leave_N_out(total_data,total_label,seed=CURRENT_TEST_RANDOM_SEED,num_subjects=NUM_TEST_SUBJECTS)
        elif len(self.cfg.DATASET.TARGET_DOMAINS) > 0:
            # Provide a list of target subjects for test set
            # pick_data_subject_ids = cfg.DATASET.SOURCE_DOMAINS
            pick_test_subjects = list(self.cfg.DATASET.TARGET_DOMAINS)
            if (set(available_subject_ids) & set(pick_test_subjects))== set(pick_test_subjects):
            # if pick_test_subject_ids in available_subject_ids:
                pick_train_subjects = [i for i in available_subject_ids if i not in pick_test_subjects]
                train_data = [total_data[train_subject] for train_subject in pick_train_subjects]
                train_label = [total_label[train_subject] for train_subject in pick_train_subjects]
                test_data = [total_data[test_subject] for test_subject in pick_test_subjects]
                test_label = [total_label[test_subject] for test_subject in pick_test_subjects]
            else:
                raise ValueError("given subject index not available in the dataset")
        else:
            raise ValueError("Need to check the .yaml configuration for how to split the train/test data")

            # specify how many subjects are used to train/valid model
        if NUM_TRAIN_VALID_SUBJECTS > -1:
            pick_train_subjects = pick_train_subjects[:NUM_TRAIN_VALID_SUBJECTS]
            train_data = train_data[:NUM_TRAIN_VALID_SUBJECTS]
            train_label = train_label[:NUM_TRAIN_VALID_SUBJECTS]

        self.pick_train_subjects = pick_train_subjects
        self.pick_test_subjects = pick_test_subjects

        return [train_data,train_label,test_data,test_label]



















