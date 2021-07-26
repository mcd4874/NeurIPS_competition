import os.path as osp

from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.ProcessDataBase import ProcessDataBase
from collections import defaultdict
from scipy.io import loadmat
import numpy as np



@DATASET_REGISTRY.register()
class ADAPTATION_DATASET(ProcessDataBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        #process and convert source data
        self.process_source_adaptation_data()


    def process_source_adaptation_data(self):
        # source
        self.source_domain_class_weight = self.generate_domain_class_weight(self.source_label_list)
        self.source_domain_num_class = [len(self.source_domain_class_weight[i]) for i in range(len(self.source_domain_class_weight))]
        self.source_data_list = self.expand_data_dim(self.source_data_list)
        for source_dataset in range(len(self.source_data_list)):
            print("source dataset {} has shape : {} ".format(source_dataset,self.source_data_list[source_dataset].shape))
        self._train_u = self._generate_datasource(self.source_data_list,self.source_label_list)
        self.source_num_domain = len(self.source_domain_num_class)
        print("number of source dataset : ",len(self.source_domain_class_weight))
        print("source dataset class weight : ",self.source_domain_class_weight)
        print("source domain num class : ",self.source_domain_num_class)


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
        target = temp['target_domain']
        source = temp['source_domain']

        print("origin source : ",source.shape)
        target = target[0][0]
        source = source[0]

        source_data_list = list()
        source_label_list = list()
        print("len of source domain : ",len(source))
        print(len(source[0]))

        for source_domain in source:
            source_domain = source_domain[0][0]
            domain_data = source_domain['source_domain_data']
            domain_label = source_domain['source_domain_label']
            domain_data = domain_data[0]
            domain_label = domain_label[0]
            source_data = []







            source_label = []
            for source_subject in range(len(domain_data)):
                source_subject_data = domain_data[source_subject]
                source_subject_label = domain_label[source_subject]

                source_subject_data = np.array(source_subject_data).astype(np.float32)
                source_subject_label = np.squeeze(np.array(source_subject_label)).astype(int)
                source_data.append(source_subject_data)
                source_label.append(source_subject_label)

            source_data = np.concatenate(source_data)
            source_label = np.concatenate(source_label)
            source_data_list.append(source_data)
            source_label_list.append(source_label)


        #convert into numpy
        target_data = list()
        target_label = list()
        target_temp_data = target['target_domain_data']
        target_temp_label = target['target_domain_label']

        # case of shape (1,num_subject,trials,chans,samples)
        #still need to double check for exceptation cases
        if len(target_temp_data) == 1 and len(target_temp_label) == 1:
            target_temp_data = target_temp_data[0]
            target_temp_label = target_temp_label[0]

        for target_subject in range(len(target_temp_data)):
            target_subject_data = target_temp_data[target_subject]
            target_subject_label = target_temp_label[target_subject]
            target_subject_data = np.array(target_subject_data).astype(np.float32)
            target_subject_label = np.squeeze(np.array(target_subject_label)).astype(int)
            target_data.append(target_subject_data)
            target_label.append(target_subject_label)

        available_subject_ids = [ i for i in range(len(target_data))]
        K_FOLD_TEST = self.cfg.DATASET.K_FOLD_TEST
        VALID_FOLD_TEST = self.cfg.DATASET.VALID_FOLD_TEST
        NUM_TEST_SUBJECTS = self.cfg.DATASET.TEST_NUM_SUBJECTS
        NUM_TRAIN_VALID_SUBJECTS = self.cfg.DATASET.NUM_TRAIN_VALID_SUBJECTS

        if self.cfg.DATASET.TEST_K_FOLDS and self.cfg.DATASET.K_FOLD_TEST > 1 and len(self.cfg.DATASET.TEST_RANDOM_SEEDS) == 0:
            train_data, train_label, pick_train_subjects,test_data, test_label,pick_test_subjects = self._pick_train_valid_cross_set(target_data,target_label,folds=K_FOLD_TEST,valid_fold=VALID_FOLD_TEST)

        elif self.cfg.DATASET.TEST_NUM_SUBJECTS > 0 and len(self.cfg.DATASET.TEST_RANDOM_SEEDS) == self.cfg.DATASET.K_FOLD_TEST:
            #Randomly select N seiekcclklekrefbcirrbulngcjijenudrgvrbkkgcvkr
            # ubjects to be test subjects and define constant random with seed.
            #make sure that the number of random seeds equal K_FOLD_TEST

            CURRENT_TEST_RANDOM_SEED = self.cfg.DATASET.TEST_RANDOM_SEEDS[VALID_FOLD_TEST]
            train_data, train_label, pick_train_subjects,test_data, test_label,pick_test_subjects = self._leave_N_out(target_data,target_label,seed=CURRENT_TEST_RANDOM_SEED,num_subjects=NUM_TEST_SUBJECTS)
        elif len(self.cfg.DATASET.TARGET_DOMAINS) > 0:
            # Provide a list of target subjects for test set
            # pick_data_subject_ids = cfg.DATASET.SOURCE_DOMAINS
            pick_test_subjects = list(self.cfg.DATASET.TARGET_DOMAINS)
            if (set(available_subject_ids) & set(pick_test_subjects))== set(pick_test_subjects):
            # if pick_test_subject_ids in available_subject_ids:
                pick_train_subjects = [i for i in available_subject_ids if i not in pick_test_subjects]
                train_data = [target_data[train_subject] for train_subject in pick_train_subjects]
                train_label = [target_label[train_subject] for train_subject in pick_train_subjects]
                test_data = [target_data[test_subject] for test_subject in pick_test_subjects]
                test_label = [target_label[test_subject] for test_subject in pick_test_subjects]
            else:
                raise ValueError("given subject index not available in the dataset")
        else:
            raise ValueError("Need to check the .yaml configuration for how to split the train/test data")

        #specify how many subjects are used to train/valid model
        if NUM_TRAIN_VALID_SUBJECTS > -1:
            pick_train_subjects = pick_train_subjects[:NUM_TRAIN_VALID_SUBJECTS]
            train_data = train_data[:NUM_TRAIN_VALID_SUBJECTS]
            train_label = train_label[:NUM_TRAIN_VALID_SUBJECTS]

        self.pick_train_subjects = pick_train_subjects
        self.pick_test_subjects = pick_test_subjects

        self.source_data_list = source_data_list
        self.source_label_list = source_label_list

        return [train_data,train_label,test_data,test_label]

    # def process_data_format(self, input_data, cfg):



















