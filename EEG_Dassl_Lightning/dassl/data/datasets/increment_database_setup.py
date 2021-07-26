import os.path as osp

from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.database import MainDataBase
from dassl.utils.tools import set_random_seed
from collections import defaultdict
from scipy.io import loadmat
import numpy as np



@DATASET_REGISTRY.register()
class INCREMENT_DATASET_SETUP(MainDataBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        #process and convert source data
        multi_dataset = self.process_source_adaptation_data()
        self.set_multi_dataset_u(multi_dataset)


    def check_dataInfo(self):
        return

    def process_source_adaptation_data(self):

        return (self.source_data_list,self.source_label_list)

    def _read_data(self,data_path):
        """
        Process data from .mat file
        Re-implement this function to process new dataset
        Given file with whole data without specify test data and test label.
        Generate train data and test data with shape (1,subjects) and subject = (trials,channels,frequency)
        .mat data format shall be

        "total_data":total_data,
        "total_label":total_label,

        "source_domain": source_list,
        "target_domain": {
            "target_domain_data": target_data,
            "target_domain_label": target_label,
            "target_label_name_map": target_label_name_map,
            "dataset_name":target_dataset_name,
            "subject_id":target_subject_ids
            }

        target_data has format [(subject_trials, channels, frequency), (subject_trials, channels, frequency) , ..]. A list of numpy array
        target_label has format [(1,subject_trials) , (1,subject trials)]. A list of numpy array
        target_label_name_map has format
        """
        temp = loadmat(data_path)
        target = temp['target_domain']
        source = temp['source_domain']

        source = source[0]
        target = target[0][0]


        source_data_list = list()
        source_label_list = list()
        source_label_name_map_list = list()


        for source_domain in source:
            source_domain = source_domain[0][0]
            domain_data = source_domain['source_domain_data']
            domain_label = source_domain['source_domain_label']
            domain_label_name_map = source_domain['source_label_name_map']
            source_label_name_map_list.append(domain_label_name_map)
            if not (isinstance(domain_data, np.ndarray)  and len(domain_data.shape) == 4):
                domain_data = domain_data[0]
                domain_label = domain_label[0]
            source_data = []
            source_label = []
            for source_subject in range(len(domain_data)):
                source_subject_data = domain_data[source_subject]
                source_subject_label = domain_label[source_subject]

                source_subject_data = np.array(source_subject_data).astype(np.float32)
                source_subject_label = np.squeeze(np.array(source_subject_label)).astype(int)

                if self.cfg.DATAMANAGER.DATASET.EA:
                    if np.iscomplexobj(source_subject_data):
                        print("problem")
                    # print("run custom EA on Source domain subject data")
                    source_subject_data = self.euclidean_alignment(source_subject_data)

                source_data.append(source_subject_data)
                source_label.append(source_subject_label)


            source_data_list.append(source_data)
            source_label_list.append(source_label)
        self.source_label_name_map_list = source_label_name_map_list

        #convert into numpy
        target_data = list()
        target_label = list()
        target_temp_data = target['target_domain_data']
        target_temp_label = target['target_domain_label']
        target_label_name_map = target['target_label_name_map']
        self._label_name_map = defaultdict()
        for i,label_name in enumerate(target_label_name_map):
            self._label_name_map[i] = label_name
        # target_label_name_map = target_label_name_map[0][0]


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

            if self.cfg.DATAMANAGER.DATASET.EA:
                # print("run custom EA on Target domain subject data")
                target_subject_data = self.euclidean_alignment(target_subject_data)

            target_data.append(target_subject_data)
            target_label.append(target_subject_label)

        current_test_fold = self.cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.CURRENT_TEST_FOLD
        current_shuffle_fold = self.cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.CURRENT_SHUFFLE_FOLD
        N_SHUFFLE_FOLD = self.cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS
        NUM_TEST_SUBJECTS = self.cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.NUM_TEST_SUBJECTS

        shuffle_seeds = self.cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_SEED
        FIX_SEED = self.cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SET_FIX_SEED
        test_subject_idx = self.cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_SUBJECT_INDEX
        specific_test_subject_idx = test_subject_idx[current_test_fold-1]
        if FIX_SEED == True:
            shuffle_seed = shuffle_seeds[0]
            available_subject_ids = np.arange(len(target_data))
        else:
            shuffle_seed = shuffle_seeds[current_shuffle_fold-1]
            set_random_seed(shuffle_seed)
            available_subject_ids = np.random.permutation(len(target_data))



        #set up increment experiment subject available
        START_NUM_TRAIN_SUGJECT = self.cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_NUM_TRAIN_SUGJECT #number of start out subjects
        assert START_NUM_TRAIN_SUGJECT > 0
        INCREMENT_UPDATE = self.cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.INCREMENT_TRAIN_SUGJECT # subjects increment
        CURRENT_INCREMENT_FOLD =  self.cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.CURRENT_INCREMENT_FOLD

        NUM_TRAIN_SUBJECT = START_NUM_TRAIN_SUGJECT+INCREMENT_UPDATE*(CURRENT_INCREMENT_FOLD-1)
        assert (len(target_data)-NUM_TRAIN_SUBJECT) >= NUM_TEST_SUBJECTS


        # pick_train_subjects = available_subject_ids[:NUM_TRAIN_SUBJECT]
        if len(specific_test_subject_idx) > 0:
            available_train_subjects = [idx for idx in available_subject_ids if idx not in specific_test_subject_idx]
            pick_test_subjects = specific_test_subject_idx
        else:
            available_train_subjects = available_subject_ids[:-NUM_TEST_SUBJECTS]
            # print("available train subjects ids : ",available_train_subjects)
            pick_test_subjects = available_subject_ids[-(NUM_TEST_SUBJECTS):]
        assert len(pick_test_subjects) == NUM_TEST_SUBJECTS

        #set fix shuffle seed to replicate shuffle and split dataset
        np.random.seed(shuffle_seed)
        available_train_seeds =  np.random.permutation(N_SHUFFLE_FOLD)
        current_train_shuffle_seed = available_train_seeds[current_shuffle_fold-1]
        np.random.seed(current_train_shuffle_seed)
        np.random.shuffle(available_train_subjects)

        pick_train_subjects = available_train_subjects[:NUM_TRAIN_SUBJECT]


        train_data = [target_data[train_subject] for train_subject in pick_train_subjects]
        train_label = [target_label[train_subject] for train_subject in pick_train_subjects]
        test_data = [target_data[test_subject] for test_subject in pick_test_subjects]
        test_label = [target_label[test_subject] for test_subject in pick_test_subjects]

        self.pick_train_subjects = pick_train_subjects
        self.pick_test_subjects = pick_test_subjects

        self.source_data_list = source_data_list
        self.source_label_list = source_label_list

        if self.cfg.DISPLAY_INFO.DATASET:
            print("len of source domain : ", len(source))
            print(len(source[0]))
            print("available subject ids : ", available_subject_ids)
            print("shuffle available subject ids : ", available_train_subjects)
            print("pick train subjects ids : ", pick_train_subjects)
            print("pick test subjects ids : ", pick_test_subjects)
            print("label map : ",self._label_name_map)

        return [train_data,train_label,test_data,test_label]