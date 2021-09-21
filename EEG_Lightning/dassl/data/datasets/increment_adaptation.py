import os.path as osp

from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.ProcessDataBase import ProcessDataBase
from collections import defaultdict
from scipy.io import loadmat
import numpy as np



@DATASET_REGISTRY.register()
class INCREMENT_ADAPTATION_DATASET(ProcessDataBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        #process and convert source data
        self.process_source_adaptation_data()


    def process_source_adaptation_data(self):
        # source
        self.source_domain_class_weight = self.generate_domain_class_weight(self.source_label_list)
        self.source_domain_num_class = [len(self.source_domain_class_weight[i]) for i in range(len(self.source_domain_class_weight))]
        self.source_num_domain = len(self.source_domain_num_class)
        self.source_domain_input_shapes = [self.source_data_list[source_domain][0][0].shape for source_domain in range(self.source_num_domain) ]

        list_train_u = list()
        for source_dataset_idx in range(len(self.source_data_list)):
            # print("source dataset {} has shape : {} ".format(source_dataset_idx,self.source_data_list[source_dataset_idx]))
            source_data = self.source_data_list[source_dataset_idx]
            source_data = self.expand_data_dim(source_data)
            source_label = self.source_label_list[source_dataset_idx]
            train_u = self._generate_datasource(source_data,source_label)
            list_train_u.append(train_u)
        self.list_train_u = list_train_u
        # self._train_u = self._generate_datasource(self.source_data_list,self.source_label_list)
        print("number of source dataset : ",len(self.source_domain_class_weight))
        print("source dataset class weight : ",self.source_domain_class_weight)
        print("source domain num class : ",self.source_domain_num_class)
        print("source domain input shapes ; ",self.source_domain_input_shapes)


    def generate_available_subject_ids_sequential(self,num_subject,fold = 1,shift = 2):

        subject_ids = [i for i in range(num_subject)]
        print("fold : ",fold)
        print("shift : :",shift)
        for i in range(1,fold):
            subject_ids = subject_ids[-shift:] + subject_ids[:-shift]
            print("shift ids : ",subject_ids)
        return subject_ids



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
        source_label_name_map_list = list()
        print("len of source domain : ",len(source))
        print(len(source[0]))

        for source_domain in source:
            source_domain = source_domain[0][0]
            domain_data = source_domain['source_domain_data']
            domain_label = source_domain['source_domain_label']
            # print("source domain : ",source_domain)
            # print("check source domain instance : ",isinstance(source_domain, np.ndarray))
            # if 'source_label_name_map' in source_domain.keys():
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

                if self.cfg.DATASET.EA:
                    print("run custom EA on Source domain subject data")
                    source_subject_data = self.euclidean_alignment(source_subject_data)

                source_data.append(source_subject_data)
                source_label.append(source_subject_label)

            # source_data = np.concatenate(source_data)
            # source_label = np.concatenate(source_label)
            source_data_list.append(source_data)
            source_label_list.append(source_label)
        self.source_label_name_map_list = source_label_name_map_list

        #convert into numpy
        target_data = list()
        target_label = list()
        target_temp_data = target['target_domain_data']
        target_temp_label = target['target_domain_label']
        # if 'target_label_name_map' in target.keys():
        target_label_name_map = target['target_label_name_map']
        self._label_name_map = defaultdict()
        for i,label_name in enumerate(target_label_name_map):
            self._label_name_map[i] = label_name
        # target_label_name_map = target_label_name_map[0][0]

        print("label map : ",self._label_name_map)

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

            if self.cfg.DATASET.EA:
                print("run custom EA on Target domain subject data")
                target_subject_data = self.euclidean_alignment(target_subject_data)

            target_data.append(target_subject_data)
            target_label.append(target_subject_label)

        # available_subject_ids = [ i for i in range(len(target_data))]



        # K_FOLD_TEST = self.cfg.DATASET.K_FOLD_TEST
        current_test_fold = self.cfg.DATASET.VALID_FOLD_TEST
        # VALID_FOLD_TEST = self.cfg.DATASET.VALID_FOLD_TEST
        NUM_TEST_SUBJECTS = self.cfg.DATASET.TEST_NUM_SUBJECTS

        #set up increment experiment subject available
        START_NUM_TRAIN_SUGJECT = self.cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.START_NUM_TRAIN_SUGJECT #number of start out subjects
        INCREMENT_UPDATE = self.cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.INCREMENT_UPDATE # subjects increment
        CURRENT_INCREMENT_FOLD =  self.cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.CURRENT_FOLD

        available_subject_ids = self.generate_available_subject_ids_sequential(len(target_data),fold=current_test_fold,shift= INCREMENT_UPDATE)
        print("availbale subject ids : ",available_subject_ids)

        NUM_TRAIN_SUBJECT = START_NUM_TRAIN_SUGJECT+INCREMENT_UPDATE*(CURRENT_INCREMENT_FOLD-1)
        assert (len(target_data)-NUM_TRAIN_SUBJECT) >= NUM_TEST_SUBJECTS
        pick_train_subjects = available_subject_ids[:NUM_TRAIN_SUBJECT]

        pick_test_subjects = available_subject_ids[-(NUM_TEST_SUBJECTS):]
        assert len(pick_test_subjects) == NUM_TEST_SUBJECTS
        # train_data = target_data[:NUM_TRAIN_SUBJECT]
        # train_label = target_label[:NUM_TRAIN_SUBJECT]
        # test_data = target_data[-(NUM_TEST_SUBJECTS):]
        # test_label = target_label[-(NUM_TEST_SUBJECTS):]

        train_data = [target_data[train_subject] for train_subject in pick_train_subjects]
        train_label = [target_label[train_subject] for train_subject in pick_train_subjects]
        test_data = [target_data[test_subject] for test_subject in pick_test_subjects]
        test_label = [target_label[test_subject] for test_subject in pick_test_subjects]

        self.pick_train_subjects = pick_train_subjects
        self.pick_test_subjects = pick_test_subjects

        self.source_data_list = source_data_list
        self.source_label_list = source_label_list

        return [train_data,train_label,test_data,test_label]

    # def process_data_format(self, input_data, cfg):



















