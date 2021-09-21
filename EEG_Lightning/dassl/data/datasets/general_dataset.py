import os.path as osp

from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import DatasetBase,EEGDatum
from dassl.data.datasets.ProcessDataBase import ProcessDataBase
from scipy.io import loadmat
import numpy as np




@DATASET_REGISTRY.register()
class GENERAL_DATASET(ProcessDataBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        # assum that number of subjects represent the domain
    def _read_data(self,data_path):
        """
        Process data from .mat file
        Re-implement this function to process new dataset
        Generate train data and test data with shape (subjects,trials,channels,frequency)
        .mat data format shall be

        "train_data":train_data,
        "train_label":train_label,
        "test_data":test_data,
         "test_label":test_label

        """
        temp = loadmat(data_path)

        total_data = temp['train_data']
        total_label = temp['train_label']

        test_data = temp['test_data']
        test_lbl = temp['test_label']

        # case of shape (1,num_subject,trials,chans,samples)
        if len(total_data) == 1 and len(total_label) == 1:
            total_data = total_data[0]
            total_label = total_label[0]

        # case of shape (1,num_subject,trials,chans,samples)
        if len(test_data) == 1 and len(test_lbl) == 1:
            test_data = test_data[0]
            test_lbl = test_lbl[0]

        for subject in range(len(total_data)):
            total_data[subject] = np.array(total_data[subject]).astype(np.float32)
            total_label[subject] = np.squeeze(np.array(total_label[subject])).astype(int)

        for subject in range(len(test_data)):
            test_data[subject] = np.array(test_data[subject]).astype(np.float32)
            test_lbl[subject] = np.squeeze(np.array(test_lbl[subject])).astype(int)



        num_trains = len(total_data)
        num_tests = len(test_data)
        self.pick_train_subjects = [ i for i in range(num_trains)]
        self.pick_test_subjects = [ i for i in range(num_trains,(num_trains+num_tests))]

        return [total_data,total_label,test_data,test_lbl]




