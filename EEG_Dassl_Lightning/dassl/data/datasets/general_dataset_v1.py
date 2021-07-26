import os.path as osp

from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase,EEGDatum
from dassl.data.datasets.ProcessDataBase import ProcessDataBase

from scipy.io import loadmat
import numpy as np




@DATASET_REGISTRY.register()
class GENERAL_DATASET(ProcessDataBase):
    # dataset_dir = 'KAGGLE_BCI'
    # file_name = 'KaggleBCI.mat'
    # domains = [0,3,4,5,6,7,8]
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
        total_label = np.array(total_label)
        total_label = np.squeeze(total_label)
        total_label = total_label.astype(int)
        test_data = temp['test_data']
        test_lbl =  temp['test_label']
        test_data = np.array(test_data)  # (subjects,trials,channels,frequency)
        test_lbl = np.array(test_lbl)
        test_lbl = test_lbl.astype(int)

        print("train data shape : {}  | train label shape : {}".format(total_data.shape,total_label.shape))
        print("test data shape : {}  | test label shape : {}".format(test_data.shape, test_lbl.shape))

        return [total_data,total_label,test_data,test_lbl]




