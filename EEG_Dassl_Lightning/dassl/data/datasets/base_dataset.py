import os
import os.path as osp
import tarfile
import zipfile
import gdown

from dassl.utils import check_isfile


import numpy as np
class EEGDatum:
    """Data instance which defines the basic attributes.

    Args:
        eeg_data (np.array): stored eeg_data with format (1,channels,freq).
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, eeg_data, label=0, domain=-1, classname=''):
        assert isinstance(eeg_data, np.ndarray)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._eeg_data = eeg_data
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def eeg_data(self):
        return self._eeg_data

    def update_eeg_data(self,data):
        assert data.shape == self._eeg_data.shape
        self._eeg_data = data

    @property
    def label(self):
        return self._label

    def update_label(self,label):
        self._label = label

    @property
    def domain(self):
        return self._domain

    def update_domain(self,domain):
        self._domain = domain

    @property
    def classname(self):
        return self._classname

    def update_classname(self,classname):
        self._classname = classname

class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # directory which contains the dataset
    domains = [] # string names of all domains

    def __init__(self, train_x=None, train_u=None,multi_dataset_u=None,val=None, test=None):
        self._train_x = train_x # labeled training data
        self._train_u = train_u # unlabeled training data (optional)
        self._multi_dataset_u = multi_dataset_u
        self._val = val # validation data (optional)
        self._test = test # test data
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname = self.get_label_classname_mapping(train_x)


    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

        # a setter function

    def set_train_u(self, train_u):
        self._train_u = train_u

    @property
    def multi_dataset_u(self):
        return self._multi_dataset_u

    # a setter function
    def set_multi_dataset_u(self,multi_dataset_u):
        self._multi_dataset_u = multi_dataset_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_label_classname_mapping(self, data_source):
        tmp = set()
        for item in data_source:
            tmp.add((item.label, item.classname))
        mapping = {label: classname for label, classname in tmp}
        return mapping

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

