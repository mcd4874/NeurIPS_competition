import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset
from dassl.data.datasets.data_util import (generate_datasource,EuclideanAlignment,LabelAlignment,
                                           DataAugmentation,get_num_classes,get_label_classname_mapping,
                                           dataset_norm,subjects_filterbank,filterBank,relabel_target)
from .datasets import build_dataset
from .samplers import build_sampler
import numpy as np
from pytorch_lightning import LightningDataModule
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
from collections import defaultdict
from scipy.linalg import sqrtm, inv
import scipy.signal as signal
import copy


# def convert_subjects_data_with_EA(subjects_data):
#     new_data = list()
#     for subject_idx in range(len(subjects_data)):
#         subject_data = subjects_data[subject_idx]
#         subject_data = euclidean_alignment(subject_data)
#         new_data.append(subject_data)
#     return new_data
def relabel_subjects(subject_label):
    new_subject_data = list()
    for subject in range(len(subject_label)):
        current_label = subject_label[subject]
        current_label = np.array([relabel_target(l) for l in current_label])
        new_subject_data.append(current_label)
    return new_subject_data

# train_x_label = [) for train_subject_label in train_x_label]


class DataManagerV1(LightningDataModule):
    def __init__(self,cfg,transform_data_func=None):
        self.cfg = cfg
        self.data_augmentation = self.cfg.DATAMANAGER.DATASET.AUGMENTATION.NAME
        self.DOMAIN_CLASS_WEIGHT = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.DOMAIN_CLASS_WEIGHT
        self.TOTAL_CLASS_WEIGHT = self.cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.TOTAL_CLASS_WEIGHT
        self.whole_class_weight = None

        self.transform_data_func = transform_data_func
        if self.transform_data_func is None:
            self.transform_data_func = generate_datasource
        self._label_name_map = None
        # self.dataset_wrapper = CustomEEGDatasetWrapper
        self.dataset_wrapper = CustomWrapper

        self._dataset = None
        self.useFilterBank = self.cfg.DATAMANAGER.DATASET.FILTERBANK.USE_FILTERBANK
        self.relabel = self.cfg.DATAMANAGER.DATASET.target_dataset_relabelled
        self.label_alignment = self.cfg.DATAMANAGER.DATASET.source_dataset_LA

        self.view_test_subject_result = True
        super(DataManagerV1, self).__init__()


    def get_require_parameter(self):
        require_parameter = dict()
        require_parameter['num_classes'] = self.num_classes
        require_parameter['total_subjects'] = self.total_subjects
        require_parameter['target_domain_class_weight'] = self.whole_class_weight
        require_parameter['num_test_subjects'] = len(self.subject_test)


        return require_parameter

    def generate_augmentation(self,data,label):
        MAX_TRIAL_MUL = self.cfg.DATAMANAGER.DATASET.AUGMENTATION.PARAMS.MAX_TRIAL_MUL
        MAX_FIX_TRIAL = self.cfg.DATAMANAGER.DATASET.AUGMENTATION.PARAMS.MAX_FIX_TRIAL
        N_SEGMENT = self.cfg.DATAMANAGER.DATASET.AUGMENTATION.PARAMS.N_SEGMENT
        spatial_dataset_name = self.cfg.DATAMANAGER.DATASET.AUGMENTATION.PARAMS.DATASET_NAME
        augmentation = DataAugmentation(data, label, max_trials_mul=MAX_TRIAL_MUL,
                                        total_fix_trials=MAX_FIX_TRIAL, spatial_dataset_name=spatial_dataset_name)
        new_data, new_label = augmentation.generate_artificial_data(self.data_augmentation, N_SEGMENT)
        return new_data,new_label

    @property
    def dataset(self):
        return self._dataset

    def update_dataset(self,dataset):
        self._dataset = dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self._dataset is None:
            self._dataset = build_dataset(self.cfg)
        self._label_name_map = self._dataset.label_name_map
        train_x_dataset = self._dataset.train_x
        val_dataset = self._dataset.val
        test_dataset = self._dataset.test

        train_x_data,train_x_label = train_x_dataset
        val_data,val_label = val_dataset
        test_data,test_label = test_dataset

        self.print_dataset_info(train_x_data, train_x_label, val_data, val_label, test_data, test_label)
        # """Create label alignment"""
        # if self.label_alignment:
        #     print(" shape {}, {}, {}".format(train_x_data[0].shape,val_data[0].shape,test_data[0].shape))
        #     tmp_data = np.concatenate([np.concatenate(train_x_data),np.concatenate(val_data),np.concatenate(test_data)])
        #     tmp_label = np.concatenate([np.concatenate(train_x_label),np.concatenate(val_label),np.concatenate(test_label)])
        #     print("target data for label alignment shape : ",tmp_data.shape)
        #     print("target label for label alignment shape : ",tmp_label.shape)
        #     self.LA = LabelAlignment(target_dataset=(tmp_data, tmp_label))

        """Data Augmentation for train_x dataset"""
        if self.data_augmentation != "":
            print("apply augmentation")
            train_x_data,train_x_label = self.generate_augmentation(train_x_data,train_x_label)


        if self.relabel:
            print("train_x_label {},{} ".format(len(train_x_label),train_x_label[0].shape))
            # train_x_label = [np.array([relabel_target(l) for l in train_subject_label]) for train_subject_label in train_x_label]

            old_label = np.unique(train_x_label[0])
            train_x_label = relabel_subjects(train_x_label)
            val_label = relabel_subjects(val_label)
            test_label = relabel_subjects(test_label)
            new_label = np.unique(train_x_label[0])
            # train_x_label = relabel_target(train_x_label)
            # val_label = relabel_target(val_label)
            # test_label = relabel_target(test_label)
            # new_label = np.unique(train_x_label[0])
            print("relabel the dataset from {} to {}".format(old_label,new_label))
        """Create class weight for train_x dataset"""
        # if self.DOMAIN_CLASS_WEIGHT:
        #     self.domain_class_weight = self.generate_domain_class_weight(train_x_label)
        if self.TOTAL_CLASS_WEIGHT:
            self.whole_class_weight = self.generate_class_weight(train_x_label)


        # """Use filter bank """
        # if self.useFilterBank:
        #     # diff = 4
        #     diff = self.cfg.DATAMANAGER.DATASET.FILTERBANK.freq_interval
        #     filter_bands = []
        #     # axis = 2
        #     for i in range(1, 9):
        #         filter_bands.append([i * diff, (i + 1) * diff])
        #     print("build filter band : ", filter_bands)
        #     filter = filterBank(
        #         filtBank=filter_bands,
        #         fs=128
        #     )
        #     source = [0, 1, 2, 3]
        #     destination = [0, 2, 3, 1]
        #
        #     train_x_data = subjects_filterbank(train_x_data,filter,source,destination)
        #     val_data = subjects_filterbank(val_data,filter,source,destination)
        #     test_data = subjects_filterbank(test_data,filter,source,destination)


        if self.cfg.DATAMANAGER.DATASET.USE_Euclidean_Aligment:
            print("run custom EA")
            if self._dataset.r_op_list is not None:
                self.train_EA = EuclideanAlignment(list_r_op=self._dataset.r_op_list,subject_ids=self._dataset.pick_train_subjects)
                self.val_EA = EuclideanAlignment(list_r_op=self._dataset.r_op_list,subject_ids=self._dataset.pick_valid_subjects)
                self.test_EA = EuclideanAlignment(list_r_op=self._dataset.r_op_list,subject_ids=self._dataset.pick_test_subjects)
            else:
                self.train_EA = EuclideanAlignment()
                self.val_EA = EuclideanAlignment()
                self.test_EA = EuclideanAlignment()
            train_x_data = self.train_EA.convert_subjects_data_with_EA(train_x_data)
            val_data = self.val_EA.convert_subjects_data_with_EA(val_data)
            test_data = self.test_EA.convert_subjects_data_with_EA(test_data)

            # print("convert trials : ",)
            # print("new trial : ",val_data[0])

        """Apply data transformation/normalization"""
        if not self.cfg.INPUT.NO_TRANSFORM:
            normalization = self.cfg.INPUT.TRANSFORMS[0]
            if normalization == 'cross_channel_norm':
                # train_x_data,train_x_label = dataset_norm(train_x_data,train_x_label)
                # val_data,val_label = dataset_norm(val_data,val_label)
                # test_data,test_label = dataset_norm(test_data,test_label)
                train_x_data = dataset_norm(train_x_data)
                val_data = dataset_norm(val_data)
                test_data = dataset_norm(test_data)
            elif normalization == 'time_norm':
                train_x_data = dataset_norm(train_x_data,norm_channels=False)
                val_data = dataset_norm(val_data,norm_channels=False)
                test_data = dataset_norm(test_data,norm_channels=False)


        train_x_data = self._expand_data_dim(train_x_data)
        val_data = self._expand_data_dim(val_data)
        test_data = self._expand_data_dim(test_data)

        #display current dataset information
        if self.cfg.DISPLAY_INFO.DATASET:
            self.print_dataset_info(train_x_data, train_x_label, val_data, val_label, test_data, test_label)


        train_items = generate_datasource(train_x_data, train_x_label, label_name_map=self._label_name_map)
        valid_items = generate_datasource(val_data, val_label, label_name_map=self._label_name_map)
        test_items = generate_datasource(test_data, test_label, label_name_map=self._label_name_map)

        list_subject_test_item = [generate_datasource([test_data[subject_idx]], [test_label[subject_idx]], label_name_map=self._label_name_map) for subject_idx in range(len(test_label))]

        #get num_classes and lab2cname
        self._num_classes = get_num_classes(train_items)
        self._lab2cname = get_label_classname_mapping(train_items)

        # train_u_dataset,\
        # multi_dataset_u,\

        self.train_x = train_items
        self.val = valid_items
        self.test = test_items
        self.subject_test = list_subject_test_item

        self._total_subjects = self._dataset.data_domains


    def train_dataloader(self):

        train_x = self.dataset_wrapper(self.cfg, self.train_x,is_train=True)
        print("train x size : ",len(train_x))

        train_loader_x = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATAMANAGER.DATALOADER.TRAIN_X.SAMPLER,
            data_source=train_x,
            batch_size=self.cfg.DATAMANAGER.DATALOADER.TRAIN_X.BATCH_SIZE,
            is_train=True
        )
        print("size of train loader : ",len(train_loader_x))
        return train_loader_x

    def val_dataloader(self):
        val = self.dataset_wrapper(self.cfg, self.val,is_train=False)

        val_loader = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATAMANAGER.DATALOADER.VALID.SAMPLER,
            data_source=val,
            batch_size=self.cfg.DATAMANAGER.DATALOADER.VALID.BATCH_SIZE,
            is_train=False
        )

        test = self.dataset_wrapper(self.cfg, self.test,is_train=False)

        test_loader = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATAMANAGER.DATALOADER.TEST.SAMPLER,
            data_source=test,
            batch_size=self.cfg.DATAMANAGER.DATALOADER.TEST.BATCH_SIZE,
            is_train=False
        )
        print("size of val loader : ",len(val_loader))
        print("size of test loader : ",len(test_loader))

        return [val_loader,test_loader]

        # train_loader_x = build_data_loader(
        #     self.cfg,
        #     sampler_type=self.cfg.DATAMANAGER.DATALOADER.TRAIN_X.SAMPLER,
        #     data_source=self.train_x,
        #     batch_size=self.cfg.DATAMANAGER.DATALOADER.TRAIN_X.BATCH_SIZE,
        #     is_train=False,
        #     dataset_wrapper=self.dataset_wrapper
        # )
        # return [val_loader,train_loader_x]
        # return val_loader
    def predict_dataloader(self):
        test = self.dataset_wrapper(self.cfg, self.test, is_train=False)
        test_loader = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATAMANAGER.DATALOADER.TEST.SAMPLER,
            data_source=test,
            batch_size=self.cfg.DATAMANAGER.DATALOADER.TEST.BATCH_SIZE,
            is_train=False
        )
        print("size of test loader : ", len(test_loader))
        return test_loader
    def test_dataloader(self) :
        if not self.view_test_subject_result:
            test = self.dataset_wrapper(self.cfg, self.test,is_train=False)
            test_loader = build_data_loader(
                self.cfg,
                sampler_type=self.cfg.DATAMANAGER.DATALOADER.TEST.SAMPLER,
                data_source=test,
                batch_size=self.cfg.DATAMANAGER.DATALOADER.TEST.BATCH_SIZE,
                is_train=False
            )
            print("size of test loader : ",len(test_loader))
        else:
            test_loader = list()
            for test in self.subject_test:
                test = self.dataset_wrapper(self.cfg, test, is_train=False)
                subject_test_loader = build_data_loader(
                    self.cfg,
                    sampler_type=self.cfg.DATAMANAGER.DATALOADER.TEST.SAMPLER,
                    data_source=test,
                    batch_size=self.cfg.DATAMANAGER.DATALOADER.TEST.BATCH_SIZE,
                    is_train=False
                )
                test_loader.append(subject_test_loader)
            print("size of test loader : ", len(test_loader))
        return test_loader

    def _expand_data_dim(self, data):
        if isinstance(data, list):
            for idx in range(len(data)):
                if len(data[idx].shape) == 3:
                    new_data = np.expand_dims(data[idx], axis=1)
                else:
                    new_data = data[idx]
                data[idx] = new_data
            return data
        elif isinstance(data, np.ndarray):
            return np.expand_dims(data, axis=2)
        else:
            raise ValueError("the data format during the process section is not correct")

    @property
    def total_subjects(self):
        return self._total_subjects

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def label_name_map(self):
        return self._label_name_map

    def generate_class_weight(self, label):
        """
        generate the weight ratio based on total labels of every subjects
        label : [subject_1,subject_2,..] and subject = (trials)
        """

        if isinstance(label, list):
            new_label = np.empty(0)
            for current_label in label:
                # print("current label shape : ",current_label.shape)
                new_label = np.concatenate([new_label, current_label])
            total = new_label.shape[0]
            labels = np.unique(new_label)
            list_ratio = []
            # print("new form label ",)
            for current_label in labels:
                current_ratio = total / len(np.where(new_label == current_label)[0])
                list_ratio.append(current_ratio)
            return list_ratio

        elif isinstance(label, np.ndarray):
            if len(label.shape) == 2:
                # label shall have shape (subjects,trials)
                label = label.reshape(label.shape[0] * label.shape[1])
            # data need to be shape (trials)
            total = label.shape[0]
            labels = np.unique(label)
            list_ratio = []
            for current_label in labels:
                current_ratio = total / len(np.where(label == current_label)[0])
                list_ratio.append(current_ratio)
            return list_ratio
        else:
            raise ValueError("the data format during the process section is not correct")

    def generate_domain_class_weight(self, label):

        """
        assume the label has shape (subjects,trials)
        """
        domain_class_weight = list()
        # domain_class_weight = defaultdict()
        for domain in range(len(label)):
            current_domain_class_weight = self.generate_class_weight(label[domain])
            domain_class_weight.append(current_domain_class_weight)
            # domain_class_weight[domain] = current_domain_class_weight
        return domain_class_weight
    def print_dataset_info(self,train_data, train_label, valid_data, valid_label,test_data,test_label):
        # print("Train data info: ")
        print("train subjects : ",self._dataset.pick_train_subjects)
        if len(train_data) > len(self._dataset.pick_train_subjects):
            unknown = -1
            for subject_idx in range(len(train_data)):
                if subject_idx< len(self._dataset.pick_train_subjects):
                    print("Train subject {} has shape : {}, with range scale ({},{}) ".format(
                        self._dataset.pick_train_subjects[subject_idx], train_data[subject_idx].shape,
                        np.max(train_data[subject_idx]), np.min(train_data[subject_idx])))
                else:
                    print("unknown Train subject {} has shape : {}, with range scale ({},{}) ".format(
                        unknown, train_data[subject_idx].shape,
                        np.max(train_data[subject_idx]), np.min(train_data[subject_idx])))
                    unknown = unknown-1

        elif len(train_data) == len(self._dataset.pick_train_subjects):
            for subject_idx in range(len(train_data)):
                print("Train subject {} has shape : {}, with range scale ({},{}) ".format(self._dataset.pick_train_subjects[subject_idx],train_data[subject_idx].shape,np.max(train_data[subject_idx]),np.min(train_data[subject_idx])))
        print("test subjects : ",self._dataset.pick_test_subjects)
        for subject_idx in range(len(test_data)):
            print("test subject {} has shape : {}, with range scale ({},{})  ".format(self._dataset.pick_test_subjects[subject_idx],test_data[subject_idx].shape,np.max(test_data[subject_idx]),np.min(test_data[subject_idx])))
        print("valid subjects : ",self._dataset.pick_valid_subjects)
        for subject_idx in range(len(valid_data)):
            print("valid subject {} has shape : {}, with range scale ({},{})  ".format(self._dataset.pick_valid_subjects[subject_idx],valid_data[subject_idx].shape,np.max(valid_data[subject_idx]),np.min(valid_data[subject_idx])))

        # for test_subject_idx in range(len(test_data)):
        #     print("pick subject id : ",self._dataset.pick_test_subjects[test_subject_idx])
        #     print("curent test subject data shape : ",test_data[test_subject_idx].shape)
        #     print("curent test subject label shape : ",test_label[test_subject_idx].shape)

        # if self.domain_class_weight is not None:
        #     print("Train data labels ratio info : ")
        #     for subject_idx in range(len(train_data)):
        #         current_subject_id = self._dataset.pick_train_subjects[subject_idx]
        #         subject_ratio = self.domain_class_weight[subject_idx]
        #         print("subject {} has labels ratio : {}".format(current_subject_id,subject_ratio))

        if self.whole_class_weight is not None:
            print("the labels ratio of whole dataset : {}".format(self.whole_class_weight))

class DataManagerV2(DataManagerV1):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup(self, stage: Optional[str] = None) :
        super().setup(stage)
        train_u_data = self._dataset.train_u
        # train_u_same_as_test = False
        if train_u_data is None:
            print("there is no provide unlabel data. We need to use test data as unlabel")
            test_dataset = self._dataset.test
            test_data, _ = test_dataset
            train_u_data = test_data
            # train_u_same_as_test = True
            subject_ids = self._dataset.pick_test_subjects
        else:
            subject_ids = [i for i in range(len(train_u_data))]

        # train_u_data = self.train_u

        if self.cfg.DATAMANAGER.DATASET.USE_Euclidean_Aligment:
            if self._dataset.r_op_list is not None:
                # if train_u_same_as_test:
                self.train_u_EA = EuclideanAlignment(list_r_op=self._dataset.r_op_list,subject_ids=subject_ids)
                # else:
                #     self.train_u_EA = EuclideanAlignment(list_r_op=self._dataset.r_op_list)
            else:
                self.train_u_EA = EuclideanAlignment()
            train_u_data = self.train_u_EA.convert_subjects_data_with_EA(train_u_data)

        """Apply data transformation/normalization"""
        if not self.cfg.INPUT.NO_TRANSFORM:
            normalization = self.cfg.INPUT.TRANSFORMS[0]
            if normalization == 'cross_channel_norm':
                train_u_data = dataset_norm(train_u_data)
            elif normalization == 'time_norm':
                train_u_data= dataset_norm(train_u_data, norm_channels=False)
        for subject_idx in range(len(train_u_data)):
            print("unlabel data subject_idx {} has shape : {}, with range scale ({},{}) ".format(
                subject_ids[subject_idx], train_u_data[subject_idx].shape,
                np.max(train_u_data[subject_idx]), np.min(train_u_data[subject_idx])))

        train_u_data = self._expand_data_dim(train_u_data)
        train_u_data = np.concatenate(train_u_data)
        self.train_u = train_u_data




    def train_dataloader(self):
        train_x_dataloader = super().train_dataloader()
        train_u = Wrapper(self.train_u)
        train_loader_u = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATAMANAGER.DATALOADER.TRAIN_U.SAMPLER,
            data_source=train_u,
            batch_size=self.cfg.DATAMANAGER.DATALOADER.TRAIN_U.BATCH_SIZE,
            is_train=True,
        )

        return {"target_loader": train_x_dataloader, "unlabel_loader": train_loader_u}

class MultiDomainDataManagerV1(DataManagerV1):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup(self, stage: Optional[str] = None) :
        super().setup(stage)
        self.list_train_u_exist = hasattr(self._dataset, 'multi_dataset_u')
        source_data_list , source_label_list=self._dataset.multi_dataset_u
        self.list_train_u = list()
        source_domain_class_weight = list()
        for source_dataset_idx in range(len(source_data_list)):
            source_data = source_data_list[source_dataset_idx]
            source_label = source_label_list[source_dataset_idx]
            print("source dataset idx : ",source_dataset_idx)

            # if self.label_alignment:
            #     source_data, source_label = self.LA.convert_source_data_with_LA(source_data, source_label)

            if self.relabel:
                print("source_label {},{} ".format(len(source_label), source_label[0].shape))
                old_label = np.unique(source_label[0])
                source_label = relabel_subjects(source_label)
                new_label = np.unique(source_label[0])
                print("relabel the dataset from {} to {}".format(old_label, new_label))
            current_domain_class_weight = self.generate_class_weight(source_label)
            # else:
            #     current_domain_class_weight = None
            source_domain_class_weight.append(current_domain_class_weight)

            if self.cfg.DATAMANAGER.DATASET.USE_Euclidean_Aligment:
                for subject_idx in range(len(source_data)):
                    print("source_data before EA subject_idx {} has shape : {}, with range scale ({},{}) ".format(
                        subject_idx, source_data[subject_idx].shape,
                        np.max(source_data[subject_idx]), np.min(source_data[subject_idx])))

                print("run custom EA")
                source_EA = EuclideanAlignment()
                source_data = source_EA.convert_subjects_data_with_EA(source_data)

                # update = list()
                # for subject_data in source_data:
                #     print("current data type : ", subject_data.dtype)
                #     if isinstance(subject_data,np.float64):
                #         print("update data type : ")
                #         subject_data = subject_data.astype(np.float32)
                #     else:
                #         print("can't update ")
                #     update.append(subject_data)
                # source_data = update
                # source_data = [subject_data.astype(np.float32) for subject_data in source_data]


            """Use filter bank """
            if self.useFilterBank:
                # diff = 4
                diff = self.cfg.DATAMANAGER.DATASET.FILTERBANK.freq_interval
                filter_bands = []
                # axis = 2
                for i in range(1, 9):
                    filter_bands.append([i * diff, (i + 1) * diff])
                print("build filter band : ", filter_bands)
                filter = filterBank(
                    filtBank=filter_bands,
                    fs=128
                )
                source = [0, 1, 2, 3]
                destination = [0, 2, 3, 1]
                source_data = subjects_filterbank(source_data, filter, source, destination)


            """Apply data transformation/normalization"""
            if not self.cfg.INPUT.NO_TRANSFORM:
                normalization = self.cfg.INPUT.TRANSFORMS[0]
                if normalization == 'cross_channel_norm':
                    source_data= dataset_norm(source_data)
                elif normalization == 'time_norm':
                    source_data= dataset_norm(source_data, norm_channels=False)
            for subject_idx in range(len(source_data)):
                print("source_data subject_idx {} has shape : {}, with range scale ({},{}) ".format(
                    subject_idx, source_data[subject_idx].shape,
                    np.max(source_data[subject_idx]), np.min(source_data[subject_idx])))

            source_data = self._expand_data_dim(source_data)
            train_u = generate_datasource(source_data,source_label)
            self.list_train_u.append(train_u)

        # source_domain_class_weight = self.generate_domain_class_weight(source_label_list)

        source_domain_num_class = [len(source_domain_class_weight[i]) for i in range(len(source_domain_class_weight))]
        source_num_domain = len(source_domain_num_class)
        source_domain_input_shapes = [source_data_list[source_domain][0][0].shape for source_domain in range(source_num_domain) ]


        # print("source domain class weight : ", source_domain_class_weight)

        self._num_source_domains = source_num_domain
        self._list_source_domain_class_weight = source_domain_class_weight
        self._list_source_domain_label_size = source_domain_num_class
        self._list_source_domain_input_shapes = source_domain_input_shapes

    def train_dataloader(self):
        train_x_dataloder = super(MultiDomainDataManagerV1, self).train_dataloader()
        # list_train_u_loader = list()
        # train_u_loaders = dict()
        train_u_loaders = list()

        if self.list_train_u_exist:
            list_sampler_type_ = self.cfg.DATAMANAGER.DATALOADER.LIST_TRAIN_U.SAMPLERS
            list_batch_size = self.cfg.DATAMANAGER.DATALOADER.LIST_TRAIN_U.BATCH_SIZES
            print("total source domain : ",self.num_source_domains)
            print("available batch for source domain : ",len(list_batch_size))
            assert self.num_source_domains == len(list_batch_size)
            for source_domain_idx in range(self.num_source_domains):
                current_train_u = self.list_train_u[source_domain_idx]
                current_batch_size = list_batch_size[source_domain_idx]
                current_sampler_type = list_sampler_type_[source_domain_idx]
                current_train_u = self.dataset_wrapper(self.cfg, current_train_u, is_train=True)

                train_loader_u = build_data_loader(
                    self.cfg,
                    sampler_type=current_sampler_type,
                    data_source=current_train_u,
                    batch_size=current_batch_size,
                    is_train=True,
                )
                # train_u_loaders[str(source_domain_idx)] = train_loader_u
                train_u_loaders.append(train_loader_u)
                # list_train_u_loader.append(train_loader_u)
        # self.list_train_u_loader = list_train_u_loader



        return {"target_loader":train_x_dataloder,"source_loader":train_u_loaders}

    @property
    def source_domains_class_weight(self):
        return self._list_source_domain_class_weight
    @property
    def source_domains_label_size(self):
        return self._list_source_domain_label_size
    @property
    def source_domains_input_shape(self):
        return self._list_source_domain_input_shapes
    @property
    def num_source_domains(self):
        return self._num_source_domains

    def get_require_parameter(self):
        require_parameter = super().get_require_parameter()
        # require_parameter = dict()
        # require_parameter['num_classes'] = self.num_classes
        require_parameter['source_domains_input_shape'] = self.source_domains_input_shape
        require_parameter['source_domains_label_size'] = self.source_domains_label_size
        require_parameter['source_domains_class_weight'] = self.source_domains_class_weight


        return require_parameter

class MultiDomainDataManagerV2(MultiDomainDataManagerV1):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup(self, stage: Optional[str] = None) :
        super().setup(stage)
        train_u_data = self._dataset.train_u
        # train_u_same_as_test = False
        if train_u_data is None:
            print("there is no provide unlabel data. We need to use test data as unlabel")
            test_dataset = self._dataset.test
            test_data, _ = test_dataset
            train_u_data = test_data
            # train_u_same_as_test = True
            subject_ids = self._dataset.pick_test_subjects
        else:
            subject_ids = [i for i in range(len(train_u_data))]

        # train_u_data = self.train_u

        if self.cfg.DATAMANAGER.DATASET.USE_Euclidean_Aligment:
            if self._dataset.r_op_list is not None:
                # if train_u_same_as_test:
                self.train_u_EA = EuclideanAlignment(list_r_op=self._dataset.r_op_list,subject_ids=subject_ids)
                # else:
                #     self.train_u_EA = EuclideanAlignment(list_r_op=self._dataset.r_op_list)
            else:
                self.train_u_EA = EuclideanAlignment()
            train_u_data = self.train_u_EA.convert_subjects_data_with_EA(train_u_data)

        """Apply data transformation/normalization"""
        if not self.cfg.INPUT.NO_TRANSFORM:
            normalization = self.cfg.INPUT.TRANSFORMS[0]
            if normalization == 'cross_channel_norm':
                train_u_data = dataset_norm(train_u_data)
            elif normalization == 'time_norm':
                train_u_data= dataset_norm(train_u_data, norm_channels=False)
        for subject_idx in range(len(train_u_data)):
            print("unlabel data subject_idx {} has shape : {}, with range scale ({},{}) ".format(
                subject_ids[subject_idx], train_u_data[subject_idx].shape,
                np.max(train_u_data[subject_idx]), np.min(train_u_data[subject_idx])))

        train_u_data = self._expand_data_dim(train_u_data)
        train_u_data = np.concatenate(train_u_data)
        self.train_u = train_u_data


    def train_dataloader(self):
        loader_dict = super(MultiDomainDataManagerV2, self).train_dataloader()
        train_u = Wrapper(self.train_u)
        train_loader_u = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATAMANAGER.DATALOADER.TRAIN_U.SAMPLER,
            data_source=train_u,
            batch_size=self.cfg.DATAMANAGER.DATALOADER.TRAIN_U.BATCH_SIZE,
            is_train=True,
        )
        loader_dict.update({
            "unlabel_loader":train_loader_u
        })
        print("loader dict : ",loader_dict)
        return loader_dict


def build_data_loader(
        cfg,
        sampler_type='default',
        data_source=None,
        batch_size=64,
        n_domain=0,
        is_train=True,
):
    if sampler_type != 'default':
        # Build sampler
        sampler = build_sampler(
            sampler_type,
            cfg=cfg,
            data_source=data_source,
            batch_size=batch_size,
            n_domain=n_domain
        )
    else:
        sampler = None

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATAMANAGER.DATALOADER.NUM_WORKERS,
        drop_last=is_train,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    return data_loader


# class

# class CustomEEGDatasetWrapper(TorchDataset):
#
#     def __init__(self, cfg, data_source, transform=None, is_train=False):
#         self.cfg = cfg
#         self.data_source = data_source
#         # transform accepts list (tuple) as input
#         self.transform = transform
#         self.is_train = is_train
#
#     def __len__(self):
#         return len(self.data_source)
#
#     def __getitem__(self, idx):
#         item = self.data_source[idx]
#         output = {
#             'label': torch.tensor(item.label),
#             'domain': torch.tensor(item.domain),
#             'eeg_data': torch.from_numpy(item.eeg_data)
#         }
#         return output

class CustomWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        # transform accepts list (tuple) as input
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        return item.eeg_data,item.label,item.domain

class Wrapper(TorchDataset):

    def __init__(self,data_source):
        self.data_source = data_source
    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        return item