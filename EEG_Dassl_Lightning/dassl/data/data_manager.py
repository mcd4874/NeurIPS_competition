import torch
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import build_transform


def build_data_loader(
    cfg,
    sampler_type='SequentialSampler',
    data_source=None,
    batch_size=64,
    n_domain=0,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain
    )

    if dataset_wrapper is None:
        # dataset_wrapper = DatasetWrapper
        print("use customEEGDatasetWrapper")
        dataset_wrapper = CustomEEGDatasetWrapper
    else:
        print("use provided dataset wrapper")
    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    # data_loader = torch.utils.data.DataLoader(
    #     dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     num_workers=cfg.DATALOADER.NUM_WORKERS,
    #     drop_last=is_train,
    #     pin_memory=False
    # )

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        # if custom_tfm_train is None:
        #     self.tfm_train = build_transform(cfg, is_train=True)
        # else:
        #     print('* Using custom transform for training')
        #     self.tfm_train = custom_tfm_train
        #
        # if custom_tfm_test is None:
        #     self.tfm_test = build_transform(cfg, is_train=False)
        # else:
        #     print('* Using custom transform for testing')
        #     self.tfm_test = custom_tfm_test

        self.tfm_train = None
        self.tfm_test = None
        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            tfm=self.tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                tfm=self.tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )



        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.VALID.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.VALID.BATCH_SIZE,
                n_domain=cfg.DATALOADER.VALID.N_DOMAIN,
                tfm=self.tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=self.tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        #Build list of test_loader for each test subject data
        list_subject_test = dataset.list_subject_test
        list_subject_test_loader = []
        for source_domain_idx in range(len(list_subject_test)):
            current_test = list_subject_test[source_domain_idx]
            train_loader_u = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=current_test,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=self.tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )
            list_subject_test_loader.append(train_loader_u)




        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS) if len(cfg.DATASET.SOURCE_DOMAINS)>0 else dataset.data_domains
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self._dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.list_subject_test_loader = list_subject_test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def dataset(self):
        return self._dataset

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        print('***** Dataset statistics *****')

        print('  Dataset: {}'.format(cfg.DATASET.NAME))

        if cfg.DATASET.SOURCE_DOMAINS:
            print('  Source domains: {}'.format(cfg.DATASET.SOURCE_DOMAINS))
        if cfg.DATASET.TARGET_DOMAINS:
            print('  Target domains: {}'.format(cfg.DATASET.TARGET_DOMAINS))

        print('  # classes: {}'.format(self.num_classes))

        print('  # train_x: {:,}'.format(len(self._dataset.train_x)))

        if self._dataset.train_u:
            print('  # train_u: {:,}'.format(len(self._dataset.train_u)))

        if self._dataset.val:
            print('  # val: {:,}'.format(len(self._dataset.val)))

        print('  # test: {:,}'.format(len(self._dataset.test)))

class MultiDomainDataManager(DataManager):
    def __init__(self, cfg):
        super().__init__(cfg)

        list_train_u_exist = hasattr(self._dataset, 'list_train_u')
        list_train_u_loader = list()
        if list_train_u_exist:
            list_sampler_type_ = cfg.DATALOADER.LIST_TRAIN_U.SAMPLERS
            list_batch_size_ = cfg.DATALOADER.LIST_TRAIN_U.BATCH_SIZES
            list_train_u = self._dataset.list_train_u
            dataset_wrapper = CustomEEGDatasetWrapper

            for source_domain_idx in range(self._dataset.source_num_domain):
                current_train_u = list_train_u[source_domain_idx]
                current_batch_size = list_batch_size_[source_domain_idx]
                current_sampler_type = list_sampler_type_[source_domain_idx]
                train_loader_u = build_data_loader(
                    cfg,
                    sampler_type=current_sampler_type,
                    data_source=current_train_u,
                    batch_size=current_batch_size,
                    tfm=self.tfm_train,
                    is_train=True,
                    dataset_wrapper=dataset_wrapper
                )
                list_train_u_loader.append(train_loader_u)
        self.list_train_u_loader = list_train_u_loader
        self._num_source_domains = len(self.list_train_u_loader)


        self._list_source_domain_class_weight = self._dataset.source_domain_class_weight
        self._list_source_domain_label_size = self._dataset.source_domain_num_class
        self._list_source_domain_input_shapes = self._dataset.source_domain_input_shapes

    @property
    def source_domains_class_weight(self):
        return self._list_source_domain_class_weight
    @property
    def source_domains_label_size(self):
        return self._list_source_domain_label_size
    @property
    def source_domains_input_shape(self):
        return self._list_source_domain_input_shapes


class CustomEEGDatasetWrapper(TorchDataset):

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

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    transform_EEG_data = self._transform_data(tfm, item.eeg_data)
                    item.update_eeg_data(transform_EEG_data)
            else:
                transform_EEG_data = self._transform_data(self.transform, item.eeg_data)
                item.update_eeg_data(transform_EEG_data)

        output = {
            'label': torch.tensor(item.label),
            'domain': torch.tensor(item.domain),
            'eeg_data': torch.from_numpy(item.eeg_data)
        }
        return output

    def _transform_data(self, tfm, data):
        return tfm(data)

