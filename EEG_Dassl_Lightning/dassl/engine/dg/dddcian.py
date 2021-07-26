import torch
from torch.nn import functional as F
import torch.nn as nn
from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.modeling.ops import ReverseGrad
from dassl.engine.trainer_tmp import SimpleNet
from torch.utils.data import Dataset as TorchDataset
from dassl.utils import MetricMeter

import numpy as np
@TRAINER_REGISTRY.register()
class CIAN(TrainerX):
    """
    Modification of Conditional Invariant model
    https://arxiv.org/abs/1807.08479
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        if n_domain <= 0:
            n_domain = self.dm.num_source_domains
        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain

        #create a cross entropy loss for whole dataset
        self.ce = nn.CrossEntropyLoss()
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                print("torch weight  : ",torch_weight)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)

        #create cross entropy losses for each domain in the dataset
        self.cce = [nn.CrossEntropyLoss() for _ in range(self.n_domain)]
        if cfg.DATASET.DOMAIN_CLASS_WEIGHT:
            domain_class_weight = self.dm.dataset.domain_class_weight
            for domain,weight in domain_class_weight.items():
                # domain_class_weight[domain] = torch.from_numpy(np.array(weight)).float().to(self.device)
                torch_weight = torch.from_numpy(np.array(weight)).float().to(self.device)
                self.cce[domain] = nn.CrossEntropyLoss(weight=torch_weight)

        self.dce = nn.CrossEntropyLoss()

        self._best_epoch_val_loss = 10000





    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == 'RandomDomainSampler'


    # def build_data_loader(self):
    #     """Create essential data-related attributes.
    #
    #     What must be done in the re-implementation
    #     of this method:
    #     1) initialize data manager
    #     2) assign as attributes the data loaders
    #     3) assign as attribute the number of classes
    #     """
    #     self.dm = DataManager(self.cfg,dataset_wrapper=CustomDatasetWrapper)
    #     self.dm._num_source_domains = self.cfg.DATALOADER.TRAIN_X.N_DOMAIN
    #     self.train_loader_x = self.dm.train_loader_x
    #     self.val_loader = self.dm.val_loader
    #     self.test_loader = self.dm.test_loader
    #     self.num_classes = self.dm.num_classes
    def build_model(self):
        cfg = self.cfg

        print('Building F')
        # print("Params key : ",cfg.MODEL.BACKBONE.PARAMS.keys())
        self.F = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print('Building CC')
        self.CC = nn.ModuleList(
            [
                nn.Linear(fdim, self.num_classes)
                for _ in range(self.dm.num_source_domains)
            ]
        )
        self.CC.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.CC)))
        print(self.dm.num_source_domains)
        self.optim_CC = build_optimizer(self.CC, cfg.OPTIM)
        self.sched_CC = build_lr_scheduler(self.optim_CC, cfg.OPTIM)
        self.register_model('CC', self.CC, self.optim_CC, self.sched_CC)

        print('Building D')
        self.D = nn.Linear(fdim, self.dm.num_source_domains)
        self.D.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.D)))
        self.optim_D = build_optimizer(self.D, cfg.OPTIM)
        self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
        self.register_model('D', self.D, self.optim_D, self.sched_D)
        self.revgrad = ReverseGrad()

        # print('Building C')
        # self.C = nn.Linear(fdim, self.num_classes)
        # self.C.to(self.device)
        # print('# params: {:,}'.format(count_num_param(self.C)))
        # self.optim_C = build_optimizer(self.C, cfg.OPTIM)
        # self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
        # self.register_model('C', self.D, self.optim_C, self.sched_C)



    def forward_backward(self, batch,backprob = True):
        input_x, label_x, domain_x = self.parse_batch_train(batch)

        input_x = torch.split(input_x, self.split_batch, 0)
        label_x = torch.split(label_x, self.split_batch, 0)
        domain_x = torch.split(domain_x, self.split_batch, 0)
        d_x = [d[0].item() for d in domain_x]
        # Step A
        loss_cc = 0
        loss_d = 0
        feat_ = []

        global_step = self.batch_idx + self.epoch * self.num_batches
        progress = global_step / (self.max_epoch * self.num_batches)
        lmda = 2 / (1 + np.exp(-10 * progress)) - 1

        for x, y, dy, d in zip(input_x, label_x, domain_x,d_x):
            feat = self.F(x)

            # C_logits = self.C(feat)

            CC_logits = self.CC[d](feat)

            loss_cc += self.cce[d](CC_logits,y)


            feat_x = self.revgrad(feat)
            D_logits = self.D(feat_x)
            loss_d += self.dce(D_logits,dy)

        loss_cc = loss_cc/self.n_domain
        loss_d = loss_d/self.n_domain
        alpha = 0.2
        loss = loss_cc + alpha*loss_d
        if backprob:
            self.model_backward_and_update(loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        loss_summary = {
            'loss_x': loss_cc.item(),
            'loss_d': loss_d.item(),
            'total_loss':loss.item()
        }
        return loss_summary

    @torch.no_grad()
    # def validate(self,full_results = False):
    def validate(self):

        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        losses = MetricMeter()

        print('Do evaluation on {} set'.format('valid set'))
        data_loader = self.val_loader
        assert data_loader is not None
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            loss = self.forward_backward(batch, backprob=False)
            losses.update(loss)
            # total_loss += loss['loss']
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['total_loss'].avg

        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        # if full_results:
        return [total_loss,losses.dict_results(),results]
        # return total_loss

    # def after_epoch(self):
    #     """
    #     save the best model for given validation loss
    #     """
    #     epoch_total_loss = self.validate()
    #     if self._best_epoch_val_loss > epoch_total_loss:
    #         print("save best model at epoch %f , Improve loss from %4f -> %4f" % (
    #         self.epoch, self._best_epoch_val_loss, epoch_total_loss))
    #         self._best_epoch_val_loss = epoch_total_loss
    #         self.save_model(epoch=self.epoch, directory=self.output_dir, is_best=True)
    #     super().after_epoch()


    def model_inference(self, input):
        f = self.F(input)
        p = 0
        for i in [0,3,4,5,6,7,8,9,10,13,14,15]:
            C_i = self.CC[i]
            z = C_i(f)
            p += F.softmax(z, 1)
        p= p/ 12
        # for C_i in self.CC:
        #     z = C_i(f)
        #     p += F.softmax(z, 1)
        # p = p / len(self.CC)
        return p

    def parse_batch_train(self, batch_x):
        input_x = batch_x['eeg_data']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        domain_x =domain_x.to(self.device)
        return input_x, label_x,domain_x
    def parse_batch_test(self, batch):
        input = batch['eeg_data']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label




