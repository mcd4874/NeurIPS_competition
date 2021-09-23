import torch
from torch.nn import functional as F
import torch.nn as nn
from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.modeling.ops import ReverseGrad
from dassl.engine.trainer_tmp import SimpleNet
from torch.utils.data import Dataset as TorchDataset
from dassl.utils import MetricMeter

import numpy as np


def genenerate_mix_feature(feature,softmax_output):
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    return op_out.view(-1, softmax_output.size(1) * feature.size(1))

@TRAINER_REGISTRY.register()
class CDAN(TrainerXU):
    """

    https://arxiv.org/abs/1705.10667 (computer vision version)
    https://www.mdpi.com/1099-4300/22/1/96 (EEG motor imaginary example)

    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)
                # print("torch weight  : ", torch_weight)
                # self.ce = nn.CrossEntropyLoss(weight=torch_weight)
        self.build_domain_discriminator()

    def build_domain_discriminator(self):
        cfg = self.cfg
        print('Building critic network')
        fdim = self.model.fdim
        d_fim = fdim * self.num_classes
        self.D =  nn.Linear(d_fim, 1)
        print('# params: {:,}'.format(count_num_param(self.D)))
        self.D.to(self.device)
        self.optim_d = build_optimizer(self.D, cfg.OPTIM)
        self.sched_d = build_lr_scheduler(self.optim_d, cfg.OPTIM)
        self.register_model('D', self.D, self.optim_d, self.sched_d)
        self.revgrad = ReverseGrad()


    # def build_model(self):
    #     cfg = self.cfg
    #
    #     print('Building F')
    #     # print("Params key : ",cfg.MODEL.BACKBONE.PARAMS.keys())
    #     self.F = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
    #     self.F.to(self.device)
    #     print('# params: {:,}'.format(count_num_param(self.F)))
    #     self.optim_F = build_optimizer(self.F, cfg.OPTIM)
    #     self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
    #     self.register_model('F', self.F, self.optim_F, self.sched_F)
    #     fdim = self.F.fdim
    #
    #     print('Building Classifier')
    #     self.C = nn.Linear(fdim, self.num_classes)
    #     self.C.to(self.device)
    #     print('# params: {:,}'.format(count_num_param(self.C)))
    #     print(self.dm.num_source_domains)
    #     self.optim_C = build_optimizer(self.C, cfg.OPTIM)
    #     self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
    #     self.register_model('C', self.CC, self.optim_C, self.sched_C)
    #
    #     print('Building D')
    #     self.D = nn.Linear(fdim, self.dm.num_source_domains)
    #     self.D.to(self.device)
    #     print('# params: {:,}'.format(count_num_param(self.D)))
    #     self.optim_D = build_optimizer(self.D, cfg.OPTIM)
    #     self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
    #     self.register_model('D', self.D, self.optim_D, self.sched_D)
    #     self.revgrad = ReverseGrad()

        # print('Building C')
        # self.C = nn.Linear(fdim, self.num_classes)
        # self.C.to(self.device)
        # print('# params: {:,}'.format(count_num_param(self.C)))
        # self.optim_C = build_optimizer(self.C, cfg.OPTIM)
        # self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
        # self.register_model('C', self.D, self.optim_C, self.sched_C)

    def forward_backward(self,batch_x,batch_u, backprob=True):
        input_x, label_x, _,input_u = self.parse_batch_train(batch_x, batch_u)
        domain_x = torch.ones(input_x.shape[0], 1).to(self.device)
        domain_u = torch.zeros(input_u.shape[0], 1).to(self.device)

        global_step = self.batch_idx + self.epoch * self.num_batches
        progress = global_step / (self.max_epoch * self.num_batches)
        lmda = 2 / (1 + np.exp(-10 * progress)) - 1



        logit_x, feat_x = self.model(input_x, return_feature=True)
        logit_u, feat_u = self.model(input_u, return_feature=True)

        loss_x = self.ce(logit_x, label_x)

        softmax_output_x = F.softmax(logit_x,dim=1)
        softmax_output_u = F.softmax(logit_u,dim=1)

        feat_x = self.revgrad(feat_x, grad_scaling=lmda)
        feat_u = self.revgrad(feat_u, grad_scaling=lmda)

        mix_feature_x = genenerate_mix_feature(feat_x,softmax_output_x)
        mix_feature_u = genenerate_mix_feature(feat_u, softmax_output_u)

        output_xd = self.D(mix_feature_x)
        output_ud = self.D(mix_feature_u)
        loss_d = self.bce(output_xd, domain_x) + self.bce(output_ud, domain_u)

        loss = loss_x + loss_d
        if backprob:
            self.model_backward_and_update(loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        loss_summary = {
            'loss_x': loss_x.item(),
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

        self.num_batches = len(data_loader)
        valid_loader_x_iter = iter(data_loader)
        loader_u_iter = iter(self.train_loader_u)
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(valid_loader_x_iter)
            except StopIteration:
                valid_loader_x_iter = iter(data_loader)
                batch_x = next(valid_loader_x_iter)

            try:
                batch_u = next(loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            input, label, domain, target = self.parse_batch_train(batch_x, batch_u)
            loss = self.forward_backward(batch_x, batch_u, backprob=False)
            losses.update(loss)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss_x'].avg

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
    #             self.epoch, self._best_epoch_val_loss, epoch_total_loss))
    #         self._best_epoch_val_loss = epoch_total_loss
    #         self.save_model(epoch=self.epoch, directory=self.output_dir, is_best=True)
    #     super().after_epoch()

    def model_inference(self, input):
        logit = self.model(input)
        probs =  F.softmax(logit,dim=1)
        return probs