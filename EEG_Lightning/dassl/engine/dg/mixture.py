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
class mixup(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.ce = nn.CrossEntropyLoss()
        # if cfg.DATASET.TOTAL_CLASS_WEIGHT:
        #     total_data_class_weight = self.dm.dataset.whole_class_weight
        #     if total_data_class_weight is not None:
        #         torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
        #         print("torch weight  : ", torch_weight)
        #         self.ce = nn.CrossEntropyLoss(weight=torch_weight)

        self.loss_fn = torch.nn.KLDivLoss('batchmean', ).cuda()
        self.mixup = cfg.TRAINER.PARAMS.MIXUP #2.0
        self.label_smooth = cfg.TRAINER.PARAMS.LABEL_SMOOTH
        self._best_epoch_val_loss = 10000

    def build_model(self):
        cfg = self.cfg

        print('Building F')
        # print("Params key : ",cfg.MODEL.BACKBONE.PARAMS.keys())
        self.F = SimpleNet(cfg, cfg.MODEL, self.num_classes, **cfg.MODEL.BACKBONE.PARAMS)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)

    def forward_backward(self, batch,backprob = True):
        lam_mu = np.random.beta(self.mixup, self.mixup)
        input_x, label_x = self.parse_batch_train(batch)

        one_hot_label_x = one_hot(label_x,self.num_classes)
        #smooth the one hot label
        one_hot_label_x -= self.label_smooth * (one_hot_label_x - 1 / (self.num_classes + 1))

        mixers = torch.randperm(input_x.shape[0]).cuda()
        x = lam_mu * input_x + (1 - lam_mu) * input_x[mixers]
        y = lam_mu * one_hot_label_x + (1 - lam_mu) * one_hot_label_x[mixers]

        y_pred = self.F(x)
        y_pred = F.log_softmax(y_pred,dim=-1)
        loss_x = self.loss_fn(y_pred,y)
        if backprob:
            self.model_backward_and_update(loss_x, 'F')
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        loss_summary = {
            'loss': loss_x.item()
        }

        return loss_summary

    @torch.no_grad()
    # def validate(self, full_results=False):
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
            one_hot_label_x = one_hot(label, self.num_classes)
            logits = self.F(input)
            y_preds = F.log_softmax(logits,dim=-1)
            loss = self.loss_fn(y_preds,one_hot_label_x)
            loss_summary = {
                'loss': loss.item()
            }
            losses.update(loss_summary)
            # total_loss += loss['loss']
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss'].avg

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
        feat = self.F(input)
        preds = F.softmax(feat, 1)
        return preds
    def parse_batch_train(self, batch):
        input = batch['eeg_data']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label



def one_hot(y: torch.Tensor, num_classes):
    """ 1-hot encodes a tensor to another similarly stored tensor"""
    if len(y.shape) > 0 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    out = torch.zeros(y.size() + torch.Size([num_classes]), device=y.device)
    return out.scatter_(-1, y.view((*y.size(), 1)), 1)