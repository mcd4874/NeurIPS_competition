from dassl.engine import TRAINER_REGISTRY,TrainerXU
from dassl.data import DataManager
from torch.utils.data import Dataset as TorchDataset
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.engine.trainer_tmp import SimpleNet
from dassl.utils import MetricMeter
import numpy as np

@TRAINER_REGISTRY.register()
class CustomMCD(TrainerXU):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_step_F = cfg.TRAINER.CustomMCD.N_STEP_F
        self._best_epoch_val_loss = 10000

        self.ce = nn.CrossEntropyLoss()
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)

    def build_model(self):
        cfg = self.cfg

        print('Building F')
        self.F = SimpleNet(cfg, cfg.MODEL, 0,**cfg.MODEL.BACKBONE.PARAMS)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print('Building C1')
        print("fdim : ",fdim)
        print("num_classes : ",self.num_classes)
        self.C1 = nn.Linear(fdim, self.num_classes)
        self.C1.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.C1)))
        self.optim_C1 = build_optimizer(self.C1, cfg.OPTIM)
        self.sched_C1 = build_lr_scheduler(self.optim_C1, cfg.OPTIM)
        self.register_model('C1', self.C1, self.optim_C1, self.sched_C1)

        print('Building C2')
        self.C2 = nn.Linear(fdim, self.num_classes)
        self.C2.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.C2)))
        self.optim_C2 = build_optimizer(self.C2, cfg.OPTIM)
        self.sched_C2 = build_lr_scheduler(self.optim_C2, cfg.OPTIM)
        self.register_model('C2', self.C2, self.optim_C2, self.sched_C2)

    def forward_backward(self, batch_x, batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, batch_u)
        input_x, label_x,_ ,input_u = parsed

        # Step A
        feat_x = self.F(input_x)
        logit_x1 = self.C1(feat_x)
        logit_x2 = self.C2(feat_x)
        # loss_x1 = F.cross_entropy(logit_x1, label_x)
        # loss_x2 = F.cross_entropy(logit_x2, label_x)

        loss_x1 = self.ce(logit_x1, label_x)
        loss_x2 = self.ce(logit_x2, label_x)

        loss_step_A = loss_x1 + loss_x2
        if backprob:
            self.model_backward_and_update(loss_step_A)

        # Step B
        with torch.no_grad():
            feat_x = self.F(input_x)
        logit_x1 = self.C1(feat_x)
        logit_x2 = self.C2(feat_x)
        # loss_x1 = F.cross_entropy(logit_x1, label_x)
        # loss_x2 = F.cross_entropy(logit_x2, label_x)
        loss_x1 = self.ce(logit_x1, label_x)
        loss_x2 = self.ce(logit_x2, label_x)
        loss_x = loss_x1 + loss_x2

        with torch.no_grad():
            feat_u = self.F(input_u)
        pred_u1 = F.softmax(self.C1(feat_u), 1)
        pred_u2 = F.softmax(self.C2(feat_u), 1)
        loss_dis = self.discrepancy(pred_u1, pred_u2)

        loss_step_B = loss_x - loss_dis
        if backprob:
            self.model_backward_and_update(loss_step_B, ['C1', 'C2'])
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        # Step C
        for _ in range(self.n_step_F):
            feat_u = self.F(input_u)
            pred_u1 = F.softmax(self.C1(feat_u), 1)
            pred_u2 = F.softmax(self.C2(feat_u), 1)
            loss_step_C = self.discrepancy(pred_u1, pred_u2)
            if backprob:
                self.model_backward_and_update(loss_step_C, 'F')

        loss_summary = {
            'loss_step_A': loss_step_A.item(),
            'loss_step_B': loss_step_B.item(),
            'loss_step_C': loss_step_C.item()
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
        total_loss = losses.meters['loss_step_A'].avg

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

    def discrepancy(self, y1, y2):
        return (y1 - y2).abs().mean()

    def model_inference(self, input):
        feat = self.F(input)
        return F.softmax(self.C1(feat),dim=1)


