from dassl.engine import TRAINER_REGISTRY,TrainerXU
from dassl.data import DataManager
from dassl.utils import MetricMeter
from torch.utils.data import Dataset as TorchDataset
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.engine.trainer_tmp import SimpleNet
import numpy as np
# from torch.optim.lr_scheduler import _LRScheduler
class PairClassifiers(nn.Module):

    def __init__(self, fdim, num_classes):
        super().__init__()
        self.c1 = nn.Linear(fdim, num_classes)
        self.c2 = nn.Linear(fdim, num_classes)

    def forward(self, x):
        z1 = self.c1(x)
        # if not self.training:
        #     return z1
        z2 = self.c2(x)
        return z1, z2


@TRAINER_REGISTRY.register()
class CustomM3SDA(TrainerXU):
    """Moment Matching for Multi-Source Domain Adaptation.

    https://arxiv.org/abs/1812.01754.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        if n_domain <= 0:
            n_domain = self.dm.num_source_domains
        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain

        self.n_step_F = cfg.TRAINER.M3SDA.N_STEP_F
        self.lmda = cfg.TRAINER.M3SDA.LMDA




    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == 'RandomDomainSampler'
        assert not cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_model(self):
        cfg = self.cfg
        print("Params : ",cfg.MODEL.BACKBONE)
        print('Building F')

        self.F = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print('Building C')
        """Need to fix this not hard code"""
        print("num source domain : ",self.dm.num_source_domains)

        self.C = nn.ModuleList(
            [
                PairClassifiers(fdim, self.num_classes)
                for _ in range(self.dm.num_source_domains)
            ]
        )
        self.C.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
        self.register_model('C', self.C, self.optim_C, self.sched_C)


    def forward_backward(self, batch_x, batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, batch_u)
        input_x, label_x, domain_x, input_u = parsed
        input_x = torch.split(input_x, self.split_batch, 0)
        label_x = torch.split(label_x, self.split_batch, 0)
        domain_x = torch.split(domain_x, self.split_batch, 0)
        domain_x = [d[0].item() for d in domain_x]
        # Step A
        loss_x = 0
        feat_x = []

        for x, y, d in zip(input_x, label_x, domain_x):
            f = self.F(x)
            z1, z2 = self.C[d](f)
            loss_x += F.cross_entropy(z1, y) + F.cross_entropy(z2, y)

            feat_x.append(f)

        loss_x /= self.n_domain

        feat_u = self.F(input_u)
        loss_msda = self.moment_distance(feat_x, feat_u)

        loss_step_A = loss_x + loss_msda * self.lmda
        if backprob:
            self.model_backward_and_update(loss_step_A)

        # Step B
        with torch.no_grad():
            feat_u = self.F(input_u)

        loss_x, loss_dis = 0, 0

        for x, y, d in zip(input_x, label_x, domain_x):
            with torch.no_grad():
                f = self.F(x)
            z1, z2 = self.C[d](f)
            loss_x += F.cross_entropy(z1, y) + F.cross_entropy(z2, y)

            z1, z2 = self.C[d](feat_u)
            p1 = F.softmax(z1, 1)
            p2 = F.softmax(z2, 1)
            loss_dis += self.discrepancy(p1, p2)

        loss_x /= self.n_domain
        loss_dis /= self.n_domain

        loss_step_B = loss_x - loss_dis
        if backprob:
            self.model_backward_and_update(loss_step_B, 'C')

        # Step C
        for _ in range(self.n_step_F):
            feat_u = self.F(input_u)

            loss_dis = 0

            for d in domain_x:
                z1, z2 = self.C[d](feat_u)
                p1 = F.softmax(z1, 1)
                p2 = F.softmax(z2, 1)
                loss_dis += self.discrepancy(p1, p2)

            loss_dis /= self.n_domain
            loss_step_C = loss_dis
            if backprob:
                self.model_backward_and_update(loss_step_C, 'F')

        loss_summary = {
            'loss_step_A': loss_step_A.item(),
            'loss_step_B': loss_step_B.item(),
            'loss_step_C': loss_step_C.item()
        }

        if backprob:
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

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

        num_batches = len(data_loader)
        valid_loader_x_iter = iter(data_loader)
        loader_u_iter = iter(self.train_loader_u)
        for self.batch_idx in range(num_batches):
            try:
                batch_x = next(valid_loader_x_iter)
            except StopIteration:
                valid_loader_x_iter = iter(data_loader)
                batch_x = next(valid_loader_x_iter)

            try:
                batch_u = next(loader_u_iter)
            except StopIteration:
                loader_u_iter = iter(self.train_loader_u)
                batch_u = next(loader_u_iter)

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

    def moment_distance(self, x, u):
        # x (list): a list of feature matrix.
        # u (torch.Tensor): feature matrix.
        x_mean = [xi.mean(0) for xi in x]
        u_mean = u.mean(0)
        dist1 = self.pairwise_distance(x_mean, u_mean)

        x_var = [xi.var(0) for xi in x]
        u_var = u.var(0)
        dist2 = self.pairwise_distance(x_var, u_var)

        return (dist1+dist2) / 2

    def pairwise_distance(self, x, u):
        # x (list): a list of feature vector.
        # u (torch.Tensor): feature vector.
        dist = 0
        count = 0

        for xi in x:
            dist += self.euclidean(xi, u)
            count += 1

        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                dist += self.euclidean(x[i], x[j])
                count += 1

        return dist / count

    def euclidean(self, input1, input2):
        return ((input1 - input2)**2).sum().sqrt()

    def discrepancy(self, y1, y2):
        return (y1 - y2).abs().mean()



    def model_inference(self, input):
        """only use c1 from each domain for classifier score"""

        f = self.F(input)
        p = 0
        for C_i in self.C:
            z,_ = C_i(f)
            p += F.softmax(z, 1)
        p = p / len(self.C)
        return p
