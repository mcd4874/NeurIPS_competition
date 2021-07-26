import torch
from torch.nn import functional as F
import torch.nn as nn
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.engine.trainer_tmp import SimpleNet
import numpy as np
from dassl.utils import MetricMeter


@TRAINER_REGISTRY.register()
class CrossGrad(TrainerX):
    """Cross-gradient training.

    https://arxiv.org/abs/1804.10745.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps_f = cfg.TRAINER.CG.EPS_F
        self.eps_d = cfg.TRAINER.CG.EPS_D
        self.alpha_f = cfg.TRAINER.CG.ALPHA_F
        self.alpha_d = cfg.TRAINER.CG.ALPHA_D

        # self.ce = nn.CrossEntropyLoss()
        # self.ce_1 = nn.CrossEntropyLoss()
        self.class_weight = None
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                print("torch weight  : ", torch_weight)
                self.class_weight = torch_weight
                # self.ce = nn.CrossEntropyLoss(weight=torch_weight)


    def build_model(self):
        cfg = self.cfg

        print('Building F')
        self.F =  SimpleNet(cfg, cfg.MODEL, self.num_classes, **cfg.MODEL.BACKBONE.PARAMS)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)

        print('Building D')
        print("num domain : ",self.dm.num_source_domains)
        self.D = SimpleNet(cfg, cfg.MODEL, self.dm.num_source_domains,**cfg.MODEL.BACKBONE.PARAMS)
        self.D.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.D)))
        self.optim_D = build_optimizer(self.D, cfg.OPTIM)
        self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
        self.register_model('D', self.D, self.optim_D, self.sched_D)

    def forward_backward(self, batch, backprob=True):
        input, label, domain = self.parse_batch_train(batch)
        if backprob:
            input.requires_grad = True

            # Compute domain perturbation
            loss_d = F.cross_entropy(self.D(input), domain)
            loss_d.backward()
            grad_d = torch.clamp(input.grad.data, min=-0.1, max=0.1)
            input_d = input.data + self.eps_f * grad_d

            # Compute label perturbation
            input.grad.data.zero_()
            # loss_f = F.cross_entropy(self.F(input), label)
            loss_f = F.cross_entropy(self.F(input), label,weight=self.class_weight)
            loss_f.backward()
            grad_f = torch.clamp(input.grad.data, min=-0.1, max=0.1)
            input_f = input.data + self.eps_d * grad_f

            input = input.detach()

            # Update label net
            # loss_f1 = F.cross_entropy(self.F(input), label)
            # loss_f2 = F.cross_entropy(self.F(input_d), label)
            loss_f1 = F.cross_entropy(self.F(input), label,weight=self.class_weight)
            loss_f2 = F.cross_entropy(self.F(input_d), label,weight=self.class_weight)
            loss_f = (1 - self.alpha_f) * loss_f1 + self.alpha_f * loss_f2
            self.model_backward_and_update(loss_f, 'F')

            # Update domain net
            loss_d1 = F.cross_entropy(self.D(input), domain)
            loss_d2 = F.cross_entropy(self.D(input_f), domain)
            loss_d = (1 - self.alpha_d) * loss_d1 + self.alpha_d * loss_d2
            self.model_backward_and_update(loss_d, 'D')
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            loss_f = F.cross_entropy(self.F(input), label)
            loss_d = F.cross_entropy(self.D(input), domain)
        loss_summary = {'loss_f': loss_f.item(), 'loss_d': loss_d.item()}



        return loss_summary

    # def model_inference(self, input):
    #     return self.F(input)

    def model_inference(self, input):
        logit = self.F(input)
        probs = F.softmax(logit, dim=1)
        return probs

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
            loss = self.forward_backward(batch, backprob=False)
            losses.update(loss)
            # total_loss += loss['loss']
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss_f'].avg

        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        # if full_results:
        return [total_loss,losses.dict_results(), results]


    # def after_epoch(self):
    #     """
    #     save the best model for given validation loss
    #     """
    #     # epoch_total_loss = self.validate()
    #     epoch_total_loss,validate_losses,validate_results = self.validate(full_results=True)
    #     #save validate information for data analysis
    #     for val_metric,value in validate_results.items():
    #         new_val_metric = "val_"+val_metric
    #         self._history[new_val_metric].append(value)
    #     for loss_name, val in validate_losses.items():
    #         new_val_loss = "val_" + loss_name
    #         self._history[new_val_loss].append(val)
    #
    #
    #     if self._best_epoch_val_loss > epoch_total_loss:
    #         print("save best model at epoch %f , Improve loss from %4f -> %4f" % (
    #             self.epoch, self._best_epoch_val_loss, epoch_total_loss))
    #         self._best_epoch_val_loss = epoch_total_loss
    #         self.save_model(epoch=self.epoch, directory=self.output_dir, is_best=True)
    #     super().after_epoch()
    # def parse_batch_train(self, batch_x):
    #     input_x = batch_x['eeg_data']
    #     label_x = batch_x['label']
    #     domain_x = batch_x['domain']
    #     input_x = input_x.to(self.device)
    #     label_x = label_x.to(self.device)
    #     domain_x =domain_x.to(self.device)
    #     return input_x, label_x,domain_x
    # def parse_batch_test(self, batch):
    #     input = batch['eeg_data']
    #     label = batch['label']
    #     input = input.to(self.device)
    #     label = label.to(self.device)
    #     return input, label


