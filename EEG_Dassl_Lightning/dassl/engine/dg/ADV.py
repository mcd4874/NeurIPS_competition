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
class ADV(TrainerX):
    """
    Adversarial Invariant representation EEG
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8981912
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        if n_domain <= 0:
            n_domain = self.dm.num_source_domains
        self.n_domain = n_domain

        #create a cross entropy loss for whole dataset
        self.ce = nn.CrossEntropyLoss()
        self.ce_1 =  nn.CrossEntropyLoss()
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                print("torch weight  : ",torch_weight)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)

        self._best_epoch_val_loss = 10000

        # self.lmda = 0.25 #within_subject
        self.lmda = cfg.TRAINER.ADV.lmda # within_subject_1

        self.build_domain_discriminator()



    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print('Building model')
        self.model = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

        fdim = self.model.fdim
        self.classifier = nn.Linear(fdim, self.num_classes)
        print('# params: {:,}'.format(count_num_param(self.classifier)))
        self.classifier.to(self.device)
        self.optim_classifier = build_optimizer(self.classifier, cfg.OPTIM)
        self.sched_classifier = build_lr_scheduler(self.optim_classifier, cfg.OPTIM)
        self.register_model('classifier', self.classifier, self.optim_classifier, self.sched_classifier)


    def build_domain_discriminator(self):
        cfg = self.cfg
        print('Building critic network')
        fdim = self.model.fdim
        print("domains : ",self.n_domain)
        self.D = nn.Linear(fdim, self.n_domain)
        print('# params: {:,}'.format(count_num_param(self.D)))
        self.D.to(self.device)
        self.optim_d = build_optimizer(self.D, cfg.OPTIM)
        self.sched_d = build_lr_scheduler(self.optim_d, cfg.OPTIM)
        self.register_model('D', self.D, self.optim_d, self.sched_d)
        self.revgrad = ReverseGrad()


    def forward_backward(self, batch,backprob = True):
        input_x, label_x, domain_x = self.parse_batch_train(batch)


        feat = self.model(input_x)
        logits = self.classifier(feat)
        loss_x = self.ce(logits, label_x)

        critic_logits = self.D(feat)
        loss_critic = self.ce_1(critic_logits, domain_x)

        loss_general = loss_x-self.lmda*loss_critic
        if backprob:
            self.model_backward_and_update(loss_general,['model','classifier'])
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr(['model','classifier'])


        with torch.no_grad():
            feat = self.model(input_x)
        critic_logits = self.D(feat)

        loss_critic = self.ce_1(critic_logits,domain_x)
        if backprob:
            self.model_backward_and_update(loss_critic,['D'])
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr(['D'])

        loss_summary = {
            'loss_x': loss_x.item(),
            'loss_d': loss_critic.item(),
            'loss_general': loss_general.item()
        }

        return loss_summary
    # def forward_backward(self, batch,backprob = True):
    #     input_x, label_x, domain_x = self.parse_batch_train(batch)
    #
    #     global_step = self.batch_idx + self.epoch * self.num_batches
    #     progress = global_step / (self.max_epoch * self.num_batches)
    #     lmda = 2 / (1 + np.exp(-10 * progress)) - 1
    #
    #     # lmda = 0.1
    #
    #     logits,feat = self.model(input_x,return_feature=True)
    #
    #     loss_x = self.ce(logits,label_x)
    #
    #     feat = self.revgrad(feat, grad_scaling=lmda)
    #
    #     critic_logits = self.D(feat)
    #
    #     loss_critic = self.ce_1(critic_logits,domain_x)
    #
    #     total_loss = loss_x+loss_critic
    #
    #     if backprob:
    #         self.model_backward_and_update(total_loss)
    #         if (self.batch_idx + 1) == self.num_batches:
    #             self.update_lr()
    #
    #     loss_summary = {
    #         'loss_x': loss_x.item(),
    #         'loss_d': loss_critic.item(),
    #         'total_loss':total_loss.item()
    #     }
    #     return loss_summary

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
    #         self.epoch, self._best_epoch_val_loss, epoch_total_loss))
    #         self._best_epoch_val_loss = epoch_total_loss
    #         self.save_model(epoch=self.epoch, directory=self.output_dir, is_best=True)
    #     super().after_epoch()


    def model_inference(self, input):
        # logit = self.model(input)
        feat = self.model(input)
        logit = self.classifier(feat)

        probs = F.softmax(logit, dim=1)
        return probs

    # def model_inference(self, input):
    #     logit = self.model(input)
    #     probs = F.softmax(logit, dim=1)
    #     return probs

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




