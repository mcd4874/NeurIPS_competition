import torch
from torch.nn import functional as F
import torch.nn as nn
from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerBase
from dassl.modeling.ops import ReverseGrad
from dassl.engine.trainer import SimpleNet
from torch.utils.data import Dataset as TorchDataset
from dassl.utils import MetricMeter
import learn2learn as l2l
import numpy as np
@TRAINER_REGISTRY.register()
class MLDG(TrainerBase):
    """
    Learning to Generalize: Meta-Learning for Domain Generalization
    https://arxiv.org/pdf/1710.03463.pdf
    """
    def __init__(self, cfg,require_parameter=None):
        self.total_subjects = require_parameter["total_subjects"]
        self.alpha = cfg.TRAINER.MLDG.alpha
        self.inner_lr = cfg.TRAINER.MLDG.inner_lr
        # self.num_test_subject = cfg.TRAINER.MLDG.num_test_subject
        self.percent_test_subject = cfg.TRAINER.MLDG.percent_test_subject #50%
        self.num_test_subject = int(self.total_subjects * self.percent_test_subject)
        if self.num_test_subject < 1:
            self.num_test_subject = cfg.TRAINER.MLDG.num_test_subject
        # self.num_inner_loop = cfg.TRAINER.MLDG.num_inner_loop
        # self.warm_up = 20
        print("num meta train subject : ",self.total_subjects-self.num_test_subject)
        print("num meta test subject : ",self.num_test_subject)
        self.check_cfg(cfg)
        self.split_batch = cfg.DATALOADER.TRAIN_X.BATCH_SIZE // self.total_subjects

        super().__init__(cfg,require_parameter)


        self.candidates = np.arange(self.total_subjects)
        index = np.random.permutation(self.candidates)
        self.meta_test_idx = index[:self.num_test_subject]
        self.meta_train_idx = index[self.num_test_subject:]



    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == 'RandomDomainSampler'
        assert cfg.DATALOADER.TRAIN_X.BATCH_SIZE % self.total_subjects ==0
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt,gamma=1.0
        )
        optimizers = [opt]
        return optimizers, [scheduler]
    def build_model(self):
        """Build and register model.

                The default builds a classification model along with its
                optimizer and scheduler.

                Custom trainers can re-implement this method if necessary.
                """

        cfg = self.cfg
        print('Building model')
        if cfg.MODEL.BACKBONE.PARAMS:
            model = SimpleNet(cfg, self.num_classes, **cfg.MODEL.BACKBONE.PARAMS)
        else:
            model = SimpleNet(cfg, self.num_classes)
        print('# params: {:,}'.format(count_num_param(model)))
        print("hyper params : ", self.hparams)


        print('Building MAML')
        # fast_lr = 0.001
        self.model = l2l.algorithms.MAML(model, lr=self.inner_lr,first_order=False)

    @torch.enable_grad()
    def meta_learn(self, batch):
        learner = self.model.clone()
        learner.train()

        input, label, domain = self.parse_batch_train(batch)
        input_x = torch.split(input, self.split_batch, 0)
        label_x = torch.split(label, self.split_batch, 0)
        # domain_x = torch.split(domain_x, self.split_batch, 0)
        # d_x = [d[0].item() for d in domain_x]

        # print("current domain : ",d_x)

        # for i in range(self.num_inner_loop):
            # domain = np.random.permutation(self.meta_train_idx)[0]
            # x, y = input_x[domain], label_x[domain]
        meta_train_loss = 0
        for domain in self.meta_train_idx:
            x = input_x[domain]
            y = label_x[domain]
            logits = learner(x)
            loss = self.loss_function(logits, y, train=True)
            meta_train_loss +=loss
        meta_train_loss /= len(self.meta_train_idx)
        learner.adapt(meta_train_loss)

        meta_test_loss = 0
        # Evaluating the adapted model
        # list_y = []
        # list_preds = []
        for domain in self.meta_test_idx:
            x, y = input_x[domain], label_x[domain]
            logits = learner(x)
            loss = self.loss_function(logits, y, train=True)
            # preds = F.softmax(logits,dim=1)
            # list_y.append(y)
            # list_preds.append(preds)
            meta_test_loss += loss
        meta_test_loss /= len(self.meta_test_idx)

        total_loss = meta_train_loss + meta_test_loss*self.alpha


        logits = learner(input)
        # preds = F.softmax(logits, dim=1)
        # acc = self.train_acc(preds,label)

        # return total_loss,acc
        return total_loss

    # def training_step(self, batch, batch_idx):
    #     train_loss,train_acc = self.meta_learn(batch)
    #
    #     self.log('Train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log('Train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     return {'loss':train_loss}

    def training_step(self, batch, batch_idx):
        train_loss = self.meta_learn(batch)

        # self.log('Train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss':train_loss}

    def training_step_end(self, outputs):
        """
        shuffle at the end of each train batch
        Args:
            outputs:

        Returns:

        """
        index = np.random.permutation(self.candidates)
        self.meta_test_idx = index[:self.num_test_subject]
        self.meta_train_idx = index[self.num_test_subject:]
        # print("rest --")
        # print("meta train to be subject : {}".format(self.meta_train_idx))
        # print("meta test to be subject : {}".format(self.meta_test_idx))
        return {'loss': outputs['loss']}

    # def training_epoch_end(self, outputs):
    #     #shuffle the train meta and test meta subjects
    #     index = np.random.permutation(self.candidates)
    #     self.meta_test_idx = index[:self.num_test_subject]
    #     self.meta_train_idx = index[self.num_test_subject:]
    #     print("update meta train to be subject : {}".format(self.meta_train_idx))
    #     print("update meta test to be subject : {}".format(self.meta_test_idx))

    def validation_step(self, batch, batch_idx):
        loss,y_logit,y = self.share_step(batch,train_mode=False)
        y_pred = F.softmax(y_logit, dim=1)
        acc = self.valid_acc(y_pred, y)
        log = {
            "val_loss": loss,
            "val_acc": acc
        }
        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def test_step(self, batch,batch_idx):
        loss, y_logit, y = self.share_step(batch, train_mode=False)
        y_pred = F.softmax(y_logit,dim=1)
        acc =   self.test_acc(y_pred,y)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def share_step(self,batch,train_mode=True):
        x, y, domain = self.parse_batch_train(batch)
        y_logits = self.model(x)
        loss = self.loss_function(y_logits, y,train=train_mode)
        return loss,y_logits,y



