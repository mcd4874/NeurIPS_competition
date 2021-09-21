from dassl.engine import TRAINER_REGISTRY
from dassl.engine.trainer import TrainerBase
from dassl.data import DataManager
from dassl.utils import MetricMeter
from torch.utils.data import Dataset as TorchDataset
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.engine.trainer import SimpleNet
import numpy as np
from dassl.modeling import build_layer
from dassl.modeling.ops import ReverseGrad
from typing import Any, Dict, List, Optional, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


import torchmetrics



@TRAINER_REGISTRY.register()
class MCD(TrainerBase):
    """
    https://arxiv.org/abs/1712.02560
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
        self.n_step_F = 5
        self.automatic_optimization = False

    def build_metrics(self):
        super(MCD, self).build_metrics()
        self.valid_acc_1 = torchmetrics.Accuracy()
        self.valid_acc_2 = torchmetrics.Accuracy()


    def forward(self, input, return_feature=False):
        f_target = self.CommonFeature(input)
        logits_target_1 = self.TargetClassifier_1(f_target)
        logits_target_2 = self.TargetClassifier_2(f_target)
        ensemble_logit = logits_target_1+logits_target_2
        probs = F.softmax(ensemble_logit, dim=1)
        if return_feature:
            return probs, ensemble_logit
        return probs
    def configure_optimizers(self):
        opt_cfg = self.cfg.OPTIM


        F_params = list(self.CommonFeature.parameters())
        F_opt = build_optimizer(F_params,opt_cfg)

        F_scheduler = build_lr_scheduler(optimizer=F_opt,optim_cfg=opt_cfg)


        C1_params = list(self.TargetClassifier_1.parameters())
        C1_opt = build_optimizer(C1_params,opt_cfg)
        C1_scheduler = build_lr_scheduler(optimizer=C1_opt,optim_cfg=opt_cfg)


        C2_params = list(self.TargetClassifier_2.parameters())
        C2_opt = build_optimizer(C2_params,opt_cfg)
        C2_scheduler = build_lr_scheduler(optimizer=C2_opt,optim_cfg=opt_cfg)


        optimizers = [F_opt,C1_opt,C2_opt]
        return optimizers
        # lr_schedulers=[F_scheduler,C1_scheduler,C2_scheduler,C_S_scheduler]
        # return optimizers, lr_schedulers

    def build_model(self):
        cfg = self.cfg
        print("Params : ", cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE)
        print('Building F')

        print('Building CommonFeature')
        backbone_info = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE
        FC_info = cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC
        # backbone_params = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS.copy()
        FREEZE: True

        self.CommonFeature = SimpleNet(backbone_info, FC_info, 0, **cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS)
        freeze_common_feature = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.FREEZE if cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.FREEZE else False
        if freeze_common_feature:
            for parameter in self.CommonFeature.parameters():
                parameter.requires_grad = False
            print("freeze feature extractor : ",)

        self.fdim = self.CommonFeature.fdim

        print('Building Target Classifier')
        self.TargetClassifier_1 = self.create_classifier(self.fdim, self.num_classes, FC_info=FC_info)
        self.TargetClassifier_2 = self.create_classifier(self.fdim, self.num_classes, FC_info=FC_info)

    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)

        f_target = self.CommonFeature(input)
        logits_target_1 = self.TargetClassifier_1(f_target)
        logits_target_2 = self.TargetClassifier_2(f_target)

        loss_target_1 = self.loss_function(logits_target_1, label, train=train_mode,weight=weight)
        loss_target_2 = self.loss_function(logits_target_2, label, train=train_mode,weight=weight)
        loss_target = loss_target_1+loss_target_2

        return loss_target,logits_target_1,logits_target_2,label

    def discrepancy(self,out1,out2):
        return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2,dim=1)))

    def training_step(self, batch, batch_idx):
        target_batch, unlabel_batch  = self.parse_batch_train(batch)
        F_opt,C1_opt,C2_opt = self.optimizers()


        loss_target,logit_1,logit_2,label = self.share_step(target_batch, train_mode=True, weight=self.class_weight)
        ensemble_logit = logit_1+logit_2
        y_pred = F.softmax(ensemble_logit, dim=1)
        y = label
        acc = self.train_acc(y_pred, y)

        #step A
        loss_A = loss_target
        F_opt.zero_grad()
        C1_opt.zero_grad()
        C2_opt.zero_grad()
        self.manual_backward(loss_A)
        F_opt.step()
        C1_opt.step()
        C2_opt.step()

        #step B
        # loss_x,_,_,_, = self.share_step(target_batch, train_mode=True)
        loss_x,_,_,_, = self.share_step(target_batch, train_mode=True,weight=self.class_weight)

        #try to use with torch.no_grad():
        f_target = self.CommonFeature(unlabel_batch)
        logit_u_1 = self.TargetClassifier_1(f_target)
        logit_u_2 = self.TargetClassifier_2(f_target)
        loss_dis = self.discrepancy(logit_u_1, logit_u_2)


        loss_B = loss_x-loss_dis
        F_opt.zero_grad()
        C1_opt.zero_grad()
        C2_opt.zero_grad()
        self.manual_backward(loss_B)
        C1_opt.step()
        C2_opt.step()

        #step C
        for _ in range(self.n_step_F):
            f_target = self.CommonFeature(unlabel_batch)
            logit_u_1 = self.TargetClassifier_1(f_target)
            logit_u_2 = self.TargetClassifier_2(f_target)
            loss_C = self.discrepancy(logit_u_1, logit_u_2)
            F_opt.zero_grad()
            C1_opt.zero_grad()
            C2_opt.zero_grad()
            self.manual_backward(loss_C)
            F_opt.step()

        self.log('Train_acc', acc, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss_A', loss_A,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_loss_B', loss_B,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_loss_C', loss_C,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        # return {'loss':total_loss}

    def parse_batch_train(self, batch):
        target_batch = batch["target_loader"]
        unlabel_batch = batch["unlabel_loader"]
        return target_batch,unlabel_batch

    def parse_target_batch(self,batch):
        input, label, domain = batch
        return input,label,domain
    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss,logit_1,logit_2,y = self.share_step(batch,train_mode=False)
        y_logit = logit_1+logit_2
        y_pred = F.softmax(y_logit, dim=1)

        y_pred_1 = F.softmax(logit_1, dim=1)
        y_pred_2 = F.softmax(logit_2, dim=1)

        if dataset_idx == 0 :
            acc = self.valid_acc(y_pred, y)
            acc_1 = self.valid_acc_1(y_pred_1,y)
            acc_2 = self.valid_acc_2(y_pred_2,y)
            log = {
                "val_loss": loss,
                "val_acc": acc,
                "val_acc_1":acc_1,
                "val_acc_2":acc_2
            }
            self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True,add_dataloader_idx=False)
        else:
            acc = self.test_acc(y_pred, y)
            log = {
                "test_loss": loss,
                "test_acc": acc
            }
            self.log_dict(log, on_step=False, on_epoch=True, prog_bar=False, logger=True,add_dataloader_idx=False)

        return {'loss': loss}

    def test_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss,logit_1,logit_2,y = self.share_step(batch,train_mode=False)
        y_logit = logit_1+logit_2
        y_pred = F.softmax(y_logit,dim=1)
        return {'loss': loss,'y_pred':y_pred,'y':y}
