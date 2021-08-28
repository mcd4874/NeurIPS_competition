from dassl.engine import TRAINER_REGISTRY
from dassl.engine.trainer import TrainerMultiAdaptation
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
import copy

import torchmetrics



@TRAINER_REGISTRY.register()
class MultiDatasetADDAV1(TrainerMultiAdaptation):
    """
    https://arxiv.org/pdf/1702.05464.pdf

    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
        self.automatic_optimization = False
        self.bce = nn.BCEWithLogitsLoss()

        # epoch 13
        self.start_adda_epoch= self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.ADDA_TRAIN_EPOCHS


    def build_metrics(self):
        super(MultiDatasetADDAV1, self).build_metrics()
        self.valid_acc_1 = torchmetrics.Accuracy()
        self.valid_acc_2 = torchmetrics.Accuracy()

    def forward(self, input, return_feature=False):
        f_target = self.CommonFeature(input)
        logits_target = self.TargetClassifier(f_target)
        probs = F.softmax(logits_target, dim=1)
        if return_feature:
            return probs, logits_target
        return probs
    def configure_optimizers(self):
        opt_cfg = self.cfg.OPTIM


        F_params = list(self.CommonFeature.parameters())
        F_opt = build_optimizer(F_params,opt_cfg)
        F_scheduler = build_lr_scheduler(optimizer=F_opt,optim_cfg=opt_cfg)


        C_T_params = list(self.TargetClassifier.parameters())
        C_T_opt = build_optimizer(C_T_params,opt_cfg)
        C_T_scheduler = build_lr_scheduler(optimizer=C_T_opt,optim_cfg=opt_cfg)



        C_S_params = list(self.SourceClassifiers.parameters())
        C_S_opt = build_optimizer(C_S_params,opt_cfg)
        C_S_scheduler = build_lr_scheduler(optimizer=C_S_opt,optim_cfg=opt_cfg)

        D_params = list(self.DomainDiscriminator.parameters())
        D_opt = build_optimizer(D_params,opt_cfg)
        D_scheduler = build_lr_scheduler(optimizer=D_opt,optim_cfg=opt_cfg)

        optimizers = [F_opt,C_T_opt,C_S_opt,D_opt]
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

        self.CommonFeature = SimpleNet(backbone_info, FC_info, 0, **cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS)
        self.fdim = self.CommonFeature.fdim

        print('Building Target Classifier')
        self.TargetClassifier = self.create_classifier(self.fdim, self.num_classes, FC_info=FC_info)

        self.fix_feature = copy.deepcopy(self.CommonFeature)

        print('Building SourceClassifiers')
        print("source domains label size : ", self.source_domains_label_size)
        source_classifier_list = []
        for num_class in self.source_domains_label_size:
            source_classifier = self.create_classifier(self.fdim, num_class, FC_info=FC_info)
            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )


        self.DomainDiscriminator = nn.Linear(self.fdim, 1)

    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)
        f_target = self.CommonFeature(input)
        logits_target = self.TargetClassifier(f_target)

        loss_target = self.loss_function(logits_target, label, train=train_mode,weight=weight)
        return loss_target,logits_target,label

    def on_train_epoch_start(self) -> None:
        if self.source_pretrain_epochs > self.current_epoch:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_SOURCE_LOSS_RATIO
        else:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO

        if self.start_adda_epoch == self.current_epoch:
            self.fix_feature = copy.deepcopy(self.CommonFeature)
    def on_validation_epoch_start(self) -> None:
        if self.start_adda_epoch > self.current_epoch:
            self.non_save_ratio = 10.0
        else:
            self.non_save_ratio = 1.0
    # def calculate_dann(self,target_feature,source_feature):
    #     #there is a problem need to concern. We assume that label target batch size is same as source batch size
    #     domain_label_target = torch.ones(target_feature.shape[0], 1,device=self.device)
    #     domain_label_source = torch.zeros(source_feature.shape[0], 1,device=self.device)
    #     feature = torch.cat([target_feature, source_feature])
    #     domain_label = torch.cat([domain_label_target, domain_label_source])
    #     domain_pred = self.DomainDiscriminator(feature)
    #     loss_d = self.bce(domain_pred, domain_label)
    #
    #     y_pred = F.sigmoid(domain_pred)
    #     y_pred = y_pred > 0.5
    #     total = torch.sum(y_pred==domain_label)
    #     acc= total/(target_feature.shape[0]+source_feature.shape[0])
    #     # print("acc : ",acc)
    #     return loss_d,acc

    def training_step(self, batch, batch_idx):
        target_batch, unlabel_batch ,list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)

        F_opt,C_T_opt,C_S_opt,D_opt = self.optimizers()

        loss_source = 0
        loss_adv_D=0
        loss_adv_M=0
        acc_d=0
        if self.start_adda_epoch > self.current_epoch:
            for u, y, d in zip(list_input_u, list_label_u, domain_u):
                # print("check range for source data : {} - {}".format(u.max(),u.min()))
                f = self.CommonFeature(u)
                logits = self.SourceClassifiers[d](f)
                domain_weight = self.source_domains_class_weight[d]
                loss_source += self.loss_function(logits, y, train=True, weight=domain_weight)
            loss_source /= len(domain_u)

            loss_target,logit_target,label= self.share_step(target_batch, train_mode=True, weight=self.class_weight)


            loss_cls = self.source_ratio*loss_source+self.target_ratio*loss_target
            F_opt.zero_grad()
            C_T_opt.zero_grad()
            C_S_opt.zero_grad()
            self.manual_backward(loss_cls)
            F_opt.step()
            C_T_opt.step()
            C_S_opt.step()
        else:
            data, _, _ = self.parse_target_batch(target_batch)
            feature_target_label = self.fix_feature(data)
            feature_target_unlabel = self.CommonFeature(unlabel_batch)

            domain_target_label = torch.zeros(feature_target_label.shape[0], 1, device=self.device)
            domain_target_unlabel = torch.ones(feature_target_unlabel.shape[0], 1, device=self.device)
            feature = torch.cat([feature_target_label, feature_target_unlabel])
            domain_label = torch.cat([domain_target_label, domain_target_unlabel])
            domain_pred = self.DomainDiscriminator(feature)
            loss_adv_D = self.bce(domain_pred, domain_label)

            y_pred = F.sigmoid(domain_pred)
            y_pred = y_pred > 0.5
            total = torch.sum(y_pred == domain_label)
            acc_d = total / (domain_target_label.shape[0] + domain_target_unlabel.shape[0])

            F_opt.zero_grad()
            D_opt.zero_grad()
            self.manual_backward(loss_adv_D)
            F_opt.zero_grad()

            D_opt.step()

            feature_target_unlabel = self.CommonFeature(unlabel_batch)
            domain_target_unlabel = torch.zeros(feature_target_unlabel.shape[0], 1, device=self.device)
            unlabel_domain_pred = self.DomainDiscriminator(feature_target_unlabel)
            loss_adv_M = self.bce(unlabel_domain_pred, domain_target_unlabel)



            F_opt.zero_grad()
            D_opt.zero_grad()
            self.manual_backward(loss_adv_M)
            D_opt.zero_grad()

            F_opt.step()

            loss_target,logit_target,label= self.share_step(target_batch, train_mode=True, weight=self.class_weight)
            loss_cls = loss_target
            F_opt.zero_grad()
            C_T_opt.zero_grad()

        y_pred = F.softmax(logit_target, dim=1)
        y = label
        acc = self.train_acc(y_pred, y)

        self.log('Train_acc', acc, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss_cls', loss_cls,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss_adv_D', loss_adv_D, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss_adv_M', loss_adv_M, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_acc_d', acc_d, on_step=False,on_epoch=True,prog_bar=True, logger=True)


    def parse_batch_train(self, batch):
        target_batch = batch["target_loader"]
        unlabel_batch = batch["unlabel_loader"]
        list_source_batches = batch["source_loader"]
        return target_batch,unlabel_batch,list_source_batches


    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss,y_logit,y = self.share_step(batch,train_mode=False)
        y_pred = F.softmax(y_logit, dim=1)

        if dataset_idx == 0 :
            acc = self.valid_acc(y_pred, y)
            log = {
                "val_loss": loss*self.non_save_ratio,
                "val_acc": acc,
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

    def test_step(self, batch, batch_idx):
        loss,y_logit,y = self.share_step(batch,train_mode=False)
        y_pred = F.softmax(y_logit,dim=1)
        return {'loss': loss,'y_pred':y_pred,'y':y}