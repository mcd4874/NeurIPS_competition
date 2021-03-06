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
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


import torchmetrics

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)

class PairClassifiers(nn.Module):

    def __init__(self, fdim, num_classes,FC_max_norm=0.25):
        super().__init__()
        self.c1 = self.create_classifier(fdim, num_classes,FC_max_norm)
        self.c2 = self.create_classifier(fdim, num_classes,FC_max_norm)
    def create_classifier(self,fdim,num_classes,FC_max_norm=0.25):
        classifier = nn.Linear(fdim, num_classes)
        if FC_max_norm>-1.0:
            print("use max norm constraint on last FC")
            classifier = LinearWithConstraint(fdim, num_classes, max_norm=FC_max_norm)
        return classifier

    def forward(self, x):
        z1 = self.c1(x)
        # if not self.training:
        #     return z1
        z2 = self.c2(x)
        return z1, z2


@TRAINER_REGISTRY.register()
class MultiDatasetM3SDA(TrainerMultiAdaptation):
    """
    https://arxiv.org/abs/1712.02560
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
        self.n_step_F = 5
        self.automatic_optimization = False
        self.lmda = 0.5

    def build_metrics(self):
        super(MultiDatasetM3SDA, self).build_metrics()
        self.valid_acc_1 = torchmetrics.Accuracy()
        self.valid_acc_2 = torchmetrics.Accuracy()

        # self.subjects_ = [for subject_id in range(self.num_test_subjects)]

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
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

        F_scheduler = build_lr_scheduler(optimizer=F_opt,optim_cfg=opt_cfg)


        C1_params = list(self.TargetClassifier_1.parameters())
        C1_opt = build_optimizer(C1_params,opt_cfg)
        C1_scheduler = build_lr_scheduler(optimizer=C1_opt,optim_cfg=opt_cfg)


        C2_params = list(self.TargetClassifier_2.parameters())
        C2_opt = build_optimizer(C2_params,opt_cfg)
        C2_scheduler = build_lr_scheduler(optimizer=C2_opt,optim_cfg=opt_cfg)

        C_S_params = list(self.SourceClassifiers.parameters())
        C_S_opt = build_optimizer(C_S_params,opt_cfg)
        C_S_scheduler = build_lr_scheduler(optimizer=C_S_opt,optim_cfg=opt_cfg)


        # opt = build_optimizer(params,opt_cfg)
        # scheduler = build_lr_scheduler(optimizer=opt,optim_cfg=opt_cfg)
        optimizers = [F_opt,C1_opt,C2_opt,C_S_opt]
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

        print('Building SourceClassifiers')
        print("source domains label size : ", self.source_domains_label_size)
        source_classifier_list = []
        for num_class in self.source_domains_label_size:
            source_classifier = self.create_classifier(self.fdim, num_class, FC_info=FC_info)
            # source_classifier = PairClassifiers(self.fdim, num_class, FC_max_norm=FC_info.max_norm)
            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )


    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)

        f_target = self.CommonFeature(input)
        logits_target_1 = self.TargetClassifier_1(f_target)
        logits_target_2 = self.TargetClassifier_2(f_target)

        loss_target_1 = self.loss_function(logits_target_1, label, train=train_mode,weight=weight)
        loss_target_2 = self.loss_function(logits_target_2, label, train=train_mode,weight=weight)
        loss_target = loss_target_1+loss_target_2

        return loss_target,logits_target_1,logits_target_2,label,f_target

    def on_train_epoch_start(self) -> None:
        if self.source_pretrain_epochs > self.current_epoch:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_SOURCE_LOSS_RATIO
        else:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO

    def discrepancy(self,out1,out2):
        return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2,dim=1)))

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


    def training_step(self, batch, batch_idx):
        target_batch, unlabel_batch ,list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)

        F_opt,C1_opt,C2_opt,C_S_opt = self.optimizers()

        loss_source = 0
        feat_x = list()
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.CommonFeature(u)
            feat_x.append(f)
            logits = self.SourceClassifiers[d](f)

            # logit_source_1,logit_source_2 = self.SourceClassifiers[d](f)
            # domain_weight = self.source_domains_class_weight[d]
            if not self.no_source_weight:
                domain_weight = self.source_domains_class_weight[d]
                # print("domain weight : ",domain_weight)
            else:
                # print("no domain weight")
                domain_weight = None
            # loss_source_1 = self.loss_function(logit_source_1, y, train=True, weight=domain_weight)
            # loss_source_2 = self.loss_function(logit_source_2, y, train=True, weight=domain_weight)
            # loss_s = loss_source_1 + loss_source_2
            # loss_source += loss_s
            loss_source += self.loss_function(logits, y, train=True, weight=domain_weight)

        loss_source /= len(domain_u)

        if self.no_target_weight:
            loss_target, logit_1, logit_2, label, f_target = self.share_step(target_batch, train_mode=True)
        else:
            loss_target,logit_1,logit_2,label,f_target = self.share_step(target_batch, train_mode=True, weight=self.class_weight)
        feat_x.append(f_target)
        feat_u = self.CommonFeature(unlabel_batch)
        loss_msda = self.moment_distance(feat_x, feat_u)


        ensemble_logit = logit_1+logit_2
        y_pred = F.softmax(ensemble_logit, dim=1)
        y = label
        acc = self.train_acc(y_pred, y)

        # ## visualize very few params
        # print("check params update extractor before ")
        # for param in self.CommonFeature.parameters():
        #     print(param[0])
        #     break
        #
        # print("check params update classifier before ")
        # for param in self.TargetClassifier_1.parameters():
        #     print(param[0])
        #     break
        #step A
        loss_A = self.source_ratio*loss_source+self.target_ratio*loss_target +  loss_msda*self.lmda
        F_opt.zero_grad()
        C1_opt.zero_grad()
        C2_opt.zero_grad()
        C_S_opt.zero_grad()
        self.manual_backward(loss_A)
        F_opt.step()
        C1_opt.step()
        C2_opt.step()
        C_S_opt.step()

        #step B
        loss_x,_,_,_,_ = self.share_step(target_batch, train_mode=True)
        # loss_x,_,_,_,_ = self.share_step(target_batch, train_mode=True,weight=self.class_weight)

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
        self.log('Train_loss_msda', loss_msda,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss_B', loss_B,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_loss_C', loss_C,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        # return {'loss':total_loss}

    def parse_batch_train(self, batch):
        target_batch = batch["target_loader"]
        unlabel_batch = batch["unlabel_loader"]
        list_source_batches = batch["source_loader"]
        return target_batch,unlabel_batch,list_source_batches


    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss,logit_1,logit_2,y,_ = self.share_step(batch,train_mode=False)
        y_logit = logit_1+logit_2
        y_pred = F.softmax(y_logit, dim=1)

        y_pred_1 = F.softmax(logit_1, dim=1)
        y_pred_2 = F.softmax(logit_2, dim=1)

        if dataset_idx == 0 :
            acc = self.valid_acc(y_pred, y)
            acc_1 = self.valid_acc_1(y_pred_1,y)
            acc_2 = self.valid_acc_2(y_pred_2,y)
            log = {
                "val_loss": loss*self.non_save_ratio,
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
        loss,logit_1,logit_2,y,_ = self.share_step(batch,train_mode=False)
        y_logit = logit_1+logit_2
        y_pred = F.softmax(y_logit,dim=1)
        return {'loss': loss,'y_pred':y_pred,'y':y}

