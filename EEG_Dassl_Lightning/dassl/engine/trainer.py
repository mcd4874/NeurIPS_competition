import time
import os.path as osp
import os
import datetime
from collections import OrderedDict,defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Union


from dassl.data import DataManager,MultiDomainDataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, resume_from_checkpoint, load_pretrained_weights, generate_path_for_multi_sub_model
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator
import pandas as pd

from dassl.modeling import build_layer

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)

class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a single classifier layer.
    """

    def __init__(self, backbone_info,FC_info=None, num_classes=0, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            backbone_info.NAME,
            pretrained=backbone_info.PRETRAINED,
            pretrained_path = backbone_info.PRETRAINED_PATH,
            **kwargs
        )
        fdim = self.backbone.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)
            if FC_info and  FC_info.max_norm > -1.0:
            # if FC_info.NAME and  FC_info.max_norm > -1.0:
                print("use max norm {} constraint on last FC".format(FC_info.max_norm))
                self.classifier = LinearWithConstraint(fdim,num_classes,max_norm=FC_info.max_norm)



        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)

        if self.classifier is None:
            return f
        y = self.classifier(f)
        if return_feature:
            return y, f
        return y

class TrainerBase(pl.LightningModule):
    def __init__(self,cfg,require_parameter=None):
        self.cfg = cfg
        # self.lr = self.cfg.OPTIM.LR
        self.require_parameter = require_parameter
        self.num_classes = require_parameter['num_classes']
        self.num_test_subjects = require_parameter['num_test_subjects']
        self._history = defaultdict(list)
        self.history_dir = cfg.history_dir
        self.output_dir = cfg.output_dir
        self.class_weight = require_parameter['target_domain_class_weight']
        self.view_test_subject_result = True
        super(TrainerBase, self).__init__()


        self.build_model()
        self.build_metrics()


        # self.load_from_checkpoint()
    def create_classifier(self,fdim,num_classes,FC_info=None):
        classifier = nn.Linear(fdim, num_classes)
        if FC_info and FC_info.max_norm > -1.0:
        # if self.cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC.NAME and self.cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC.max_norm > -1.0:
            print("use max norm constraint on last FC")
            classifier = LinearWithConstraint(fdim, num_classes, max_norm=FC_info.max_norm)
        return classifier

    def build_model(self):
        """Build and register model.

                The default builds a classification model along with its
                optimizer and scheduler.

                Custom trainers can re-implement this method if necessary.
                """
        cfg = self.cfg
        print('Building model')
        backbone_info = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE
        FC_info = cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC
        self.model = SimpleNet(backbone_info,FC_info, self.num_classes, **cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS)
        # if backbone_info.PARAMS:
        #     self.model = SimpleNet(cfg, self.num_classes, **cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS)
        # else:
        #     self.model = SimpleNet(cfg,self.num_classes)
        print('# params: {:,}'.format(count_num_param(self.model)))
        # print("hyper params : ",self.hparams)

    # def parse_batch_test(self, batch):
    #     input = batch['eeg_data']
    #     label = batch['label']
    #     # domain = batch['domain']
    #     return input, label

    # def parse_batch_train(self, batch):
    #     input = batch['eeg_data']
    #     label = batch['label']
    #     domain = batch['domain']
    #     return input, label,domain
    def parse_batch_train(self, batch):
        # input = batch['eeg_data']
        # label = batch['label']
        # domain = batch['domain']
        input, label, domain = batch
        return input, label,domain

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     opt,gamma=1.0
        # )
        params = self.model.parameters()
        opt_cfg = self.cfg.OPTIM
        opt = build_optimizer(params,opt_cfg)
        scheduler = build_lr_scheduler(optimizer=opt,optim_cfg=opt_cfg)
        optimizers = [opt]
        # lr_schedulers = {'scheduler': scheduler, 'monitor': 'metric_to_track'}
        # return optimizers, lr_schedulers
        return optimizers, [scheduler]

    def build_metrics(self):
        #need to redo the torchmetric for metrics
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        self.test_acc = torchmetrics.Accuracy()
        self.test_F1 = torchmetrics.F1(average="micro")

        self.test_avg_class_acc= torchmetrics.Accuracy(average='macro',num_classes=self.num_classes)
        self.test_class_acc = torchmetrics.Accuracy(average='none',num_classes=self.num_classes)

        # self.num_test_subjects

        # self.confusion_matrix =  torchmetrics.ConfusionMatrix(num_classes=self.num_classes)
        self.train_avg_loss = torchmetrics.AverageMeter()
        self.valid_avg_loss = torchmetrics.AverageMeter()
        self.test_avg_loss = torchmetrics.AverageMeter()

    def loss_function(self, y_pred,y,train=True,weight=None):
        if train:
            loss = F.cross_entropy(y_pred,y,weight=weight)
        else:
            loss = F.cross_entropy(y_pred,y)
        return loss


    def forward(self,input,return_feature=False):
        logit = self.model(input)
        probs = F.softmax(logit,dim=1)
        if return_feature:
            return probs,logit
        return probs

    def on_train_start(self) -> None:
        if self.class_weight:
            self.class_weight = torch.FloatTensor(self.class_weight).to(self.device)
        print("target class weight : ", self.class_weight)

    def training_step(self, batch, batch_idx):
        loss,y_logit,y = self.share_step(batch,train_mode=True,weight=self.class_weight)
        y_pred = F.softmax(y_logit,dim=1)
        acc = self.train_acc(y_pred,y)
        self.log('Train_acc', acc,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_loss', loss,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        return {'loss':loss}

    def share_step(self,batch,train_mode=True,weight=None):
        x, y, domain = self.parse_batch_train(batch)
        y_logits = self.model(x)
        loss = self.loss_function(y_logits, y,train=train_mode,weight=weight)
        return loss,y_logits,y

    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss, y_logit, y = self.share_step(batch, train_mode=False)
        y_pred = F.softmax(y_logit, dim=1)
        if dataset_idx == 0 :
            acc = self.valid_acc(y_pred, y)
            log = {
                "val_loss": loss,
                "val_acc": acc
            }
            self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True,add_dataloader_idx=False)
        else:
            acc = self.test_acc(y_pred, y)
            log = {
                "test_loss": loss,
                "test_acc": acc
            }
            self.log_dict(log, on_step=False, on_epoch=True, prog_bar=False, logger=True,add_dataloader_idx=False)

        return {'loss': loss,'log':log}

    # def validation_step(self, batch, batch_idx):
    #     loss,y_logit,y = self.share_step(batch,train_mode=False)
    #     y_pred = F.softmax(y_logit, dim=1)
    #     acc = self.valid_acc(y_pred, y)
    #     log = {
    #         "val_loss": loss,
    #         "val_acc": acc
    #     }
    #     self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #
    #     return {'loss': loss}

    # def test_step(self, batch,batch_idx):
    #     loss, y_logit, y = self.share_step(batch, train_mode=False)
    #     y_pred = F.softmax(y_logit,dim=1)
    #     acc =   self.test_acc(y_pred,y)
    #     self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
    #     self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def test_step(self, batch, batch_idx):
        loss, y_logit, y, = self.share_step(batch, train_mode=False)
        y_pred = F.softmax(y_logit,dim=1)

        return {'loss': loss,'y_pred':y_pred,'y':y}
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        for batch_result in outputs:
            batch_loss = batch_result['loss']
            batch_y_pred = batch_result['y_pred']
            batch_y = batch_result['y']
            self.test_acc.update(batch_y_pred,batch_y)
            self.test_avg_loss.update(batch_loss)
            self.test_avg_class_acc.update(batch_y_pred,batch_y)
            self.test_class_acc.update(batch_y_pred,batch_y)
            # self.test_F1.update(batch_y_pred,batch_y)
            # self.confusion_matrix.update(batch_y_pred,batch_y)

        # confusion_matrix = self.confusion_matrix.compute()
        acc = self.test_acc.compute()
        avg_class_acc = self.test_avg_class_acc.compute()
        classes_acc = self.test_class_acc.compute()
        F1 = self.test_F1.compute()
        loss = self.test_avg_loss.compute()
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # self.log('test_F1', F1, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        self.log('test_classes_avg_acc', avg_class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        print("individual classes acc : ",classes_acc)
        for label in range (self.num_classes):
            # class_precision=confusion_matrix[classes][classes]
            # classes_total = torch.sum(confusion_matrix[classes,:])
            # print("class ",classes)
            # print("class total ",classes_total)
            # print("class precision : ",class_precision/classes_total)
            # class_acc = class_precision/classes_total
            class_acc = classes_acc[label]
            format = 'class_{}_acc'.format(label)
            self.log(format, class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

            # print("true labe : ",y)
        # print("confusion matrix : ",matrix)

class TrainerMultiAdaptation(TrainerBase):
    def __init__(self,cfg,require_parameter=None):
        self.source_domains_input_shape = require_parameter['source_domains_input_shape']
        self.source_domains_label_size = require_parameter['source_domains_label_size']
        self.source_domains_class_weight = require_parameter['source_domains_class_weight']
        super().__init__(cfg,require_parameter)

        self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO
        self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO
        self.source_pretrain_epochs= self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_PRE_TRAIN_EPOCHS

        #this is a trick to deal with pretrain source model without saving the target model
        #only save the target model after finish pretrain
        self.non_save_ratio = 1.0
    def build_temp_layer(self, layer_info):
        embedding_layer_info = layer_info
        layer_name = embedding_layer_info.NAME
        layer_params = embedding_layer_info.PARAMS
        return [layer_name, layer_params]
    def build_model(self):
        cfg = self.cfg
        print("Params : ", cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE)
        print('Building F')



        print('Building TargetFeature')
        backbone_info = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE
        FC_info = cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC
        backbone_params = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS.copy()

        self.TargetFeature = SimpleNet(backbone_info,FC_info, 0, **cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS)

        print('Building Temporal Layer')
        layer_infp = cfg.LIGHTNING_MODEL.COMPONENTS.LAYER
        layer_name, layer_params = self.build_temp_layer(layer_infp)
        self.TemporalLayer = build_layer(layer_name, verbose=True, **layer_params)
        self.fdim2 = self.TemporalLayer.fdim

        print('Building Target Classifier')
        self.TargetClassifier = self.create_classifier(self.fdim2, self.num_classes,FC_info=FC_info)

        print(' Building SourceFeatures')
        # special case for only 1 source domain
        print("source domain input shape : ",self.source_domains_input_shape)
        list_num_ch = [input_shape[0] for input_shape in self.source_domains_input_shape]
        print("list num ch for source domains : ", list_num_ch)
        source_feature_list = []
        for num_ch in list_num_ch:
            backbone_params['num_ch'] = num_ch
            source_feature = SimpleNet(backbone_info,FC_info,0, **backbone_params)
            source_feature_list.append(source_feature)
        self.SourceFeatures = nn.ModuleList(
            source_feature_list
        )
        print('# params: {:,}'.format(count_num_param(self.SourceFeatures)))

        print('Building SourceClassifiers')
        print("source domains label size : ", self.source_domains_label_size)
        source_classifier_list = []
        for num_class in self.source_domains_label_size:
            # source_classifier = nn.Linear(fdim2, num_class)
            source_classifier = self.create_classifier(self.fdim2, num_class,FC_info=FC_info)

            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )

    def configure_optimizers(self):
        params = list(self.TargetFeature.parameters()) + \
                 list(self.TemporalLayer.parameters()) + \
                 list(self.TargetClassifier.parameters()) + \
                 list(self.SourceFeatures.parameters()) + \
                 list(self.SourceClassifiers.parameters())
        opt_cfg = self.cfg.OPTIM
        opt = build_optimizer(params,opt_cfg)
        scheduler = build_lr_scheduler(optimizer=opt,optim_cfg=opt_cfg)
        # opt = torch.optim.Adam(params, lr=self.lr)
        #
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     opt, gamma=1.0
        # )
        optimizers = [opt]
        lr_schedulers=[scheduler]
        # lr_schedulers = {'scheduler': scheduler, 'monitor': 'metric_to_track'}
        return optimizers, lr_schedulers

    def build_metrics(self):
        super().build_metrics()
        self.train_source_loss = torchmetrics.AverageMeter()
        self.train_target_loss = torchmetrics.AverageMeter()

    def forward(self, input, return_feature=False):
        f_target = self.TargetFeature(input)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.TargetClassifier(temp_layer_target)
        probs = F.softmax(logits_target, dim=1)

        if return_feature:
            return probs, logits_target
        return probs

    def parse_target_batch(self,batch):
        # input = batch['eeg_data']
        # label = batch['label']
        # domain = batch['domain']
        input, label, domain = batch
        return input,label,domain


    def parse_batch_train(self, batch):
        target_batch = batch["target_loader"]
        list_source_batches = batch["source_loader"]
        return target_batch,list_source_batches

    # def parse_source_batches(self,list_batch_u):
    #     list_input_u = list()
    #     list_label_u = list()
    #     for batch_u in list_batch_u:
    #         input_u = batch_u['eeg_data']
    #         label_u = batch_u['label']
    #         list_input_u.append(input_u)
    #         list_label_u.append(label_u)
    #     domain_u = [d for d in range(len(list_batch_u))]
    #     return list_input_u,list_label_u,domain_u

    def parse_source_batches(self,list_batch_u):
        list_input_u = list()
        list_label_u = list()
        for batch_u in list_batch_u:
            # input_u = batch_u['eeg_data']
            # label_u = batch_u['label']
            input_u,label_u,_ = batch_u
            list_input_u.append(input_u)
            list_label_u.append(label_u)
        domain_u = [d for d in range(len(list_batch_u))]
        return list_input_u,list_label_u,domain_u

    def loss_function(self, y_pred,y,train=True,weight=None):
        if train:
            loss = F.cross_entropy(y_pred,y,weight=weight)
        else:
            loss = F.cross_entropy(y_pred,y)
        return loss
    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)
        f_target = self.TargetFeature(input)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.TargetClassifier(temp_layer_target)
        loss_target = self.loss_function(logits_target, label, train=train_mode,weight=weight)
        return loss_target,logits_target,label, temp_layer_target
    def on_train_start(self) -> None:
        super(TrainerMultiAdaptation, self).on_train_start()
        if self.source_domains_class_weight:
            self.source_domains_class_weight = [torch.FloatTensor(weight).to(self.device) for weight in
                                                self.source_domains_class_weight]
        print("source domain weight : ",self.source_domains_class_weight)

    def on_train_epoch_start(self) -> None:
        if self.source_pretrain_epochs > self.current_epoch:
            # self.source_ratio = 1.0
            # self.target_ratio = 0.0
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_SOURCE_LOSS_RATIO
        else:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO

    def on_validation_epoch_start(self) -> None:
        if self.source_pretrain_epochs > self.current_epoch:
            self.non_save_ratio = 10.0
        else:
            self.non_save_ratio = 1.0

    def training_step(self, batch, batch_idx):

        target_batch,list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)
        loss_source = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f)
            logits = self.SourceClassifiers[d](temp_layer)
            domain_weight = self.source_domains_class_weight[d]
            loss_source += self.loss_function(logits, y,train=True,weight=domain_weight)
        loss_source /= len(domain_u)

        loss_target,logits_target, label, _= self.share_step(target_batch,train_mode=True,weight=self.class_weight)

        total_loss = loss_source*self.source_ratio+loss_target*self.target_ratio


        y_pred = F.softmax(logits_target, dim=1)
        # print("y pred ",y_pred)
        acc = self.train_acc(y_pred,label)

        self.log('Train_acc', acc, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss', total_loss, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        return {'loss':total_loss}

    # def validation_step(self, batch, batch_idx):
    #     loss,y_logit,y,_ = self.share_step(batch,train_mode=False)
    #     y_pred = F.softmax(y_logit, dim=1)
    #     acc = self.valid_acc(y_pred, y)
    #     log = {
    #         "val_loss": loss,
    #         "val_acc": acc
    #     }
    #     self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #
    #     return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss,y_logit,y,_ = self.share_step(batch,train_mode=False)
        y_pred = F.softmax(y_logit, dim=1)
        if dataset_idx == 0 :
            acc = self.valid_acc(y_pred, y)
            log = {
                "val_loss": loss*self.non_save_ratio,
                "val_acc": acc
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
        loss, y_logit, y,_ = self.share_step(batch, train_mode=False)
        y_pred = F.softmax(y_logit,dim=1)
        # acc =   self.test_acc(y_pred,y)
        # self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # matrix = self.confusion_matrix(y_pred,y)
        # for classes in range (self.num_classes):
        #     class_precision=matrix[classes][classes]
        #     classes_total = torch.sum(matrix[classes,:])
        #     print("class ",classes)
        #     print("class total ",classes_total)
        #     print("class precision : ",class_precision/classes_total)
        #     print("true labe : ",y)
        # print("confusion matrix : ",matrix)
        return {'loss': loss,'y_pred':y_pred,'y':y}
        # return {'loss':loss,'y'}
    # def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     for batch_result in outputs:
    #         batch_loss = batch_result['loss']
    #         batch_y_pred = batch_result['y_pred']
    #         batch_y = batch_result['y']
    #         self.test_acc.update(batch_y_pred,batch_y)
    #         self.test_avg_loss.update(batch_loss)
    #         self.test_avg_class_acc.update(batch_y_pred,batch_y)
    #         self.test_class_acc.update(batch_y_pred,batch_y)
    #         # self.test_F1.update(batch_y_pred,batch_y)
    #         # self.confusion_matrix.update(batch_y_pred,batch_y)
    #
    #     # confusion_matrix = self.confusion_matrix.compute()
    #     acc = self.test_acc.compute()
    #     avg_class_acc = self.test_avg_class_acc.compute()
    #     classes_acc = self.test_class_acc.compute()
    #     F1 = self.test_F1.compute()
    #     loss = self.test_avg_loss.compute()
    #     self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
    #     self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
    #     # self.log('test_F1', F1, on_step=False, on_epoch=True, prog_bar=True, logger=False)
    #
    #     self.log('test_classes_avg_acc', avg_class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
    #     print("individual classes acc : ",classes_acc)
    #     for label in range (self.num_classes):
    #         # class_precision=confusion_matrix[classes][classes]
    #         # classes_total = torch.sum(confusion_matrix[classes,:])
    #         # print("class ",classes)
    #         # print("class total ",classes_total)
    #         # print("class precision : ",class_precision/classes_total)
    #         # class_acc = class_precision/classes_total
    #         class_acc = classes_acc[label]
    #         format = 'class_{}_acc'.format(label)
    #         self.log(format, class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
    #
    #         # print("true labe : ",y)
    #     # print("confusion matrix : ",matrix)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # print("current output : ",outputs)
        # print("current output dataset : ",outputs[dataset_idx])
        # n_test_subjects = len(outputs)
        # self.test_acc = torchmetrics.Accuracy()
        # self.test_class_acc = torchmetrics.Accuracy(average='none',num_classes=self.num_classes)

        subject_id = 0
        for output in outputs:
            self.subject_test_acc = torchmetrics.Accuracy().to(self.device)
            self.subject_test_class_acc = torchmetrics.Accuracy(average='none', num_classes=self.num_classes).to(self.device)
            for batch_result in output:
        # for batch_result in outputs:
                batch_loss = batch_result['loss']
                batch_y_pred = batch_result['y_pred']
                batch_y = batch_result['y']
                self.test_acc.update(batch_y_pred,batch_y)
                self.test_avg_loss.update(batch_loss)
                self.test_avg_class_acc.update(batch_y_pred,batch_y)
                self.test_class_acc.update(batch_y_pred,batch_y)

                self.subject_test_acc.update(batch_y_pred,batch_y)
                self.subject_test_class_acc.update(batch_y_pred,batch_y)
            # self.test_F1.update(batch_y_pred,batch_y)
            # self.confusion_matrix.update(batch_y_pred,batch_y)

            if self.view_test_subject_result:
                subject_acc = self.subject_test_acc.compute()
                # print("subject acc :",subject_acc)
                self.log('sub_{}_test_acc'.format(subject_id), subject_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
                subject_classes_acc = self.subject_test_class_acc.compute()
                for label in range(self.num_classes):
                    class_acc = subject_classes_acc[label]
                    format = 'sub_{}_class_{}_acc'.format(subject_id,label)
                    self.log(format, class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            subject_id +=1
        # confusion_matrix = self.confusion_matrix.compute()
        acc = self.test_acc.compute()
        avg_class_acc = self.test_avg_class_acc.compute()
        classes_acc = self.test_class_acc.compute()
        F1 = self.test_F1.compute()
        loss = self.test_avg_loss.compute()
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # self.log('test_F1', F1, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        self.log('test_classes_avg_acc', avg_class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # print("individual classes acc : ",classes_acc)
        for label in range (self.num_classes):
            class_acc = classes_acc[label]
            format = 'class_{}_acc'.format(label)
            self.log(format, class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

            # print("true labe : ",y)
        # print("confusion matrix : ",matrix)