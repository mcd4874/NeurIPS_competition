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


import torchmetrics



@TRAINER_REGISTRY.register()
class MultiDatasetDannV1(TrainerMultiAdaptation):
    """
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
        self.bce = nn.BCEWithLogitsLoss()
        self.lmda =cfg.LIGHTNING_MODEL.TRAINER.DANN.lmda
        self.max_epoch = self.cfg.OPTIM.MAX_EPOCH

    def forward(self, input, return_feature=False):
        f_target = self.CommonFeature(input)
        logits_target = self.TargetClassifier(f_target)
        probs = F.softmax(logits_target, dim=1)
        if return_feature:
            return probs, logits_target
        return probs
    def configure_optimizers(self):
        params = list(self.CommonFeature.parameters()) + \
                 list(self.TargetClassifier.parameters()) + \
                 list(self.SourceClassifiers.parameters())+\
                 list(self.DomainDiscriminator.parameters())

        opt_cfg = self.cfg.OPTIM
        opt = build_optimizer(params,opt_cfg)
        scheduler = build_lr_scheduler(optimizer=opt,optim_cfg=opt_cfg)
        optimizers = [opt]
        lr_schedulers=[scheduler]
        return optimizers, lr_schedulers

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
        self.revgrad = ReverseGrad()

    def calculate_dann(self,target_feature,source_feature):
        #there is a problem need to concern. We assume that label target batch size is same as source batch size
        domain_label_target = torch.ones(target_feature.shape[0], 1,device=self.device)
        domain_label_source = torch.zeros(source_feature.shape[0], 1,device=self.device)
        feature = torch.cat([target_feature, source_feature])
        domain_label = torch.cat([domain_label_target, domain_label_source])
        domain_pred = self.DomainDiscriminator(feature)
        loss_d = self.bce(domain_pred, domain_label)
        return loss_d


    def calculate_lmda_factor(self,batch_idx,current_epoch,num_batches,max_epoch,num_pretrain_epochs=0,lmda_scale = 1.0):
        epoch = current_epoch-num_pretrain_epochs
        if epoch < 0:
            lmda = 1.0
        else:
            total_epoch = max_epoch-num_pretrain_epochs
            global_step = batch_idx + epoch * num_batches
            progress = global_step / (total_epoch * num_batches)
            lmda = 2 / (1 + np.exp(-10 * progress)) - 1
            lmda = lmda * lmda_scale  # modify the scale of lmda
        return lmda

    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)
        f_target = self.CommonFeature(input)
        logits_target = self.TargetClassifier(f_target)
        loss_target = self.loss_function(logits_target, label, train=train_mode,weight=weight)
        return loss_target,logits_target,label, f_target

    def on_train_epoch_start(self) -> None:
        if self.source_pretrain_epochs > self.current_epoch:
            self.source_ratio = 1.0
            self.target_ratio = 0.0
            self.d_ratio = 0.0
        else:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO
            self.d_ratio = 1.0

    def training_step(self, batch, batch_idx):
        target_batch, unlabel_batch ,list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)

        loss_source = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.CommonFeature(u)
            logits = self.SourceClassifiers[d](f)
            domain_weight = self.source_domains_class_weight[d]
            loss_source += self.loss_function(logits, y, train=True, weight=domain_weight)
        loss_source /= len(domain_u)

        loss_target, logits_target, label, feat_target = self.share_step(target_batch, train_mode=True, weight=self.class_weight)

        # print("unlabel batch : ",unlabel_batch)
        feat_source = self.CommonFeature(unlabel_batch)
        # total_loss = loss_source+loss_target

        lmda = self.calculate_lmda_factor(batch_idx,self.current_epoch,self.trainer.num_training_batches,self.max_epoch,num_pretrain_epochs=self.source_pretrain_epochs,lmda_scale=self.lmda)

        # feat_source = torch.cat(feat_source, 0)


        feat_target = self.revgrad(feat_target, grad_scaling=lmda)
        feat_source = self.revgrad(feat_source, grad_scaling=lmda)

        # test to combine 2 vector and calculate loss
        loss_d = self.calculate_dann(target_feature=feat_target, source_feature=feat_source)

        total_loss = loss_source * self.source_ratio + loss_target * self.target_ratio + loss_d*self.d_ratio

        y_pred = F.softmax(logits_target, dim=1)
        y = label
        acc = self.train_acc(y_pred, y)

        self.log('Train_acc', acc, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss', total_loss,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('loss_d', loss_d*self.d_ratio, on_step=False,on_epoch=True,prog_bar=True, logger=True)

        return {'loss':total_loss}

    def parse_batch_train(self, batch):
        target_batch = batch["target_loader"]
        unlabel_batch = batch["unlabel_loader"]
        list_source_batches = batch["source_loader"]
        return target_batch,unlabel_batch,list_source_batches




