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
class MultiDatasetDann(TrainerMultiAdaptation):
    """

    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
        self.bce = nn.BCEWithLogitsLoss()
        self.lmda =cfg.LIGHTNING_MODEL.TRAINER.DANN.lmda
        self.max_epoch = self.cfg.OPTIM.MAX_EPOCH

    def configure_optimizers(self):
        params = list(self.TargetFeature.parameters()) + \
                 list(self.TemporalLayer.parameters()) + \
                 list(self.TargetClassifier.parameters()) + \
                 list(self.SourceFeatures.parameters()) + \
                 list(self.SourceClassifiers.parameters()) +\
                 list(self.DomainDiscriminator.parameters())

        opt = torch.optim.Adam(params, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=1.0
        )
        optimizers = [opt]
        # lr_schedulers = {'scheduler': scheduler, 'monitor': 'metric_to_track'}
        lr_schedulers = [scheduler]
        return optimizers, lr_schedulers

    def build_model(self):
        super(MultiDatasetDann, self).build_model()
        self.DomainDiscriminator = nn.Linear(self.fdim2, 1)
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
        total_epoch = max_epoch-num_pretrain_epochs
        global_step = batch_idx + epoch * num_batches
        progress = global_step / (total_epoch * num_batches)
        lmda = 2 / (1 + np.exp(-10 * progress)) - 1
        lmda = lmda * lmda_scale  # modify the scale of lmda
        return lmda

    def share_step(self, batch, train_mode=True):
        input, label, domain = self.parse_target_batch(batch)
        f_target = self.TargetFeature(input)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.TargetClassifier(temp_layer_target)
        loss_target = self.loss_function(logits_target, label, train=train_mode)
        return loss_target, logits_target, label, temp_layer_target

    def training_step(self, batch, batch_idx):
        target_batch,list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)

        loss_source = 0
        feat_source = []
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f)
            feat_source.append(temp_layer)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_source += self.loss_function(logits, y,train=True)
        loss_source /= len(domain_u)

        loss_target,logits_target, label, feat_target = self.share_step(target_batch,train_mode=True)

        total_loss = loss_source+loss_target

        lmda = self.calculate_lmda_factor(batch_idx,self.current_epoch,self.trainer.num_training_batches,self.max_epoch,num_pretrain_epochs=0,lmda_scale=self.lmda)

        feat_source = torch.cat(feat_source, 0)


        feat_target = self.revgrad(feat_target, grad_scaling=lmda)
        feat_source = self.revgrad(feat_source, grad_scaling=lmda)

        # test to combine 2 vector and calculate loss
        loss_d = self.calculate_dann(target_feature=feat_target, source_feature=feat_source)

        total_loss += loss_d

        y_pred = F.softmax(logits_target, dim=1)
        y = label
        acc = self.train_acc(y_pred, y)

        self.log('Train_acc', acc, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss', total_loss,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('loss_d', loss_d, on_step=False,on_epoch=True,prog_bar=True, logger=True)

        return {'loss':total_loss}





