from dassl.engine import TRAINER_REGISTRY,TrainerMultiAdaptation
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
from dassl.modeling import build_layer

from dassl.engine.da.heterogeneous.heterogeneous_adaptation import HeterogeneousModelAdaptation

@TRAINER_REGISTRY.register()
class HeterogeneousModelAdaptationDSBN(HeterogeneousModelAdaptation):
    """

    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.target_dsbn_idx = len(self.dm.source_domains_label_size)
        print("current target dsbn idx : ", self.target_dsbn_idx)

    def build_temp_layer(self, cfg):
        embedding_layer_info = cfg.MODEL.LAYER
        layer_name = embedding_layer_info.NAME
        layer_params = embedding_layer_info.PARAMS
        total_domain = layer_params.total_domain
        check_total_domain = len(self.dm.source_domains_label_size) + 1
        if total_domain != check_total_domain:
            print("there is problem with the provided total domain : ", total_domain)
            layer_params.total_domain = check_total_domain
        print("total domain for DSBN : ", layer_params.total_domain)
        return [layer_name, layer_params]

    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed
        loss_u = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f,d)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_u += self.cce[d](logits, y)
        # print("loss U :",loss_u)
        # print("num domain : ",len(domain_u))
        loss_u /= len(domain_u)
        # print("check range for target data : {} - {}".format(input_x.max(), input_x.min()))
        f_target = self.TargetFeature(input_x)
        temp_layer_target = self.TemporalLayer(f_target,self.target_dsbn_idx)
        logits_target = self.TargetClassifier(temp_layer_target)
        if backprob:
            loss_x = self.ce(logits_target,label_x)
        else:
            loss_x = self.val_ce(logits_target, label_x)
        total_loss = loss_x+loss_u
        loss_summary = {
            'total_loss': total_loss.item(),
            'loss_x': loss_x.item(),
            'loss_u': loss_u.item()
        }
        # print("loss x :",loss_x)

        if backprob:
            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        # else:
            # f_target = self.TargetFeature(input_x)
            # temp_layer_target = self.TemporalLayer(f_target)
            # logits_target = self.TargetClassifier(temp_layer_target)
            # loss_x = self.val_ce(logits_target, label_x)
            # loss_summary = {
            #     'loss_x': loss_x.item()
            # }

        return loss_summary

    def model_inference(self, input,return_feature=False):
        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f, self.target_dsbn_idx)
        logits = self.TargetClassifier(temp_layer)
        result = F.softmax(logits, 1)
        if return_feature:
            return result,temp_layer
        return result
