from dassl.engine import TRAINER_REGISTRY
# from dassl.engine.trainer import TrainerMultiAdaptation
from dassl.engine.da.heterogeneous.multi_dataset_adaptation_v1 import MultiDatasetAdaptationV1
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.engine.trainer import SimpleNet

from dassl.optim import build_optimizer
from dassl.optim import build_lr_scheduler


@TRAINER_REGISTRY.register()
class MultiDatasetAdaptationV2(MultiDatasetAdaptationV1):
    """
    Apply EEGNet for multi-source dataset and 1 target dataset
    the source and target datasets share all 3 conv layers in EEGNet
    Each dataset has its own classifier
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
    def training_step(self, batch, batch_idx):

        target_batch, list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)
        loss_source = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.CommonFeature(u)
            logits = self.SourceClassifiers[d](f)
            if not self.no_source_weight:
                domain_weight = self.source_domains_class_weight[d]
            else:
                domain_weight = None
            loss_source += self.loss_function(logits, y, train=True, weight=domain_weight)
        # loss_source /= len(domain_u)

        if self.no_target_weight:
            # print("no target weight")
            loss_target, logits_target, label, _ = self.share_step(target_batch, train_mode=True)
        else:
            loss_target, logits_target, label, _ = self.share_step(target_batch, train_mode=True, weight=self.class_weight)
        # loss_target, logits_target, label, _ = self.share_step(target_batch, train_mode=True)

        # total_loss = loss_source * self.source_ratio + loss_target * self.target_ratio

        total_loss = (loss_target+loss_source)/ (len(domain_u)+1)


        y_pred = F.softmax(logits_target, dim=1)
        acc = self.train_acc(y_pred, label)
        self.log('Train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': total_loss}
