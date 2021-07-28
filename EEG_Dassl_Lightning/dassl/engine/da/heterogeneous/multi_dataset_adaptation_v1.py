from dassl.engine import TRAINER_REGISTRY
from dassl.engine.trainer import TrainerMultiAdaptation
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.engine.trainer import SimpleNet





@TRAINER_REGISTRY.register()
class MultiDatasetAdaptationV1(TrainerMultiAdaptation):
    """
    Apply EEGNet for multi-source dataset and 1 target dataset
    the source and target datasets share all 3 conv layers in EEGNet
    Each dataset has its own classifier
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)

    def build_model(self):
        cfg = self.cfg
        print("Params : ", cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE)
        print('Building F')



        print('Building CommonFeature')
        backbone_info = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE
        FC_info = cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC
        # backbone_params = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS.copy()

        self.CommonFeature = SimpleNet(backbone_info,FC_info, 0, **cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS)
        self.fdim = self.CommonFeature.fdim

        print('Building Target Classifier')
        self.TargetClassifier = self.create_classifier(self.fdim, self.num_classes,FC_info=FC_info)

        print('Building SourceClassifiers')
        print("source domains label size : ", self.source_domains_label_size)
        source_classifier_list = []
        for num_class in self.source_domains_label_size:
            source_classifier = self.create_classifier(self.fdim, num_class,FC_info=FC_info)
            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )
    def forward(self, input, return_feature=False):
        f_target = self.CommonFeature(input)
        logits_target = self.TargetClassifier(f_target)
        probs = F.softmax(logits_target, dim=1)
        if return_feature:
            return probs, logits_target
        return probs
    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)
        f_target = self.CommonFeature(input)
        logits_target = self.TargetClassifier(f_target)
        loss_target = self.loss_function(logits_target, label, train=train_mode,weight=weight)
        return loss_target,logits_target,label, f_target
    def training_step(self, batch, batch_idx):

        target_batch, list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)
        loss_source = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.CommonFeature(u)
            logits = self.SourceClassifiers[d](f)
            domain_weight = self.source_domains_class_weight[d]
            loss_source += self.loss_function(logits, y, train=True, weight=domain_weight)
        loss_source /= len(domain_u)

        loss_target, logits_target, label, _ = self.share_step(target_batch, train_mode=True, weight=self.class_weight)

        total_loss = loss_source * self.source_ratio + loss_target * self.target_ratio

        y_pred = F.softmax(logits_target, dim=1)
        acc = self.train_acc(y_pred, label)
        self.log('Train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': total_loss}
    def configure_optimizers(self):
        params = list(self.CommonFeature.parameters()) + \
                 list(self.TargetClassifier.parameters()) + \
                 list(self.SourceClassifiers.parameters())

        opt = torch.optim.Adam(params, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=1.0
        )
        optimizers = [opt]
        lr_schedulers=[scheduler]
        # lr_schedulers = {'scheduler': scheduler, 'monitor': 'metric_to_track'}
        return optimizers, lr_schedulers