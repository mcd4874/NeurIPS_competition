from dassl.engine import TRAINER_REGISTRY,TrainerMultiAdaptation
from dassl.data import DataManager
from dassl.utils import MetricMeter
from torch.utils.data import Dataset as TorchDataset
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.modeling.ops import ReverseGrad
from dassl.engine.trainer_tmp import SimpleNet
from dassl.modeling import build_layer

import numpy as np



@TRAINER_REGISTRY.register()
class HeterogeneousDANN(TrainerMultiAdaptation):
    """
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.bce = nn.BCEWithLogitsLoss()
        self.lmda = cfg.TRAINER.HeterogeneousDANN.lmda
        print("current max lmda : ",self.lmda)

    def build_temp_layer(self, cfg):
        embedding_layer_info = cfg.MODEL.LAYER
        layer_name = embedding_layer_info.NAME
        layer_params = embedding_layer_info.PARAMS
        return [layer_name, layer_params]
    def build_model(self):
        cfg = self.cfg
        print("Params : ",cfg.MODEL.BACKBONE)
        print('Building F')

        backbone_params = cfg.MODEL.BACKBONE.PARAMS.copy()

        print(backbone_params)

        print('Building TargetFeature')

        self.TargetFeature = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
        self.TargetFeature.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.TargetFeature)))
        self.optim_TargetFeature = build_optimizer(self.TargetFeature, cfg.OPTIM)
        self.sched_TargetFeature = build_lr_scheduler(self.optim_TargetFeature, cfg.OPTIM)
        self.register_model('TargetFeature', self.TargetFeature, self.optim_TargetFeature, self.sched_TargetFeature)
        fdim1 = self.TargetFeature.fdim

        print(' Building SourceFeatures')
        #special case for only 1 source domain
        source_domain_input_shapes = self.dm.source_domains_input_shape
        print(source_domain_input_shapes)
        list_num_ch = [input_shape[0] for input_shape in source_domain_input_shapes]
        print("list num ch for source domains : ",list_num_ch)
        source_feature_list = []
        for num_ch in list_num_ch:
            backbone_params['num_ch'] = num_ch
            source_feature = SimpleNet(cfg, cfg.MODEL, 0, **backbone_params)
            source_feature_list.append(source_feature)
        self.SourceFeatures = nn.ModuleList(
            source_feature_list
        )
        self.SourceFeatures.to(self.device)

        print('# params: {:,}'.format(count_num_param(self.SourceFeatures)))
        self.optim_SourceFeatures = build_optimizer(self.SourceFeatures, cfg.OPTIM)
        self.sched_SourceFeatures = build_lr_scheduler(self.optim_SourceFeatures, cfg.OPTIM)
        self.register_model('SourceFeatures', self.SourceFeatures, self.optim_SourceFeatures, self.sched_SourceFeatures)


        print('Building Temporal Layer')
        layer_name, layer_params = self.build_temp_layer(cfg)
        self.TemporalLayer = build_layer(layer_name, verbose=True, **layer_params)

        # self.TemporalLayer = TemporalLayer(**backbone_params)
        self.TemporalLayer.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.TemporalLayer)))
        self.optim_TemporalLayer = build_optimizer(self.TemporalLayer, cfg.OPTIM)
        self.sched_TemporalLayer = build_lr_scheduler(self.optim_TemporalLayer, cfg.OPTIM)
        self.register_model('TemporalLayer', self.TemporalLayer, self.optim_TemporalLayer,
                            self.sched_TemporalLayer)

        fdim2 = self.TemporalLayer.fdim

        print("fdim2 : ",fdim2)

        print('Building SourceClassifiers')
        print("source domains label size : ",self.dm.source_domains_label_size)
        source_classifier_list = []
        for num_class in self.dm.source_domains_label_size:
            # source_classifier = nn.Linear(fdim2, num_class)
            source_classifier = self.create_classifier(fdim2, num_class)

            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )
        self.SourceClassifiers.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.SourceClassifiers)))
        self.optim_SourceClassifiers = build_optimizer(self.SourceClassifiers, cfg.OPTIM)
        self.sched_SourceClassifiers = build_lr_scheduler(self.optim_SourceClassifiers, cfg.OPTIM)
        self.register_model('SourceClassifiers', self.SourceClassifiers, self.optim_SourceClassifiers,
                            self.sched_SourceClassifiers)

        print('Building Target Classifier')
        self.TargetClassifier = self.create_classifier(fdim2,self.dm.num_classes)
        # self.TargetClassifier = nn.Linear(fdim2, self.dm.num_classes)
        self.TargetClassifier.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.TargetClassifier)))
        self.optim_TargetClassifier = build_optimizer(self.TargetClassifier, cfg.OPTIM)
        self.sched_TargetClassifier = build_lr_scheduler(self.optim_TargetClassifier, cfg.OPTIM)
        self.register_model('TargetClassifier', self.TargetClassifier, self.optim_TargetClassifier, self.sched_TargetClassifier)

        print('Building DomainDiscriminator')
        self.DomainDiscriminator = nn.Linear(fdim2, 1)
        self.DomainDiscriminator.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.DomainDiscriminator)))
        self.optim_DomainDiscriminator = build_optimizer(self.DomainDiscriminator, cfg.OPTIM)
        self.sched_DomainDiscriminator = build_lr_scheduler(self.optim_DomainDiscriminator, cfg.OPTIM)
        self.register_model('DomainDiscriminator', self.DomainDiscriminator, self.optim_DomainDiscriminator,
                            self.sched_DomainDiscriminator)

        self.revgrad = ReverseGrad()


    def calculate_dann(self,target_feature,source_feature):
        #there is a problem need to concern. We assume that label target batch size is same as source batch size
        domain_label_target = torch.ones(target_feature.shape[0], 1).to(self.device)
        domain_label_source = torch.zeros(source_feature.shape[0], 1).to(self.device)
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

    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed
        loss_u = 0
        temp_feat_u = []
        # domain_label_u = []
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f)
            temp_feat_u.append(temp_layer)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_u += self.cce[d](logits, y)

        loss_u /= len(domain_u)
        f_target = self.TargetFeature(input_x)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.TargetClassifier(temp_layer_target)
        if backprob:
            loss_x = self.ce(logits_target,label_x)
        else:
            loss_x = self.val_ce(logits_target, label_x)

        feat_u = torch.cat(temp_feat_u, 0)


        # global_step = self.batch_idx + self.epoch * self.num_batches
        # progress = global_step / (self.max_epoch * self.num_batches)
        # lmda = 2 / (1 + np.exp(-10 * progress)) - 1
        # lmda = lmda* self.lmda # modify the scale of lmda
        lmda = self.calculate_lmda_factor(self.batch_idx,self.epoch,self.num_batches,self.max_epoch,num_pretrain_epochs=self.pre_train_epochs,lmda_scale=self.lmda)
        n_iter = self.epoch * self.num_batches + self.batch_idx
        self.write_scalar('train/lmda', lmda, n_iter)

        feat_x = self.revgrad(temp_layer_target, grad_scaling=lmda)
        feat_u = self.revgrad(feat_u, grad_scaling=lmda)

        #test to combine 2 vector and calculate loss
        loss_d = self.calculate_dann(target_feature=feat_x,source_feature=feat_u)

        #old way of calculate seperate domain loss for target and source
        # output_xd = self.DomainDiscriminator(feat_x)
        # output_ud = self.DomainDiscriminator(feat_u)
        # loss_d = self.bce(output_xd, domain_label_x) + self.bce(output_ud, domain_label_u)

        total_loss = loss_x + loss_u + loss_d
        loss_summary = {
            'total_loss': total_loss.item(),
            'loss_x': loss_x.item(),
            'loss_u': loss_u.item(),
            'loss_d': loss_d.item(),
            'lmda_factor': lmda
        }

        if backprob:
            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        return loss_summary

    def model_inference(self, input, return_feature=False):
        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f)
        logits = self.TargetClassifier(temp_layer)
        result = F.softmax(logits, 1)
        if return_feature:
            return result, temp_layer
        return result
