from dassl.engine import TRAINER_REGISTRY
from dassl.engine.trainer import TrainerMultiAdaptation,SimpleNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from dassl.utils import count_num_param
from dassl.modeling import build_layer


from dassl.modeling import build_head, build_backbone

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)

class ComponentNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a single classifier layer.
    """

    def __init__(self, backbone_info, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            backbone_info.NAME,
            pretrained=backbone_info.PRETRAINED,
            pretrained_path = backbone_info.PRETRAINED_PATH,
            **kwargs
        )
        fdim = self.backbone.out_features
        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x):
        f = self.backbone(x)
        return f

@TRAINER_REGISTRY.register()
class ComponentAdaptation(TrainerMultiAdaptation):
    """
    Apply EEGNet for multi-source dataset and 1 target dataset
    Each dataset has its own 1st temporal filter layer and spatial filter layer
    The output from the spatial filter layer isn't 1. We aim to compress the channel to n
    Another convolution layer that project n channels to 1 channels and the 2nd temporal filter layer are shared among all dataset
    Each dataset has its own classifier
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)

    def build_model(self):
        cfg = self.cfg
        print("Params : ", cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE)
        print('Building F')

        # backbone_params = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS.copy()
        main_feature_backbone = cfg.LIGHTNING_MODEL.COMPONENTS.MAIN_COMPONENT.BACKBONE
        target_component_backbone = cfg.LIGHTNING_MODEL.COMPONENTS.TARGET_COMPONENT.BACKBONE
        source_component_backbone = cfg.LIGHTNING_MODEL.COMPONENTS.SOURCE_COMPONENT.BACKBONE
        source_component_params = source_component_backbone.PARAMS.copy()
        FC_info = cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC

        # print(backbone_params)

        print('Building TargetFeature')

        self.TargetFeature = ComponentNet(target_component_backbone,**target_component_backbone.PARAMS)

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
            source_component_params['num_ch'] = num_ch
            source_feature = ComponentNet(source_component_backbone,**source_component_params)
            source_feature_list.append(source_feature)
        self.SourceFeatures = nn.ModuleList(
            source_feature_list
        )
        print('# params: {:,}'.format(count_num_param(self.SourceFeatures)))

        print('Building SourceClassifiers')
        print("source domains label size : ", self.source_domains_label_size)
        source_classifier_list = []
        for num_class in self.source_domains_label_size:
            source_classifier = self.create_classifier(self.fdim2, num_class,FC_info=FC_info)

            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )

        self.MainFeature = ComponentNet(main_feature_backbone,**main_feature_backbone.PARAMS)

    def training_step(self, batch, batch_idx):

        target_batch,list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)
        loss_source = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            source_f = self.SourceFeatures[d](u)
            common_f = self.MainFeature(source_f)
            temp_layer = self.TemporalLayer(common_f)
            logits = self.SourceClassifiers[d](temp_layer)

            domain_weight = self.source_domains_class_weight[d]
            loss_source += self.loss_function(logits, y,train=True,weight=domain_weight)
        loss_source /= len(domain_u)

        loss_target,logits_target, label, _= self.share_step(target_batch,train_mode=True,weight=self.class_weight)

        total_loss = loss_source*self.source_ratio+loss_target*self.target_ratio


        y_pred = F.softmax(logits_target, dim=1)
        acc = self.train_acc(y_pred,label)

        self.log('Train_acc', acc, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss', total_loss, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        return {'loss':total_loss}
    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)
        f_target = self.TargetFeature(input)
        common_f = self.MainFeature(f_target)
        temp_layer_target = self.TemporalLayer(common_f)
        logits_target = self.TargetClassifier(temp_layer_target)
        loss_target = self.loss_function(logits_target, label, train=train_mode,weight=weight)
        return loss_target,logits_target,label, temp_layer_target




