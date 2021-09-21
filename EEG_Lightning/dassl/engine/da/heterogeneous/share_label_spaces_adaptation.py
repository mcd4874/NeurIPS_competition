from dassl.engine import TRAINER_REGISTRY,TrainerMultiAdaptation
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



@TRAINER_REGISTRY.register()
class ShareLabelModelAdaptation(TrainerMultiAdaptation):
    """
        Build each individual first 2 layers of EEGNET for each dataset. All datasets  share a common classifier and a temporal filter layers
        """

    def __init__(self, cfg):
        super().__init__(cfg)


    def build_temp_layer(self, cfg):
        embedding_layer_info = cfg.MODEL.LAYER
        layer_name = embedding_layer_info.NAME
        layer_params = embedding_layer_info.PARAMS
        return [layer_name, layer_params]


    # def check_cfg(self, cfg):
    #     assert cfg.DATALOADER.TRAIN_U.SAMPLER == 'RandomDomainSampler'

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
        self.TemporalLayer.to(self.device)


        print('# params: {:,}'.format(count_num_param(self.TemporalLayer)))
        self.optim_TemporalLayer = build_optimizer(self.TemporalLayer, cfg.OPTIM)
        self.sched_TemporalLayer = build_lr_scheduler(self.optim_TemporalLayer, cfg.OPTIM)
        self.register_model('TemporalLayer', self.TemporalLayer, self.optim_TemporalLayer,
                            self.sched_TemporalLayer)

        fdim2 = self.TemporalLayer.fdim

        print("fdim2 : ",fdim2)

        print('Building Classifier')
        self.Classifier = self.create_classifier(fdim2,self.dm.num_classes)
        self.Classifier.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.Classifier)))
        self.optim_Classifier = build_optimizer(self.Classifier, cfg.OPTIM)
        self.sched_Classifier = build_lr_scheduler(self.optim_Classifier, cfg.OPTIM)
        self.register_model('Classifier', self.Classifier, self.optim_Classifier, self.sched_Classifier)

    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed
        # if backprob:
        loss_u = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f)
            logits = self.Classifier(temp_layer)
            loss_u += self.cce[d](logits, y)

        loss_u /= len(domain_u)
        f_target = self.TargetFeature(input_x)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.Classifier(temp_layer_target)
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

        if backprob:
            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        return loss_summary
    #
    # @torch.no_grad()
    # def validate(self):
    #     """A generic testing pipeline."""
    #     self.set_model_mode('eval')
    #     self.evaluator.reset()
    #     losses = MetricMeter()
    #
    #     print('Do evaluation on {} set'.format('valid set'))
    #     data_loader = self.val_loader
    #     assert data_loader is not None
    #
    #     num_batches = len(data_loader)
    #     valid_loader_x_iter = iter(data_loader)
    #
    #     list_train_loader_u_iter = [iter(train_loader_u) for train_loader_u in self.list_train_loader_u]
    #     for self.batch_idx in range(num_batches):
    #         try:
    #             batch_x = next(valid_loader_x_iter)
    #         except StopIteration:
    #             valid_loader_x_iter = iter(data_loader)
    #             batch_x = next(valid_loader_x_iter)
    #
    #         list_batch_u = list()
    #         for train_loader_u_iter_idx in range(len(list_train_loader_u_iter)):
    #             train_loader_u_iter = list_train_loader_u_iter[train_loader_u_iter_idx]
    #             # batch_u = next(train_loader_u_iter)
    #             try:
    #                 batch_u = next(train_loader_u_iter)
    #             except StopIteration:
    #                 train_loader_u_iter = iter(self.list_train_loader_u[train_loader_u_iter_idx])
    #                 list_train_loader_u_iter[train_loader_u_iter_idx] = train_loader_u_iter
    #                 batch_u = next(train_loader_u_iter)
    #             list_batch_u.append(batch_u)
    #
    #         input, label, _, _,_,_ = self.parse_batch_train(batch_x, list_batch_u)
    #         loss = self.forward_backward(batch_x, list_batch_u, backprob=False)
    #         losses.update(loss)
    #         output = self.model_inference(input)
    #         self.evaluator.process(output, label)
    #
    #     results = self.evaluator.evaluate()
    #     total_loss = losses.meters['loss_x'].avg
    #     val_losses = losses.dict_results()
    #
    #     for k, v in results.items():
    #         tag = '{}/{}'.format('validation', k)
    #         self.write_scalar(tag, v, self.epoch)
    #
    #     for k, v in val_losses.items():
    #         tag = '{}/{}'.format('validation', k)
    #         self.write_scalar(tag, v, self.epoch)
    #     return [total_loss,losses.dict_results(),results]

    def model_inference(self, input,return_feature=False):
        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f)
        logits = self.Classifier(temp_layer)
        result = F.softmax(logits, 1)
        if return_feature:
            return result,temp_layer
        return result

    def get_model_architecture(self):
        model_architecture = {
            "backbone": self.TargetFeature,
            "layer_1": self.TemporalLayer,
            "classifier_layer":self.Classifier
        }
        return model_architecture