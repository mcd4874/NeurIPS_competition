from dassl.engine import TRAINER_REGISTRY,TrainerXU
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


class TemporalLayer(nn.Module):
    def __init__(self, kern_legnth=64, num_ch=22, samples = 256,F1=8, D=2, F2=16,drop_prob=0.4):
        super().__init__()
        self.c3 = nn.Sequential (
            #conv_separable_depth"
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 16), stride=1, bias=False,groups=(F2), padding=(0, 8)),
            #conv_separable_point
            nn.Conv2d(F2, F2, (1, 1), bias=False,stride=1,padding=(0, 0))
        )
        self.b3 = nn.BatchNorm2d(F2, momentum=0.1, affine=True, eps=1e-3)
        self.d3 = nn.Dropout(drop_prob)
        self._out_features = F2*(samples//32)

    def forward(self,input):
        h3 = self.d3(F.avg_pool2d(F.elu(self.b3(self.c3(input))),(1,8)) )
        flatten = torch.flatten(h3, start_dim=1)
        return flatten

    @property
    def fdim(self):
        return self._out_features


@TRAINER_REGISTRY.register()
class BaseModelAdaptation(TrainerXU):
    """
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        n_source_domain = self.dm.dataset.source_num_domain
        n_source_batch_size = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
        self.split_batch = n_source_batch_size // n_source_domain
        self.n_source_domain = n_source_domain





    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_U.SAMPLER == 'RandomDomainSampler'

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
        list_num_ch = [22]
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

        self.TemporalLayer = TemporalLayer(**backbone_params)
        self.TemporalLayer.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.TemporalLayer)))
        self.optim_TemporalLayer = build_optimizer(self.TemporalLayer, cfg.OPTIM)
        self.sched_TemporalLayer = build_lr_scheduler(self.optim_TemporalLayer, cfg.OPTIM)
        self.register_model('TemporalLayer', self.TemporalLayer, self.optim_TemporalLayer,
                            self.sched_TemporalLayer)

        fdim2 = self.TemporalLayer.fdim

        print("fdim2 : ",fdim2)


        print('Building SourceClassifiers')
        source_classifier_list = []
        for num_class in self.dm.dataset.source_domain_num_class:
            source_classifier = nn.Linear(fdim2, num_class)
            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )
        self.SourceClassifiers.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.SourceClassifiers)))
        self.optim_SourceClassifiers = build_optimizer(self.SourceClassifiers, cfg.OPTIM)
        self.sched_SourceClassifiers = build_lr_scheduler(self.optim_SourceClassifiers, cfg.OPTIM)
        self.register_model('SourceClassifiers', self.SourceClassifiers, self.optim_SourceClassifiers, self.sched_SourceClassifiers)

        print('Building TargetClassifier')
        self.TargetClassifier = nn.Linear(fdim2, self.dm.num_classes)
        self.TargetClassifier.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.TargetClassifier)))
        self.optim_TargetClassifier = build_optimizer(self.TargetClassifier, cfg.OPTIM)
        self.sched_TargetClassifier = build_lr_scheduler(self.optim_TargetClassifier, cfg.OPTIM)
        self.register_model('TargetClassifier', self.TargetClassifier, self.optim_TargetClassifier, self.sched_TargetClassifier)

    def forward_backward(self, batch_x, batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, batch_u)
        input_x, label_x, domain_x, input_u,label_u,domain_u = parsed
        input_u = torch.split(input_u, self.split_batch, 0)
        label_u = torch.split(label_u, self.split_batch, 0)
        domain_u = torch.split(domain_u, self.split_batch, 0)
        domain_u = [d[0].item() for d in domain_u]
        loss_u = 0
        for u, y, d in zip(input_u, label_u, domain_u):
            # print("test : ")
            # print(u.shape)
            f = self.SourceFeatures[d](u)
            # print(f.shape)
            temp_layer = self.TemporalLayer(f)
            # print(temp_layer.shape)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_u += F.cross_entropy(logits, y)

        loss_u /= len(domain_u)


        f_target = self.TargetFeature(input_x)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.TargetClassifier(temp_layer_target)
        loss_x = F.cross_entropy(logits_target,label_x)

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

    @torch.no_grad()
    # def validate(self,full_results = False):
    def validate(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        losses = MetricMeter()

        print('Do evaluation on {} set'.format('valid set'))
        data_loader = self.val_loader
        assert data_loader is not None

        num_batches = len(data_loader)
        valid_loader_x_iter = iter(data_loader)
        loader_u_iter = iter(self.train_loader_u)
        for self.batch_idx in range(num_batches):
            try:
                batch_x = next(valid_loader_x_iter)
            except StopIteration:
                valid_loader_x_iter = iter(data_loader)
                batch_x = next(valid_loader_x_iter)

            try:
                batch_u = next(loader_u_iter)
            except StopIteration:
                loader_u_iter = iter(self.train_loader_u)
                batch_u = next(loader_u_iter)

            input, label, domain, target,target_label,target_domain = self.parse_batch_train(batch_x, batch_u)
            loss = self.forward_backward(batch_x, batch_u, backprob=False)
            losses.update(loss)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss_x'].avg

        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        # if full_results:
        return [total_loss,losses.dict_results(),results]
        # return total_loss

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['eeg_data']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_u = batch_u['eeg_data']
        label_u = batch_u['label']
        domain_u = batch_u['domain']

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        label_u = label_u.to(self.device)

        return input_x, label_x, domain_x, input_u,label_u,domain_u





    def model_inference(self, input):

        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f)
        logits = self.TargetClassifier(temp_layer)
        result = F.softmax(logits, 1)
        return result

    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        # data_loader = self.val_loader if split == 'val' else self.test_loader
        data_loader = self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results

    # def detail_test_report(self):
