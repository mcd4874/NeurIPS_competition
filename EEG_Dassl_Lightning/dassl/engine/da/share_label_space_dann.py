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
class ShareLabelDANN(TrainerMultiAdaptation):
    """
        Build each individual first 2 layers of EEGNET for each dataset. All datasets  share a common classifier and a temporal filter layers
        """

    def __init__(self, cfg):
        super().__init__(cfg)
        n_source_domain = self.dm.dataset.source_num_domain
        n_source_batch_size = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
        self.split_batch = n_source_batch_size // n_source_domain
        self.n_source_domain = n_source_domain

        # create a cross entropy loss for target dataset
        self.ce = nn.CrossEntropyLoss()
        # self.ce_1 =  nn.CrossEntropyLoss()
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                print("target dataset has classes weight  : ", torch_weight)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)

        # create a cross entropy loss for each source domain dataset
        self.cce = [nn.CrossEntropyLoss() for _ in range(self.n_source_domain)]
        if cfg.DATASET.DOMAIN_CLASS_WEIGHT:
            domain_class_weight = self.dm.source_domains_class_weight
            for domain, weight in domain_class_weight.items():
                print("source domain {} dataset has class weight : {}".format(domain,weight))
                torch_weight = torch.from_numpy(np.array(weight)).float().to(self.device)
                self.cce[domain] = nn.CrossEntropyLoss(weight=torch_weight)

        self.val_ce = nn.CrossEntropyLoss()

        self.bce = nn.BCEWithLogitsLoss()

        self.lmda = cfg.TRAINER.ShareLabelDANN.lmda

        print("current lmda ratio: {} for ShareLabelDANN".format(self.lmda))






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

        self.TemporalLayer = TemporalLayer(**backbone_params)
        self.TemporalLayer.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.TemporalLayer)))
        self.optim_TemporalLayer = build_optimizer(self.TemporalLayer, cfg.OPTIM)
        self.sched_TemporalLayer = build_lr_scheduler(self.optim_TemporalLayer, cfg.OPTIM)
        self.register_model('TemporalLayer', self.TemporalLayer, self.optim_TemporalLayer,
                            self.sched_TemporalLayer)

        fdim2 = self.TemporalLayer.fdim

        print("fdim2 : ",fdim2)

        print('Building Classifier')
        self.Classifier = nn.Linear(fdim2, self.dm.num_classes)
        self.Classifier.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.Classifier)))
        self.optim_Classifier = build_optimizer(self.Classifier, cfg.OPTIM)
        self.sched_Classifier = build_lr_scheduler(self.optim_Classifier, cfg.OPTIM)
        self.register_model('Classifier', self.Classifier, self.optim_Classifier, self.sched_Classifier)

        print('Building DomainDiscriminator')
        self.DomainDiscriminator = nn.Linear(fdim2, 1)
        self.DomainDiscriminator.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.DomainDiscriminator)))
        self.optim_DomainDiscriminator = build_optimizer(self.DomainDiscriminator, cfg.OPTIM)
        self.sched_DomainDiscriminator = build_lr_scheduler(self.optim_DomainDiscriminator, cfg.OPTIM)
        self.register_model('DomainDiscriminator', self.DomainDiscriminator, self.optim_DomainDiscriminator, self.sched_DomainDiscriminator)

        self.revgrad = ReverseGrad()

    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed



        if backprob:
            loss_u = 0
            temp_feat_u = []
            domain_label_u = []
            for u, y, d in zip(list_input_u, list_label_u, domain_u):
                # print("test : ")
                # print(u.shape)
                # print('input u shape : ',u.shape)
                f = self.SourceFeatures[d](u)
                # print(f.shape)
                temp_layer = self.TemporalLayer(f)
                temp_feat_u.append(temp_layer)
                # print(temp_layer.shape)
                logits = self.Classifier(temp_layer)
                loss_u += self.cce[d](logits, y)

                current_domain_u = torch.zeros(u.shape[0], 1).to(self.device)
                domain_label_u.append(current_domain_u)

            loss_u /= len(domain_u)
            f_target = self.TargetFeature(input_x)
            temp_layer_target = self.TemporalLayer(f_target)
            logits_target = self.Classifier(temp_layer_target)
            loss_x = self.ce(logits_target,label_x)


            # lmda = 0.1
            # lmda = 0.5

            lmda = self.lmda

            domain_label_x = torch.ones(input_x.shape[0], 1).to(self.device)
            domain_label_u = torch.cat(domain_label_u,0)
            feat_u = torch.cat(temp_feat_u,0)

            feat_x = self.revgrad(temp_layer_target, grad_scaling=lmda)
            feat_u = self.revgrad(feat_u, grad_scaling=lmda)
            output_xd = self.DomainDiscriminator(feat_x)
            output_ud = self.DomainDiscriminator(feat_u)
            loss_d = self.bce(output_xd, domain_label_x) + self.bce(output_ud, domain_label_u)

            total_loss = loss_x + loss_u + loss_d
            loss_summary = {
                'total_loss': total_loss.item(),
                'loss_x': loss_x.item(),
                'loss_u': loss_u.item(),
                'loss_d':loss_d.item()
            }


            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            f_target = self.TargetFeature(input_x)
            temp_layer_target = self.TemporalLayer(f_target)
            logits_target = self.Classifier(temp_layer_target)
            loss_x = self.val_ce(logits_target, label_x)
            loss_summary = {
                'loss_x': loss_x.item()
            }

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

        list_train_loader_u_iter = [iter(train_loader_u) for train_loader_u in self.list_train_loader_u]
        for self.batch_idx in range(num_batches):
            try:
                batch_x = next(valid_loader_x_iter)
            except StopIteration:
                valid_loader_x_iter = iter(data_loader)
                batch_x = next(valid_loader_x_iter)

            list_batch_u = list()
            for train_loader_u_iter_idx in range(len(list_train_loader_u_iter)):
                train_loader_u_iter = list_train_loader_u_iter[train_loader_u_iter_idx]
                # batch_u = next(train_loader_u_iter)
                try:
                    batch_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(self.list_train_loader_u[train_loader_u_iter_idx])
                    list_train_loader_u_iter[train_loader_u_iter_idx] = train_loader_u_iter
                    batch_u = next(train_loader_u_iter)
                list_batch_u.append(batch_u)

            input, label, _, _,_,_ = self.parse_batch_train(batch_x, list_batch_u)
            loss = self.forward_backward(batch_x, list_batch_u, backprob=False)
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


    def model_inference(self, input):

        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f)
        logits = self.Classifier(temp_layer)
        result = F.softmax(logits, 1)
        return result

    def get_model_architecture(self):
        model_architecture = {
            "backbone": self.TargetFeature,
            "layer_1": self.TemporalLayer,
            "classifier_layer": self.Classifier
        }
        return model_architecture
