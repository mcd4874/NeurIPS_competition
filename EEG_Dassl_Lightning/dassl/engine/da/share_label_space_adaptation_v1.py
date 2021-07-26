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





@TRAINER_REGISTRY.register()
class ShareLabelModelAdaptationV1(TrainerMultiAdaptation):
    """
    Build each individual EEGNET for each dataset. All datasets only share common classifier
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




        fdim2 = self.TargetFeature.fdim

        print("fdim2 : ",fdim2)

        print('Building Classifier')
        self.Classifier = nn.Linear(fdim2, self.dm.num_classes)
        self.Classifier.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.Classifier)))
        self.optim_Classifier = build_optimizer(self.Classifier, cfg.OPTIM)
        self.sched_Classifier = build_lr_scheduler(self.optim_Classifier, cfg.OPTIM)
        self.register_model('Classifier', self.Classifier, self.optim_Classifier, self.sched_Classifier)

    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed
        if backprob:
            loss_u = 0
            for u, y, d in zip(list_input_u, list_label_u, domain_u):
                # print("test : ")
                # print(u.shape)
                # print('input u shape : ',u.shape)
                f = self.SourceFeatures[d](u)
                # print(f.shape)
                # print(temp_layer.shape)
                logits = self.Classifier(f)
                loss_u += self.cce[d](logits, y)

            loss_u /= len(domain_u)
            f_target = self.TargetFeature(input_x)
            logits_target = self.Classifier(f_target)
            loss_x = self.ce(logits_target,label_x)
            total_loss = loss_x+loss_u
            loss_summary = {
                'total_loss': total_loss.item(),
                'loss_x': loss_x.item(),
                'loss_u': loss_u.item()
            }


            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            f_target = self.TargetFeature(input_x)
            logits_target = self.Classifier(f_target)
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
        logits = self.Classifier(f)
        result = F.softmax(logits, 1)
        return result

    def get_model_architecture(self):
        model_architecture = {
            "backbone": self.TargetFeature,
            "classifier_layer":self.Classifier
        }
        return model_architecture
