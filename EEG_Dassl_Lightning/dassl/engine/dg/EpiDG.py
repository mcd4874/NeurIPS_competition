import torch
from torch.nn import functional as F
import torch.nn as nn
from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.modeling.ops import ReverseGrad
from dassl.engine.trainer_tmp import SimpleNet
from torch.utils.data import Dataset as TorchDataset
from dassl.utils import MetricMeter

import numpy as np
@TRAINER_REGISTRY.register()
class EpiDG(TrainerX):
    """
    Episodic Training for Domain Generalization
    https://arxiv.org/abs/1902.00113
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        if n_domain <= 0:
            n_domain = self.dm.num_source_domains
        self.n_domain = n_domain
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.split_batch = batch_size // n_domain

        #create a cross entropy loss for whole dataset
        self.ce = nn.CrossEntropyLoss()
        # self.ce_1 =  nn.CrossEntropyLoss()
        self.val_ce = nn.CrossEntropyLoss()
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                print("torch weight  : ",torch_weight)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)

        # create cross entropy losses for each domain in the dataset
        self.cce = [nn.CrossEntropyLoss() for _ in range(self.n_domain)]
        if cfg.DATASET.DOMAIN_CLASS_WEIGHT:
            domain_class_weight = self.dm.dataset.domain_class_weight
            for domain, weight in domain_class_weight.items():
                # domain_class_weight[domain] = torch.from_numpy(np.array(weight)).float().to(self.device)
                torch_weight = torch.from_numpy(np.array(weight)).float().to(self.device)
                self.cce[domain] = nn.CrossEntropyLoss(weight=torch_weight)


        self.candidates = np.arange(self.n_domain)
        self.warn_up_AGG = cfg.TRAINER.EpiDG.warn_up_AGG
        self.start_train_feature = cfg.TRAINER.EpiDG.start_train_feature
        self.start_train_classifier = cfg.TRAINER.EpiDG.start_train_classifier
        self.warm_up_DS = cfg.TRAINER.EpiDG.warm_up_DS

        self.loss_weight_epir = cfg.TRAINER.EpiDG.loss_weight_epir
        self.loss_weight_epif = cfg.TRAINER.EpiDG.loss_weight_epif
        self.loss_weight_epic = cfg.TRAINER.EpiDG.loss_weight_epic

    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == 'RandomDomainSampler'



    def build_model(self):
        cfg = self.cfg

        print('Building Feature')
        self.Feature = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
        self.Feature.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.Feature)))
        self.optim_Feature = build_optimizer(self.Feature, cfg.OPTIM)
        self.sched_Feature = build_lr_scheduler(self.optim_Feature, cfg.OPTIM)
        self.register_model('Feature', self.Feature, self.optim_Feature, self.sched_Feature)
        fdim = self.Feature.fdim

        self.Classifier = nn.Linear(fdim, self.num_classes)
        self.Classifier.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.Classifier)))
        self.optim_Classifier = build_optimizer(self.Classifier, cfg.OPTIM)
        self.sched_Classifier = build_lr_scheduler(self.optim_Classifier, cfg.OPTIM)
        self.register_model('Classifier', self.Classifier, self.optim_Classifier, self.sched_Classifier)

        self.CR =  nn.Linear(fdim, self.num_classes)
        self.CR.to(self.device)


        print("num domains : ",self.dm.num_source_domains)
        print('Building FF')
        self.FF = nn.ModuleList(
            [
                SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
                for _ in range(self.dm.num_source_domains)
            ]
        )
        self.FF.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.FF)))
        print(self.dm.num_source_domains)
        self.optim_FF = build_optimizer(self.FF, cfg.OPTIM)
        self.sched_FF = build_lr_scheduler(self.optim_FF, cfg.OPTIM)
        self.register_model('FF', self.FF, self.optim_FF, self.sched_FF)



        print('Building CC')
        self.CC = nn.ModuleList(
            [
                nn.Linear(fdim, self.num_classes)
                for _ in range(self.dm.num_source_domains)
            ]
        )
        self.CC.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.CC)))
        print(self.dm.num_source_domains)
        self.optim_CC = build_optimizer(self.CC, cfg.OPTIM)
        self.sched_CC = build_lr_scheduler(self.optim_CC, cfg.OPTIM)
        self.register_model('CC', self.CC, self.optim_CC, self.sched_CC)


    # def forward_backward(self, batch,backprob = True):
    #     input_x, label_x, domain_x = self.parse_batch_train(batch)
    #
    #     # AGG_loss = 0.0
    #     epir_loss = 0.0
    #     epic_loss = 0.0
    #     epif_loss = 0.0
    #     DS_loss = 0.0
    #
    #     total_loss = 0.0
    #
    #     # train General model warm up
    #     feat = self.Feature(input_x)
    #     logits = self.Classifier(feat)
    #     AGG_loss = self.ce(logits, label_x)
    #     total_loss += AGG_loss
    #
    #     #train Domain Specific model
    #     input_x = torch.split(input_x, self.split_batch, 0)
    #     label_x = torch.split(label_x, self.split_batch, 0)
    #     domain_x = torch.split(domain_x, self.split_batch, 0)
    #     d_x = [d[0].item() for d in domain_x]
    #
    #     if self.epoch <= self.warm_up_DS:
    #         for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
    #             FF = self.FF[d](x)
    #             CC_logits = self.CC[d](FF)
    #             DS_loss += self.ce(CC_logits, y)
    #         DS_loss/=self.n_domain
    #
    #         if backprob:
    #             self.model_backward_and_update(DS_loss, ['FF', 'CC'])
    #             if (self.batch_idx + 1) == self.num_batches:
    #                 self.update_lr( ['FF', 'CC'])
    #
    #
    #
    #
    #
    #     if self.epoch >= self.warn_up_AGG:
    #
    #
    #
    #         candidates = list(self.candidates)
    #         index_val = np.random.choice(candidates, size=1)[0]
    #         candidates.remove(index_val)
    #         index_trn = np.random.choice(candidates, size=1)[0]
    #         assert index_trn != index_val
    #
    #         for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
    #             if d == index_val:
    #                 assert d != index_trn
    #                 #update the feature based on random domain specific classifier
    #                 if self.epoch >= self.start_train_feature:
    #                     c_feat = self.Feature(x)
    #                     output_logits = self.CC[index_trn](c_feat)
    #                     epif_loss += self.ce(output_logits, y)
    #
    #                 #update the classifier based on random domain specific feature
    #                 if self.epoch >= self.start_train_classifier:
    #                     s_feat = self.FF[index_trn](x)
    #                     output_logits = self.Classifier(s_feat)
    #                     epic_loss += self.ce(output_logits, y)
    #
    #             # update the feature based on randomclassifier
    #             feat = self.Feature(x)
    #             random_logits = self.CR(feat)
    #             epir_loss += self.ce(random_logits, y)
    #
    #         # epif_loss /= (self.n_domain-1)
    #         # epic_loss /= (self.n_domain-1)
    #         epir_loss /=self.n_domain
    #
    #         # for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
    #         #     if d == index_val:
    #         #         assert d != index_trn
    #         #
    #         #         #update the feature based on random domain specific classifier
    #         #         if self.epoch >= self.start_train_feature:
    #         #             feat = self.Feature(x)
    #         #             output_logits = self.CC[index_trn](feat)
    #         #             epif_loss = self.ce(output_logits, y)
    #         #
    #         #         #update the classifier based on random domain specific feature
    #         #         if self.epoch >= self.start_train_classifier:
    #         #             feat = self.FF[index_trn](x)
    #         #             output_logits = self.Classifier(feat)
    #         #             epic_loss = self.ce(output_logits, y)
    #         #
    #         #     # update the feature based on randomclassifier
    #         #     feat = self.Feature(x)
    #         #     random_logits = self.CR(feat)
    #         #     epir_loss += self.ce(random_logits, y)
    #         # epir_loss /=self.n_domain
    #
    #         # for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
    #         #     if d == index_val:
    #         #         assert d != index_trn
    #         #
    #         #         #update the feature based on random domain specific classifier
    #         #         if self.epoch >= self.start_train_feature:
    #         #             feat = self.Feature(x)
    #         #             output_logits = self.CC[index_trn](feat)
    #         #             epif_loss = self.ce(output_logits, y)
    #         #
    #         #         #update the classifier based on random domain specific feature
    #         #         if self.epoch >= self.start_train_classifier:
    #         #             feat = self.FF[index_trn](x)
    #         #             output_logits = self.Classifier(feat)
    #         #             epic_loss = self.ce(output_logits, y)
    #         #
    #         #     # update the feature based on randomclassifier
    #         #     feat = self.Feature(x)
    #         #     random_logits = self.CR(feat)
    #         #     epir_loss += self.ce(random_logits, y)
    #         # epir_loss /=self.n_domain
    #
    #         total_loss+= epif_loss*self.loss_weight_epif + epic_loss*self.loss_weight_epic + epir_loss*self.loss_weight_epir
    #
    #     if backprob:
    #         self.model_backward_and_update(total_loss, ['Feature', 'Classifier'])
    #         if (self.batch_idx + 1) == self.num_batches:
    #             self.update_lr( ['Feature', 'Classifier'])
    #     loss_summary = {
    #         'loss_general': AGG_loss.item(),
    #         'loss_domain_specific': DS_loss if isinstance(DS_loss,float) else DS_loss.item(),
    #         'loss_random': epir_loss if isinstance(epir_loss,float) else epir_loss.item(),
    #         'loss_episodic_feature': epic_loss if isinstance(epic_loss,float) else epic_loss.item(),
    #         'loss_episodic_classifier': epif_loss if isinstance(epif_loss,float) else epif_loss.item()
    #     }
    #     return loss_summary


    def forward_backward(self, batch,backprob = True):
        input_x, label_x, domain_x = self.parse_batch_train(batch)

        # AGG_loss = 0.0
        epir_loss = 0.0
        epic_loss = 0.0
        epif_loss = 0.0
        DS_loss = 0.0

        total_loss = 0.0

        # train General model warm up
        feat = self.Feature(input_x)
        logits = self.Classifier(feat)
        AGG_loss = self.ce(logits, label_x)
        total_loss += AGG_loss

        #train Domain Specific model
        input_x = torch.split(input_x, self.split_batch, 0)
        label_x = torch.split(label_x, self.split_batch, 0)
        domain_x = torch.split(domain_x, self.split_batch, 0)
        d_x = [d[0].item() for d in domain_x]

        if self.epoch <= self.warm_up_DS:
            for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
                FF = self.FF[d](x)
                CC_logits = self.CC[d](FF)
                DS_loss += self.cce[d](CC_logits, y)
            DS_loss/=self.n_domain

            if backprob:
                self.model_backward_and_update(DS_loss, ['FF', 'CC'])
                if (self.batch_idx + 1) == self.num_batches:
                    self.update_lr( ['FF', 'CC'])





        if self.epoch >= self.warn_up_AGG:



            candidates = list(self.candidates)
            index_val = np.random.choice(candidates, size=1)[0]
            candidates.remove(index_val)
            index_trn = np.random.choice(candidates, size=1)[0]
            assert index_trn != index_val

            for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
                if d != index_trn:
                    #update the feature based on random domain specific classifier
                    if self.epoch >= self.start_train_feature:
                        c_feat = self.Feature(x)
                        output_logits = self.CC[index_trn](c_feat)
                        epif_loss += self.ce(output_logits, y)

                    #update the classifier based on random domain specific feature
                    if self.epoch >= self.start_train_classifier:
                        s_feat = self.FF[index_trn](x)
                        output_logits = self.Classifier(s_feat)
                        epic_loss += self.ce(output_logits, y)

                # update the feature based on randomclassifier
                feat = self.Feature(x)
                random_logits = self.CR(feat)
                epir_loss += self.ce(random_logits, y)

            epif_loss /= (self.n_domain-1)
            epic_loss /= (self.n_domain-1)
            epir_loss /=self.n_domain


            total_loss+= epif_loss*self.loss_weight_epif + epic_loss*self.loss_weight_epic + epir_loss*self.loss_weight_epir

        if backprob:
            self.model_backward_and_update(total_loss, ['Feature', 'Classifier'])
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr( ['Feature', 'Classifier'])
        loss_summary = {
            'loss_general': AGG_loss.item(),
            'total_loss': total_loss if isinstance(total_loss,float) else total_loss.item(),
            'loss_domain_specific': DS_loss if isinstance(DS_loss,float) else DS_loss.item(),
            'loss_random': epir_loss if isinstance(epir_loss,float) else epir_loss.item(),
            'loss_episodic_feature': epic_loss if isinstance(epic_loss,float) else epic_loss.item(),
            'loss_episodic_classifier': epif_loss if isinstance(epif_loss,float) else epif_loss.item()
        }
        return loss_summary

    @torch.no_grad()
    def validate(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        losses = MetricMeter()

        print('Do evaluation on {} set'.format('valid set'))
        data_loader = self.val_loader
        assert data_loader is not None
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            loss = self.forward_backward(batch, backprob=False)
            losses.update(loss)
            # total_loss += loss['loss']
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss_general'].avg
        # total_loss = losses.meters['total_loss'].avg


        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        return [total_loss,losses.dict_results(),results]




    def model_inference(self, input):
        feat = self.Feature(input)
        logits = self.Classifier(feat)
        probs = F.softmax(logits, dim=1)
        return probs





