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
from dassl.utils import MetricMeter,AverageMeter
import time
import datetime


import numpy as np


@TRAINER_REGISTRY.register()
class MetaReg(TrainerX):
    """
    Feature-Critic Networks for Heterogeneous Domain Generalisation
    https://arxiv.org/pdf/1901.11448.pdf
    """
    def __init__(self, cfg):

        self.train_full_model_epoch = cfg.TRAINER.MetaReg.train_full_model_epoch
        self.regularize_ratio = cfg.TRAINER.MetaReg.regularize_ratio
        self.meta_train_step = cfg.TRAINER.MetaReg.meta_train_step
        self.inner_lr = cfg.TRAINER.MetaReg.inner_lr

        super().__init__(cfg)

        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        if n_domain <= 0:
            n_domain = self.dm.num_source_domains
        self.n_domain = n_domain
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.split_batch = batch_size // n_domain

        self.valid_ce =  nn.CrossEntropyLoss()

        #create a cross entropy loss for whole dataset
        self.ce = nn.CrossEntropyLoss()
        # self.ce_1 =  nn.CrossEntropyLoss()
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
        index = np.random.permutation(self.candidates)
        self.meta_train_idx = index[0]
        self.meta_test_idx = index[1]




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

        # for name, param in self.Feature.state_dict().items():
        #     print(name)
        # self.Feature.backbone.c1.weight.register_hook(lambda x: print('grad accumulated in Feature'))
        #
        self.DomainClassifier = nn.ModuleList(
            [
                nn.Linear(fdim, self.num_classes)
                for _ in range(self.dm.num_source_domains)
            ]
        )
        self.DomainClassifier.to(self.device)

        # self.Classifier.weight.register_hook(lambda x: print('grad accumulated in classifier'))

        print('# params: {:,}'.format(count_num_param(self.DomainClassifier)))
        self.optim_DomainClassifier = build_optimizer(self.DomainClassifier, cfg.OPTIM)
        self.sched_DomainClassifier = build_lr_scheduler(self.optim_DomainClassifier, cfg.OPTIM)
        self.register_model('DomainClassifier', self.DomainClassifier, self.optim_DomainClassifier, self.sched_DomainClassifier)


        self.MainClasifier = nn.Linear(fdim, self.num_classes)
        self.MainClasifier.to(self.device)
        self.optim_MainClasifier = build_optimizer(self.MainClasifier, cfg.OPTIM)
        self.sched_MainClasifier = build_lr_scheduler(self.optim_DomainClassifier, cfg.OPTIM)
        self.register_model('MainClasifier', self.MainClasifier, self.optim_MainClasifier,
                            self.sched_MainClasifier)

        self.omega = nn.Linear(fdim*self.num_classes, 1)
        self.omega.to(self.device)

        # self.omega.weight.register_hook(lambda x: print('grad accumulated in fc1 omega'))

        print('# params: {:,}'.format(count_num_param(self.omega)))
        self.optim_omega = build_optimizer(self.omega, cfg.OPTIM)
        self.sched_omega = build_lr_scheduler(self.optim_omega, cfg.OPTIM)
        self.register_model('omega', self.omega, self.optim_omega, self.sched_omega)

        self.temp_classifier = nn.Linear(fdim, self.num_classes)
        self.temp_classifier.to(self.device)

        # self.temp_optm = build_optimizer(self.temp_classifier, cfg.OPTIM)

        self.temp_optm=torch.optim.SGD(self.temp_classifier.parameters(), lr=self.inner_lr, momentum=0.9)



    def forward_backward(self, batch,backprob = True):
        input_x, label_x, domain_x = self.parse_batch_train(batch)
        general_loss = 0.0
        loss_reg = 0.0
        loss_x = 0.0

        if not backprob:
            feat = self.Feature(input_x)
            logits = self.MainClasifier(feat)
            loss_x = self.valid_ce(logits, label_x)

            loss_summary = {
                "loss_x": loss_x if isinstance(loss_x, float) else loss_x.item(),
            }
        else:

            if self.epoch >= self.train_full_model_epoch:
                #Train Full Model with regularizer on main classifier
                self.omega.eval()
                feat = self.Feature(input_x)
                logits = self.MainClasifier(feat)
                loss_x = self.ce(logits,label_x)
                loss_reg = self.omega(torch.abs(torch.flatten(self.MainClasifier.weight)))

                general_loss = loss_x+self.regularize_ratio*loss_reg
                # for param in self.omega.parameters():
                #     print("sum of params before udpate omega : ",param.sum())
                self.model_backward_and_update(general_loss, ['Feature', 'MainClasifier'])
                # for param in self.omega.parameters():
                #     print("sum of params before udpate omega : ",param.sum())
                if (self.batch_idx + 1) == self.num_batches:
                    self.update_lr(['Feature', 'MainClasifier'])

                loss_summary = {
                    "general_loss": general_loss.item(),
                    "loss_x": loss_x if isinstance(loss_x, float) else loss_x.item(),
                    "loss_reg": loss_reg if isinstance(loss_reg, float) else loss_reg.item()
                }

            else:
                # TRAIN STEP 1, regular training (line 2-7 in MetaReg algo)
                input_x = torch.split(input_x, self.split_batch, 0)
                label_x = torch.split(label_x, self.split_batch, 0)
                domain_x = torch.split(domain_x, self.split_batch, 0)
                d_x = [d[0].item() for d in domain_x]

                for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
                    feat = self.Feature(x)
                    logits = self.DomainClassifier[d](feat)
                    loss = self.cce[d](logits, y)
                    general_loss += loss

                general_loss /= self.n_domain


                self.model_backward_and_update(general_loss, ['Feature', 'DomainClassifier'])


                #TRAIN STEP 2: pick 2 domain a and b and perform meta-train (line 8-13)

                self.model_zero_grad(['omega'])
                index = np.random.permutation(self.candidates)
                self.meta_train_idx = index[0]
                self.meta_test_idx = index[1]
                self.reinit_temp_model()
                for l in range(self.meta_train_step):
                    x, y = input_x[self.meta_train_idx], label_x[self.meta_train_idx]
                    feat = self.Feature(x)
                    logits = self.temp_classifier(feat)
                    loss = self.cce[self.meta_train_idx](logits, y)
                    reg_loss = self.omega(torch.abs(torch.flatten(self.temp_classifier.weight)))
                    total_loss = loss + reg_loss
                    self.temp_optm.zero_grad()
                    # print("before backward loss+rel loss")
                    # for param in self.omega.parameters():
                    #     print("sum of grad before update omega : ", param.grad.data.sum())
                    #     print()

                    # print("current reg : ",reg_loss)
                    # self.model_backward(total_loss,create_graph=True)
                    # total_loss.backward()
                    self.model_backward(total_loss)

                    # for param in self.omega.parameters():
                    #     print("sum of weight gradient after update omega : ", param.grad.data.sum())
                    #     print()
                    self.temp_optm.step()

                #TRAIN STEP 3 : optimize the regularizer
                # average accumulate grad in omega
                for p in self.omega.parameters():
                    if p.grad is not None:
                        p.grad.div_(self.meta_train_step)

                x, y = input_x[self.meta_test_idx], label_x[self.meta_test_idx]

                feat = self.Feature(x)
                logits = self.temp_classifier(feat)
                meta_test_loss = self.cce[self.meta_test_idx](logits, y)

                # self.model_zero_grad(['omega'])
                # self.model_zero_grad(['omega'])
                # print("before update meta loss")
                # print("step 3 ")
                # for param in self.omega.parameters():
                #     print("sum of grad before update omega : ", param.grad.data.sum())
                #     print("sum of params before udpate omega : ",param.sum())
                self.model_backward(meta_test_loss)
                # for param in self.omega.parameters():
                #     print("sum of grad after update omega : ", param.grad.data.sum())
                #     print("sum of params after udpate omega : ", param.sum())
                self.model_update(['omega'])




                if (self.batch_idx + 1) == self.num_batches:
                    self.update_lr(['Feature', 'DomainClassifier','omega'])
                loss_summary = {
                    "general_loss": general_loss.item(),
                    "loss_x":loss if isinstance(loss,float) else loss.item(),
                    "loss_reg": reg_loss if isinstance(reg_loss,float) else reg_loss.item(),
                    "total_loss":total_loss.item()
                }

        return loss_summary


    # def forward_backward(self, batch,backprob = True):
    #     input_x, label_x, domain_x = self.parse_batch_train(batch)
    #     general_loss = 0.0
    #     loss_reg = 0.0
    #     if self.epoch >= self.train_full_model_epoch:
    #         #Train Full Model with regularizer on main classifier
    #         feat = self.Feature(input_x)
    #         logits = self.MainClasifier(feat)
    #         loss = self.ce(logits,label_x)
    #         loss_reg = self.omega(torch.abs(torch.flatten(self.MainClasifier.weight)))
    #
    #         general_loss = loss+loss_reg
    #
    #         if backprob:
    #             self.model_backward_and_update(general_loss, ['Feature', 'MainClasifier'])
    #             if (self.batch_idx + 1) == self.num_batches:
    #                 self.update_lr(['Feature', 'MainClasifier'])
    #
    #     else:
    #         # TRAIN STEP 1, regular training (line 2-7 in MetaReg algo)
    #         input_x = torch.split(input_x, self.split_batch, 0)
    #         label_x = torch.split(label_x, self.split_batch, 0)
    #         domain_x = torch.split(domain_x, self.split_batch, 0)
    #         d_x = [d[0].item() for d in domain_x]
    #
    #         for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
    #             feat = self.Feature(x)
    #             logits = self.DomainClassifier[d](feat)
    #             loss = self.cce[d](logits, y)
    #             general_loss += loss
    #
    #         general_loss /= self.n_domain
    #
    #         if backprob:
    #             self.model_backward_and_update(general_loss, ['Feature', 'DomainClassifier'])
    #             if (self.batch_idx + 1) == self.num_batches:
    #                 self.update_lr(['Feature', 'DomainClassifier'])
    #
    #
    #
    #
    #
    #     loss_summary = {
    #         "loss_main": general_loss.item(),
    #         "loss_reg": loss_reg if isinstance(loss_reg,float) else loss_reg.item()
    #     }
    #
    #     return loss_summary

    def reinit_temp_model(self):
        current_state_dict = self.DomainClassifier[self.meta_train_idx].state_dict()
        self.temp_classifier.load_state_dict(current_state_dict)
        self.temp_optm=torch.optim.SGD(self.temp_classifier.parameters(), lr=self.inner_lr, momentum=0.9)

    # def meta_training_forward_backward(self,batch):
    #
    #     input_x, label_x, domain_x = self.parse_batch_train(batch)
    #     # train Domain Specific model
    #     input_x = torch.split(input_x, self.split_batch, 0)
    #     label_x = torch.split(label_x, self.split_batch, 0)
    #     domain_x = torch.split(domain_x, self.split_batch, 0)
    #     d_x = [d[0].item() for d in domain_x]
    #     x, y = input_x[self.meta_train_idx], label_x[self.meta_train_idx]
    #     feat = self.Feature(x)
    #     logits = self.temp_classifier(feat)
    #     loss = self.cce[self.meta_train_idx](logits, y)
    #
    #     reg_loss = self.omega(torch.abs(torch.flatten(self.temp_classifier.weight)))
    #
    #     total_loss = loss+reg_loss
    #
    #     self.temp_optm.zero_grad()
    #     print("before backward loss+rel loss")
    #     # for param in self.omega.parameters():
    #     #     print("sum of weight before update omega : ", param.grad.data.sum())
    #     #     print()
    #     # self.model_backward(total_loss,create_graph=True)
    #     total_loss.backward(retain_graph=True)
    #
    #     for param in self.omega.parameters():
    #         print("sum of weight after update omega : ", param.grad.data.sum())
    #         print()
    #     self.temp_optm.step()
    #
    # def meta_testing_forward_backward(self,batch):
    #     input_x, label_x, domain_x = self.parse_batch_train(batch)
    #
    #     input_x = torch.split(input_x, self.split_batch, 0)
    #     label_x = torch.split(label_x, self.split_batch, 0)
    #     domain_x = torch.split(domain_x, self.split_batch, 0)
    #     d_x = [d[0].item() for d in domain_x]
    #
    #     x, y = input_x[self.meta_test_idx], label_x[self.meta_test_idx]
    #
    #     feat = self.Feature(x)
    #     logits = self.temp_classifier(feat)
    #     meta_test_loss = self.cce[self.meta_train_idx](logits, y)
    #
    #     # self.model_zero_grad(['omega'])
    #     for param in self.omega.parameters():
    #         print("sum of weight before update omega : ",param.grad.data.sum())
    #         print()
    #     self.model_backward(meta_test_loss)
    #     self.model_update(['omega'])
    #     for param in self.omega.parameters():
    #         print("sum of weight after update omega : ",param.grad.data.sum())





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

        total_loss = losses.meters['loss_x'].avg
        # total_loss = losses.meters['loss_main'].avg


        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        return [total_loss,losses.dict_results(),results]


    def run_epoch(self):
        """
        Customize the process of how to train model in an epoch
        """
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()
            self.epoch_losses = losses

        # # deal with meta training/testing phase
        # self.reinit_temp_model()
        # for self.batch_idx, batch in enumerate(self.train_loader_x):
        #     self.meta_training_forward_backward(batch)
        #
        # for self.batch_idx, batch in enumerate(self.train_loader_x):
        #     self.meta_testing_forward_backward(batch)
        #
        # index = np.random.permutation(self.candidates)
        # self.meta_train_idx = index[0]
        # self.meta_test_idx = index[1]


    def model_inference(self, input):
        feat = self.Feature(input)

        # if self.epoch < self.train_full_model_epoch:
        #
        # else:

        logits = self.MainClasifier(feat)
        probs = F.softmax(logits, dim=1)
        return probs





