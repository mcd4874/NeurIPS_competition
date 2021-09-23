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
import learn2learn as l2l
import numpy as np
@TRAINER_REGISTRY.register()
class MLDG(TrainerX):
    """
    Learning to Generalize: Meta-Learning for Domain Generalization
    https://arxiv.org/pdf/1710.03463.pdf
    """
    def __init__(self, cfg):
        self.alpha = cfg.TRAINER.MLDG.alpha
        self.inner_lr = cfg.TRAINER.MLDG.inner_lr
        self.num_test_subject = cfg.TRAINER.MLDG.num_test_subject
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
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                print("torch weight  : ",torch_weight)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)

        # create a cross entropy loss for each domain dataset
        self.cce = [nn.CrossEntropyLoss() for _ in range(self.n_domain)]
        if cfg.DATASET.DOMAIN_CLASS_WEIGHT:
            domain_class_weight = self.dm.dataset.domain_class_weight
            for domain, weight in domain_class_weight.items():
                # domain_class_weight[domain] = torch.from_numpy(np.array(weight)).float().to(self.device)
                torch_weight = torch.from_numpy(np.array(weight)).float().to(self.device)
                self.cce[domain] = nn.CrossEntropyLoss(weight=torch_weight)


        self.val_ce = nn.CrossEntropyLoss()


        self.candidates = np.arange(self.n_domain)
        index = np.random.permutation(self.candidates)
        self.meta_test_idx = index[:self.num_test_subject]
        self.meta_train_idx = index[self.num_test_subject:]

        # self.meta_test_idx = index[0]
        # self.alpha = 0.1 #within_subject

        # self.alpha = 1.0 #within_subject_1






    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == 'RandomDomainSampler'
    def build_model(self):
        cfg = self.cfg

        print('Building Feature')
        # params_model = cfg.MODEL.BACKBONE.PARAMS
        # print("params model : ",params_model)
        # params_model['num_ch'] = 34
        # print("new params set up ", params_model)
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes, **cfg.MODEL.BACKBONE.PARAMS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))



        print('Building MAML')
        # fast_lr = 0.001
        self.maml = l2l.algorithms.MAML(self.model, lr=self.inner_lr)
        self.maml.to(self.device)


        self.optim_maml = build_optimizer(self.maml, cfg.OPTIM)
        self.sched_maml = build_lr_scheduler(self.optim_maml, cfg.OPTIM)
        self.register_model('maml', self.maml, self.optim_maml, self.sched_maml)

        self.learner1 = None
        self.learner2 = None

        # self.register_model('model', self.model)

    def check_equal_model(self,model1,model2):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                return False
        return True

    def check_equal_grad(self,model1,model2):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.grad.data.ne(p2.grad.data).sum() > 0:
                return False
        return True

    def check_model_params(self,modelList):
        for model_info in modelList:
            # name = model_info['name']
            # model = model_info['model']
            name = model_info[0]
            model = model_info[1]
            print(name)
            for p in model.parameters():
                print("sum params : ",p.data.sum())


    def forward_backward(self, batch,backprob = True):
        if self.batch_idx == 0:
            index = np.random.permutation(self.candidates)
            self.meta_test_idx = index[:self.num_test_subject]
            self.meta_train_idx = index[self.num_test_subject:]
            print("update meta test to be subject : {}".format(self.meta_test_idx))


        input_x, label_x, domain_x = self.parse_batch_train(batch)
        learner = self.maml.clone()

        #check model and maml params



        meta_train_loss = 0.0
        meta_test_loss = 0.0

        #clone maml

        if backprob:
            # train Domain Specific model
            input_x = torch.split(input_x, self.split_batch, 0)
            label_x = torch.split(label_x, self.split_batch, 0)
            domain_x = torch.split(domain_x, self.split_batch, 0)
            d_x = [d[0].item() for d in domain_x]

            for domain in self.meta_train_idx:
                x,y = input_x[domain],label_x[domain]
                logits = learner(x)
                if self.cfg.DATASET.DOMAIN_CLASS_WEIGHT:
                    loss = self.cce[domain](logits, y)
                else:
                    loss = self.ce(logits, y)
                meta_train_loss += loss

            # for x, y, dy, d in zip(input_x, label_x, domain_x, d_x):
            #     if d != self.meta_test_idx:
            #         logits =learner(x)
            #         if self.cfg.DATASET.DOMAIN_CLASS_WEIGHT:
            #             loss = self.cce[d](logits, y)
            #         else:
            #             loss = self.ce(logits, y)
            #         meta_train_loss+= loss

            self.model_zero_grad(['maml'])

            meta_train_loss /= len(self.meta_train_idx)
            # print("equal model and clone maml 1 ", self.check_equal_model(self.model, learner))
            # print("equal model and maml 1 ",self.check_equal_model(self.model,self.maml))
            # print("equal maml and clone maml 1 ", self.check_equal_model(self.maml, learner))

            # print("equal grad model and clone maml 1 ", self.check_equal_grad(self.model, learner))
            # print("equal grad model and maml 1 ", self.check_equal_grad(self.model, self.maml))
            # print("equal grad maml and clone maml 1 ", self.check_equal_grad(self.model, self.maml))

            # self.check_model_params([["model",self.model],["maml",self.maml],["clone",learner]])
            self.learner1 = learner
            learner.adapt(meta_train_loss)
            self.learner2 = learner
            # print("equal model and clone maml 2 ", self.check_equal_model(self.model, learner))
            # print("equal model and maml 2 ", self.check_equal_model(self.model, self.maml))
            # print("equal maml and clone maml 2 ", self.check_equal_model(self.maml, learner))

            # print("equal grad model and clone maml 2 ", self.check_equal_grad(self.model, learner))
            # print("equal grad model and maml 2 ", self.check_equal_grad(self.model, self.maml))
            # print("equal grad maml and clone maml 2 ", self.check_equal_grad(self.maml, learner))

            # self.check_model_params([["model", self.model], ["maml", self.maml], ["clone", learner]])
            #meta test stage
            for domain in self.meta_test_idx:
                x, y = input_x[domain], label_x[domain]
                logits = learner(x)
                if self.cfg.DATASET.DOMAIN_CLASS_WEIGHT:
                    loss = self.cce[domain](logits, y)
                else:
                    loss = self.ce(logits, y)
                meta_test_loss +=loss
            meta_test_loss /= len(self.meta_test_idx)
            final_loss = meta_train_loss + self.alpha*meta_test_loss
            # print("final loss : ",final_loss)
            self.model_backward(final_loss)
            self.model_update(['maml'])

            # print("equal model and clone maml 3 ", self.check_equal_model(self.model, learner))
            # print("equal model and maml 3 ", self.check_equal_model(self.model, self.maml))
            # print("equal maml and clone maml 3 ", self.check_equal_model(self.maml, learner))
            #
            # print("equal grad model and clone maml 3 ", self.check_equal_grad(self.model, learner))
            # print("equal grad model and maml 3 ", self.check_equal_grad(self.model, self.maml))
            # print("equal grad maml and clone maml 3 ", self.check_equal_grad(self.model, self.maml))


            # self.check_model_params([["model", self.model], ["maml", self.maml], ["clone", learner]])
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr( ['maml'])

                # index = np.random.permutation(self.candidates)
                # self.meta_test_idx = index[:self.num_test_subject]
                # self.meta_train_idx = index[self.num_test_subject:]
                # print("update meta test to be subject : {}".format(self.meta_test_idx))

            loss_summary = {
                'meta_train_loss': meta_train_loss.item(),
                'meta_test_loss': meta_test_loss.item(),
                'total_loss': final_loss.item()
            }
        else:
            logits = learner(input_x)
            val_loss = self.val_ce(logits,label_x)

            input_x = torch.split(input_x, self.split_batch, 0)
            label_x = torch.split(label_x, self.split_batch, 0)

            for domain in self.meta_train_idx:
                x,y = input_x[domain],label_x[domain]
                logits = self.learner1(x)
                loss = self.val_ce(logits,y)
                meta_train_loss+=loss

            meta_train_loss /= len(self.meta_train_idx)
            for domain in self.meta_test_idx:
                x,y = input_x[domain],label_x[domain]
                logits = self.learner2(x)
                loss = self.val_ce(logits,y)
                meta_test_loss+=loss
            meta_test_loss /= len(self.meta_test_idx)

            total_loss = meta_train_loss + self.alpha*meta_test_loss


            loss_summary = {
                'meta_train_loss': meta_train_loss.item(),
                'meta_test_loss': meta_test_loss.item(),
                'total_loss': total_loss.item(),
                "general_loss":val_loss.item()
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
        # total_loss = losses.meters['total_loss'].avg
        total_loss = losses.meters['general_loss'].avg


        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        return [total_loss,losses.dict_results(),results]




    def model_inference(self, input):
        logits = self.maml(input)
        probs = F.softmax(logits, dim=1)
        return probs





