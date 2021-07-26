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

class Critic_Network_Flatten_FTF(nn.Module):
    def __init__(self, fdim):
        super(Critic_Network_Flatten_FTF, self).__init__()
        self.fc1 = nn.Linear(fdim, fdim//2)
        self.fc2 = nn.Linear(fdim//2, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = nn.functional.softplus(self.fc2(x))
        return torch.mean(x)

@TRAINER_REGISTRY.register()
class FCDG(TrainerX):
    """
    Feature-Critic Networks for Heterogeneous Domain Generalisation
    https://arxiv.org/pdf/1901.11448.pdf
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
        self.meta_train_idx = index[:4]
        self.meta_valid_idx = index[4:]
        self.beta = 10
        self.alpha = 0.001
        self.heldout_p = 100

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
        self.Classifier = nn.Linear(fdim, self.num_classes)
        self.Classifier.to(self.device)

        # self.Classifier.weight.register_hook(lambda x: print('grad accumulated in classifier'))

        print('# params: {:,}'.format(count_num_param(self.Classifier)))
        self.optim_Classifier = build_optimizer(self.Classifier, cfg.OPTIM)
        self.sched_Classifier = build_lr_scheduler(self.optim_Classifier, cfg.OPTIM)
        self.register_model('Classifier', self.Classifier, self.optim_Classifier, self.sched_Classifier)


        # self.omega =  nn.Linear(fdim, 1)
        # self.omega.to(self.device)

        self.omega = Critic_Network_Flatten_FTF(fdim)
        self.omega.to(self.device)

        # self.omega.fc1.weight.register_hook(lambda x: print('grad accumulated in fc1 omega'))

        print('# params: {:,}'.format(count_num_param(self.omega)))
        self.optim_omega = build_optimizer(self.omega, cfg.OPTIM)
        self.sched_omega = build_lr_scheduler(self.optim_omega, cfg.OPTIM)
        self.register_model('omega', self.omega, self.optim_omega, self.sched_omega)

        self.temp_old_feature = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
        self.temp_old_feature.to(self.device)

        self.temp_new_feature = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
        self.temp_new_feature.to(self.device)





    def forward_backward(self, batch,backprob = True):
        meta_train_loss_main = 0.0
        meta_train_loss_dg = 0.0
        meta_loss_held_out = 0.0




        input_x, label_x, domain_x = self.parse_batch_train(batch)

        # train Domain Specific model
        input_x = torch.split(input_x, self.split_batch, 0)
        label_x = torch.split(label_x, self.split_batch, 0)
        domain_x = torch.split(domain_x, self.split_batch, 0)
        d_x = [d[0].item() for d in domain_x]

        for i in self.meta_train_idx:
            x,y = input_x[i],label_x[i]
            feat = self.Feature(x)
            logits = self.Classifier(feat)
            loss_main = self.ce(logits,y)
            meta_train_loss_main +=loss_main
            loss_dg = self.beta * self.omega(feat)
            meta_train_loss_dg +=loss_dg

        meta_train_loss_main /= len(self.meta_train_idx)
        meta_train_loss_dg /= len(self.meta_train_idx)

        if backprob:
            self.model_zero_grad(['Feature','Classifier'])
            # print('before backward main loss ')
            self.model_backward(meta_train_loss_main,retain_graph=True)

            grad_theta = [theta_i.grad for theta_i in self.Feature.parameters()]
            theta_updated_old = {}
            num_grad = 0
            for i, (k, v) in enumerate(self.Feature.state_dict().items()):
                if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    theta_updated_old[k] = v
                    continue
                elif grad_theta[num_grad] is None:
                    num_grad += 1
                    theta_updated_old[k] = v
                else:
                    theta_updated_old[k] = v - self.alpha * grad_theta[num_grad]
                    num_grad += 1

            # print("before backward dg ")
            self.model_backward(meta_train_loss_dg,create_graph=True)


            grad_theta = [theta_i.grad for theta_i in self.Feature.parameters()]
            theta_updated_new = {}
            num_grad = 0
            for i, (k, v) in enumerate(self.Feature.state_dict().items()):
                if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    theta_updated_new[k] = v
                    continue
                elif grad_theta[num_grad] is None:
                    num_grad += 1
                    theta_updated_new[k] = v
                else:
                    theta_updated_new[k] = v - self.alpha * grad_theta[num_grad]
                    num_grad += 1
            self.temp_old_feature.load_state_dict(theta_updated_old)
            self.temp_new_feature.load_state_dict(theta_updated_new)

            # #copy model and grad into temp old
            # for target_param, param in zip(self.temp_old_feature.parameters(),self.Feature.parameters()):
            #     target_param.data.copy_(param.data - self.alpha*param.grad)
            #
            # # print('feature params : ',self.Feature.parameters())
            # # print('feature state dicts : ', self.Feature.state_dict())
            #
            # print("before backward dg ")
            # self.model_backward(meta_train_loss_dg,create_graph=True)
            #
            # # copy model and grad into new old
            # for target_param, param in zip(self.temp_new_feature.parameters(), self.Feature.parameters()):
            #     target_param.data.copy_(param.data - self.alpha * param.grad)

            #deal with meta-test step
            for i in self.meta_valid_idx:
                x, y = input_x[i], label_x[i]
                feat_old = self.temp_old_feature(x)
                feat_new = self.temp_new_feature(x)
                logits_old = self.Classifier(feat_old)
                logits_new = self.Classifier(feat_new)
                loss_old= self.ce(logits_old, y)
                loss_new = self.ce(logits_new, y)
                reward = loss_old - loss_new
                # calculate the updating rule of omega, here is the max function of h.
                utility = torch.tanh(reward)
                # so, here is the min value transfering to the backpropogation.
                loss_held_out = - utility.sum()
                meta_loss_held_out += loss_held_out * self.heldout_p

            meta_loss_held_out /= len(self.meta_valid_idx)

            self.model_update(['Feature','Classifier'])
            self.model_zero_grad(['omega'])
            # print("before backward meta loss")
            self.model_backward(meta_loss_held_out)
            # print("after backward meta loss ")
            self.model_update(['omega'])

            torch.cuda.empty_cache()

            if (self.batch_idx + 1) == self.num_batches:
                index = np.random.permutation(self.candidates)
                self.meta_train_idx = index[:4]
                self.meta_valid_idx = index[4:]
                self.update_lr(['Feature', 'Classifier','omega'])

        loss_summary = {
            "loss_main": meta_train_loss_main.item(),
            "loss_dg": meta_train_loss_dg.item(),
            "loss_hold_out": meta_loss_held_out if isinstance(meta_loss_held_out,float) else meta_loss_held_out.item()
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
        total_loss = losses.meters['loss_main'].avg

        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        return [total_loss,losses.dict_results(),results]




    def model_inference(self, input):
        feat = self.Feature(input)
        logits = self.Classifier(feat)
        probs = F.softmax(logits, dim=1)
        return probs





