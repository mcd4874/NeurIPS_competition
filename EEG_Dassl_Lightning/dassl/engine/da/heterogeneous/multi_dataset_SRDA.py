from dassl.engine import TRAINER_REGISTRY
from dassl.engine.trainer import TrainerMultiAdaptation
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
from dassl.modeling.ops import ReverseGrad
from typing import Any, Dict, List, Optional, Union


import torchmetrics
def sample_unit_vec(shape, n):
    mean = torch.zeros(shape)
    std = torch.ones(shape)
    dis = torch.distributions.Normal(mean, std)
    samples = dis.sample_n(n)
    samples = samples.view(n, -1)
    samples_norm = torch.norm(samples, 2, 1).view(n, 1)
    samples = samples/samples_norm
    return samples.view(n,*shape)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

# class

# class VATLoss(nn.Module):
#
#     def __init__(self, xi=10.0, eps=1.0, ip=1):
#         """VAT loss
#         :param xi: hyperparameter of VAT (default: 10.0)
#         :param eps: hyperparameter of VAT (default: 1.0)
#         :param ip: iteration times of computing adv noise (default: 1)
#         """
#         super(VATLoss, self).__init__()
#         self.xi = xi
#         self.eps = eps
#         self.ip = ip
#
#     def loss_fn(self,pred_hat,pred):
#         logp_hat = F.log_softmax(pred_hat, dim=1)
#         adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
#         return adv_distance
#
#     def forward(self, model, x):
#         with torch.no_grad():
#             pred = F.softmax(model(x), dim=1)
#
#         d = sample_unit_vec(x.shape[1:], x.shape[0]).to(x.device)
#
#         # prepare random unit tensor
#         # d = torch.rand(x.shape).sub(0.5).to(x.device)
#         # d = _l2_normalize(d)
#
#
#         # with _disable_tracking_bn_stats(model):
#         # calc adversarial direction
#         for _ in range(self.ip):
#             d.requires_grad_()
#             pred_hat = model(x + self.xi * d)
#             adv_distance = self.loss_fn(pred_hat,pred)
#             adv_distance.backward()
#             d = _l2_normalize(d.grad)
#             model.zero_grad()
#
#         # calc LDS
#         r_adv = d * self.eps
#         pred_hat = model(x + r_adv)
#         logp_hat = F.log_softmax(pred_hat, dim=1)
#         lds = F.kl_div(logp_hat, pred, reduction='batchmean')
#
#         return lds

class VATAttack(object):
    def __init__(self, epsilon=1.0, zeta=1e-6, num_k=1):
        """
        Fast approximation method in virtual adversarial training
        :param model: nn.Module
        :param epsilon: float
        :param zeta: float
        :param num_k: int, number of iterations
        """
        # self.model = model
        self.epsilon = epsilon
        self.zeta = zeta
        self.num_k = num_k

    def loss_fn(self, out1, out2):
        # if out2.data.type()=='torch.cuda.LongTensor' or out2.data.type()=='torch.LongTensor':
        return nn.KLDivLoss()(F.softmax(out1, dim=1), out2)
        # else:
        #     return torch.mean(F.softmax(out2, dim=1) *
        #                       (torch.log(F.softmax(out2, dim=1) + 1e-6) - torch.log(F.softmax(out1, dim=1) + 1e-6)))
    def perturb(self,G,C,data, epsilons=None, zetas=None):
        """
        Given examples (X_nat), returns their adversarial
        counterparts with an attack length of epsilon.
        """

        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons
        if zetas is not None:
            self.zeta = zetas

        feature = G(data)

        # X = np.copy(X_nat)
        d = sample_unit_vec(feature.shape[1:], feature.shape[0]).to(feature.device)

        # X_var = to_var(torch.from_numpy(X), requires_grad=True)
        # d_var = to_var(d, requires_grad=True)
        pred = C(feature)
        for i in range(self.num_k):
            d.requires_grad_()
            r_var = self.zeta * d
            pert = C(feature + r_var)
            loss = self.loss_fn(pert, pred)
            loss.backward()
            d = _l2_normalize(d.grad)
            G.zero_grad()
            C.zero_grad()
        # print("current d : ",d)

        r_adv = d * self.epsilon
        return r_adv
            # d_var = to_var(d_var.grad.data, requires_grad=True)

        # X += self.epsilon * (self.zeta * d_var).data.cpu().sign().numpy()
        # X = np.clip(X, 0, 1)
        # return X

    # def perturb(self, feature, epsilons=None, zetas=None):
    #     """
    #     Given examples (X_nat), returns their adversarial
    #     counterparts with an attack length of epsilon.
    #     """
    #     # Providing epsilons in batch
    #     if epsilons is not None:
    #         self.epsilon = epsilons
    #     if zetas is not None:
    #         self.zeta = zetas
    #
    #     X = np.copy(X_nat)
    #     d = sample_unit_vec(X.shape[1:], X.shape[0])
    #
    #     X_var = to_var(torch.from_numpy(X), requires_grad=True)
    #     d_var = to_var(d, requires_grad=True)
    #
    #     for i in range(self.num_k):
    #         r_var = self.zeta * d_var
    #         ref = self.model(X_var)
    #         pert = self.model(X_var + r_var)
    #         loss = self.loss_fn(pert, ref)
    #         loss.backward()
    #         d_var = to_var(d_var.grad.data, requires_grad=True)
    #
    #     X += self.epsilon * (self.zeta * d_var).data.cpu().sign().numpy()
    #     X = np.clip(X, 0, 1)
    #     return X

@TRAINER_REGISTRY.register()
class MultiDatasetSRDA(TrainerMultiAdaptation):
    """
    https://arxiv.org/abs/1712.02560
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
        self.n_step_F = 5
        self.automatic_optimization = False

    def build_metrics(self):
        super(MultiDatasetSRDA, self).build_metrics()
        self.VATAttack = VATAttack()

    def forward(self, input, return_feature=False):
        f_target = self.CommonFeature(input)
        logit = self.TargetClassifier(f_target)
        probs = F.softmax(logit, dim=1)
        if return_feature:
            return probs, logit
        return probs
    def configure_optimizers(self):
        opt_cfg = self.cfg.OPTIM


        F_params = list(self.CommonFeature.parameters())
        F_opt = build_optimizer(F_params,opt_cfg)
        F_scheduler = build_lr_scheduler(optimizer=F_opt,optim_cfg=opt_cfg)


        C_T_params = list(self.TargetClassifier.parameters())
        C_T_opt = build_optimizer(C_T_params,opt_cfg)
        C_T_scheduler = build_lr_scheduler(optimizer=C_T_opt,optim_cfg=opt_cfg)

        C_S_params = list(self.SourceClassifiers.parameters())
        C_S_opt = build_optimizer(C_S_params,opt_cfg)
        C_S_scheduler = build_lr_scheduler(optimizer=C_S_opt,optim_cfg=opt_cfg)


        # opt = build_optimizer(params,opt_cfg)
        # scheduler = build_lr_scheduler(optimizer=opt,optim_cfg=opt_cfg)
        optimizers = [F_opt,C_T_opt,C_S_opt]
        return optimizers
        # lr_schedulers=[F_scheduler,C1_scheduler,C2_scheduler,C_S_scheduler]
        # return optimizers, lr_schedulers

    def build_model(self):
        cfg = self.cfg
        print("Params : ", cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE)
        print('Building F')

        print('Building CommonFeature')
        backbone_info = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE
        FC_info = cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC
        # backbone_params = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS.copy()

        self.CommonFeature = SimpleNet(backbone_info, FC_info, 0, **cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS)
        self.fdim = self.CommonFeature.fdim

        print('Building Target Classifier')
        self.TargetClassifier = self.create_classifier(self.fdim, self.num_classes, FC_info=FC_info)

        print('Building SourceClassifiers')
        print("source domains label size : ", self.source_domains_label_size)
        source_classifier_list = []
        for num_class in self.source_domains_label_size:
            source_classifier = self.create_classifier(self.fdim, num_class, FC_info=FC_info)
            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )


    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)
        f_target = self.CommonFeature(input)
        logit = self.TargetClassifier(f_target)
        loss_target = self.loss_function(logit, label, train=train_mode,weight=weight)

        return loss_target,logit,label

    def on_train_epoch_start(self) -> None:
        if self.source_pretrain_epochs > self.current_epoch:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_SOURCE_LOSS_RATIO
        else:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO

    # def discrepancy(self,out1,out2):
    #     return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2,dim=1)))

    def ce_discrepancy(self,out1, out2):
        return - torch.mean(F.softmax(out2, dim=1) * torch.log(F.softmax(out1, dim=1) + 1e-6))

    def training_step(self, batch, batch_idx):
        target_batch, unlabel_batch ,list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)

        F_opt,C_T_opt,C_S_opt = self.optimizers()

        loss_source = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.CommonFeature(u)
            logits = self.SourceClassifiers[d](f)
            domain_weight = self.source_domains_class_weight[d]
            loss_source += self.loss_function(logits, y, train=True, weight=domain_weight)
        loss_source /= len(domain_u)

        loss_target,target_logit,label= self.share_step(target_batch, train_mode=True, weight=self.class_weight)
        y_pred = F.softmax(target_logit, dim=1)
        y = label
        acc = self.train_acc(y_pred, y)

        #step A
        loss_A = self.source_ratio*loss_source+self.target_ratio*loss_target
        F_opt.zero_grad()
        C_T_opt.zero_grad()
        C_S_opt.zero_grad()
        self.manual_backward(loss_A)
        F_opt.step()
        C_T_opt.step()
        C_S_opt.step()

        F_opt.zero_grad()
        C_T_opt.zero_grad()
        C_S_opt.zero_grad()



        #step B

        # f_ul = self.CommonFeature(unlabel_batch)
        # output_ul = self.TargetClassifier(f_ul)

        r_adv = self.VATAttack.perturb(self.CommonFeature,self.TargetClassifier,data=unlabel_batch)

        f_ul = self.CommonFeature(unlabel_batch)
        output_ul = self.TargetClassifier(f_ul)

        f_ul_adv = f_ul+r_adv
        output_ul_adv = self.TargetClassifier(f_ul_adv)
        loss_attack = self.ce_discrepancy(output_ul_adv, output_ul)
        F_opt.zero_grad()
        C_T_opt.zero_grad()
        self.manual_backward(loss_attack)
        F_opt.step()


        self.log('Train_acc', acc, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss_A', loss_A,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False,on_epoch=True,prog_bar=True, logger=True)
        self.log('Train_loss_adv', loss_attack,on_step=False,on_epoch=True, prog_bar=True, logger=True)

    def parse_batch_train(self, batch):
        target_batch = batch["target_loader"]
        unlabel_batch = batch["unlabel_loader"]
        list_source_batches = batch["source_loader"]
        return target_batch,unlabel_batch,list_source_batches


    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss,y_logit,y = self.share_step(batch,train_mode=False)
        y_pred = F.softmax(y_logit, dim=1)


        if dataset_idx == 0 :
            acc = self.valid_acc(y_pred, y)
            log = {
                "val_loss": loss*self.non_save_ratio,
                "val_acc": acc,
            }
            self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True,add_dataloader_idx=False)
        else:
            acc = self.test_acc(y_pred, y)
            log = {
                "test_loss": loss,
                "test_acc": acc
            }
            self.log_dict(log, on_step=False, on_epoch=True, prog_bar=False, logger=True,add_dataloader_idx=False)

        return {'loss': loss}

    def test_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss,y_logit,y = self.share_step(batch,train_mode=False)
        y_pred = F.softmax(y_logit,dim=1)
        return {'loss': loss,'y_pred':y_pred,'y':y}