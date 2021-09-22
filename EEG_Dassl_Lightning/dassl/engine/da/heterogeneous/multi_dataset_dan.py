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


import torchmetrics

from dassl.utils.kernel import GaussianKernel
from typing import Optional, Sequence

class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks (ICML 2015) <https://arxiv.org/pdf/1502.02791>`_
    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as
    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},
    :math:`k` is a kernel function in the function space
    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}
    where :math:`k_{u}` is a single kernel.
    Using kernel trick, MK-MMD can be computed as
    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s})\\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t})\\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}).\\
    Args:
        kernels (tuple(torch.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False
    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`
    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar
    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.
    .. note::
        The kernel values will add up when there are multiple kernels.
    Examples::
        # >>> from dalib.modules.kernels import GaussianKernel
        # >>> feature_dim = 1024
        # >>> batch_size = 10
        # >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        # >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        # >>> # features from source domain and target domain
        # >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        # >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        # print("index matrix : ",self.index_matrix)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        # print("kernel matrix : ",kernel_matrix)
        l = (kernel_matrix * self.index_matrix).sum()
        # print("l : ",l)
        loss = (l + 2. / float(batch_size - 1))

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


@TRAINER_REGISTRY.register()
class MultiDatasetDan(TrainerMultiAdaptation):
    """

    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
        self.bce = nn.BCEWithLogitsLoss()
        self.max_epoch = self.cfg.OPTIM.MAX_EPOCH

        self.trade_off = cfg.LIGHTNING_MODEL.TRAINER.DAN.trade_off
        print("trade off ratio : ", self.trade_off)
        alpha = cfg.LIGHTNING_MODEL.TRAINER.DAN.GaussianKernel.alpha
        sigma = cfg.LIGHTNING_MODEL.TRAINER.DAN.GaussianKernel.sigma
        track_running_stats = cfg.LIGHTNING_MODEL.TRAINER.DAN.GaussianKernel.track_running_stats
        linear = cfg.LIGHTNING_MODEL.TRAINER.DAN.linear

        if len(sigma) == 0:
            # sigma = None
            # define loss function
            print("alpha range : ", alpha)
            self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
                kernels=[GaussianKernel(alpha=k, track_running_stats=track_running_stats) for k in alpha],
                linear=linear
            )
        else:
            print("sigma range : ", sigma)
            self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
                kernels=[GaussianKernel(sigma=s, track_running_stats=track_running_stats) for s in sigma],
                linear=linear
            )




    def build_model(self):
        cfg = self.cfg
        print("Params : ", cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE)
        print('Building F')

        print('Building CommonFeature')
        backbone_info = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE
        FC_info = cfg.LIGHTNING_MODEL.COMPONENTS.LAST_FC

        self.CommonFeature = SimpleNet(backbone_info, FC_info, 0, **cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.PARAMS)
        freeze_common_feature = cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.FREEZE if cfg.LIGHTNING_MODEL.COMPONENTS.BACKBONE.FREEZE else False
        if freeze_common_feature:
            for parameter in self.CommonFeature.parameters():
                parameter.requires_grad = False
            print("freeze feature extractor : ",)

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

    def forward(self, input, return_feature=False):
        f_target = self.CommonFeature(input)
        logits_target = self.TargetClassifier(f_target)
        probs = F.softmax(logits_target, dim=1)

        if return_feature:
            return probs, logits_target
        return probs

    def configure_optimizers(self):
        params = list(self.CommonFeature.parameters()) + \
                 list(self.TargetClassifier.parameters()) + \
                 list(self.SourceClassifiers.parameters())
        opt_cfg = self.cfg.OPTIM
        opt = build_optimizer(params,opt_cfg)
        scheduler = build_lr_scheduler(optimizer=opt,optim_cfg=opt_cfg)
        optimizers = [opt]
        lr_schedulers=[scheduler]
        return optimizers, lr_schedulers

    def share_step(self,batch,train_mode = True,weight=None):
        input, label, domain = self.parse_target_batch(batch)
        f_target = self.CommonFeature(input)
        logits_target = self.TargetClassifier(f_target)
        loss_target = self.loss_function(logits_target, label, train=train_mode,weight=weight)
        return loss_target, logits_target,f_target, label

    def parse_batch_train(self, batch):
        target_batch = batch["target_loader"]
        unlabel_batch = batch["unlabel_loader"]
        list_source_batches = batch["source_loader"]
        return target_batch,unlabel_batch,list_source_batches

    def on_train_epoch_start(self) -> None:
        if self.source_pretrain_epochs > self.current_epoch:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.PRETRAIN_SOURCE_LOSS_RATIO
        else:
            self.target_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.TARGET_LOSS_RATIO
            self.source_ratio = self.cfg.LIGHTNING_MODEL.TRAINER.EXTRA.SOURCE_LOSS_RATIO

    def training_step(self, batch, batch_idx):
        target_batch, unlabel_batch ,list_source_batches = self.parse_batch_train(batch)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_source_batches)

        loss_source = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.CommonFeature(u)
            logits = self.SourceClassifiers[d](f)
            domain_weight = self.source_domains_class_weight[d]
            loss_source += self.loss_function(logits, y, train=True, weight=domain_weight)
        loss_source /= len(domain_u)

        loss_target, logit_target, f_target,label = self.share_step(target_batch, train_mode=True,
                                                                  weight=self.class_weight)
        y_pred = F.softmax(logit_target, dim=1)
        y = label
        acc = self.train_acc(y_pred, y)

        total_loss = self.source_ratio*loss_source+self.target_ratio*loss_target

        f_unlabel = self.CommonFeature(unlabel_batch)

        transfer_loss = self.mkmmd_loss(f_target, f_unlabel)
        total_loss = total_loss + self.trade_off * transfer_loss

        self.log('Train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_source_loss', loss_source, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train_target_loss', loss_target, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('transfer_loss', transfer_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        loss, logit, _,y = self.share_step(batch,train_mode=False)
        y_pred = F.softmax(logit, dim=1)
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
        loss, logit, _,y = self.share_step(batch,train_mode=False)
        y_pred = F.softmax(logit,dim=1)
        return {'loss': loss,'y_pred':y_pred,'y':y}

