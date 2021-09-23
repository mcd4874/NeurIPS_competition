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
from dassl.modeling import build_layer
import os.path as osp
import numpy as np

from typing import Optional, Sequence
from dassl.utils.kernel import GaussianKernel

from dassl.utils.analysis import collect_feature,visualize,calculate

from dassl.utils.data_helper import ForeverDataIterator


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
class HeterogeneousDAN(TrainerMultiAdaptation):
    """
    https://arxiv.org/pdf/1502.02791.pdf
    Deep Adaptation Networks
    code reference https://github.com/thuml/Transfer-Learning-Library
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.bce = nn.BCEWithLogitsLoss()


        self.trade_off = cfg.TRAINER.DAN.trade_off
        print("trade off ratio : ",self.trade_off)
        alpha = cfg.TRAINER.DAN.GaussianKernel.alpha
        sigma = cfg.TRAINER.DAN.GaussianKernel.sigma
        track_running_stats = cfg.TRAINER.DAN.GaussianKernel.track_running_stats
        linear = cfg.TRAINER.DAN.linear
        # self.lmda = cfg.TRAINER.HeterogeneousDANN.lmda
        # print("current max lmda : ",self.lmda)

        if len(sigma) ==0:
            # sigma = None
        # define loss function
            print("alpha range : ",alpha)
            self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
                kernels=[GaussianKernel(alpha=k,track_running_stats=track_running_stats) for k in alpha],
                linear=linear
            )
        else:
            print("sigma range : ",sigma)
            self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
                kernels=[GaussianKernel(sigma=s,track_running_stats=track_running_stats) for s in sigma],
                linear=linear
            )
        # self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        #     kernels=[GaussianKernel(alpha=2 ** k) for k in range(-1, 2)],
        #     linear=False
        # )

    def build_temp_layer(self, cfg):
        embedding_layer_info = cfg.MODEL.LAYER
        layer_name = embedding_layer_info.NAME
        layer_params = embedding_layer_info.PARAMS
        return [layer_name, layer_params]
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

        # self.TemporalLayer = TemporalLayer(**backbone_params)
        self.TemporalLayer.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.TemporalLayer)))
        self.optim_TemporalLayer = build_optimizer(self.TemporalLayer, cfg.OPTIM)
        self.sched_TemporalLayer = build_lr_scheduler(self.optim_TemporalLayer, cfg.OPTIM)
        self.register_model('TemporalLayer', self.TemporalLayer, self.optim_TemporalLayer,
                            self.sched_TemporalLayer)

        fdim2 = self.TemporalLayer.fdim

        print("fdim2 : ",fdim2)

        print('Building SourceClassifiers')
        print("source domains label size : ",self.dm.source_domains_label_size)
        source_classifier_list = []
        for num_class in self.dm.source_domains_label_size:
            # source_classifier = nn.Linear(fdim2, num_class)
            source_classifier = self.create_classifier(fdim2, num_class)

            source_classifier_list.append(source_classifier)
        self.SourceClassifiers = nn.ModuleList(
            source_classifier_list
        )
        self.SourceClassifiers.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.SourceClassifiers)))
        self.optim_SourceClassifiers = build_optimizer(self.SourceClassifiers, cfg.OPTIM)
        self.sched_SourceClassifiers = build_lr_scheduler(self.optim_SourceClassifiers, cfg.OPTIM)
        self.register_model('SourceClassifiers', self.SourceClassifiers, self.optim_SourceClassifiers,
                            self.sched_SourceClassifiers)

        print('Building Target Classifier')
        self.TargetClassifier = self.create_classifier(fdim2,self.dm.num_classes)
        # self.TargetClassifier = nn.Linear(fdim2, self.dm.num_classes)
        self.TargetClassifier.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.TargetClassifier)))
        self.optim_TargetClassifier = build_optimizer(self.TargetClassifier, cfg.OPTIM)
        self.sched_TargetClassifier = build_lr_scheduler(self.optim_TargetClassifier, cfg.OPTIM)
        self.register_model('TargetClassifier', self.TargetClassifier, self.optim_TargetClassifier, self.sched_TargetClassifier)


        print('Building DomainDiscriminator')
        self.DomainDiscriminator = nn.Linear(fdim2, 1)
        self.DomainDiscriminator.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.DomainDiscriminator)))
        self.optim_DomainDiscriminator = build_optimizer(self.DomainDiscriminator, cfg.OPTIM)
        self.sched_DomainDiscriminator = build_lr_scheduler(self.optim_DomainDiscriminator, cfg.OPTIM)
        self.register_model('DomainDiscriminator', self.DomainDiscriminator, self.optim_DomainDiscriminator,
                            self.sched_DomainDiscriminator)

        self.revgrad = ReverseGrad()


    def calculate_dann(self,target_feature,source_feature):
        #there is a problem need to concern. We assume that label target batch size is same as source batch size
        domain_label_target = torch.ones(target_feature.shape[0], 1).to(self.device)
        domain_label_source = torch.zeros(source_feature.shape[0], 1).to(self.device)
        feature = torch.cat([target_feature, source_feature])
        domain_label = torch.cat([domain_label_target, domain_label_source])
        domain_pred = self.DomainDiscriminator(feature)
        loss_d = self.bce(domain_pred, domain_label)
        return loss_d

    def calculate_lmda_factor(self,batch_idx,current_epoch,num_batches,max_epoch,num_pretrain_epochs=0,lmda_scale = 1.0):
        epoch = current_epoch-num_pretrain_epochs
        total_epoch = max_epoch-num_pretrain_epochs
        global_step = batch_idx + epoch * num_batches
        progress = global_step / (total_epoch * num_batches)
        lmda = 2 / (1 + np.exp(-10 * progress)) - 1
        lmda = lmda * lmda_scale  # modify the scale of lmda
        return lmda

    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed
        loss_u = 0
        temp_feat_u = []
        # domain_label_u = []
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f)
            temp_feat_u.append(temp_layer)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_u += self.cce[d](logits, y)

        loss_u /= len(domain_u)

        f_target = self.TargetFeature(input_x)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.TargetClassifier(temp_layer_target)
        if backprob:
            loss_x = self.ce(logits_target,label_x)
        else:
            loss_x = self.val_ce(logits_target, label_x)

        feat_u = torch.cat(temp_feat_u, 0)


        # global_step = self.batch_idx + self.epoch * self.num_batches
        # progress = global_step / (self.max_epoch * self.num_batches)
        # lmda = 2 / (1 + np.exp(-10 * progress)) - 1
        # lmda = lmda* self.lmda # modify the scale of lmda
        # lmda = self.calculate_lmda_factor(self.batch_idx,self.epoch,self.num_batches,self.max_epoch,num_pretrain_epochs=self.pre_train_epochs,lmda_scale=self.lmda)
        # n_iter = self.epoch * self.num_batches + self.batch_idx
        # self.write_scalar('train/lmda', lmda, n_iter)

        # feat_x = self.revgrad(temp_layer_target, grad_scaling=lmda)
        # feat_u = self.revgrad(feat_u, grad_scaling=lmda)

        #test to combine 2 vector and calculate loss
        # loss_d = self.calculate_dann(target_feature=feat_x,source_feature=feat_u)

        #old way of calculate seperate domain loss for target and source
        # output_xd = self.DomainDiscriminator(feat_x)
        # output_ud = self.DomainDiscriminator(feat_u)
        # loss_d = self.bce(output_xd, domain_label_x) + self.bce(output_ud, domain_label_u)

        # total_loss = loss_x + loss_u + loss_d
        feat_x = temp_layer_target
        # transfer_loss = self.mkmmd_loss(feat_u, feat_x)
        # total_loss = loss_x + loss_u + self.trade_off*transfer_loss
        # loss_summary = {
        #     'total_loss': total_loss.item(),
        #     'loss_x': loss_x.item(),
        #     'loss_u': loss_u.item(),
        #     'loss_mmd':transfer_loss.item(),
        #     # 'loss_d': loss_d.item(),
        #     # 'lmda_factor': lmda
        # }

        if backprob:
            transfer_loss = self.mkmmd_loss(feat_u, feat_x)
            total_loss = loss_x + loss_u + self.trade_off * transfer_loss
            loss_summary = {
                'total_loss': total_loss.item(),
                'loss_x': loss_x.item(),
                'loss_u': loss_u.item(),
                'loss_mmd': self.trade_off *transfer_loss.item(),
                # 'loss_d': loss_d.item(),
                # 'lmda_factor': lmda
            }
            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            total_loss = loss_x + loss_u
            loss_summary = {
                'total_loss': total_loss.item(),
                'loss_x': loss_x.item(),
                'loss_u': loss_u.item(),
                # 'loss_d': loss_d.item(),
                # 'lmda_factor': lmda
            }
        return loss_summary

    def build_feature_extractor(self,extractor_type="target"):
        extractor = None
        if extractor_type=="target":
            extractor = nn.Sequential(self.TargetFeature, self.TemporalLayer)
        elif extractor_type == "source":
            extractor = nn.Sequential(self.SourceFeatures[0], self.TemporalLayer)
        return extractor

    def get_feature(self,tmp_x,tmp_u,tmp_x_label,tmp_u_label,x_feature_extractor="target",u_feature_extractor="source",x_domain_name='train_target',u_domain_name='source'):
        t_f = []
        s_f = []
        x_feature_extractor = self.build_feature_extractor(extractor_type=x_feature_extractor)
        u_feature_extractor = self.build_feature_extractor(extractor_type=u_feature_extractor)
        with torch.no_grad():
            for x in tmp_x:
                target_feature = x_feature_extractor(x)
                t_f.append(target_feature)
            for u in tmp_u:
                source_feature = u_feature_extractor(u)
                s_f.append(source_feature)
        # source_feature = s_feature_extractor(u).detach().cpu()
        # plot t-SNE
        target_feature = torch.cat(t_f, dim=0)
        source_feature = torch.cat(s_f, dim=0)

        target_label = torch.cat(tmp_x_label, dim=0)
        source_label = torch.cat(tmp_u_label, dim=0)

        target_domain = torch.ones((target_feature.shape[0], 1))
        source_domain = torch.zeros((source_feature.shape[0], 1))

        target_domain_name = np.array([x_domain_name] * target_feature.shape[0])
        source_domain_name = np.array([u_domain_name] * source_feature.shape[0])

        feature = torch.cat([source_feature,target_feature], dim=0)
        label = torch.cat([source_label,target_label], dim=0)
        domain = torch.cat([source_domain,target_domain], dim=0)
        domain_name = np.concatenate([source_domain_name, target_domain_name], axis=0)

        return feature,label,domain,domain_name

    def generate_tsne(self,feature,label,domain_name,file_name='TSNE.png'):
        np_feature = feature.detach().cpu().numpy()
        np_label = label.detach().cpu().numpy()
        np_domain_name = domain_name
        visualize(np_feature,np_label,np_domain_name,filename=file_name)

    # def generate_tsne(self,tmp_x,tmp_u,x_label,u_label,x_feature_extractor="target",u_feature_extractor="source",x_domain_name='train_target',u_domain_name='source',file_name='TSNE.png',bs=32):
    #
    #     # x = torch.cat(tmp_x, dim=0)
    #     # u = torch.cat(tmp_u, dim=0)
    #     t_f = []
    #     s_f = []
    #     x_feature_extractor = self.build_feature_extractor(extractor_type=x_feature_extractor)
    #     u_feature_extractor = self.build_feature_extractor(extractor_type=u_feature_extractor)
    #     with torch.no_grad():
    #         for x in tmp_x:
    #             target_feature = x_feature_extractor(x).detach().cpu()
    #             t_f.append(target_feature)
    #         for u in tmp_u:
    #             source_feature = u_feature_extractor(u).detach().cpu()
    #             s_f.append(source_feature)
    #     # source_feature = s_feature_extractor(u).detach().cpu()
    #     # plot t-SNE
    #     target_feature = torch.cat(t_f, dim=0).numpy()
    #     source_feature = torch.cat(s_f, dim=0).numpy()
    #     target_label = np.concatenate(x_label,axis=0)
    #     source_label = np.concatenate(u_label,axis=0)
    #
    #     tSNE_filename = osp.join(self.output_dir, file_name)
    #     visualize(source_feature, target_feature,source_label=source_label,target_label=target_label,source_domain=u_domain_name,target_domain=x_domain_name ,filename=tSNE_filename)
    #     print("Saving t-SNE to", tSNE_filename)

    @torch.no_grad()
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

        # tmp_x,tmp_u = [],[]

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
                try:
                    batch_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(self.list_train_loader_u[train_loader_u_iter_idx])
                    list_train_loader_u_iter[train_loader_u_iter_idx] = train_loader_u_iter
                    batch_u = next(train_loader_u_iter)
                list_batch_u.append(batch_u)

            input, label, _, tmp_input_U,_,_ = self.parse_batch_train(batch_x, list_batch_u)
            loss = self.forward_backward(batch_x, list_batch_u, backprob=False)
            losses.update(loss)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

            # tmp_x.append(input)
            # tmp_u.append(tmp_input_U[0])


        # self.generate_tsne(tmp_x,tmp_u)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss_x'].avg
        val_losses = losses.dict_results()
        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        for k, v in val_losses.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        return [total_loss, losses.dict_results(), results]

    def model_inference(self, input, return_feature=False):
        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f)
        logits = self.TargetClassifier(temp_layer)
        result = F.softmax(logits, 1)
        if return_feature:
            return result, temp_layer
        return result

    # def after_epoch(self):
    # def before_epoch(self):
    # def before_train(self):
    #     """do something"""
    #     folder = "before_train"
    #     outdir = os.path.join(self.output_dir, folder)
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)
    #     source_loader_u_iter = ForeverDataIterator(self.list_train_loader_u[0])
    #     train_loader_x_iter = ForeverDataIterator(self.train_loader_x)
    #     val_loader_iter = ForeverDataIterator(self.val_loader)
    #     test_loader_iter = ForeverDataIterator(self.test_loader)
    #
    #     # plot and play with t-sne to compare the source dataset vs train dataset/valid dataset/ test dataset
    #     # num_batches = self.num_batches
    #     self.set_model_mode('eval')
    #
    #     # tmp_x,tmp_u,tmp_val,tmp_test = [],[],[],[]
    #     # tmp_x_label,tmp_u_label,tmp_val_label,tmp_test_label = [],[],[],[]
    #
    #     # compare train dataset vs source dataset
    #     # max_batches = max(len(train_loader_x_iter),len(source_loader_u_iter))
    #     # for idx in range(max_batches):
    #     #     train_batch_x = next(train_loader_x_iter)
    #     #     x, x_label, _ = self.parse_target_batch(train_batch_x)
    #     #     tmp_x.append(x)
    #     #     tmp_x_label.append(x_label.cpu().numpy())
    #     #     source_batch_u = next(source_loader_u_iter)
    #     #     u, u_label, _ = self.parse_target_batch(source_batch_u)
    #     #     tmp_u.append(u)
    #     #     tmp_u_label.append(u_label.cpu().numpy())
    #     # tmp_x, tmp_x_label, tmp_u, tmp_u_label = self.get_data_pair(train_loader_x_iter, source_loader_u_iter)
    #     # feature,label,domain,domain_name = self.get_feature(tmp_x,tmp_u,tmp_x_label,tmp_u_label,x_feature_extractor="target",u_feature_extractor="source",x_domain_name="train_target", u_domain_name="source")
    #     # self.generate_tsne(feature,label,domain_name)
    #     # A_dist = calculate(feature,domain,self.device)
    #     # print("A dist for before train target vs source data : ",A_dist)
    #
    #     # self.generate_tsne(tmp_x, tmp_u, tmp_x_label, tmp_u_label, x_feature_extractor="target",
    #     #                    u_feature_extractor="source", ,
    #     #                    file_name=os.path.join(folder, 'train_source_TSNE_1.png'))
    #
    #     # tmp_x, tmp_x_label, tmp_test, tmp_test_label = self.get_data_pair(train_loader_x_iter, test_loader_iter)
    #     # self.generate_tsne(tmp_x, tmp_test, tmp_x_label, tmp_test_label, x_feature_extractor="target",
    #     #                    u_feature_extractor="target", x_domain_name="train_target", u_domain_name="test_target",
    #     #                    file_name=os.path.join(folder, 'train_test_TSNE_1.png'))
    #
    #     ########
    #     # tmp_x, tmp_x_label, tmp_test, tmp_test_label = self.get_data_pair(train_loader_x_iter, test_loader_iter)
    #     # feature,label,domain,domain_name = self.get_feature(tmp_x,tmp_test,tmp_x_label,tmp_test_label,x_feature_extractor="target",u_feature_extractor="target",x_domain_name="train_target", u_domain_name="test_target")
    #     # # self.generate_tsne(feature,label,domain_name,file_name=os.path.join(outdir, 'train_test_TSNE.png'))
    #     # A_dist = calculate(feature,domain,self.device)
    #     # print("A dist for before train target vs test target : ",A_dist)
    #     #
    #     # tmp_x, tmp_x_label = self.get_data(train_loader_x_iter)
    #     # tmp_test, tmp_test_label = self.get_data(test_loader_iter)
    #     # feature, label, domain, domain_name = self.get_feature(tmp_x, tmp_test, tmp_x_label, tmp_test_label,
    #     #                                                        x_feature_extractor="target",
    #     #                                                        u_feature_extractor="target",
    #     #                                                        x_domain_name="train_target",
    #     #                                                        u_domain_name="test_target")
    #     # self.generate_tsne(feature, label, domain_name, file_name=os.path.join(outdir, 'train_test_TSNE.png'))
    #
    #
    #     tmp_test, tmp_test_label = self.get_data(test_loader_iter)
    #     x_feature_extractor = self.build_feature_extractor(extractor_type="target")
    #     t_f = []
    #     with torch.no_grad():
    #         for x in tmp_test:
    #             target_feature = x_feature_extractor(x)
    #             t_f.append(target_feature)
    #     target_feature = torch.cat(t_f, dim=0)
    #     target_label = torch.cat(tmp_test_label, dim=0)
    #     target_domain_name = np.array(["train_target"] * target_feature.shape[0])
    #     self.generate_tsne(target_feature, target_label, target_domain_name, file_name=os.path.join(outdir, 'test_TSNE.png'))
    #
    #
    #     #
    #     # tmp_val, tmp_val_label, tmp_test, tmp_test_label = self.get_data_pair(val_loader_iter, test_loader_iter)
    #     # self.generate_tsne(tmp_val, tmp_test, tmp_val_label, tmp_test_label, x_feature_extractor="target",
    #     #                    u_feature_extractor="target", x_domain_name="val_target", u_domain_name="test_target",
    #     #                    file_name=os.path.join(folder, 'val_test_TSNE_1.png'))
    #     #
    #     # tmp_test, tmp_test_label, tmp_u, tmp_u_label = self.get_data_pair(test_loader_iter, source_loader_u_iter)
    #     # self.generate_tsne(tmp_test, tmp_u, tmp_test_label, tmp_u_label, x_feature_extractor="target",
    #     #                    u_feature_extractor="source", x_domain_name="test_target", u_domain_name="source",
    #     #                    file_name=os.path.join(folder, 'test_source_TSNE_1.png'))
    #
    #     # for x_idx in range(len(train_loader_x_iter)):
    #     #     train_batch_x = next(train_loader_x_iter)
    #     #     x, x_label, _ = self.parse_target_batch(train_batch_x)
    #     #     tmp_x.append(x)
    #     #     tmp_x_label.append(x_label.cpu().numpy())
    #     #
    #     # for u_idx in range(len(source_loader_u_iter)):
    #     #     source_batch_u = next(source_loader_u_iter)
    #     #     u, u_label, _ = self.parse_target_batch(source_batch_u)
    #     #     tmp_u.append(u)
    #     #     tmp_u_label.append(u_label.cpu().numpy())
    #     #
    #     #
    #     # for val_idx in range(len(val_loader_iter)):
    #     #     val_batch = next(val_loader_iter)
    #     #     val_data,val_label,_ = self.parse_target_batch(val_batch)
    #     #     tmp_val.append(val_data)
    #     #     tmp_val_label.append(val_label.cpu().numpy())
    #     #
    #     #
    #     # for test_idx in range(len(test_loader_iter)):
    #     #     test_batch = next(test_loader_iter)
    #     #     test_data,test_label = self.parse_batch_test(test_batch)
    #     #     tmp_test.append(test_data)
    #     #     tmp_test_label.append(test_label.cpu().numpy())
    #
    #     # self.generate_tsne(tmp_x,tmp_u,'x_u_TSNE.png')
    #     # self.generate_tsne(tmp_val,tmp_u,'val_u_TSNE.png')
    #     # self.generate_tsne(tmp_test,tmp_u,'test_u_TSNE.png')
    #     # self.generate_tsne(tmp_x,tmp_test,'x_test_TSNE.png')
    #
    #     # self.generate_tsne(tmp_x, tmp_u,tmp_x_label,tmp_u_label,x_feature_extractor="target",u_feature_extractor="source",x_domain_name="train_target",u_domain_name="source", file_name=os.path.join(folder,'train_source_TSNE_1.png'))
    #     # self.generate_tsne(tmp_x, tmp_test,tmp_x_label,tmp_test_label,x_feature_extractor="target",u_feature_extractor="target",x_domain_name="train_target",u_domain_name="test_target", file_name=os.path.join(folder,'train_test_TSNE_1.png'))
    #     # self.generate_tsne(tmp_val, tmp_test,tmp_val_label,tmp_test_label,x_feature_extractor="target",u_feature_extractor="target",x_domain_name="val_target",u_domain_name="test_target", file_name=os.path.join(folder,'val_test_TSNE_1.png'))
    #     # self.generate_tsne(tmp_test, tmp_u,tmp_test_label,tmp_u_label,x_feature_extractor="target",u_feature_extractor="source",x_domain_name="test_target",u_domain_name="source", file_name=os.path.join(folder,'test_source_TSNE_1.png'))
    #     #
    #     # self.generate_tsne(tmp_val, tmp_u,tmp_x_label,tmp_u_label, 'val_u_TSNE_1.png')
    #     # self.generate_tsne(tmp_test, tmp_u,tmp_x_label,tmp_u_label, 'test_u_TSNE_1.png')
    #
    #     super().before_train()


    def get_data_pair(self,loader_x_iter,loader_u_iter):
        loader_x_iter.reset()
        loader_u_iter.reset()
        tmp_x, tmp_u= [], []
        tmp_x_label, tmp_u_label= [], []
        max_batches = max(len(loader_x_iter), len(loader_u_iter))
        for idx in range(max_batches):
            train_batch_x = next(loader_x_iter)
            x, x_label, _ = self.parse_target_batch(train_batch_x)
            tmp_x.append(x)
            tmp_x_label.append(x_label)
            source_batch_u = next(loader_u_iter)
            u, u_label, _ = self.parse_target_batch(source_batch_u)
            tmp_u.append(u)
            tmp_u_label.append(u_label)
        return tmp_x,tmp_x_label,tmp_u,tmp_u_label

    def get_data(self,loader_iter):
        loader_iter.reset()
        tmp_x, tmp_x_label = [],[]
        for x_idx in range(len(loader_iter)):
            train_batch_x = next(loader_iter)
            x, x_label, _ = self.parse_target_batch(train_batch_x)
            tmp_x.append(x)
            tmp_x_label.append(x_label)
        return tmp_x,tmp_x_label

    # def after_train(self):
    #     super().after_train()
    #     # super().after_epoch()
    #     folder = "after_train"
    #     outdir = os.path.join(self.output_dir,folder)
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)
    #     source_loader_u_iter = ForeverDataIterator(self.list_train_loader_u[0])
    #     train_loader_x_iter = ForeverDataIterator(self.train_loader_x)
    #     val_loader_iter = ForeverDataIterator(self.val_loader)
    #     test_loader_iter = ForeverDataIterator(self.test_loader)
    #
    #     list_test_subject_loader = self.get_test_subject_data_loader()
    #
    #     #plot and play with t-sne to compare the source dataset vs train dataset/valid dataset/ test dataset
    #     # num_batches = self.num_batches
    #     self.set_model_mode('eval')
    #
    #     # tmp_x,tmp_u,tmp_val,tmp_test = [],[],[],[]
    #     # tmp_x_label,tmp_u_label,tmp_val_label,tmp_test_label = [],[],[],[]
    #
    #     #compare train dataset vs source dataset
    #     # max_batches = max(len(train_loader_x_iter),len(source_loader_u_iter))
    #     # for idx in range(max_batches):
    #     #     train_batch_x = next(train_loader_x_iter)
    #     #     x, x_label, _ = self.parse_target_batch(train_batch_x)
    #     #     tmp_x.append(x)
    #     #     tmp_x_label.append(x_label.cpu().numpy())
    #     #     source_batch_u = next(source_loader_u_iter)
    #     #     u, u_label, _ = self.parse_target_batch(source_batch_u)
    #     #     tmp_u.append(u)
    #     #     tmp_u_label.append(u_label.cpu().numpy())
    #
    #     # tmp_x, tmp_x_label, tmp_u, tmp_u_label = self.get_data_pair(train_loader_x_iter, source_loader_u_iter)
    #     # feature, label, domain, domain_name = self.get_feature(tmp_x, tmp_u, tmp_x_label, tmp_u_label,
    #     #                                                        x_feature_extractor="target",
    #     #                                                        u_feature_extractor="source",
    #     #                                                        x_domain_name="train_target", u_domain_name="source")
    #     # self.generate_tsne(feature, label, domain_name)
    #     # A_dist = calculate(feature, domain, self.device)
    #     # print("A dist for after train target vs source data : ", A_dist)
    #
    #     # tmp_x,tmp_x_label,tmp_u,tmp_u_label = self.get_data_pair(train_loader_x_iter,source_loader_u_iter)
    #     # self.generate_tsne(tmp_x, tmp_u,tmp_x_label,tmp_u_label,x_feature_extractor="target",u_feature_extractor="source",x_domain_name="train_target",u_domain_name="source", file_name=os.path.join(folder,'train_source_TSNE_1.png'))
    #     #
    #
    #
    #
    #     # tmp_x,tmp_x_label,tmp_test,tmp_test_label = self.get_data_pair(train_loader_x_iter,test_loader_iter)
    #     # self.generate_tsne(tmp_x, tmp_test, tmp_x_label, tmp_test_label, x_feature_extractor="target",
    #     #                    u_feature_extractor="target", x_domain_name="train_target", u_domain_name="test_target",
    #     #                    file_name=os.path.join(folder, 'train_test_TSNE_1.png'))
    #     #
    #
    #     #######
    #     # tmp_x, tmp_x_label, tmp_test, tmp_test_label = self.get_data_pair(train_loader_x_iter, test_loader_iter)
    #     # feature,label,domain,domain_name = self.get_feature(tmp_x,tmp_test,tmp_x_label,tmp_test_label,x_feature_extractor="target",u_feature_extractor="target",x_domain_name="train_target", u_domain_name="test_target")
    #     # # self.generate_tsne(feature,label,domain_name,file_name=os.path.join(outdir, 'train_test_TSNE.png'))
    #     # A_dist = calculate(feature,domain,self.device)
    #     # print("A dist for after train target vs test target : ",A_dist)
    #     #
    #     # tmp_x, tmp_x_label = self.get_data(train_loader_x_iter)
    #     # tmp_test, tmp_test_label = self.get_data( test_loader_iter)
    #     # feature,label,domain,domain_name = self.get_feature(tmp_x,tmp_test,tmp_x_label,tmp_test_label,x_feature_extractor="target",u_feature_extractor="target",x_domain_name="train_target", u_domain_name="test_target")
    #     # self.generate_tsne(feature,label,domain_name,file_name=os.path.join(outdir, 'train_test_TSNE.png'))
    #
    #     # tmp_test, tmp_test_label = self.get_data(test_loader_iter)
    #     # x_feature_extractor = self.build_feature_extractor(extractor_type="target")
    #     # t_f = []
    #     # with torch.no_grad():
    #     #     for x in tmp_test:
    #     #         target_feature = x_feature_extractor(x)
    #     #         t_f.append(target_feature)
    #     # target_feature = torch.cat(t_f, dim=0)
    #     # target_label = torch.cat(tmp_test_label, dim=0)
    #     # target_domain_name = np.array(["train_target"] * target_feature.shape[0])
    #     # self.generate_tsne(target_feature, target_label, target_domain_name, file_name=os.path.join(outdir, 'test_TSNE.png'))
    #     idx =0
    #     for subject_loader in list_test_subject_loader:
    #
    #         subject_test_iter = ForeverDataIterator(subject_loader)
    #         tmp_test, tmp_test_label = self.get_data(subject_test_iter)
    #         x_feature_extractor = self.build_feature_extractor(extractor_type="target")
    #         t_f = []
    #         with torch.no_grad():
    #             for x in tmp_test:
    #                 target_feature = x_feature_extractor(x)
    #                 t_f.append(target_feature)
    #         target_feature = torch.cat(t_f, dim=0)
    #         target_label = torch.cat(tmp_test_label, dim=0)
    #         target_domain_name = np.array(["train_target"] * target_feature.shape[0])
    #         self.generate_tsne(target_feature, target_label, target_domain_name, file_name=os.path.join(outdir, 'test_{}_TSNE.png'.format(idx)))
    #         idx+=1
    #
    #
    #     # tmp_val,tmp_val_label,tmp_test,tmp_test_label = self.get_data_pair(val_loader_iter,test_loader_iter)
    #     # self.generate_tsne(tmp_val, tmp_test, tmp_val_label, tmp_test_label, x_feature_extractor="target",
    #     #                    u_feature_extractor="target", x_domain_name="val_target", u_domain_name="test_target",
    #     #                    file_name=os.path.join(folder, 'val_test_TSNE_1.png'))
    #     #
    #     # tmp_test,tmp_test_label,tmp_u,tmp_u_label = self.get_data_pair(test_loader_iter,source_loader_u_iter)
    #     # self.generate_tsne(tmp_test, tmp_u, tmp_test_label, tmp_u_label, x_feature_extractor="target",
    #     #                    u_feature_extractor="source", x_domain_name="test_target", u_domain_name="source",
    #     #                    file_name=os.path.join(folder, 'test_source_TSNE_1.png'))
    #
    #     # for x_idx in range(len(train_loader_x_iter)):
    #     #     train_batch_x = next(train_loader_x_iter)
    #     #     x, x_label, _ = self.parse_target_batch(train_batch_x)
    #     #     tmp_x.append(x)
    #     #     tmp_x_label.append(x_label.cpu().numpy())
    #     #
    #     # for u_idx in range(len(source_loader_u_iter)):
    #     #     source_batch_u = next(source_loader_u_iter)
    #     #     u, u_label, _ = self.parse_target_batch(source_batch_u)
    #     #     tmp_u.append(u)
    #     #     tmp_u_label.append(u_label.cpu().numpy())
    #     #
    #     #
    #     # for val_idx in range(len(val_loader_iter)):
    #     #     val_batch = next(val_loader_iter)
    #     #     val_data,val_label,_ = self.parse_target_batch(val_batch)
    #     #     tmp_val.append(val_data)
    #     #     tmp_val_label.append(val_label.cpu().numpy())
    #     #
    #     #
    #     # for test_idx in range(len(test_loader_iter)):
    #     #     test_batch = next(test_loader_iter)
    #     #     test_data,test_label = self.parse_batch_test(test_batch)
    #     #     tmp_test.append(test_data)
    #     #     tmp_test_label.append(test_label.cpu().numpy())
    #
    #
    #     # self.generate_tsne(tmp_x,tmp_u,'x_u_TSNE.png')
    #     # self.generate_tsne(tmp_val,tmp_u,'val_u_TSNE.png')
    #     # self.generate_tsne(tmp_test,tmp_u,'test_u_TSNE.png')
    #     # self.generate_tsne(tmp_x,tmp_test,'x_test_TSNE.png')
    #
    #     # self.generate_tsne(tmp_x, tmp_u,tmp_x_label,tmp_u_label,x_feature_extractor="target",u_feature_extractor="source",x_domain_name="train_target",u_domain_name="source", file_name=os.path.join(folder,'train_source_TSNE_1.png'))
    #     # self.generate_tsne(tmp_x, tmp_test,tmp_x_label,tmp_test_label,x_feature_extractor="target",u_feature_extractor="target",x_domain_name="train_target",u_domain_name="test_target", file_name=os.path.join(folder,'train_test_TSNE_1.png'))
    #     # self.generate_tsne(tmp_val, tmp_test,tmp_val_label,tmp_test_label,x_feature_extractor="target",u_feature_extractor="target",x_domain_name="val_target",u_domain_name="test_target", file_name=os.path.join(folder,'val_test_TSNE_1.png'))
    #     # self.generate_tsne(tmp_test, tmp_u,tmp_test_label,tmp_u_label,x_feature_extractor="target",u_feature_extractor="source",x_domain_name="test_target",u_domain_name="source", file_name=os.path.join(folder,'test_source_TSNE_1.png'))
    #     #
    #     # self.generate_tsne(tmp_val, tmp_u,tmp_x_label,tmp_u_label, 'val_u_TSNE_1.png')
    #     # self.generate_tsne(tmp_test, tmp_u,tmp_x_label,tmp_u_label, 'test_u_TSNE_1.png')

import os
    #
    #
    #
