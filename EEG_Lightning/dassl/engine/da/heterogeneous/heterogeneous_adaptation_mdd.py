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
import math
import numpy as np


#assume that only deal with 1 source vs 1 target domain and the label spaces size are divisible by batch size
def mdd_loss(target_feature, source_features,target_label,source_labels,target_label_size,source_label_size, target_weight=1, source_weight=1):
    target_softmax_out = nn.Softmax(dim=1)(target_feature)
    source_softmax_out = nn.Softmax(dim=1)(source_features)
    target_bs = target_softmax_out.size()[0]
    source_bs = source_softmax_out.size()[0]
    # print("target bs : ",target_softmax_out[:5,])
    # print("source bs : ",source_softmax_out[:5,])
    # torch.linalg.norm
    if target_bs == source_bs:
        loss = torch.norm((target_softmax_out - source_softmax_out).abs(), 2, 1).sum() / float(target_bs)
        # print("target first 10 label : ",target_label)
        # print("source first 10 label : ",source_labels)

    elif target_bs < source_bs:
        # print("target size : ",target_softmax_out.size())
        round_mul = math.ceil(source_bs/target_bs)
        # print("round mul : ",round_mul)
        remain = source_bs%target_bs
        # print("remain : ",remain)
        new_target_out = torch.tile(target_softmax_out,(round_mul,1))
        # print(" new target out size : ",new_target_out.size())
        if remain>0:
            new_target_out = new_target_out[:target_bs*(round_mul-1)+remain,]
            # print(" new update target out size : ", new_target_out.size())

        #need to do stuff
        loss = torch.norm((new_target_out - source_softmax_out).abs(), 2, 1).sum() / float(target_bs)
    else:
        #need to do stuff
        # print("target size : ",target_softmax_out.size())
        round_mul = math.ceil(target_bs /source_bs )
        # print("round mul : ",round_mul)
        remain = target_bs % source_bs
        # print("remain : ",remain)
        new_source_out = torch.tile(source_softmax_out, (round_mul, 1))
        # print(" new target out size : ",new_target_out.size())
        if remain > 0:
            new_source_out = new_source_out[:target_bs * (round_mul - 1) + remain, ]
            # print(" new update target out size : ", new_target_out.size())

        # need to do stuff
        loss = torch.norm((target_softmax_out - new_source_out).abs(), 2, 1).sum() / float(target_bs)

    # print("target label: ",target_label)
    # print("source label:  ",source_labels)

    # split_target = target_bs//target_label_size
    # split_source = source_bs//source_label_size
    # print("diff loss : ",loss)
    match_target_loss = get_pari_loss1(target_label,target_softmax_out,target_label_size)
    match_source_loss = get_pari_loss1(source_labels,source_softmax_out,source_label_size)
    # print("pair loss : ",loss)
    # print("target pair loss : ",match_target_loss)
    # print("source pair loss : ",match_source_loss)

    return loss + target_weight * match_target_loss + source_weight * match_source_loss



def get_pari_loss1(labels, features,label_space_size):
    # print("feature require grad : ",features.requires_grad)
    # clone_f = features.clone().detach()
    # print("clone feature require grad : ",clone_f.requires_grad)
    #
    # res = features.unsqueeze(1) - clone_f
    # res = res.view(res.shape[0] * res.shape[1], res.shape[2])
    # print("update calculate require grad : ",res.requires_grad)
    # bs = res.shape[0]
    # loss = torch.norm((res).abs(), 2, 0).sum()
    # print("final pairwise dist loses : ",loss)
    #
    # n = features.size(0)
    # dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
    # dist = dist + dist.t()
    # dist.addmm_(1, -2, features, features.t())
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # print("dist formula : ",dist.sum())
    #
    # loss = loss/bs
    n = features.size()[0]
    splits = n//label_space_size
    feature_by_label = torch.split(features,splits,dim=0)
    total_loss = 0

    # label_groups = torch.split(labels,splits,dim=0)
    # for idx in range(len(feature_by_label)):
    #     feature = feature_by_label[idx]
    #     label = label_groups[idx]
    #     print(" label split group : ",label)
    for feature in feature_by_label:
        n = feature.size()[0]
        # clone_f = feature.clone().detach()
        # cdist = torch.cdist(feature, clone_f, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        cdist = torch.cdist(feature, feature.detach(), p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        # print("cdist : ", cdist)
        loss = cdist.sum() / (n * (n - 1))
        total_loss = total_loss+loss
        # print("final cdist loss : ", loss)
    total_loss = total_loss/label_space_size
    # print("total loss : ",total_loss)
    return total_loss
    # loss = 0
    # count = 0
    # for i in range(len(labels)):
    #     for j in range(i + 1, len(labels)):
    #         if (labels[i] == labels[j]):
    #             count += 1
    #             loss += torch.norm((features[i] - features[j]).abs(), 2, 0).sum()
    # return loss / count


@TRAINER_REGISTRY.register()
class HeterogeneousMDD(TrainerMultiAdaptation):
    """
    Maximum Density Divergence for Domain Adaptation
    https://arxiv.org/pdf/2004.12615.pdf
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

        self.lmda = cfg.TRAINER.HeterogeneousMDD.lmda
        self.mdd_weight = cfg.TRAINER.HeterogeneousMDD.mdd_weight
        self.fix_lmda = cfg.TRAINER.HeterogeneousMDD.fix_lmda
        print("current max lmda : ",self.lmda)

        # print("source batch : ",self.cfg.DATALOADER.VALID.BATCH_SIZE)

        self.use_entropy = cfg.TRAINER.HeterogeneousMDD.use_entropy
        # use projection if the source label spaces are not the same as the target label spaces
        self.use_projection = cfg.TRAINER.HeterogeneousMDD.use_projection



    # def check_cfg(self, cfg):
    #     assert cfg.DATALOADER.TRAIN_U.SAMPLER == 'RandomDomainSampler'

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
        embedding_layer_info = cfg.MODEL.LAYER
        layer_name = embedding_layer_info.NAME
        layer_params = embedding_layer_info.PARAMS
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
        # input_spaces = fdim2
        input_spaces = fdim2 * self.dm.num_classes
        self.DomainDiscriminator = nn.Linear(input_spaces, 1)
        self.DomainDiscriminator.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.DomainDiscriminator)))
        self.optim_DomainDiscriminator = build_optimizer(self.DomainDiscriminator, cfg.OPTIM)
        self.sched_DomainDiscriminator = build_lr_scheduler(self.optim_DomainDiscriminator, cfg.OPTIM)
        self.register_model('DomainDiscriminator', self.DomainDiscriminator, self.optim_DomainDiscriminator,
                            self.sched_DomainDiscriminator)

        self.revgrad = ReverseGrad()


        print("build projection layer if source domains have different label spaces")
        #assume all source domain have same label spaces
        source_domain_label_size = self.dm.source_domains_label_size[0]
        current_input_spaces = fdim2 * source_domain_label_size
        print("current input spaces is {} and output sapces is {}".format(current_input_spaces,input_spaces))
        self.source_projection = nn.Linear(current_input_spaces, input_spaces)
        self.source_projection.to(self.device)

        # source_projection_list = []
        # for num_class in self.dm.source_domains_label_size:
        #     # source_classifier = nn.Linear(fdim2, num_class)
        #     current_input_spaces = fdim2*num_class
        #     source_projection = nn.Linear(current_input_spaces, input_spaces)
        #     source_projection_list.append(source_projection)
        # self.SourceProjections = nn.ModuleList(
        #     source_projection_list
        # )

        print('# params: {:,}'.format(count_num_param(self.source_projection)))
        self.optim_source_projection = build_optimizer(self.source_projection, cfg.OPTIM)
        self.sched_source_projection = build_lr_scheduler(self.optim_source_projection, cfg.OPTIM)
        self.register_model('SourceProjection', self.source_projection, self.optim_source_projection,
                            self.sched_source_projection)

    def calculate_lmda_factor(self,batch_idx,current_epoch,num_batches,max_epoch,num_pretrain_epochs=0,lmda_scale = 1.0):
        epoch = current_epoch-num_pretrain_epochs
        total_epoch = max_epoch-num_pretrain_epochs
        global_step = batch_idx + epoch * num_batches
        progress = global_step / (total_epoch * num_batches)
        lmda = 2 / (1 + np.exp(-10 * progress)) - 1
        lmda = lmda * lmda_scale  # modify the scale of lmda
        return lmda
    def generate_entropy(self,softmax_output):
        epsilon = 1e-5
        entropy = -softmax_output * torch.log(softmax_output + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def genenerate_mix_feature(self,feature, softmax_output):
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        return op_out.view(-1, softmax_output.size(1) * feature.size(1))

    def DANN(self,target_feature,source_feature):
        #there is a problem need to concern. We assume that label target batch size is same as source batch size
        domain_label_target = torch.ones(target_feature.shape[0], 1).to(self.device)
        domain_label_source = torch.zeros(source_feature.shape[0], 1).to(self.device)
        feature = torch.cat([target_feature, source_feature])
        domain_label = torch.cat([domain_label_target, domain_label_source])
        domain_pred = self.DomainDiscriminator(feature)
        loss_d = self.bce(domain_pred, domain_label)
        return loss_d


    def CDAN(self,target_feature, target_softmax_out, source_feature, source_softmax_out, entropy=None,lmda = None):
        target_bs = target_feature.shape[0]
        source_bs = source_feature.shape[0]

        domain_label_target = torch.ones(target_bs, 1).to(self.device)
        domain_label_source = torch.zeros(source_bs, 1).to(self.device)

        # feature = torch.cat([source_feature,target_feature])
        domain_label = torch.cat([domain_label_source,domain_label_target])
        # softmax_out = torch.cat([source_softmax_out,target_softmax_out])
        #calculate entropy
        if entropy is not None:
            target_entropy = self.generate_entropy(target_softmax_out)
            source_entropy = self.generate_entropy(source_softmax_out)
            entropy = torch.cat([source_entropy,target_entropy])

        # detach_softmax_out = softmax_out.detach()
        # mix_feature = self.genenerate_mix_feature(feature, detach_softmax_out)

        target_softmax_out = target_softmax_out.detach()
        source_softmax_out = source_softmax_out.detach()

        target_mix_feature = self.genenerate_mix_feature(target_feature, target_softmax_out)
        source_mix_feature = self.genenerate_mix_feature(source_feature, source_softmax_out)
        if self.use_projection:
            source_mix_feature = self.source_projection(source_mix_feature)

        mix_feature =  torch.cat([source_mix_feature,target_mix_feature])
        domain_out = self.DomainDiscriminator(mix_feature)

        if entropy is not None:
            # print("apply entropy")
            entropy = self.revgrad(entropy,lmda)
            entropy = 1.0 + torch.exp(-entropy)
            source_mask = torch.ones_like(entropy)
            source_mask[source_bs:] = 0
            source_weight = entropy * source_mask
            # print("source weight : ",source_weight)
            target_mask = torch.ones_like(entropy)
            target_mask[0:source_bs] = 0
            target_weight = entropy * target_mask
            # print("target wegith : ",target_weight)
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            # print("weight : ",weight)

            return torch.sum(weight.view(-1, 1) * nn.BCEWithLogitsLoss(reduction='none')(domain_out, domain_label)) / torch.sum(
                weight).detach().item()
        else:
            # print("not apply entropy")
            return nn.BCEWithLogitsLoss()(domain_out, domain_label)



    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed

        # if not backprob:
            # print("current val batch x : ",input_x.size()[0])
            # print("list vad batch u ",[batch_u.size()[0] for batch_u in list_input_u])

        loss_u = 0
        temp_feat_u = []
        temp_softmax_u = []
        source_label = []
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f)
            temp_feat_u.append(temp_layer)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_u += self.cce[d](logits, y)

            softmax_u= F.softmax(logits, dim=1)
            temp_softmax_u.append(softmax_u)

            source_label.append(y)
        loss_u /= len(domain_u)

        f_target = self.TargetFeature(input_x)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.TargetClassifier(temp_layer_target)
        softmax_output_x = F.softmax(logits_target, dim=1)

        if backprob:
            loss_x = self.ce(logits_target,label_x)
        else:
            loss_x = self.val_ce(logits_target, label_x)

        temp_feat_u = torch.cat(temp_feat_u, 0)
        softmax_output_u = torch.cat(temp_softmax_u,0)
        source_label = torch.cat(source_label, 0)

        # domain_label_x = torch.ones(input_x.shape[0], 1).to(self.device)
        # domain_label_u = torch.zeros(temp_feat_u.shape[0], 1).to(self.device)

        #calculate mdd loss
        loss_mdd = mdd_loss(temp_layer_target,temp_feat_u,label_x,source_label,self.dm.num_classes,self.dm.source_domains_label_size[0])

        # global_step = self.batch_idx + self.epoch * self.num_batches
        # progress = global_step / (self.max_epoch * self.num_batches)
        # lmda = 2 / (1 + np.exp(-10 * progress)) - 1
        # lmda = lmda* self.lmda # modify the scale of lmda
        #calculate lmda factor for reversed gradient
        lmda = self.calculate_lmda_factor(self.batch_idx,self.epoch,self.num_batches,self.max_epoch,num_pretrain_epochs=self.pre_train_epochs,lmda_scale=self.lmda)
        if self.fix_lmda:
            lmda = self.lmda
        n_iter = self.epoch * self.num_batches + self.batch_idx
        self.write_scalar('train/lmda', lmda, n_iter)


        feat_x = self.revgrad(temp_layer_target, grad_scaling=lmda)
        feat_u = self.revgrad(temp_feat_u, grad_scaling=lmda)

        #calculate adversarial loss
        loss_d = self.CDAN(feat_x,softmax_output_x,feat_u,softmax_output_u,entropy=self.use_entropy,lmda = lmda)

        # output_xd = self.DomainDiscriminator(feat_x)
        # output_ud = self.DomainDiscriminator(feat_u)
        # loss_d = self.bce(output_xd, domain_label_x) + self.bce(output_ud, domain_label_u)

        total_loss = loss_x + loss_u + loss_d+self.mdd_weight*loss_mdd
        loss_summary = {
            'total_loss': total_loss.item(),
            'loss_x': loss_x.item(),
            'loss_u': loss_u.item(),
            'loss_d': loss_d.item(),
            'loss_mdd':self.mdd_weight*loss_mdd.item(),
            'lmda_factor':lmda
        }

        # print("loss x :",loss_x)
        if backprob:
            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()


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
        val_losses = losses.dict_results()
        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        for k, v in val_losses.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        return [total_loss, losses.dict_results(), results]


    def model_inference(self, input):
        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f)
        logits = self.TargetClassifier(temp_layer)
        result = F.softmax(logits, 1)
        return result
