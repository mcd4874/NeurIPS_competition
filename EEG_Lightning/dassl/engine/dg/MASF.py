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
from pytorch_metric_learning import miners, losses
import numpy as np
@TRAINER_REGISTRY.register()
class MASF(TrainerX):
    """
    Domain Generalization via Model-Agnostic Learning of Semantic Features
    https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, cfg):

        self.beta1 = cfg.TRAINER.MASF.beta1
        self.beta2 = cfg.TRAINER.MASF.beta2
        self.inter_lr = cfg.TRAINER.MASF.inner_lr  # 0.001

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

        self.val_ce = nn.CrossEntropyLoss()

        self.miner = miners.MultiSimilarityMiner()
        self.triplet_loss_func = losses.TripletMarginLoss()

        self.candidates = np.arange(self.n_domain)
        index = np.random.permutation(self.candidates)
        self.meta_train_idx = index[1:3]
        self.meta_test_idx = index[:1]
        # self.alpha = 0.1 #within_subject

        # self.alpha = 1.0 #within_subject_1






    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == 'RandomDomainSampler'
    def build_model(self):
        cfg = self.cfg

        print('Building Feature')
        self.Feature  = SimpleNet(cfg, cfg.MODEL, 0, **cfg.MODEL.BACKBONE.PARAMS)
        self.Feature.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.Feature)))
        fdim = self.Feature.fdim


        print('Building Feature MAML')
        # fast_lr = 0.001
        self.feature_maml = l2l.algorithms.MAML(self.Feature, lr=self.inter_lr)
        self.feature_maml.to(self.device)


        self.optim_feature_maml = build_optimizer(self.feature_maml, cfg.OPTIM)
        self.sched_feature_maml = build_lr_scheduler(self.optim_feature_maml, cfg.OPTIM)
        self.register_model('feature_maml', self.feature_maml, self.optim_feature_maml, self.sched_feature_maml)

        self.classifier = nn.Linear(fdim, self.num_classes)

        print('Building Classifier MAML')
        # fast_lr = 0.001
        self.classifier_maml = l2l.algorithms.MAML(self.classifier, lr=self.inter_lr)
        self.classifier_maml.to(self.device)

        self.optim_classifier_maml = build_optimizer(self.classifier_maml, cfg.OPTIM)
        self.sched_classifier_maml = build_lr_scheduler(self.optim_classifier_maml, cfg.OPTIM)
        self.register_model('classifier_maml', self.classifier_maml, self.optim_classifier_maml, self.sched_classifier_maml)


        self.embedding = nn.Linear(fdim, int(fdim//2))
        self.embedding.to(self.device)
        self.optim_embedding = build_optimizer(self.embedding, cfg.OPTIM)
        self.sched_embedding = build_lr_scheduler(self.optim_embedding, cfg.OPTIM)
        self.register_model('embedding', self.embedding, self.optim_embedding, self.sched_embedding)

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


    # def generate_b

    def generate_soft_label_dist(self,input,label,cls,eps=1e-16,temp = 1.5):
        mask = (label == float(cls)).float()
        mask = torch.unsqueeze(mask, dim=-1)
        product = torch.mul(input, mask)
        logits_sum = torch.sum(product, dim=0)
        print("mask : ",mask)
        print("logit sum : ",logits_sum)
        num1 = torch.sum(mask, dim=0)
        average_feature_cls = logits_sum * 1.0 / (num1 + eps)  # add eps for prevent un-sampled class resulting in NAN
        # soft_prob_cls = F.softmax(average_feature_cls/temp, dim=-1)
        soft_prob_cls = F.softmax(average_feature_cls/temp)
        soft_prob_cls = torch.clamp(soft_prob_cls, min=1e-8, max=1.0)

        return soft_prob_cls

    def global_loss_func(self,batch_a,batch_b):
        input_a,label_a = batch_a
        input_b, label_b = batch_b

        n_classes = self.num_classes
        kd_loss = 0.0
        eps = 1e-16
        temp = 1.5

        count_classes = 0

        #still need to implement bool case where we have labels in 1 set but not the other
        for cls in range(n_classes):
            print("current cls : ",cls)
            mask_a = (label_a == float(cls)).float()
            mask_a = torch.unsqueeze(mask_a, dim=-1)
            num_a = torch.sum(mask_a, dim=0)

            mask_b = (label_b == float(cls)).float()
            mask_b = torch.unsqueeze(mask_b, dim=-1)
            num_b = torch.sum(mask_b, dim=0)

            if num_a>0.0 and num_b>0.0:
                count_classes+=1
                soft_prob_cls_a = self.generate_soft_label_dist(input_a,label_a,cls,eps,temp)
                print("softmax a : ", soft_prob_cls_a)
                print("num a : ", num_a)
                soft_prob_cls_b = self.generate_soft_label_dist(input_b, label_b, cls,eps, temp)
                print("softmax b : ", soft_prob_cls_b)
                print("num b : ",num_b)

                KL_a_b = torch.sum(soft_prob_cls_a*torch.log(soft_prob_cls_a/soft_prob_cls_b))

                # KL_a_b = F.kl_div(soft_prob_cls_a, soft_prob_cls_b)
                print("KL a b :",KL_a_b)
                KL_b_a = torch.sum(soft_prob_cls_b*torch.log(soft_prob_cls_b/soft_prob_cls_a))
                # KL_b_a = F.kl_div(soft_prob_cls_b, soft_prob_cls_a)
                print("KL b a :",KL_b_a)

                KL_avg_loss = (KL_a_b+KL_b_a)/2
                kd_loss += KL_avg_loss

            else:
                print("missing label in one domain")
        kd_loss /=count_classes

        return kd_loss

    # def generate_soft_label_dist(self,input,label,cls,classifier_learner,eps=1e-16,temp = 1.5):
    #     mask = (label == float(cls)).float()
    #     mask = torch.unsqueeze(mask, dim=-1)
    #     product = torch.mul(input, mask)
    #     logits_sum = torch.sum(product, dim=0)
    #     # print("logit sum : ",logits_sum)
    #     num1 = torch.sum(mask, dim=0)
    #     average_feature_cls = logits_sum * 1.0 / (num1 + eps)  # add eps for prevent un-sampled class resulting in NAN
    #     soft_logit_cls = classifier_learner(average_feature_cls)
    #     soft_prob_cls = F.softmax(soft_logit_cls/temp, dim=-1)
    #     soft_prob_cls = torch.clamp(soft_prob_cls, min=1e-8, max=1.0)
    #
    #     return soft_prob_cls
    #
    # def global_loss_func(self,batch_a,batch_b,classifier_learner):
    #     input_a,label_a = batch_a
    #     input_b, label_b = batch_b
    #
    #     n_classes = self.num_classes
    #     kd_loss = 0.0
    #     eps = 1e-16
    #     temp = 2.0
    #
    #     count_classes = 0
    #
    #     #still need to implement bool case where we have labels in 1 set but not the other
    #     for cls in range(n_classes):
    #         mask_a = (label_a == float(cls)).float()
    #         mask_a = torch.unsqueeze(mask_a, dim=-1)
    #         num_a = torch.sum(mask_a, dim=0)
    #
    #         mask_b = (label_b == float(cls)).float()
    #         mask_b = torch.unsqueeze(mask_b, dim=-1)
    #         num_b = torch.sum(mask_b, dim=0)
    #
    #         if num_a>0.0 and num_b>0.0:
    #             count_classes+=1
    #             soft_prob_cls_a = self.generate_soft_label_dist(input_a,label_a,cls,classifier_learner,eps,temp)
    #             soft_prob_cls_b = self.generate_soft_label_dist(input_b, label_b, cls, classifier_learner,eps, temp)
    #
    #             print("softmax a : ",soft_prob_cls_a)
    #             print("num a : ",num_a)
    #             print("softmax b : ", soft_prob_cls_b)
    #             print("num b : ",num_b)
    #
    #             KL_a_b = torch.sum(soft_prob_cls_a*torch.log(soft_prob_cls_a/soft_prob_cls_b))
    #
    #             # KL_a_b = F.kl_div(soft_prob_cls_a, soft_prob_cls_b)
    #             print("KL a b :",KL_a_b)
    #             KL_b_a = torch.sum(soft_prob_cls_b*torch.log(soft_prob_cls_b/soft_prob_cls_a))
    #             # KL_b_a = F.kl_div(soft_prob_cls_b, soft_prob_cls_a)
    #             print("KL b a :",KL_b_a)
    #
    #             KL_avg_loss = (KL_a_b+KL_b_a)/2
    #             kd_loss += KL_avg_loss
    #
    #         else:
    #             print("missing label in one domain")
    #     kd_loss /=count_classes
    #
    #     return kd_loss

    # def local_loss_func(self):

    def forward_backward(self, batch,backprob = True):
        input_x, label_x, domain_x = self.parse_batch_train(batch)
        feature_learner = self.feature_maml.clone()
        classifier_learner = self.classifier_maml.clone()
        #check model and maml params

#

        meta_train_loss = 0.0
        global_loss = 0.0

        #clone maml

        if backprob:
            # train Domain Specific model
            input_x = torch.split(input_x, self.split_batch, 0)
            label_x = torch.split(label_x, self.split_batch, 0)
            domain_x = torch.split(domain_x, self.split_batch, 0)
            d_x = [d[0].item() for d in domain_x]

            # print('meta train : ',self.meta_train_idx)
            # print('meta test : ',self.meta_test_idx)
            # print(len(input_x))
            # meta_train_x = input_x[self.meta_train_idx]
            # meta_train_label_x = label_x[self.meta_train_idx]
            # d_x = d_x[self.meta_train_idx]

            # meta_test_x = input_x[self.meta_test_idx]
            # meta_test_label_x = label_x[self.meta_test_idx]

            for domain in self.meta_train_idx:
                x,y = input_x[domain],label_x[domain]
                feat = feature_learner(x)
                logits = classifier_learner(feat)
                loss = self.ce(logits, y)
                meta_train_loss += loss


            self.model_zero_grad(['feature_maml','classifier_maml','embedding'])

            meta_train_loss /= (len(self.meta_train_idx))
            # print("equal model and clone maml 1 ", self.check_equal_model(self.model, learner))
            # print("equal model and maml 1 ",self.check_equal_model(self.model,self.maml))
            # print("equal maml and clone maml 1 ", self.check_equal_model(self.maml, learner))

            # print("equal grad model and clone maml 1 ", self.check_equal_grad(self.model, learner))
            # print("equal grad model and maml 1 ", self.check_equal_grad(self.model, self.maml))
            # print("equal grad maml and clone maml 1 ", self.check_equal_grad(self.model, self.maml))

            # self.check_model_params([["model",self.model],["maml",self.maml],["clone",learner]])

            classifier_learner.adapt(meta_train_loss)
            feature_learner.adapt(meta_train_loss)

            # print("equal model and clone maml 2 ", self.check_equal_model(self.model, learner))
            # print("equal model and maml 2 ", self.check_equal_model(self.model, self.maml))
            # print("equal maml and clone maml 2 ", self.check_equal_model(self.maml, learner))

            # print("equal grad model and clone maml 2 ", self.check_equal_grad(self.model, learner))
            # print("equal grad model and maml 2 ", self.check_equal_grad(self.model, self.maml))
            # print("equal grad maml and clone maml 2 ", self.check_equal_grad(self.maml, learner))

            # self.check_model_params([["model", self.model], ["maml", self.maml], ["clone", learner]])
            #meta test stage
            full_feat = []
            full_label = []
            #global loss
            meta_test_x = input_x[self.meta_test_idx[0]]
            meta_test_label_x = label_x[self.meta_test_idx[0]]
            feat_b = feature_learner(meta_test_x)

            logit_b = classifier_learner(feat_b)

            full_feat.append(feat_b)
            full_label.append(meta_test_label_x)

            for domain in self.meta_train_idx:
                x,y = input_x[domain],label_x[domain]
                feat_a =feature_learner(x)
                # global_loss_b_a = self.global_loss_func((feat_a,y),(feat_b,meta_test_label_x),classifier_learner)
                logit_a = classifier_learner(feat_a)
                global_loss_b_a = self.global_loss_func((logit_a,y),(logit_b,meta_test_label_x))
                global_loss+= global_loss_b_a

                full_feat.append(feat_a)
                full_label.append(y)

            global_loss /= len(self.meta_train_idx)

            print(global_loss)

            #local loss
            full_feat = torch.cat(full_feat,dim=0)
            full_label = torch.cat(full_label,dim=0)
            embedding_feat = self.embedding(full_feat)

            hard_pairs = self.miner(embedding_feat,full_label)
            local_loss = self.triplet_loss_func(embedding_feat,full_label,hard_pairs)

            meta_test_loss = self.beta1*global_loss + self.beta2*local_loss

            final_loss =  meta_train_loss+meta_test_loss


            self.model_backward_and_update(final_loss, ['feature_maml','classifier_maml','embedding'])
            self.model_update(['feature_maml','classifier_maml','embedding'])



            # print("equal model and clone maml 3 ", self.check_equal_model(self.model, learner))
            # print("equal model and maml 3 ", self.check_equal_model(self.model, self.maml))
            # print("equal maml and clone maml 3 ", self.check_equal_model(self.maml, learner))
            #
            # print("equal grad model and clone maml 3 ", self.check_equal_grad(self.model, learner))
            # print("equal grad model and maml 3 ", self.check_equal_grad(self.model, self.maml))
            # print("equal grad maml and clone maml 3 ", self.check_equal_grad(self.model, self.maml))


            # self.check_model_params([["model", self.model], ["maml", self.maml], ["clone", learner]])
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr( ['feature_maml','classifier_maml'])

                index = np.random.permutation(self.candidates)
                self.meta_train_idx = index[1:3]
                self.meta_test_idx = index[:1]
                print("update meta test to be subject : {}".format(self.meta_test_idx))

            loss_summary = {
                'meta_train_loss': meta_train_loss.item(),
                'global_loss':global_loss.item(),
                'local_loss':local_loss.item(),
                'meta_test_loss': meta_test_loss.item(),
                'total_loss': final_loss.item()
            }
        else:
            feat = self.feature_maml(input_x)
            logits = self.classifier_maml(feat)
            val_loss = self.val_ce(logits,label_x)

            loss_summary = {
                'meta_train_loss': 0,
                'meta_test_loss': 0,
                'total_loss': val_loss.item()
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
        total_loss = losses.meters['total_loss'].avg

        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        return [total_loss,losses.dict_results(),results]




    def model_inference(self, input):
        feat = self.feature_maml(input)
        logits = self.classifier_maml(feat)
        probs = F.softmax(logits, dim=1)
        return probs





