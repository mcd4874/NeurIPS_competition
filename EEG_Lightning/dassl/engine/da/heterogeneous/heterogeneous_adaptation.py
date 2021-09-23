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
from dassl.modeling import build_layer


# class TemporalLayer(nn.Module):
#     def __init__(self, kern_legnth=64, num_ch=22, samples = 256,F1=8, D=2, F2=16,drop_prob=0.4):
#         super().__init__()
#         self.c3 = nn.Sequential (
#             #conv_separable_depth"
#             nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 16), stride=1, bias=False,groups=(F2), padding=(0, 8)),
#             #conv_separable_point
#             nn.Conv2d(F2, F2, (1, 1), bias=False,stride=1,padding=(0, 0))
#         )
#         self.b3 = nn.BatchNorm2d(F2, momentum=0.1, affine=True, eps=1e-3)
#         self.d3 = nn.Dropout(drop_prob)
#         self._out_features = F2*(samples//32)
#
#     def forward(self,input):
#         h3 = self.d3(F.avg_pool2d(F.elu(self.b3(self.c3(input))),(1,8)) )
#         flatten = torch.flatten(h3, start_dim=1)
#         return flatten
#
#     @property
#     def fdim(self):
#         return self._out_features


@TRAINER_REGISTRY.register()
class HeterogeneousModelAdaptation(TrainerMultiAdaptation):
    """

    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_temp_layer(self, cfg):
        embedding_layer_info = cfg.MODEL.LAYER
        layer_name = embedding_layer_info.NAME
        layer_params = embedding_layer_info.PARAMS
        return [layer_name, layer_params]

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

    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed
        loss_u = 0
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            # print("check range for source data : {} - {}".format(u.max(),u.min()))
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_u += self.cce[d](logits, y)
        # print("loss U :",loss_u)
        # print("num domain : ",len(domain_u))
        loss_u /= len(domain_u)
        # print("check range for target data : {} - {}".format(input_x.max(), input_x.min()))
        f_target = self.TargetFeature(input_x)
        temp_layer_target = self.TemporalLayer(f_target)
        logits_target = self.TargetClassifier(temp_layer_target)
        if backprob:
            loss_x = self.ce(logits_target,label_x)
        else:
            loss_x = self.val_ce(logits_target, label_x)
        total_loss = loss_x+loss_u
        loss_summary = {
            'total_loss': total_loss.item(),
            'loss_x': loss_x.item(),
            'loss_u': loss_u.item()
        }

        if backprob:
            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        return loss_summary



    def model_inference(self, input,return_feature=False):
        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f)
        logits = self.TargetClassifier(temp_layer)
        result = F.softmax(logits, 1)
        if return_feature:
            return result,temp_layer
        return result

    # @torch.no_grad()
    # def get_EasyTL(self):
    #     self.set_model_mode('eval')
    #     self.evaluator.reset()
    #     import scipy
    #     import time
    #     from dassl.engine.da.heterogeneous.easyTL import EasyTL
    #
    #     train_data = []
    #     train_label = []
    #     for batch_idx, batch in enumerate(self.train_loader_x):
    #         input_x, label_x = self.parse_batch_test(batch)
    #         _, feature_x = self.model_inference(input_x, return_feature=True)
    #
    #         train_data.append(feature_x.cpu().numpy())
    #         train_label.append(label_x.cpu().numpy())
    #     train_data = np.concatenate(train_data)
    #     train_label = np.concatenate(train_label)
    #
    #     test_data = []
    #     test_label = []
    #     for batch_idx, batch in enumerate(self.test_loader):
    #         input_x, label_x = self.parse_batch_test(batch)
    #         _, feature_x = self.model_inference(input_x, return_feature=True)
    #
    #         test_data.append(feature_x.cpu().numpy())
    #         test_label.append(label_x.cpu().numpy())
    #     test_data = np.concatenate(test_data)
    #     test_label = np.concatenate(test_label)
    #     Xs = train_data
    #     Ys = train_label
    #     Xt = test_data
    #     Yt = test_label
    #     print("original test label uniue : ", np.unique(test_label))
    #     t0 = time.time()
    #     Acc1, _ = EasyTL(Xs, Ys, Xt, Yt, 'raw')
    #     t1 = time.time()
    #     print("Time Elapsed: {:.2f} sec".format(t1 - t0))
    #     print("EasyTL(c) ACC : {:.1f} %".format(Acc1 * 100))
    #     Acc2, yt_prob = EasyTL(Xs, Ys, Xt, Yt)
    #     self.evaluator.process(yt_prob, Yt)
    #     results = self.evaluator.evaluate()
    #
    #     t2 = time.time()
    #     print("Time Elapsed: {:.2f} sec".format(t2 - t1))
    #
    #     print('EasyTL(c) Acc: {:.1f} % || EasyTL Acc: {:.1f} %'.format(Acc1 * 100, Acc2 * 100))
    #     print("EAsyTL acc results : ", results)
    #     return Acc2


    # @torch.no_grad()
    # def test(self):
    #     self.set_model_mode('eval')
    #     self.evaluator.reset()
    #     #do something
    #     #get train data
    #     # train_loader_x_iter = ForeverDataIterator(self.train_loader_x)
    #     # for self.batch_idx in range(len(train_loader_x_iter)):
    #
    #     # train_data = []
    #     # train_label = []
    #     # for batch_idx, batch in enumerate(self.train_loader_x):
    #     #     input_x = batch['eeg_data']
    #     #     label_x = batch['label']
    #     #
    #     #     train_data.append(input_x.numpy())
    #     #     train_label.append(label_x.numpy())
    #     # train_data = np.concatenate(train_data)
    #     # train_label=np.concatenate(train_label)
    #     #
    #     # test_data = []
    #     # test_label = []
    #     # for batch_idx, batch in enumerate(self.test_loader):
    #     #     input_x = batch['eeg_data']
    #     #     label_x = batch['label']
    #     #     test_data.append(input_x.numpy())
    #     #     test_label.append(label_x.numpy())
    #     # test_data = np.concatenate(test_data)
    #     # test_label = np.concatenate(test_label)
    #     #
    #     # #assume both train and test data in format (trials,1,n_channels,samples) for raw data.
    #     # #format to (trials,n_channels*samples) before use TL easy
    #     #
    #     # train_data = np.squeeze(train_data)
    #     # train_data = train_data.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2])
    #     # test_data = np.squeeze(test_data)
    #     # test_data = test_data.reshape(test_data.shape[0],test_data.shape[1]*test_data.shape[2])
    #     import scipy
    #     import time
    #     from dassl.engine.da.heterogeneous.easyTL import EasyTL
    #     # # Xs = train_data / np.tile(np.sum(train_data, axis=1).reshape(-1, 1), [1, train_data.shape[1]])
    #     # # Xs = scipy.stats.mstats.zscore(Xs)
    #     # # Xt = test_data / np.tile(np.sum(test_data, axis=1).reshape(-1, 1), [1, test_data.shape[1]])
    #     # # Xt = scipy.stats.mstats.zscore(Xt)
    #     # #
    #     # # Ys =  train_label
    #     # # Yt = test_label
    #     # #
    #     # # t0 = time.time()
    #     # # Acc1, _ = EasyTL(Xs, Ys, Xt, Yt, 'raw')
    #     # # t1 = time.time()
    #     # # print("Time Elapsed: {:.2f} sec".format(t1 - t0))
    #     # # Acc2, _ = EasyTL(Xs, Ys, Xt, Yt)
    #     # # t2 = time.time()
    #     # # print("Time Elapsed: {:.2f} sec".format(t2 - t1))
    #     # #
    #     # # print('EasyTL(c) Acc: {:.1f} % || EasyTL Acc: {:.1f} %'.format(Acc1 * 100, Acc2 * 100))
    #     #
    #     #
    #     # #try with feature representation
    #     train_data = []
    #     train_label = []
    #     for batch_idx, batch in enumerate(self.train_loader_x):
    #         input_x, label_x = self.parse_batch_test(batch)
    #         _,feature_x = self.model_inference(input_x,return_feature=True)
    #
    #         train_data.append(feature_x.cpu().numpy())
    #         train_label.append(label_x.cpu().numpy())
    #     train_data = np.concatenate(train_data)
    #     train_label = np.concatenate(train_label)
    #
    #     test_data = []
    #     test_label = []
    #     for batch_idx, batch in enumerate(self.test_loader):
    #         input_x, label_x = self.parse_batch_test(batch)
    #         _, feature_x = self.model_inference(input_x, return_feature=True)
    #
    #         test_data.append(feature_x.cpu().numpy())
    #         test_label.append(label_x.cpu().numpy())
    #     test_data = np.concatenate(test_data)
    #     test_label = np.concatenate(test_label)
    #     Xs = train_data
    #     Ys = train_label
    #     Xt = test_data
    #     Yt = test_label
    #     print("original test label uniue : ",np.unique(test_label))
    #     t0 = time.time()
    #     Acc1, _ = EasyTL(Xs, Ys, Xt, Yt, 'raw')
    #     t1 = time.time()
    #     print("Time Elapsed: {:.2f} sec".format(t1 - t0))
    #     print("EasyTL(c) ACC : {:.1f} %".format(Acc1 * 100))
    #     Acc2, yt_prob = EasyTL(Xs, Ys, Xt, Yt)
    #     self.evaluator.process(yt_prob, Yt)
    #     results = self.evaluator.evaluate()
    #
    #
    #     t2 = time.time()
    #     print("Time Elapsed: {:.2f} sec".format(t2 - t1))
    #
    #     print('EasyTL(c) Acc: {:.1f} % || EasyTL Acc: {:.1f} %'.format(Acc1 * 100, Acc2 * 100))
    #     print("EAsyTL acc results : ",results)
    #     # print("yt_pred size : ",yt_pred.shape)
    #     # print("check EasyTL acc 1 ",np.mean(yt_pred == Yt.flatten()))
    #     # print("unique yt_pred ",np.unique(yt_pred))
    #     # print("unique Yt ",np.unique(Yt))
    #
    #     # print("check EasyTL acc 2 ",np.mean((yt_pred-1) == (Yt).flatten()))
    #
    #
    #     #get test data
    #     result =  super().test()
    #     result["TL_accuracy"] = Acc2
    #     return result
    #     # """A generic testing pipeline."""
    #     # self.set_model_mode('eval')
    #     # self.evaluator.reset()
    #     #
    #     # split = self.cfg.TEST.SPLIT
    #     # print('Do evaluation on {} set'.format(split))
    #     # # data_loader = self.val_loader if split == 'val' else self.test_loader
    #     # data_loader = self.test_loader
    #     # assert data_loader is not None
    #     #
    #     # for batch_idx, batch in enumerate(data_loader):
    #     #     input, label = self.parse_batch_test(batch)
    #     #     output = self.model_inference(input)
    #     #     self.evaluator.process(output, label)
    #     #
    #     # results = self.evaluator.evaluate()
    #     #
    #     # for k, v in results.items():
    #     #     tag = '{}/{}'.format(split, k)
    #     #     self.write_scalar(tag, v, self.epoch)
    #     #
    #     # #run
    #     #
    #     # return results


