import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from dassl.utils.torchtools import load_pretrained_backbone
from dassl.modeling.backbone.build import BACKBONE_REGISTRY
from dassl.modeling.backbone.backbone import Backbone
import math
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class DeepConvNet(Backbone):

    def __init__(self, F1 = 25, F2 = 25,F3 = 50, F4 = 100, F5 = 200,kern_legnth = 5,Chans = 64, Samples = 128, dropoutRate = 0.5,momentum = 0.1,eps = 1e-05,pooling_length = 2, pool_stride = 2):
        super().__init__()
        self.block1 =  nn.Sequential(
            Conv2dWithConstraint(in_channels=1,out_channels=F1,kernel_size = (1,kern_legnth),stride=1,bias=False,max_norm=2.0),
            Conv2dWithConstraint(in_channels=F1, out_channels=F2, kernel_size = (Chans, 1), stride=1, bias=False,max_norm=2.0),
            nn.BatchNorm2d(F2, momentum=momentum, eps=eps),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pooling_length), stride=(1, pool_stride)),
            nn.Dropout(dropoutRate)
        )

        w = int((((Samples -(kern_legnth)+1) - pooling_length)/pool_stride +1))
        # print("w : ",w)
        self.block2 = nn.Sequential(
            Conv2dWithConstraint(in_channels=F2,out_channels=F3,kernel_size=(1,kern_legnth),max_norm=2.0),
            nn.BatchNorm2d(F3, momentum=momentum, eps=eps),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pooling_length), stride=(1, pool_stride)),
            nn.Dropout(dropoutRate)
        )
        w1 = int((((w -(kern_legnth)+1) - pooling_length)/pool_stride +1))
        print("w1 : ",w1)

        self.block3 = nn.Sequential(
            Conv2dWithConstraint(in_channels=F3,out_channels=F4, kernel_size=(1, kern_legnth), max_norm=2.0),
            nn.BatchNorm2d(F4, momentum=momentum, eps=eps),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pooling_length), stride=(1, pool_stride)),
            nn.Dropout(dropoutRate)
        )
        w2 = math.floor((((w1 -(kern_legnth)+1) - pooling_length)/pool_stride +1))
        # print("w2 : ",w2)

        self.block4 = nn.Sequential(
            Conv2dWithConstraint(in_channels=F4,out_channels=F5, kernel_size=(1, kern_legnth), max_norm=2.0),
            nn.BatchNorm2d(F5, momentum=momentum, eps=eps),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pooling_length), stride=(1, pool_stride)),
            nn.Dropout(dropoutRate)
        )

        w3 = math.floor((((w2 -(kern_legnth)+1) - pooling_length)/pool_stride +1))
        # print("w3 : ",w3)

        self._out_features = w3*F5
        # print("output feature shape : ",self._out_features)

        self.flatten = nn.Flatten()
    def forward(self,input):
        h1 = self.block1(input)
        # print("h1 shape : ",h1.shape)
        h2 = self.block2(h1)
        # print("h2 shape : ",h2.shape)
        h3 = self.block3(h2)
        # print("h3 shape : ",h3.shape)
        h4 = self.block4(h3)
        # print("h4 shape : ",h4.shape)
        flatten =  self.flatten(h4)
        # print("flatten shape : ",flatten.shape)
        return flatten
@BACKBONE_REGISTRY.register()
def deepconvnet(pretrained=False,pretrained_path = '', **kwargs):
    model = DeepConvNet(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model