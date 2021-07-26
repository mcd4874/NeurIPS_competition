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

class ShallowConvNet(Backbone):
    def __init__(self,F1 = 40, F2 = 40,kern_legnth = 31,Chans = 64, Samples = 128, dropoutRate = 0.5,momentum = 0.1,eps = 1e-05,pooling_length = 35, pool_stride = 7):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kern_legnth), stride=1, bias=False)
        w1 = Samples -(kern_legnth-1)
        self.c2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(Chans, 1), stride=1, bias=False)
        w2 = w1
        self.b1 = nn.BatchNorm2d(F2,momentum=momentum,eps=eps)
        self.pool = nn.AvgPool2d(kernel_size=(1,pooling_length),stride=(1,pool_stride))
        self.drop = nn.Dropout(dropoutRate)

        self._out_features = math.ceil((w2-pooling_length)/pool_stride) * F2


    def forward(self,input):
        h1 = self.c2(self.c1(input))
        h2 = torch.square(self.b1(h1))
        h3 = torch.log(torch.clamp(self.pool(h2),min = 1e-7,max=100000))
        d =  self.drop(h3)
        flatten = torch.flatten(d, start_dim=1)
        return flatten

@BACKBONE_REGISTRY.register()
def shallowConvNet(pretrained=False,pretrained_path = '', **kwargs):
    model = ShallowConvNet(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model