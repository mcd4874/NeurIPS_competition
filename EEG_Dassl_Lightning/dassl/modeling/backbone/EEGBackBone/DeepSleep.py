import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from dassl.utils.torchtools import load_pretrained_backbone
from dassl.modeling.backbone.build import BACKBONE_REGISTRY
from dassl.modeling.backbone.backbone import Backbone
# class Conv2dWithConstraint(nn.Conv2d):
#     def __init__(self, *args, max_norm=1.0, **kwargs):
#         self.max_norm = max_norm
#         super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
#     def forward(self, x):
#         self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
#         return super(Conv2dWithConstraint, self).forward(x)
#
#
# # do to max norm on the final dense layer
# class LinearWithConstraint(nn.Linear):
#     def __init__(self, *args, max_norm=0.25, **kwargs):
#         self.max_norm = max_norm
#         super(LinearWithConstraint, self).__init__(*args, **kwargs)
#
#     def forward(self, x):
#         self.weight.data = torch.renorm(
#             self.weight.data, p=2, dim=0, maxnorm=self.max_norm
#         )
#         return super(LinearWithConstraint, self).forward(x)

class EEGNETSleep(Backbone):
    """
    I implement deep sleep model from
    https://arxiv.org/pdf/1707.03321.pdf
    """
    def __init__(self, kern_length_1=64, num_ch=22, samples = 256,F1=8,F2=8,drop_prob=0.4,avg_pool_1 = 4, avg_pool_2 = 8, kern_length_2 = 16):
        super().__init__()
        self.avg_pool_1 = avg_pool_1
        self.avg_pool_2 = avg_pool_2
        self.kern_length_1 = kern_length_1
        self.kern_length_2 = kern_length_2

        self.spatial = nn.Conv2d(in_channels=1, out_channels=num_ch, kernel_size=(num_ch, 1), stride=1,
                                       bias=False, padding=(0, 0))

        self.temporal_1 = nn.Conv2d(in_channels=num_ch, out_channels=F1, kernel_size=(1, kern_length_1), stride=1, bias=False,
                            padding=(0, (kern_length_1 // 2)))

        self.temporal_2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(1, kern_length_2), stride=1, bias=False,
                            padding=(0, (kern_length_2 // 2)))
        # self.b1 = nn.BatchNorm2d(F1)
        self.b1 = nn.BatchNorm2d(num_ch)
        self.b2 = nn.BatchNorm2d(F1)
        samples = samples//self.avg_pool_1

        self.b3 = nn.BatchNorm2d(F2)
        self.d3 = nn.Dropout(drop_prob)
        samples = samples//self.avg_pool_2

        self._out_features = F2*samples
    def forward(self,input):

        h1 = self.b1(self.spatial(input))
        h2 = F.avg_pool2d(F.relu(self.b2(self.temporal_1(h1))),(1,self.avg_pool_1))
        h3 = self.d3(F.avg_pool2d(F.relu(self.b3(self.temporal_2(h2))),(1,self.avg_pool_2)) )
        flatten = torch.flatten(h3, start_dim=1)
        return flatten

@BACKBONE_REGISTRY.register()
def deepsleep(pretrained=False,pretrained_path = '', **kwargs):
    print("params set up : ",kwargs)
    model = EEGNETSleep(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model