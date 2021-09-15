import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from dassl.utils.torchtools import load_pretrained_backbone
from dassl.modeling.backbone.build import BACKBONE_REGISTRY
from dassl.modeling.backbone.backbone import Backbone
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


# do to max norm on the final dense layer
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)

class EEGNETSleep(Backbone):
    """
    I modify EEGNet to mimic the architecture of this paper for sleep model
    https://arxiv.org/pdf/1707.03321.pdf
    """
    def __init__(self, kern_legnth=64, num_ch=22, samples = 256,F1=8,F2=8,drop_prob=0.4,avg_pool_1 = 4, avg_pool_2 = 8, sep_kern_length = 16):
        super().__init__()
        self.avg_pool_1 = avg_pool_1
        self.avg_pool_2 = avg_pool_2
        self.sep_kern_length = sep_kern_length

        self.spatial = Conv2dWithConstraint(in_channels=1, out_channels=num_ch, kernel_size=(num_ch, 1), stride=1,
                                       bias=False, groups=1, padding=(0, 0), max_norm=1.0)

        # self.b1 = nn.BatchNorm2d(F1)
        self.b1 = nn.BatchNorm2d(num_ch)

        self.temporal = nn.Conv2d(in_channels=num_ch, out_channels=F1, kernel_size=(1, kern_legnth), stride=1, bias=False,
                            padding=(0, (kern_legnth // 2)))
        self.b2 = nn.BatchNorm2d(F1)
        self.d2 = nn.Dropout(drop_prob)
        samples = samples//self.avg_pool_1

        # self.c3 = nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 16), stride=1, bias=False,
        #                     groups=(nfeatl), padding=(0, 16 // 2))
        self.c3 = nn.Sequential (
            #conv_separable_depth"
            nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(1, self.sep_kern_length), stride=1, bias=False,groups=(F1), padding=(0, self.sep_kern_length//2)),
            #conv_separable_point
            nn.Conv2d(F2, F2, (1, 1), bias=False,stride=1,padding=(0, 0))
        )
        self.b3 = nn.BatchNorm2d(F2)
        self.d3 = nn.Dropout(drop_prob)
        samples = samples//self.avg_pool_2

        self._out_features = F2*samples
    def forward(self,input):

        h1 = self.b1(self.spatial(input))
        h2 = self.d2(F.avg_pool2d(F.elu(self.b2(self.temporal(h1))),(1,self.avg_pool_1)) )
        h3 = self.d3(F.avg_pool2d(F.elu(self.b3(self.c3(h2))),(1,self.avg_pool_2)) )
        flatten = torch.flatten(h3, start_dim=1)
        return flatten

@BACKBONE_REGISTRY.register()
def eegnetsleep(pretrained=False,pretrained_path = '', **kwargs):
    print("params set up : ",kwargs)
    model = EEGNETSleep(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model