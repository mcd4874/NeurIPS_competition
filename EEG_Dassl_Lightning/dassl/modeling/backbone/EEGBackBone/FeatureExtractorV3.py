import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from dassl.utils.torchtools import load_pretrained_backbone
from dassl.modeling.backbone.build import BACKBONE_REGISTRY
from dassl.modeling.backbone.backbone import Backbone
import math
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def calculate_pad(x,k,s=(1,1),d=(1,1)):
    ih, iw = x[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    return pad_h,pad_w

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
        specific for EEGNet (1,kernel). Not guarantee for other work
    """

    def __init__(self, in_channels, out_channels,kernel_size,stride=1,padding=(0,0),dilation=1, groups=1, bias=False):
        #force to use default stride=1 and dilation=1
        self.pad_h, self.pad_w = padding
        super(Conv2dSame, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,groups=groups,bias=bias,
                                padding=(0, 0))


    def forward(self, x):
        x = F.pad(x, [self.pad_w // 2, self.pad_w - self.pad_w // 2, self.pad_h // 2, self.pad_h - self.pad_h // 2])
        # print("pad shape : ",x.shape)
        # print("pad x : ",x)
        result = super(Conv2dSame, self).forward(x)
        # print("result shape : ",result.shape)
        return result
        # return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


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
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class VarPool(nn.Module):

    def __init__(self,pool_size,var_type='LogVarLayer',axis=3):
        super(VarPool, self).__init__()
        # self.input_dim = int(input_dim)
        self.pool_size = int(pool_size)

        # self.strideFactor = int(self.input_dim//self.pool_size)
        if var_type=='LogVarLayer':
            self.temporalLayer = LogVarLayer(dim=axis)
        else:
            self.temporalLayer = VarLayer(dim=axis)


    def forward(self, x):
        """
        assume x has shape (n_batch,n_filter,1,samples)
        """
        # print("init x shape : ",x.shape)
        x = x.reshape([*x.shape[0:2], int(x.shape[3]//self.pool_size), self.pool_size])
        x = self.temporalLayer(x)
        x = x.reshape([*x.shape[0:2],1,x.shape[2]])
        return x

class EXTRACTORV3(Backbone):
    def __init__(self, kern_legnth=64, num_ch=22, samples = 256,F1=8, D=2, F2=16,drop_prob=0.4,avg_pool_1 = 4, avg_pool_2 = 8, sep_kern_length = 16):
        super().__init__()
        self.avg_pool_1 = avg_pool_1
        self.avg_pool_2 = avg_pool_2
        self.sep_kern_length = sep_kern_length
        input_dim_1 = (1, 1, num_ch, samples)
        pad_h, pad_w = calculate_pad(input_dim_1, k=(1, kern_legnth))
        if pad_h % 2 == 1 or pad_w % 2 == 1:
            print("use conv2d Same for even kern size")
            self.c1 = Conv2dSame(in_channels=1, out_channels=F1, kernel_size=(1, kern_legnth), bias=False,
                                 padding=(pad_h, pad_w))
        else:
            print("use norm conv2d")
            self.c1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kern_legnth), bias=False,
                                padding=(0, (kern_legnth // 2)))
        # self.b1 = nn.BatchNorm2d(F1)
        self.b1 = nn.BatchNorm2d(F1)

        # self.c2 = Conv2dWithConstraint(in_channels=F1, out_channels=F1 * D, kernel_size=(num_ch, 1), stride=1,
        #                                bias=False, groups=F1, padding=(0, 0), max_norm=0.5)
        self.c2 = Conv2dWithConstraint(in_channels=F1, out_channels=F1 * D, kernel_size=(num_ch, 1),
                                       bias=False, groups=F1, padding=(0, 0), max_norm=1.0)
        # self.b2 = nn.BatchNorm2d(F1 * D)
        self.b2 = nn.BatchNorm2d(F2)
        # self.p2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        # self.p2 = VarPool(pool_size=avg_pool_1)
        self.d2 = nn.Dropout(drop_prob)
        # samples = samples//self.avg_pool_1
        self._out_features = (F2,1,samples)

    def forward(self,input):
        h1 = self.b1(self.c1(input))
        h2 = self.d2(F.elu(self.b2(self.c2(h1))) )
        return h2

@BACKBONE_REGISTRY.register()
def extractorv3(pretrained=False,pretrained_path = '', **kwargs):
    model = EXTRACTORV3(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model