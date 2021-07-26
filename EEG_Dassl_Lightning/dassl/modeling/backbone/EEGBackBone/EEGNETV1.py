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



        # return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))


class EEGNETV1(Backbone):
    def __init__(self, kern_legnth=64, num_ch=22, samples = 256,F1=8, D=2, F2=16,drop_prob=0.4,avg_pool_1 = 4, avg_pool_2 = 8, sep_kern_length = 16):
        super().__init__()
        self.avg_pool_1 = avg_pool_1
        self.avg_pool_2 = avg_pool_2
        self.sep_kern_length = sep_kern_length
        # print("kern pad : ",kern_legnth//2)
        # print("round kern pad : ",int(kern_legnth//2))
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
        self.p2 = VarPool(pool_size=avg_pool_1)
        self.d2 = nn.Dropout(drop_prob)
        # samples = samples//4
        samples = samples//self.avg_pool_1

        # self.c3 = nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 16), stride=1, bias=False,
        #                     groups=(nfeatl), padding=(0, 16 // 2))
        input_dim_2 = (1, 1, num_ch, samples)
        pad_h, pad_w = calculate_pad(input_dim_2, k=(1, sep_kern_length))
        if pad_h % 2 == 1 or pad_w % 2 == 1:
            self.c3 = nn.Sequential(
                # conv_separable_depth"
                Conv2dSame(in_channels=F2, out_channels=F2, kernel_size=(1, self.sep_kern_length), bias=False,
                           groups=(F2), padding=(pad_h, pad_w)),
                # conv_separable_point
                nn.Conv2d(F2, F2, (1, 1), bias=False, padding=(0, 0))
            )
        else:
            # conv_separable_depth"
            self.c3 = nn.Sequential(
                nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, self.sep_kern_length), stride=1, bias=False,
                          groups=(F2), padding=(0, self.sep_kern_length // 2)),
                # conv_separable_point
                nn.Conv2d(F2, F2, (1, 1), bias=False, padding=(0, 0))
            )
        # self.b3 = nn.BatchNorm2d(F1 * D)
        self.b3 = nn.BatchNorm2d(F2, momentum=0.1, affine=True, eps=1e-3)
        # self.p3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.p3 = VarPool(pool_size=avg_pool_2)
        self.d3 = nn.Dropout(drop_prob)
        # samples = samples//8
        samples = samples//self.avg_pool_2

        self._out_features = F2*samples
        # self.classifier = nn.Linear(nfeatr, nb_classes)
    def forward(self,input):
        # print("input shape : ",input.shape)
        h1 = self.b1(self.c1(input))
        # print("h1 shape : ",h1.shape)
        tmp_2 = F.elu(self.b2(self.c2(h1)))
        # print("tmp shape : ",tmp_2.shape)
        h2 = self.d2( self.p2(tmp_2))
        # print("h2 shape : ",h2.shape)
        tmp_3 = F.elu(self.b3(self.c3(h2)))
        h3 = self.d3(self.p3(tmp_3))
        flatten = torch.flatten(h3, start_dim=1)
        return flatten

@BACKBONE_REGISTRY.register()
def eegnet_v1(pretrained=False,pretrained_path = '', **kwargs):
    print("params set up : ",kwargs)
    model = EEGNETV1(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model