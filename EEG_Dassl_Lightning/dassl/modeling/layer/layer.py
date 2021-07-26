import functools
import torch.nn as nn
import torch
from torch.nn import functional as F
from .build import LAYER_REGISTRY

class TemporalLayer(nn.Module):
    def __init__(self,  kern_legnth=64, num_ch=22, samples = 256,F1=8, D=2, F2=16,drop_prob=0.4,avg_pool_1 = 4, avg_pool_2 = 8, sep_kern_length = 16):
        super().__init__()
        self.avg_pool_2 = avg_pool_2
        self.sep_kern_length = sep_kern_length
        self.c3 = nn.Sequential (
            #conv_separable_depth"
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, self.sep_kern_length), stride=1, bias=False,groups=(F2), padding=(0, self.sep_kern_length//2)),
            #conv_separable_point
            nn.Conv2d(F2, F2, (1, 1), bias=False,stride=1,padding=(0, 0))
        )
        self.b3 = nn.BatchNorm2d(F2, momentum=0.1, affine=True, eps=1e-5)
        self.d3 = nn.Dropout(drop_prob)
        self._out_features = F2*(samples//(avg_pool_1*avg_pool_2))

    def forward(self,input):
        h3 = self.d3(F.avg_pool2d(F.elu(self.b3(self.c3(input))),(1,self.avg_pool_2)) )
        flatten = torch.flatten(h3, start_dim=1)
        return flatten

    @property
    def fdim(self):
        return self._out_features

class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        # print("current domain label : ",domain_label)
        bn = self.bns[domain_label]
        return bn(x)


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class DomainBatchTemporalLayer(nn.Module):
    def __init__(self,  kern_legnth=64, num_ch=22, samples = 256,F1=8, D=2, F2=16,drop_prob=0.4,avg_pool_1 = 4, avg_pool_2 = 8, sep_kern_length = 16,total_domain=2):
        super().__init__()
        self.avg_pool_2 = avg_pool_2
        self.sep_kern_length = sep_kern_length
        self.c3 = nn.Sequential (
            #conv_separable_depth"
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, self.sep_kern_length), stride=1, bias=False,groups=(F2), padding=(0, self.sep_kern_length//2)),
            #conv_separable_point
            nn.Conv2d(F2, F2, (1, 1), bias=False,stride=1,padding=(0, 0))
        )
        self.domain_batch_norm = DomainSpecificBatchNorm2d(num_features=F2,num_classes=total_domain,momentum=0.1,affine=True)
        # self.b3 = nn.BatchNorm2d(F2, momentum=0.1, affine=True, eps=1e-5)
        self.d3 = nn.Dropout(drop_prob)
        self._out_features = F2*(samples//(avg_pool_1*avg_pool_2))

    def forward(self,input,domain_label):
        conv = self.c3(input)
        batch_norm = self.domain_batch_norm(conv,domain_label)
        h3 = self.d3(F.avg_pool2d(F.elu(batch_norm),(1,self.avg_pool_2)) )
        flatten = torch.flatten(h3, start_dim=1)
        return flatten

    @property
    def fdim(self):
        return self._out_features


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

class LogVarTemporalLayer(nn.Module):
    def __init__(self,  kern_legnth=64, num_ch=22, samples = 256,F1=8, D=2, F2=16,drop_prob=0.4,avg_pool_1 = 4, avg_pool_2 = 8, sep_kern_length = 16):
        super().__init__()
        self.avg_pool_2 = avg_pool_2
        self.sep_kern_length = sep_kern_length
        # input_dim_1 = (1, 1, num_ch, samples)
        # pad_h, pad_w = calculate_pad(input_dim_1, k=(1, kern_legnth))
        input_dim_2 = (1, F2, 1, samples)
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
        self.b3 = nn.BatchNorm2d(F2)
        # self.p3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.p3 = VarPool(pool_size=(avg_pool_2 * avg_pool_1))
        self.d3 = nn.Dropout(drop_prob)
        samples = samples // (avg_pool_2 *avg_pool_1)

        self._out_features = F2 * samples

    def forward(self,input):
        h3 = self.d3(self.p3(F.elu(self.b3(self.c3(input)))))
        flatten = torch.flatten(h3, start_dim=1)
        return flatten

    @property
    def fdim(self):
        return self._out_features

@LAYER_REGISTRY.register()
def EEGNetConv3(**kwargs):
    return TemporalLayer(**kwargs)

@LAYER_REGISTRY.register()
def EEGNetConv3DSBN(**kwargs):
    return DomainBatchTemporalLayer(**kwargs)
@LAYER_REGISTRY.register()
def VarConv3(**kwargs):
    return LogVarTemporalLayer(**kwargs)