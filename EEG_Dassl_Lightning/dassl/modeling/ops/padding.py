""" Padding Helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from typing import List, Tuple

import torch.nn.functional as F


# Calculate symmetric padding for a convolution
# def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
#     padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
#     return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# # Can SAME padding for given args be done statically?
# def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
#     return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0
#
#
# # Dynamically pad input x with 'SAME' padding for conv with specified args
# def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
#     ih, iw = x.size()[-2:]
#     pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
#     if pad_h > 0 or pad_w > 0:
#         x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
#     return x
#
#
# def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
#     dynamic = False
#     if isinstance(padding, str):
#         # for any string padding, the padding will be calculated for you, one of three ways
#         padding = padding.lower()
#         if padding == 'same':
#             # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
#             if is_static_pad(kernel_size, **kwargs):
#                 # static case, no extra overhead
#                 padding = get_padding(kernel_size, **kwargs)
#             else:
#                 # dynamic 'SAME' padding, has runtime/GPU memory overhead
#                 padding = 0
#                 dynamic = True
#         elif padding == 'valid':
#             # 'VALID' padding, same as padding=0
#             padding = 0
#         else:
#             # Default to PyTorch style 'same'-ish symmetric padding
#             padding = get_padding(kernel_size, **kwargs)
#     return padding, dynamic
import torch
import torch.nn as nn
import torch.nn.functional as F

# self.c1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kern_legnth), bias=False,
#                             padding=(0, (kern_legnth)// 2))
# def conv2d_same(
#         x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
#         padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
#     x = pad_same(x, weight.shape[-2:], stride, dilation)
#     return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


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
        print("pad shape : ",x.shape)
        print("pad x : ",x)
        result = super(Conv2dSame, self).forward(x)
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
from dassl.modeling.backbone.backbone import Backbone
class EEGNET(Backbone):
    def __init__(self, kern_legnth=64, num_ch=22, samples = 256,F1=8, D=2, F2=16,drop_prob=0.4,avg_pool_1 = 4, avg_pool_2 = 8, sep_kern_length = 16):
        super().__init__()
        self.avg_pool_1 = avg_pool_1
        self.avg_pool_2 = avg_pool_2
        self.sep_kern_length = sep_kern_length

        input_dim_1 = (1,1,num_ch,samples)
        pad_h,pad_w = calculate_pad(input_dim_1,k=(1,kern_legnth))
        if pad_h % 2 == 1 or pad_w % 2 == 1:
            print("use conv2d Same for even kern size")
            self.c1 = Conv2dSame(in_channels=1, out_channels=F1, kernel_size=(1, kern_legnth), bias=False,padding=(pad_h,pad_w))
        else:
            self.c1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kern_legnth), bias=False,
                                padding=(0, (kern_legnth // 2)))
        # self.b1 = nn.BatchNorm2d(F1)
        self.b1 = nn.BatchNorm2d(F1)
        self.c2 = Conv2dWithConstraint(in_channels=F1, out_channels=F1 * D, kernel_size=(num_ch, 1), stride=1,
                                       bias=False, groups=F1, padding=(0, 0), max_norm=1.0)
        # self.b2 = nn.BatchNorm2d(F1 * D)
        self.b2 = nn.BatchNorm2d(F2)
        # self.p2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.d2 = nn.Dropout(drop_prob)
        # samples = samples//4
        samples = samples//self.avg_pool_1

        # self.c3 = nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 16), stride=1, bias=False,
        #                     groups=(nfeatl), padding=(0, 16 // 2))
        if pad_h % 2 == 1 or pad_w % 2 == 1:
            input_dim_2 = (1, 1, num_ch, samples)
            pad_h, pad_w = calculate_pad(input_dim_2, k=(1, kern_legnth))
            self.c3 = nn.Sequential(
                # conv_separable_depth"
                Conv2dSame(in_channels=F2, out_channels=F2, kernel_size=(1, self.sep_kern_length), bias=False,
                          groups=(F2), padding=(pad_h, pad_w)),
                # conv_separable_point
                nn.Conv2d(F2, F2, (1, 1), bias=False, padding=(0, 0))
            )
        else:
            # conv_separable_depth"
            self.c3 = nn.Sequential (
                nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, self.sep_kern_length), stride=1, bias=False,groups=(F2), padding=(0, self.sep_kern_length//2)),
                #conv_separable_point
                nn.Conv2d(F2, F2, (1, 1), bias=False,padding=(0, 0))
            )
        # self.b3 = nn.BatchNorm2d(F1 * D)
        self.b3 = nn.BatchNorm2d(F2, momentum=0.1, affine=True, eps=1e-3)
        # self.p3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.d3 = nn.Dropout(drop_prob)
        samples = samples//self.avg_pool_2
        self._out_features = F2*samples
    def forward(self,input):
        h1 = self.b1(self.c1(input))
        h2 = self.d2(F.avg_pool2d(F.elu(self.b2(self.c2(h1))),(1,self.avg_pool_1)) )
        h3 = self.d3(F.avg_pool2d(F.elu(self.b3(self.c3(h2))),(1,self.avg_pool_2)) )
        flatten = torch.flatten(h3, start_dim=1)
        return flatten


#test the padding
eegnet = EEGNET()
x = torch.ones((2,1,1,64))
kern = (1,16)
pad_h, pad_w = calculate_pad(x.shape, k=kern)

c1 = Conv2dSame(in_channels=1,out_channels=1,kernel_size=kern,padding=(pad_h,pad_w))

y = c1(x)
print(y.shape)