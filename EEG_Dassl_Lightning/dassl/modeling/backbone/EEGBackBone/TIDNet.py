import torch
from torch.nn import *
from torch.nn.utils import weight_norm

from dassl.utils.torchtools import load_pretrained_backbone
from dassl.modeling.backbone.build import BACKBONE_REGISTRY
from dassl.modeling.backbone.backbone import Backbone
import math

# class Expand(Module):
#     def __init__(self, axis=-1):
#         super().__init__()
#         self.axis = axis
#
#     def forward(self, x):
#         return x.unsqueeze(self.axis)
#
#
# class Squeeze(Module):
#     def __init__(self, axis=-1):
#         super().__init__()
#         self.axis = axis
#
#     def forward(self, x):
#         return x.squeeze(self.axis)


# class Permute(Module):
#     def __init__(self, axes):
#         super().__init__()
#         self.axes = axes
#
#     def forward(self, x):
#         return x.permute(self.axes)


# class Flatten(Module):
#     def forward(self, x):
#         return x.contiguous().view(x.size(0), -1)




class BatchNormZG(BatchNorm2d):
    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()


class ConvBlock2D(Module):
    """
    Implements Convolution block with order:
    Convolution, dropout, activation, batch-norm
    """

    def __init__(self, in_filters, out_filters, kernel, stride=(1, 1), padding=0, dilation=1, groups=1, do_rate=0.5,
                 batch_norm=True, activation=LeakyReLU, residual=False):
        super().__init__()
        self.kernel = kernel
        self.activation = activation()
        self.residual = residual

        self.conv = Conv2d(in_filters, out_filters, kernel, stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=not batch_norm)
        self.dropout = Dropout2d(p=do_rate)
        self.batch_norm = BatchNormZG(out_filters) if residual else BatchNorm2d(out_filters) if batch_norm else \
            lambda x: x

    def forward(self, input, **kwargs):
        res = input
        input = self.conv(input, **kwargs)
        input = self.dropout(input)
        input = self.activation(input)
        input = self.batch_norm(input)
        return input + res if self.residual else input


class LinearBlock(Module):
    """
    Implements fully connected block with order:
    FC, dropout, activation, batch-norm
    """

    def __init__(self, incoming, outgoing, do_rate=0.5, batch_norm=True, activation=LeakyReLU):
        super().__init__()
        self.activation = activation()
        self.fc = Linear(incoming, outgoing, bias=not batch_norm)
        self.dropout = Dropout(p=do_rate)
        self.batch_norm = BatchNorm1d(outgoing) if batch_norm else lambda x: x

    def forward(self, input, **kwargs):
        input = self.fc(input, **kwargs)
        input = self.dropout(input)
        input = self.activation(input)
        input = self.batch_norm(input)
        return input


class DenseFilter(Module):
    def __init__(self, in_features, growth_rate, filter_len=5, do=0.5, bottleneck=2, activation=LeakyReLU, dim=-2):
        super().__init__()
        dim = dim if dim > 0 else dim + 4
        if dim < 2 or dim > 3:
            raise ValueError('Only last two dimensions supported')
        kernel = (filter_len, 1) if dim == 2 else (1, filter_len)

        self.net = Sequential(
            BatchNorm2d(in_features),
            activation(),
            Conv2d(in_features, bottleneck * growth_rate, 1),
            BatchNorm2d(bottleneck * growth_rate),
            activation(),
            Conv2d(bottleneck * growth_rate, growth_rate, kernel, padding=tuple((k // 2 for k in kernel))),
            Dropout2d(do)
        )

    def forward(self, x):
        return torch.cat((x, self.net(x)), dim=1)


class TemporalTransition(Module):
    def __init__(self, in_features, out_features=None, max_span=31, pooling=2, activation=LeakyReLU):
        super().__init__()
        t3 = max_span + 1 - max_span % 2
        t2 = (t3 // 2) + 1 - (t3 // 2) % 2
        t1 = (t3 // 10) + 1 - (t3 // 10) % 2
        if out_features is None:
            out_features = in_features
        self.t1 = Sequential(
            BatchNorm2d(in_features),
            activation(),
            Conv2d(in_features, max(1, out_features - 2*(out_features//3)), (1, t1), padding=(0, t1 // 2))
        )
        self.t2 = Sequential(
            BatchNorm2d(in_features),
            activation(),
            Conv2d(in_features, max(1, out_features//3), (1, t2), padding=(0, t2 // 2))
        )
        self.t3 = Sequential(
            BatchNorm2d(in_features),
            activation(),
            Conv2d(in_features, max(1, out_features//3), (1, t3), padding=(0, t3 // 2))
        )
        self.pool = MaxPool2d((1, pooling), (1, pooling))

    def forward(self, x):
        x = torch.cat((self.t1(x), self.t2(x), self.t3(x)), dim=1)
        return self.pool(x)


class DenseSpatialFilter(Module):
    def __init__(self, channels, growth, depth, in_ch=1, bottleneck=4, dropout_rate=0.0, activation=LeakyReLU,
                 collapse=True):
        super().__init__()
        self.net = Sequential(*[
            DenseFilter(in_ch + growth * d, growth, bottleneck=bottleneck, do=dropout_rate,
                        activation=activation) for d in range(depth)
        ])
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        if collapse:
            self.channel_collapse = ConvBlock2D(n_filters, n_filters, (channels, 1), do_rate=0)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse(x).squeeze(-2)
        return x


class SpatialFilter(Module):
    def __init__(self, channels, filters, depth, in_ch=1, dropout_rate=0.0, activation=LeakyReLU, batch_norm=True,
                 residual=False):
        super().__init__()
        kernels = [(channels // depth, 1) for _ in range(depth-1)]
        kernels += [(channels - sum(x[0] for x in kernels) + depth-1, 1)]
        self.filter = Sequential(
            ConvBlock2D(in_ch, filters, kernels[0], do_rate=dropout_rate/depth, activation=activation,
                        batch_norm=batch_norm),
            *[ConvBlock2D(filters, filters, kernel, do_rate=dropout_rate/depth, activation=activation,
                          batch_norm=batch_norm)
              for kernel in kernels[1:]]
        )
        self.residual = Conv1d(channels * in_ch, filters, 1) if residual else None

    def forward(self, x):
        res = x
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        elif self.residual:
            res = res.contiguous().view(res.shape[0], -1, res.shape[3])
        x = self.filter(x).squeeze(-2)
        return x + self.residual(res) if self.residual else x


class TemporalFilter(Module):
    def __init__(self, channels, filters, depth, temp_len, dropout=0., activation=LeakyReLU, residual='netwise'):
        super().__init__()
        temp_len = temp_len + 1 - temp_len % 2
        self.residual_style = str(residual)
        net = list()

        for i in range(depth):
            dil = depth - i
            conv = weight_norm(Conv2d(channels if i == 0 else filters, filters, kernel_size=(1, temp_len),
                                      dilation=dil, padding=(0, dil * (temp_len - 1) // 2)))
            net.append(Sequential(
                conv,
                activation(),
                Dropout2d(dropout)
            ))
        if self.residual_style.lower() == 'netwise':
            self.net = Sequential(*net)
            self.residual = Conv2d(channels, filters, (1, 1))
        elif residual.lower() == 'dense':
            self.net = net

    def forward(self, x):
        if self.residual_style.lower() == 'netwise':
            # if not torch.isfinite(self.net(x)).all():
            #     print("anamoly for net(x) ")
            #
            #     for name, param in self.net.named_parameters():
            #         print(name, torch.isfinite(param.grad).all())
            #
            #         print(param)

            results = self.net(x) + self.residual(x)
            return results
        elif self.residual_style.lower() == 'dense':
            for l in self.net:
                x = torch.cat((x, l(x)), dim=1)
            return x
class TIDNet(Backbone):

    """
    todo: there are problem with gradient when train deep model with long epochs
    Cause Gradient vanishing in temporal layer
    """
    def __init__(self, s_growth=24, t_filters=32, channels=22, samples=1500, do=0.4, pooling=20,
                 temp_layers=2, spat_layers=2, temp_span=0.05, bottleneck=3, summary=-1):
        super().__init__()
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.temp_len = math.ceil(temp_span * samples)

        self.temporal = Sequential(
            TemporalFilter(1, t_filters, depth=temp_layers, temp_len=self.temp_len),
            MaxPool2d((1, pooling)),
            Dropout2d(do),
        )
        summary = samples // pooling if summary == -1 else summary

        self.spatial = DenseSpatialFilter(channels, s_growth, spat_layers, in_ch=t_filters, dropout_rate=do,
                                          bottleneck=bottleneck)
        self.extract_features = Sequential(
            AdaptiveAvgPool1d(int(summary)),
            Flatten()
        )

        self._out_features = (t_filters + s_growth * spat_layers) * summary


    def forward(self, input):
        x = self.temporal(input)
        x = self.spatial(x)
        return self.extract_features(x)

@BACKBONE_REGISTRY.register()
def tidnet(pretrained=False,pretrained_path = '', **kwargs):

    model = TIDNet(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model