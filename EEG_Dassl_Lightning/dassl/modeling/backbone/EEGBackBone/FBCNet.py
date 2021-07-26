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
#%% Support classes for FBNet Implementation
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


class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FBCNet(Backbone):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s.
        The data input is in a form of batch x 1 x chan x time x filterBand
        https://arxiv.org/pdf/2104.01233.pdf
        reference code : https://github.com/ravikiran-mane/FBCNet
    '''

    def SCB(self, m, nChan, nBands):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(nBands, m * nBands, (nChan, 1), groups=nBands,
                                 max_norm=2, padding=0),
            nn.BatchNorm2d(m * nBands),
            swish()
        )

    # def LastBlock(self, inF, outF,  *args, **kwargs):
    #     return nn.Sequential(
    #         LinearWithConstraint(inF, outF, max_norm=0.5, *args, **kwargs),
    #         nn.LogSoftmax(dim=1))

    # def __init__(self, nChan, nTime, nClass=2, nBands=9, m=32,
    #              temporalLayer='LogVarLayer', strideFactor=4, doWeightNorm=True, *args, **kwargs):
    def __init__(self, nChan, nBands=9, m=32,
                     temporalLayer='LogVarLayer', strideFactor=4):
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands)

        # Formulate the temporal agreegator
        # self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)
        if temporalLayer=='LogVarLayer':
            self.temporalLayer = LogVarLayer(dim=3)
        else:
            self.temporalLayer = VarLayer(dim=3)

        # The final fully connected layer
        # self.lastLayer = self.LastBlock(self.m * self.nBands * self.strideFactor, nClass)
        # samples = samples//8
        samples = nBands*m*strideFactor

        self._out_features = samples

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        # print("update x shape : ",x.shape)
        # x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)
        # x = self.lastLayer(x)
        return x



@BACKBONE_REGISTRY.register()
def fbcnet(pretrained=False,pretrained_path = '', **kwargs):
    print("params set up : ",kwargs)
    model = FBCNet(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model