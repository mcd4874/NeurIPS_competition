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

class ComponentExtractor(Backbone):
    def __init__(self, kern_legnth=64, num_ch=22,output_chans=22, samples = 256,F1=8,D=2,h_pad=0,w_pad=0):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kern_legnth), stride=1, bias=False,
                            padding=(0, (kern_legnth // 2)))
        # self.b1 = nn.BatchNorm2d(F1)
        self.b1 = nn.BatchNorm2d(F1)

        #calculate the appropriate kernel size
        # (input+top_pad+bottom_pad-kernel_size)/stride +1 = outout_channel
        stride = 1
        kernel_size = (num_ch + 2*h_pad) - (output_chans-1) * stride
        print("appropriate kernel size : ",kernel_size)
        F2 = F1*D
        self.c2 = Conv2dWithConstraint(in_channels=F1, out_channels=F2, kernel_size=(kernel_size, 1), stride=1,
                                       bias=False, groups=F1, padding=(h_pad, w_pad), max_norm=1.0)
        self.b2 = nn.BatchNorm2d(F2)
        self._out_features = (F2,output_chans,samples)
    def forward(self,input):
        # print("source model input size : ",input.shape)
        h1 = self.b1(self.c1(input))
        # print("h1 source model size : ",h1.shape)
        h2 = F.elu(self.b2(self.c2(h1)))
        # print("output source model : ",h2.shape)
        return h2

class MainFeature(Backbone):
    def __init__(self,num_ch=22, samples = 256,F2=16,drop_prob=0.4,avg_pool_1 = 4):
        super().__init__()
        self.avg_pool_1 = avg_pool_1
        # self.c1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kern_legnth), stride=1, bias=False,padding=(0, (kern_legnth // 2)))
        # self.b1 = nn.BatchNorm2d(F1)

        self.c2 = Conv2dWithConstraint(in_channels=F2, out_channels=F2, kernel_size=(num_ch, 1), stride=1,
                                       bias=False, groups=F2, padding=(0, 0), max_norm=1.0)
        self.b2 = nn.BatchNorm2d(F2)
        self.d2 = nn.Dropout(drop_prob)
        samples = samples//self.avg_pool_1

        self._out_features = (F2,1,samples)
    def forward(self,input):
        # if source_model:
        #     # print("input shape : ",input.shape)
        h2 = self.d2(F.avg_pool2d(F.elu(self.b2(self.c2(input))),(1,self.avg_pool_1)) )
        # else:
        # h1 = self.b1(self.c1(input))
        # h2 = self.d2(F.avg_pool2d(F.elu(self.b2(self.c2(h1))),(1,self.avg_pool_1)) )
        return h2

@BACKBONE_REGISTRY.register()
def componentExtractor(pretrained=False,pretrained_path = '', **kwargs):
    # print("kwargs : ",kwargs)
    params = kwargs

    is_component_extractor = params['is_component_extractor']
    update_params = {}
    for k, v in params.items():
        if k!='is_component_extractor' :
            update_params[k] = v
    if is_component_extractor:
        model = ComponentExtractor(**update_params)
    else:
        model = MainFeature(**update_params)
    # print("pretrain : ",pretrained)
    # print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model