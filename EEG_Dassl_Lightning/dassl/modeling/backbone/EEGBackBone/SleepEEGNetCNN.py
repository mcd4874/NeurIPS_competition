import torch
import torch.nn as nn
import torch.nn.functional as F
from dassl.utils.torchtools import load_pretrained_backbone
from dassl.modeling.backbone.build import BACKBONE_REGISTRY
from dassl.modeling.backbone.backbone import Backbone



class SleepEEGNetCNN(Backbone):
    """
    I implement CNN part of SleepEEGNet
    https://arxiv.org/pdf/1903.02108.pdf
    """
    def __init__(self,num_ch=17, samples = 256,drop_prob=0.4):
        super().__init__()

        ######### CNNs with small filter size at the first layer #########
        self.spatial_layer= nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(num_ch, 1), stride=1,
                     bias=False, padding=(0, 0))

        # Convolution
        pad_1_0 = (((samples-1)*6+50)-samples)//2
        self.layer_1_0 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(1,50),stride=6,padding=(0,pad_1_0))
        self.layer_1_0_pooling = nn.MaxPool2d(kernel_size=(1,8),stride=(1,8))
        self.layer_1_0_dropout = nn.Dropout(drop_prob)

        layer_1_0_sample = int((samples-8)/8) +1

        #Convolution
        pad_2_0 = (((layer_1_0_sample-1)*1+8)-layer_1_0_sample)//2
        self.layer_2_0 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(1,8),stride=1,padding=(0,pad_2_0))
        self.layer_3_0 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,8),stride=1,padding=(0,pad_2_0))
        self.layer_4_0 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,8),stride=1,padding=(0,pad_2_0))
        self.layer_4_0_pooling = nn.MaxPool2d(kernel_size=(1,4),stride=(1,4))

        layer_4_0_sample = int((layer_1_0_sample - 4) / 4) + 1
        ######### CNNs with small filter size at the first layer #########

        # Convolution
        pad_1_1 = (((samples-1)*50+400)-samples)//2
        self.layer_1_1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(1,400),stride=50,padding=pad_1_1)
        self.layer_1_1_pooling = nn.MaxPool2d(kernel_size=(1,4),stride=(1,4))
        self.layer_1_1_dropout = nn.Dropout(drop_prob)
        layer_1_1_sample = int((samples-4)/4) +1

        #Convolution
        pad_2_1 = (((layer_1_1_sample-1)*1+8)-layer_1_1_sample)//2
        self.layer_2_1 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(1,6),stride=1,padding=(0,pad_2_1))
        self.layer_3_1 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,6),stride=1,padding=(0,pad_2_1))
        self.layer_4_1 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,6),stride=1,padding=(0,pad_2_1))
        self.layer_4_1_pooling = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
        layer_4_1_sample = int((layer_1_1_sample - 2) / 2) + 1


        self.final_dropout = nn.Dropout(drop_prob)

        self._out_features = layer_4_0_sample*128+layer_4_1_sample*128
    def forward(self,input):

        h_0 = self.spatial_layer(input)

        h1 = self.layer_1_0_dropout(self.layer_1_0_pooling(F.relu(self.layer_1_0(h_0))))
        h2 = self.layer_4_0_pooling(self.layer_4_0(self.layer_3_0(self.layer_2_0(h1))))
        flatten_1 = torch.flatten(h2, start_dim=1)

        h3 = self.layer_1_1_dropout(self.layer_1_1_pooling(F.relu(self.layer_1_1(h_0))))
        h4 = self.layer_4_1_pooling(self.layer_4_1(self.layer_3_1(self.layer_2_1(h3))))
        flatten_2 = torch.flatten(h4, start_dim=1)

        combine = torch.cat([flatten_1,flatten_2])
        combine = self.final_dropout(combine)

        return combine

@BACKBONE_REGISTRY.register()
def sleepeegnetcnn(pretrained=False,pretrained_path = '', **kwargs):
    print("params set up : ",kwargs)
    model = SleepEEGNetCNN(**kwargs)
    print("pretrain : ",pretrained)
    print('pretrain path :',pretrained_path)
    if pretrained and pretrained_path!= '':
        print("load pretrain model ")
        load_pretrained_backbone(model,pretrained_path)

    return model