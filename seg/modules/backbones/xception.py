"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import os
__all__ = ['xception_base']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=2, LabelFrozen=0, out_indices=(0, 1, 2, 3), **kwargs):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.out_indices = out_indices

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,1,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        # self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        # self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        # self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        # self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        # self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        # self.block12=Block(728,1024,2,1,start_with_relu=True,grow_first=False)

        # self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        # self.bn3 = nn.BatchNorm2d(1536)

        # #do relu here
        # self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        # self.bn4 = nn.BatchNorm2d(2048)
        if LabelFrozen==1:
            for p in self.parameters():
                p.requires_grad = False
        elif LabelFrozen==0:
            for p in self.parameters():
                p.requires_grad = True
        else:
            print('## bug in LabelFrozen')
        # self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x1 = self.block3(x1)

        x1 = self.block4(x1)

        x2 = self.block5(x1)
        x3 = self.block6(x2)

        # x3 = self.block7(x2)
        # x3 = self.block8(x3)
        # x3 = self.block9(x3)

        # x4 = self.block10(x3)
        # x4 = self.block11(x4)
        # x4 = self.block12(x4)

        outs = []
        for i, feats in enumerate([x1, x2, x3]):
            if i in self.out_indices: 
                outs.append(feats)

        return tuple(outs)

    # def logits(self, features):
    #     x = self.relu(features)

    #     x = F.adaptive_avg_pool2d(x, (1, 1))
    #     x = x.view(x.size(0), -1)
    #     # x = self.last_linear(x)
    #     x = self.fc(x)
    #     return x

    def forward(self, input):
        return self.features(input)
        # x = self.logits(x)
        # return x




'''build resnet'''
def BuildXception(xception_type, **kwargs):
    # parse args
    default_args = {
        'use_stem': True,
        'norm_cfg': None,
        'pretrained': True,
        'out_indices': (0, 1, 2),
        'pretrained_model_path': '',
        'use_avg_for_downsample': False,
        'num_classes':2,
        'LabelFrozen':0, 
    }
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain args for instanced resnet
    xception_args = {}
    xception_args.update(default_args)
    # obtain the instanced resnet
    model = Xception(**xception_args)
    # load weights of pretrained model
    # if default_args['use_stem']: resnet_type = resnet_type + 'stem'
    if default_args['pretrained'] and os.path.exists(default_args['pretrained_model_path']):
        pretrained_dict_G = torch.load(default_args['pretrained_model_path'])
        model = preTrained(model, pretrained_dict_G)

        # if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        # else: state_dict = checkpoint
        # model.load_state_dict(state_dict, strict=False)
    elif default_args['pretrained']:
        settings = pretrained_settings['xception']['imagenet']
        pretrained_dict_G = model_zoo.load_url(settings['url'])

        # pretrained_dict_G = model_zoo.load_url(model_urls[resnet_type])
        model = preTrained(model, pretrained_dict_G)


        # if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        # else: state_dict = checkpoint
        # model.load_state_dict(state_dict, strict=False)
    # return the model
    return model



def preTrained(modelG, pretrained_dict_G):
    model_dict_G = modelG.state_dict()
    for name, weights in pretrained_dict_G.items():
        if 'pointwise' in name:
            pretrained_dict_G[name] = weights.unsqueeze(-1).unsqueeze(-1)
    pretrained_dict_G = {k: v for k, v in pretrained_dict_G.items() if k in model_dict_G}
    # print('##pretrained_dict:',pretrained_dict.keys())
    # del pretrained_dict.['fc2']
    model_dict_G.update(pretrained_dict_G)
    modelG.load_state_dict(model_dict_G)
    # for name, param in modelG.named_parameters():
    #     if param.requires_grad:
    #         pass
    #         print("requires_grad True : ", name)
    #     else:
    #         print("requires_grad False : ", name)
    return modelG







# BuildXception('xception')


def xception_base(num_classes=2, trainType='imagenet', LabelFrozen=0):

    modelG = Xception(num_classes, LabelFrozen)
    # modelG = torch.nn.DataParallel(modelG)
    # print('##model:', model)
    model_dict_G = modelG.state_dict()
    # print('##model_dict:', model_dict.keys())
    # ('module.fc2.weight', tensor([[-3.9120e-02, 1.6058e-02, -1.8196e-02, -3.6930e-02, -1.5812e-02]]))
    # pretrained_dict_G = \
    # torch.load('/home/wj/08Face/LightCNN-master/model_para/LightCNN_29Layers_V2_checkpoint.pth.tar')['state_dict']


    settings = pretrained_settings['xception'][trainType]
    pretrained_dict_G = model_zoo.load_url(settings['url'])
    for name, weights in pretrained_dict_G.items():
        if 'pointwise' in name:
            pretrained_dict_G[name] = weights.unsqueeze(-1).unsqueeze(-1)

    #    for k, v in pretrained_dict.items():
    #        if k in model_dict:
    #            print('##k:',k)
    # print('\n##pretrained_dict:', pretrained_dict.keys())

    pretrained_dict_G = {k: v for k, v in pretrained_dict_G.items() if k in model_dict_G}
    # print('##pretrained_dict:',pretrained_dict.keys())
    # del pretrained_dict.['fc2']
    model_dict_G.update(pretrained_dict_G)
    modelG.load_state_dict(model_dict_G)

    modelG.fc = nn.Linear(2048, num_classes)

    modelG.input_space = settings['input_space']
    modelG.input_size = settings['input_size']
    modelG.input_range = settings['input_range']
    modelG.mean = settings['mean']
    modelG.std = settings['std']

    # modelG.last_linear = modelG.fc
    # del modelG.fc
    # print('##model:', modelG)
    # exit()
    return modelG