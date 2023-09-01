'''
Function:
    Base model for all supported models
Author:
    Jian Wang
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from .objectcontext import ObjectContextBlock
from .spatialgather import SpatialGatherModule
from ..base import BaseModel

class OCRNetBase(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(OCRNetBase, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg

        # build bottleneck
        bottleneck_cfg = cfg['bottleneck']
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_cfg['in_channels'], bottleneck_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (bottleneck_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        # build spatial gather module
        spatialgather_cfg = {
            'scale': cfg['spatialgather']['scale']
        }
        self.spatial_gather_module = SpatialGatherModule(**spatialgather_cfg)
        # build object context block
        ocb_cfg = cfg['objectcontext']
        self.object_context_block = ObjectContextBlock(
            in_channels=ocb_cfg['in_channels'], 
            transform_channels=ocb_cfg['transform_channels'], 
            scale=ocb_cfg['scale'],
            align_corners=align_corners,
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )

        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        self.auxiliary_decoder = nn.Sequential(
            nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (auxiliary_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(auxiliary_cfg['dropout']),
            nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )

        # build decoder
        decoder_cfg = cfg['decoder']
        self.segDecoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()

    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'auxiliary_decoder': self.auxiliary_decoder,
            'bottleneck': self.bottleneck,
            'spatial_gather_module': self.spatial_gather_module,
            'object_context_block': self.object_context_block,
            'decoder': self.decoder
        }

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True

    def pretrain(self, cfg, unFreezeLayers):
        if cfg['FROZEN']==1:
            for p in self.parameters():
                p.requires_grad = False            
            
            for unFreezeLayer in unFreezeLayers:
                self.unfreeze(unFreezeLayer)
            print('## unfreeze seg layers and freeze cls layers')
            print('## only train seg layers')
        elif cfg['FROZEN']==2:
            for p in self.parameters():
                p.requires_grad = True
            for FreezeLayer in unFreezeLayers:
                self.freeze(FreezeLayer)                    
            print('## freeze seg layers and unfreeze cls layers')
            print('## only train cls layers')
        elif cfg['FROZEN']==0:
            for p in self.parameters():
                p.requires_grad = True
            print('## unfreeze all layers')
            print('## train all layers')
        else:
            print('## bug in Label FROZEN:',cfg['FROZEN'])


class OCRNetSABase(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(OCRNetSABase, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg

        # build bottleneck
        bottleneck_cfg = cfg['bottleneck']
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_cfg['in_channels'], bottleneck_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (bottleneck_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        # build spatial gather module
        spatialgather_cfg = {
            'scale': cfg['spatialgather']['scale']
        }
        self.spatial_gather_module = SpatialGatherModule(**spatialgather_cfg)
        # build object context block
        ocb_cfg = cfg['objectcontext']
        self.object_context_block = ObjectContextBlock(
            in_channels=ocb_cfg['in_channels'], 
            transform_channels=ocb_cfg['transform_channels'], 
            scale=ocb_cfg['scale'],
            align_corners=align_corners,
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )

        # build auxiliary decoder
        auxiliary_cfg = cfg['auxiliary']
        self.auxiliary_decoder = nn.Sequential(
            nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (auxiliary_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(auxiliary_cfg['dropout']),
            nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )

        # build decoder
        decoder_cfg = cfg['decoder']
        self.segDecoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
    

        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        
        self.Qf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.Kf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.Vf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.W_z = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512)
            )
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'auxiliary_decoder': self.auxiliary_decoder,
            'bottleneck': self.bottleneck,
            'spatial_gather_module': self.spatial_gather_module,
            'object_context_block': self.object_context_block,
            'decoder': self.decoder
        }

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True

    def pretrain(self, cfg, Layers):
        if cfg['FROZEN']==1:
            for p in self.parameters():
                p.requires_grad = False            
            
            for unFreezeLayer in Layers:
                self.unfreeze(unFreezeLayer)
            print('## unfreeze seg layers and freeze cls layers')
            print('## only train seg layers')
        elif cfg['FROZEN']==2:
            for p in self.parameters():
                p.requires_grad = True
            for FreezeLayer in Layers:
                self.freeze(FreezeLayer)                    
            print('## freeze seg layers and unfreeze cls layers')
            print('## only train cls layers')
        elif cfg['FROZEN']==3:     
            for p in self.parameters():
                p.requires_grad = True
            for FreezeLayer in Layers:
                self.freeze(FreezeLayer)         
            print('## only freeze backbone network')
            print('## train other layers')         
        elif cfg['FROZEN']==0:
            for p in self.parameters():
                p.requires_grad = True
            print('## unfreeze all layers')
            print('## train all layers')
        else:
            print('## bug in Label FROZEN:',cfg['FROZEN'])

# unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

    def SAFusion(self, Q_feat_s, KV_feat_s):
        Q = self.Qf(Q_feat_s).view(Q_feat_s.size(0),Q_feat_s.size(1),-1)      ##Q: torch.Size([32, 512, 361])
        Q = Q.permute(0,2,1)                                                        ##Q: torch.Size([32, 361, 512])
        K = self.Kf(KV_feat_s).view(KV_feat_s.size(0),KV_feat_s.size(1),-1)      ##K: torch.Size([32, 512, 9])
        V = self.Vf(KV_feat_s).view(KV_feat_s.size(0),KV_feat_s.size(1),-1)
        V = V.permute(0,2,1)                                                        ##V: torch.Size([32, 9, 512])
        attn = torch.matmul(Q, K)     # Q.bmm(K) #/math.sqrt(Q.size(-1))   ## [19*19, 3*3]                     ##QK: torch.Size([32, 361, 9])

        # N = attn.size(-1) 
        # attn_div_C = attn / N
        attn = self.softmax(attn)
        attn_div_C = self.dropout(attn)

        QKV = torch.matmul(attn_div_C, V) #attn_div_C.bmm(V).permute(0,2,1)                                      ##QKV: torch.Size([32, 361, 512])  --> torch.Size([32, 512, 361])
        QKV = QKV.permute(0, 2, 1).contiguous()
        QKV = QKV.view(QKV.size(0), QKV.size(1), Q_feat_s.size(-1), -1)
        QKV = self.W_z(QKV)
        QKV_out = QKV+Q_feat_s
        return QKV_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)