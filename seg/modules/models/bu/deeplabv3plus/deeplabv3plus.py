'''
Function:
    Implementation of Deeplabv3plus
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .aspp import DepthwiseSeparableASPP


'''Deeplabv3plus'''
class Deeplabv3Plus(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(Deeplabv3Plus, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build aspp net
        aspp_cfg = {
            'in_channels': cfg['aspp']['in_channels'],
            'out_channels': cfg['aspp']['out_channels'],
            'dilations': cfg['aspp']['dilations'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.aspp_net = DepthwiseSeparableASPP(**aspp_cfg)
        # build shortcut
        shortcut_cfg = cfg['shortcut']
        self.shortcut = nn.Sequential(
            nn.Conv2d(shortcut_cfg['in_channels'], shortcut_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (shortcut_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg),
            DepthwiseSeparableConv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
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
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()

################## add by wj
        # self.clsDecoder = nn.Sequential(
        #     nn.Conv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=3, stride=2, padding=1),
        #     nn.Dropout2d(decoder_cfg['dropout']),
        #     nn.Conv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        #     nn.AdaptiveAvgPool2d((1,1))
        # )
        self.ReLu = nn.ReLU(inplace=True)

        self.featEncoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=0),
            # nn.AdaptiveAvgPool2d((1,1))
        )

        self.Qf = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, bias=False)
        self.Kf = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, bias=False)
        self.Vf = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, bias=False)
        self.W_z = nn.Sequential(
                nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(decoder_cfg['in_channels'])
            )

        self.getScore = nn.Sequential(    
                nn.Linear( decoder_cfg['in_channels'], decoder_cfg['in_channels']),
                nn.BatchNorm1d(decoder_cfg['in_channels']),
                nn.ReLU(inplace=True),
                nn.Linear( decoder_cfg['in_channels'], cfg['num_classes'])
        )
        self.maskDecoder = nn.Sequential(
            nn.Conv2d(cfg['num_classes'], 1, kernel_size=1, stride=1, padding=0)
        )
    def SAFusion(self, Q_feat_s, KV_feat_s):
        Q = self.Qf(Q_feat_s).view(Q_feat_s.size(0),Q_feat_s.size(1),-1)      ##Q: torch.Size([32, 512, 361])
        Q = Q.permute(0,2,1)                                                        ##Q: torch.Size([32, 361, 512])
        K = self.Kf(KV_feat_s).view(KV_feat_s.size(0),KV_feat_s.size(1),-1)      ##K: torch.Size([32, 512, 9])
        V = self.Vf(KV_feat_s).view(KV_feat_s.size(0),KV_feat_s.size(1),-1)
        V = V.permute(0,2,1)                                                        ##V: torch.Size([32, 9, 512])
        attn = torch.matmul(Q, K)     # Q.bmm(K) #/math.sqrt(Q.size(-1))   ## [19*19, 3*3]                     ##QK: torch.Size([32, 361, 9])

        N = attn.size(-1) 
        attn_div_C = attn / N
        # attn = self.softmax2(attn)
        # attn = self.dropout(attn)

        QKV = torch.matmul(attn_div_C, V) #attn_div_C.bmm(V).permute(0,2,1)                                      ##QKV: torch.Size([32, 361, 512])  --> torch.Size([32, 512, 361])
        QKV = QKV.permute(0, 2, 1).contiguous()
        QKV = QKV.view(QKV.size(0), QKV.size(1), Q_feat_s.size(-1), -1)
        QKV = self.W_z(QKV)
        QKV_out = QKV+Q_feat_s
        return QKV_out


    def clsDecoder(self, feats, preds):
        dim_bs, dim_c, dim_h, dim_w = feats.size()
 
        # print('##pred_mask:',torch.sum(preds,dim=1,keepdim=True).size(),torch.sum(preds,dim=1,keepdim=True))
        # pred_mask = torch.sum(preds,dim=1,keepdim=True)
        # pred_mask = preds[:,1,:,:].unsqueeze(1)
        pred_mask = self.maskDecoder(preds) 
        pred_mask_1 = pred_mask.expand(dim_bs,dim_c,dim_h,dim_w) 

        feats_1 = self.featEncoder(feats)
        # print('##feats_1:',feats_1.size())

        h, w = feats_1.size(2), feats_1.size(3)
        pred_mask_2 = F.interpolate(pred_mask_1, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        # print('##pred_mask_2:',pred_mask_2.size())
        
        Q_feat = torch.mul(feats_1, pred_mask_2)
        feats_2 = self.ReLu(self.SAFusion(Q_feat, feats_1))
        # feats_2 = self.ReLu(feats_2)
        feats_3 = F.adaptive_avg_pool2d(feats_2, (1, 1))
        feats_end = feats_3.view(feats_3.size(0), -1)
        score = self.getScore(feats_end)
        return score
##feats_1: torch.Size([2, 560, 16, 16])
##pred_mask_2: torch.Size([2, 560, 16, 16])

####################################

#         self.clsDecoder = nn.Sequential(
#             DepthwiseSeparableConv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=3, stride=2, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg),
#             DepthwiseSeparableConv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=3, stride=2, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg),
#             nn.Dropout2d(decoder_cfg['dropout']),
#             nn.Conv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
#             # F.adaptive_avg_pool2d((1,1))
#             nn.AdaptiveAvgPool2d((1,1))
#         )
# ####################################


    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x1, x2, x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to aspp
        aspp_out = self.aspp_net(x4)
        aspp_out = F.interpolate(aspp_out, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=self.align_corners)
        # feed to shortcut
        shortcut_out = self.shortcut(x1)
        # feed to decoder
        features = torch.cat([aspp_out, shortcut_out], dim=1)
        preds = self.decoder(features)
        
        
        # print('##aspp_out:',aspp_out.size())
        # print('##shortcut_out:',shortcut_out.size())
        # print('##targets:',targets['segmentation'].size())
        # print('##features:',features.size())
        # print('##preds:',preds.size())
        predsLabel = self.clsDecoder(features, preds)
        # predsLabel = self.clsDecoder(preds).squeeze(3).squeeze(2)
        # print('##predsLabel:',predsLabel.size())

##aspp_out: torch.Size([2, 512, 64, 64])
##shortcut_out: torch.Size([2, 48, 64, 64])
##features: torch.Size([2, 560, 64, 64])
##preds: torch.Size([2, 2, 64, 64])

##x: torch.Size([2, 3, 256, 256])
##targets: torch.Size([2, 256, 256])

##predsLabel: torch.Size([2, 2])

############add by wj
        if self.mode == 'TRAIN' and targets!=None:
            # feed to backbone network
            x1_2, x2_2, x3_2, x4_2 = self.transforminputs(self.backbone_net(images2), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to aspp
            aspp_out_2 = self.aspp_net(x4_2)
            aspp_out_2 = F.interpolate(aspp_out_2, size=(x1_2.size(2), x1_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            # feed to shortcut
            shortcut_out_2 = self.shortcut(x1_2)
            # feed to decoder
            features_2 = torch.cat([aspp_out_2, shortcut_out_2], dim=1)
            preds_2 = self.decoder(features_2)

            # print('##clsDecoder:',self.clsDecoder(features_2, preds_2).size())

            predsLabel_2 = self.clsDecoder(features_2, preds_2)#.squeeze(3).squeeze(2)

        # print('##preds_2:',preds_2.size())
        # exit()
##preds_2: torch.Size([2, 2, 64, 64])
############ add by wj



        # feed to auxiliary decoder and return according to the mode
        if self.mode == 'TRAIN' and targets!=None:
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux = self.auxiliary_decoder(x3)
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds, 'loss_inv': preds_2, 'loss_aux': preds_aux, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds, predsLabel
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net, 
            'aspp_net': self.aspp_net, 
            'shortcut': self.shortcut,
            'decoder': self.decoder,
            'auxiliary_decoder': self.auxiliary_decoder
        }