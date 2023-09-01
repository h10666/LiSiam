'''
Function:
    Implementation of OCRNet
Author:
    Jian Wang
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .ocrnetbase import OCRNetBase, OCRNetSABase


''' dual steam '''
class SiamV1(OCRNetBase):
    def __init__(self, cfg, **kwargs):
        super(SiamV1, self).__init__(cfg, **kwargs)    

################## add by wj
        decoder_cfg = cfg['decoder']
        self.clsEncoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.ReLu = nn.ReLU(inplace=True)
        self.getScore = nn.Sequential(    
                nn.Linear( 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear( 512, cfg['num_classes'])
        )
        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices')) 
        # feed to auxiliary decoder
        preds_aux = self.auxiliary_decoder(x3)
        # feed to bottleneck
        feats0 = self.bottleneck(x4)
        # feed to ocr module
        context = self.spatial_gather_module(feats0, preds_aux)
        feats = self.object_context_block(feats0, context)

        feats_seg = self.segE(feats)
        feats_cls = self.clsE(feats)
        # feed to segDecoder
        preds = self.segDecoder(feats_seg)         ## [BS, 2, 64, 64] (mask)
        # predsLabel = self.clsDecoder(feats_cls, feats_seg)
        predsLabel = self.getScore(self.clsEncoder(feats).squeeze(3).squeeze(2))    #.view(feats.size(0),-1)


################ add by wj
        if self.mode == 'TRAIN' and targets!=None:
            x3_2, x4_2 = self.transforminputs(self.backbone_net(images2), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to auxiliary decoder
            preds_aux_2 = self.auxiliary_decoder(x3_2)
            # feed to bottleneck
            feats0_2 = self.bottleneck(x4_2)
            # feed to ocr module
            context_2 = self.spatial_gather_module(feats0_2, preds_aux_2)
            feats_2 = self.object_context_block(feats0_2, context_2)

            feats_seg_2 = self.segE(feats_2)
            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder
            preds_2 = self.segDecoder(feats_seg_2)     ## [BS, 2, 64, 64]
            # predsLabel_2 = self.clsDecoder(feats_cls_2, feats_seg_2)
            predsLabel_2 = self.getScore(self.clsEncoder(feats_2).squeeze(3).squeeze(2))

        if self.mode == 'TRAIN' and targets!=None:
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_aux': preds_aux, 'loss_cls': preds, 'loss_cls_2': preds_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds, predsLabel


''' make mask by 1*1 conv and sigmoid '''
class SiamV2(OCRNetBase):
    def __init__(self, cfg, **kwargs):
        super(SiamV2, self).__init__(cfg, **kwargs)    

################## add by wj
        decoder_cfg = cfg['decoder']
        # self.clsEncoder = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1,1))
        # )
        # self.ReLu = nn.ReLU(inplace=True)
        self.getScore = nn.Sequential(    
                nn.Linear( 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear( 512, cfg['num_classes'])
        )


        self.Qf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.Kf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.Vf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.W_z = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512)
            )
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)    
        self.sig = nn.Sigmoid()  
        self.ReLu = nn.ReLU(inplace=False)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.ReLU(inplace=True)         
            )
        self.segE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.ReLU(inplace=True)         
            )
        self.maskE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], 1, kernel_size=1, stride=1, padding=0)    
            )
        

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net, self.segE]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, feats_seg):
        dim_bs,dim_c,dim_h,dim_w = feats_seg.size()
        mask = self.maskE(feats_seg)
        # mask = mask - mask.mean(dim=1, keepdim=True)            #.sum(dim=1, keepdim=True)
        # std = mask.std(dim=1, keepdim=True) + 1e-5
        # mask = mask/std
        mask = self.sig(mask)
        mask = mask.expand(dim_bs,dim_c,dim_h,dim_w)


##feats_seg: torch.Size([16, 512, 32, 32])
##mask 0 : torch.Size([16, 1, 32, 32])

        Q_feat = torch.mul(feats_cls, mask)
        # print('##Q_feat:',Q_feat.size())
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, feats_cls)))
        sa_feats = self.ReLu(self.SAFusion(Q_feat, feats_cls))
        # print('##sa_feats:',sa_feats.size())
        # exit()

        feats_end = F.adaptive_avg_pool2d(sa_feats, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

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
####################################

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices')) 
        # feed to auxiliary decoder
        preds_aux = self.auxiliary_decoder(x3)
        # feed to bottleneck
        feats0 = self.bottleneck(x4)
        # feed to ocr module
        context = self.spatial_gather_module(feats0, preds_aux)
        feats = self.object_context_block(feats0, context)

        feats_seg = self.segE(feats)
        feats_cls = self.clsE(feats)
        # feed to segDecoder
        preds = self.segDecoder(feats_seg)         ## [BS, 2, 64, 64] (mask)
        predsLabel = self.clsDecoder(feats_cls, feats_seg)
        # predsLabel = self.getScore(self.clsEncoder(feats).squeeze(3).squeeze(2))    #.view(feats.size(0),-1)



################ add by wj
        if self.mode == 'TRAIN' and targets!=None:
            x3_2, x4_2 = self.transforminputs(self.backbone_net(images2), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to auxiliary decoder
            preds_aux_2 = self.auxiliary_decoder(x3_2)
            # feed to bottleneck
            feats0_2 = self.bottleneck(x4_2)
            # feed to ocr module
            context_2 = self.spatial_gather_module(feats0_2, preds_aux_2)
            feats_2 = self.object_context_block(feats0_2, context_2)

            feats_seg_2 = self.segE(feats_2)
            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder
            preds_2 = self.segDecoder(feats_seg_2)     ## [BS, 2, 64, 64]
            predsLabel_2 = self.clsDecoder(feats_cls_2, feats_seg_2)
            # predsLabel_2 = self.getScore(self.clsEncoder(feats_2).squeeze(3).squeeze(2))
###############

        if self.mode == 'TRAIN' and targets!=None:
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_aux': preds_aux, 'loss_cls': preds, 'loss_cls_2': preds_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds, predsLabel



'''   '''
class SiamV3(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(SiamV3, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(1)


        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.ReLU(inplace=True)         
            )
        self.segE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.ReLU(inplace=True)         
            )
        self.maskE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], 1, kernel_size=1, stride=1, padding=0)    
            )
        self.getScore = nn.Sequential(    
                nn.Linear( 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear( 512, cfg['num_classes'])
        )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net, self.segE]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, feats_seg):
        dim_bs,dim_c,dim_h,dim_w = feats_seg.size()
        mask = self.maskE(feats_seg)
        # mask = mask - mask.mean(dim=1, keepdim=True)            #.sum(dim=1, keepdim=True)
        # std = mask.std(dim=1, keepdim=True) + 1e-5
        # mask = mask/std
        mask = self.BN(mask)
        mask = self.sig(mask)
        mask = mask.expand(dim_bs,dim_c,dim_h,dim_w)


##feats_seg: torch.Size([16, 512, 32, 32])
##mask 0 : torch.Size([16, 1, 32, 32])

        Q_feat = torch.mul(feats_cls, mask)
        # print('##Q_feat:',Q_feat.size())
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, feats_cls)))
        sa_feats = self.ReLu(self.SAFusion(Q_feat, feats_cls))
        # print('##sa_feats:',sa_feats.size())
        # exit()

        feats_end = F.adaptive_avg_pool2d(sa_feats, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score


####################################

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices')) 
        # feed to auxiliary decoder
        preds_aux = self.auxiliary_decoder(x3)
        # feed to bottleneck
        feats0 = self.bottleneck(x4)
        # feed to ocr module
        context = self.spatial_gather_module(feats0, preds_aux)
        feats = self.object_context_block(feats0, context)

        feats_seg = self.segE(feats)
        feats_cls = self.clsE(feats)
        # feed to segDecoder
        preds = self.segDecoder(feats_seg)         ## [BS, 2, 64, 64] (mask)
        predsLabel = self.clsDecoder(feats_cls, feats_seg)
        # predsLabel = self.getScore(self.clsEncoder(feats).squeeze(3).squeeze(2))    #.view(feats.size(0),-1)


################ add by wj
        if self.mode == 'TRAIN' and targets!=None:
            x3_2, x4_2 = self.transforminputs(self.backbone_net(images2), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to auxiliary decoder
            preds_aux_2 = self.auxiliary_decoder(x3_2)
            # feed to bottleneck
            feats0_2 = self.bottleneck(x4_2)
            # feed to ocr module
            context_2 = self.spatial_gather_module(feats0_2, preds_aux_2)
            feats_2 = self.object_context_block(feats0_2, context_2)

            feats_seg_2 = self.segE(feats_2)
            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder
            preds_2 = self.segDecoder(feats_seg_2)     ## [BS, 2, 64, 64]
            predsLabel_2 = self.clsDecoder(feats_cls_2, feats_seg_2)
            # predsLabel_2 = self.getScore(self.clsEncoder(feats_2).squeeze(3).squeeze(2))
###############

        if self.mode == 'TRAIN' and targets!=None:
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_aux': preds_aux, 'loss_cls': preds, 'loss_cls_2': preds_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds, predsLabel



''' single layer for segE/clsE '''
class Siam8V0(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam8V0, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(1)

        # nn.MaxPool2d(3,strides,1)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            nn.MaxPool2d(3,2,1)
            # nn.ReLU(inplace=True),
            # nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.ReLU(inplace=True)         
            )
        self.segE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.ReLU(inplace=True)         
            )
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.maxpoolSA = nn.MaxPool2d(3,2,1)
        self.maskE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], 1, kernel_size=1, stride=1, padding=0)    
            )
        self.getScore = nn.Sequential(    
                nn.Linear( 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear( 512, cfg['num_classes'])
        )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net, self.segE]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, feats_seg):
        dim_bs,dim_c,dim_h,dim_w = feats_seg.size()
        # mask = self.maskE(feats_seg)
        # # mask = mask - mask.mean(dim=1, keepdim=True)            #.sum(dim=1, keepdim=True)
        # # std = mask.std(dim=1, keepdim=True) + 1e-5
        # # mask = mask/std
        # mask = self.BN(mask)
        # mask = self.sig(mask)
        # mask = mask.expand(dim_bs,dim_c,dim_h,dim_w)

        mask = self.maskE(self.maxpool(feats_seg))
        mask = self.BN(mask)
        mask = self.sig(mask)
        Q_feat =  feats_cls*mask

        # Q_feat = torch.mul(feats_cls, mask)
        sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, feats_cls)))

        feats_end = F.adaptive_avg_pool2d(sa_feats, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

##feats_cls: torch.Size([12, 512, 32, 32])
##feats_seg: torch.Size([12, 512, 64, 64])
##mask: torch.Size([12, 1, 32, 32])
##Q_feat: torch.Size([12, 512, 32, 32])
##sa_feats: torch.Size([12, 512, 16, 16])

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        # print('##targets:',targets.keys())
        # exit()
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices')) 
        # feed to auxiliary decoder
        preds_aux = self.auxiliary_decoder(x3)
        # feed to bottleneck
        feats0 = self.bottleneck(x4)
        # feed to ocr module
        context = self.spatial_gather_module(feats0, preds_aux)
        feats = self.object_context_block(feats0, context)

        feats_seg = self.segE(feats)
        feats_cls = self.clsE(feats)

        # feed to segDecoder
        preds = self.segDecoder(feats_seg)         ## [BS, 2, 64, 64] (mask)
        predsLabel = self.clsDecoder(feats_cls, feats_seg)

        # predsLabel = self.getScore(self.clsEncoder(feats).squeeze(3).squeeze(2))    #.view(feats.size(0),-1)
##feats: torch.Size([12, 512, 64, 64])
##feats_seg 0: torch.Size([12, 512, 64, 64])
##feats_cls 0: torch.Size([12, 512, 32, 32])
##preds: torch.Size([12, 2, 64, 64])


################ add by wj
        if self.mode == 'TRAIN' and targets!=None:
            x3_2, x4_2 = self.transforminputs(self.backbone_net(images2), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to auxiliary decoder
            preds_aux_2 = self.auxiliary_decoder(x3_2)
            # feed to bottleneck
            feats0_2 = self.bottleneck(x4_2)
            # feed to ocr module
            context_2 = self.spatial_gather_module(feats0_2, preds_aux_2)
            feats_2 = self.object_context_block(feats0_2, context_2)

            feats_seg_2 = self.segE(feats_2)
            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder
            preds_2 = self.segDecoder(feats_seg_2)     ## [BS, 2, 64, 64]
            predsLabel_2 = self.clsDecoder(feats_cls_2, feats_seg_2)
            # predsLabel_2 = self.getScore(self.clsEncoder(feats_2).squeeze(3).squeeze(2))
###############

        if self.mode == 'TRAIN' and targets!=None:
            targets['loss_inv'] =  preds
            preds_out = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_out_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_aux': preds_aux, 'loss_cls': preds_out, 'loss_cls_2': preds_out_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds, predsLabel


''' seg mask --> real/fake mask '''
class Siam9V0(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V0, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.ReLU(inplace=True)
        # self.BN = nn.BatchNorm2d(1)

        # nn.MaxPool2d(3,strides,1)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.MaxPool2d(3,2,1)
            # nn.ReLU(inplace=True),
            # nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(decoder_cfg['in_channels']),
            # nn.ReLU(inplace=True)         
            )
        # self.segE = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True)         
        #     )
        # self.maxpool = nn.MaxPool2d(3,2,1)
        self.maxpoolSA = nn.MaxPool2d(3,2,1)
        # self.maskE = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], 1, kernel_size=1, stride=1, padding=0)    
        #     )
        self.getScore = nn.Sequential(    
                nn.Linear( 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear( 512, cfg['num_classes'])
        )

        # unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net, self.segE]
        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        # dim_bs,dim_c,dim_h,dim_w = feats_seg.size()
        # mask = self.maskE(feats_seg)
        # # mask = mask - mask.mean(dim=1, keepdim=True)            #.sum(dim=1, keepdim=True)
        # # std = mask.std(dim=1, keepdim=True) + 1e-5
        # # mask = mask/std
        # mask = self.BN(mask)
        # mask = self.sig(mask)
        # mask = mask.expand(dim_bs,dim_c,dim_h,dim_w)

        # mask = self.maskE(self.maxpool(feats_seg))
        # mask = self.BN(mask)
        # mask = self.sig(mask)
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg


        # Q_feat = torch.mul(feats_cls, mask)
        sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))

        feats_end = F.adaptive_avg_pool2d(sa_feats, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

##feats_cls: torch.Size([12, 512, 32, 32])
##feats_seg: torch.Size([12, 512, 64, 64])
##mask: torch.Size([12, 1, 32, 32])
##Q_feat: torch.Size([12, 512, 32, 32])
##sa_feats: torch.Size([12, 512, 16, 16])

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        # print('##targets:',targets.keys())
        # exit()
        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices')) 
        # feed to auxiliary decoder
        preds_aux = self.auxiliary_decoder(x3)
        # feed to bottleneck
        feats0 = self.bottleneck(x4)
        # feed to ocr module
        context = self.spatial_gather_module(feats0, preds_aux)
        feats = self.object_context_block(feats0, context)

        # feats_seg = self.segE(feats)
        feats_cls = self.clsE(feats)
        # print('##feats_cls:',feats_cls.size(),feats_cls.size(3))


        # feed to segDecoder
        preds = self.segDecoder(feats)         ## [BS, 2, 64, 64] (mask)
        # mask_foreg = preds[:,1,:,:]
        # mask_backg = preds[:,0,:,:]

        mask_foreg = F.interpolate(preds[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(preds[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)

        # exit()
##feats_cls: torch.Size([12, 512, 32, 32]) 32
##mask_foreg: torch.Size([12, 64, 64])
##mask_backg: torch.Size([12, 64, 64])
##mask_foreg: torch.Size([12, 1, 32, 32])
##mask_backg: torch.Size([12, 1, 32, 32])
##predsLabel: torch.Size([12, 2])



################ add by wj
        if self.mode == 'TRAIN' and targets!=None:
            x3_2, x4_2 = self.transforminputs(self.backbone_net(images2), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to auxiliary decoder
            preds_aux_2 = self.auxiliary_decoder(x3_2)
            # feed to bottleneck
            feats0_2 = self.bottleneck(x4_2)
            # feed to ocr module
            context_2 = self.spatial_gather_module(feats0_2, preds_aux_2)
            feats_2 = self.object_context_block(feats0_2, context_2)

            # feats_seg_2 = self.segE(feats_2)
            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder
            # preds_2 = self.segDecoder(feats_seg_2)     ## [BS, 2, 64, 64]
            # predsLabel_2 = self.clsDecoder(feats_cls_2, feats_seg_2)
            # predsLabel_2 = self.getScore(self.clsEncoder(feats_2).squeeze(3).squeeze(2))

            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            # mask_foreg_2 = preds_2[:,1,:,:]
            # mask_backg_2 = preds_2[:,0,:,:]

            mask_foreg_2 = F.interpolate(preds_2[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(preds_2[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
        # print('##preds_aux_2:')
###############

        if self.mode == 'TRAIN' and targets!=None:
            targets['loss_inv'] =  preds
            preds_out = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_out_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux_2 = F.interpolate(preds_aux_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_aux': preds_aux_2, 'loss_cls': preds_out, 'loss_cls_2': preds_out_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds, predsLabel