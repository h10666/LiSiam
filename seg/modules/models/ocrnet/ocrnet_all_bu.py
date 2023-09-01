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
from .objectcontext import ObjectContextBlock
from .spatialgather import SpatialGatherModule
from .ocrnetbase import OCRNetBase, OCRNetSABase
## change Cls and Seg Encoder
'''OCRNet v7'''

# class OCRNetBase(BaseModel):
#     def __init__(self, cfg, **kwargs):
#         super(OCRNetBase, self).__init__(cfg, **kwargs)
#         align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg

#         # build bottleneck
#         bottleneck_cfg = cfg['bottleneck']
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(bottleneck_cfg['in_channels'], bottleneck_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
#             BuildNormalization(norm_cfg['type'], (bottleneck_cfg['out_channels'], norm_cfg['opts'])),
#             BuildActivation(act_cfg['type'], **act_cfg['opts']),
#         )
#         # build spatial gather module
#         spatialgather_cfg = {
#             'scale': cfg['spatialgather']['scale']
#         }
#         self.spatial_gather_module = SpatialGatherModule(**spatialgather_cfg)
#         # build object context block
#         ocb_cfg = cfg['objectcontext']
#         self.object_context_block = ObjectContextBlock(
#             in_channels=ocb_cfg['in_channels'], 
#             transform_channels=ocb_cfg['transform_channels'], 
#             scale=ocb_cfg['scale'],
#             align_corners=align_corners,
#             norm_cfg=copy.deepcopy(norm_cfg),
#             act_cfg=copy.deepcopy(act_cfg),
#         )

#         # build auxiliary decoder
#         auxiliary_cfg = cfg['auxiliary']
#         self.auxiliary_decoder = nn.Sequential(
#             nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
#             BuildNormalization(norm_cfg['type'], (auxiliary_cfg['out_channels'], norm_cfg['opts'])),
#             BuildActivation(act_cfg['type'], **act_cfg['opts']),
#             nn.Dropout2d(auxiliary_cfg['dropout']),
#             nn.Conv2d(auxiliary_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
#         )

#         # build decoder
#         decoder_cfg = cfg['decoder']
#         self.segDecoder = nn.Sequential(
#             nn.Dropout2d(decoder_cfg['dropout']),
#             nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
#         )
#         # freeze normalization layer if necessary
#         if cfg.get('is_freeze_norm', False): self.freezenormalization()

#     '''return all layers'''
#     def alllayers(self):
#         return {
#             'backbone_net': self.backbone_net,
#             'auxiliary_decoder': self.auxiliary_decoder,
#             'bottleneck': self.bottleneck,
#             'spatial_gather_module': self.spatial_gather_module,
#             'object_context_block': self.object_context_block,
#             'decoder': self.decoder
#         }

#     def freeze(self, layer):
#         for child in layer.children():
#             for param in child.parameters():
#                 param.requires_grad = False
#     def unfreeze(self, layer):
#         for child in layer.children():
#             for param in child.parameters():
#                 param.requires_grad = True

#     def pretrain(self, cfg, unFreezeLayers):
#         if cfg['FROZEN']==1:
#             for p in self.parameters():
#                 p.requires_grad = False            
            
#             for unFreezeLayer in unFreezeLayers:
#                 self.unfreeze(unFreezeLayer)
#             print('##unfreeze seg layers and freeze cls layers')
#         elif cfg['FROZEN']==0:
#             for p in self.parameters():
#                 p.requires_grad = True
#             print('##unfreeze all layers')
#         else:
#             print('## bug in Label FROZEN:',cfg['FROZEN'])






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


'''  '''
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
        self.ReLu = nn.ReLU(inplace=True)

          


        # self.segE = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     nn.ReLU(inplace=True)
        #     )
        # self.clsE2 = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True)         
        #     )
        # self.clsE1 = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True)         
        #     )
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


'''  '''
class SiamV4(OCRNetBase):
    def __init__(self, cfg, **kwargs):
        super(SiamV4, self).__init__(cfg, **kwargs)    

################## add by wj
        decoder_cfg = cfg['decoder']

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
        # self.sig = nn.Sigmoid()  
        self.BN = nn.BatchNorm2d(1)

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
        mask = self.BN(mask)
        mask = self.ReLu(mask)
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


''' remove expand and add * '''
class SiamV5(OCRNetBase):
    def __init__(self, cfg, **kwargs):
        super(SiamV5, self).__init__(cfg, **kwargs)    

################## add by wj
        decoder_cfg = cfg['decoder']

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
        self.BN = nn.BatchNorm2d(1)

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
        mask = self.BN(mask)
        mask = self.sig(mask)
        # mask = mask.expand(dim_bs,dim_c,dim_h,dim_w)


##feats_seg: torch.Size([16, 512, 32, 32])
##mask 0 : torch.Size([16, 1, 32, 32])

        Q_feat =  feats_cls*mask      # torch.mul(feats_cls, mask)
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





''' simple cls/seg Encoder '''
class SiamV6(OCRNetBase):
    def __init__(self, cfg, **kwargs):
        super(SiamV6, self).__init__(cfg, **kwargs)    

################## add by wj
        decoder_cfg = cfg['decoder']

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
        self.BN = nn.BatchNorm2d(1)

        self.ReLu = nn.ReLU(inplace=False)


        self.segE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)
        self.clsE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)


        # self.clsE = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True)         
        #     )
        # self.segE = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),
        #     # nn.ReLU(inplace=True)         
        #     )
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
        mask = self.BN(mask)
        mask = self.sig(mask)
        # mask = mask.expand(dim_bs,dim_c,dim_h,dim_w)


##feats_seg: torch.Size([16, 512, 32, 32])
##mask 0 : torch.Size([16, 1, 32, 32])

        Q_feat =  feats_cls*mask      # torch.mul(feats_cls, mask)
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


''' simple cls/seg Encoder '''
''' reproduce best '''
class Siam6V1(OCRNetBase):
    def __init__(self, cfg, **kwargs):
        super(Siam6V1, self).__init__(cfg, **kwargs)    

################## add by wj
        decoder_cfg = cfg['decoder']

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
        self.BN = nn.BatchNorm2d(1)

        self.ReLu = nn.ReLU(inplace=False)


        self.segE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)
        self.clsE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)

        self.maskE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], 1, kernel_size=1, stride=1, padding=0)    
            )
        
        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net, self.segE]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, feats_seg):
        dim_bs,dim_c,dim_h,dim_w = feats_seg.size()
        # mask = self.maskE(feats_seg)
        # mask = self.BN(mask)
        # mask = self.sig(mask)

        mask = self.ReLu(feats_seg)
##feats_seg: torch.Size([16, 512, 32, 32])
##mask 0 : torch.Size([16, 1, 32, 32])

        # Q_feat =  feats_cls*mask      # torch.mul(feats_cls, mask)
        Q_feat =  torch.mul(feats_cls, mask)
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

''' simple cls/seg Encoder '''
''' reproduce best '''
'''  softmax --> /N'''
class Siam6V2(OCRNetBase):
    def __init__(self, cfg, **kwargs):
        super(Siam6V2, self).__init__(cfg, **kwargs)    

################## add by wj
        decoder_cfg = cfg['decoder']

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
        self.BN = nn.BatchNorm2d(1)

        self.ReLu = nn.ReLU(inplace=False)


        self.segE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)
        self.clsE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)

        self.maskE = nn.Conv2d(decoder_cfg['in_channels'], 1, kernel_size=1, stride=1, padding=0)    
            # nn.Sequential()
        
        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net, self.segE]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, feats_seg):
        dim_bs,dim_c,dim_h,dim_w = feats_seg.size()
        # mask = self.maskE(feats_seg)
        # mask = self.BN(mask)
        # mask = self.sig(mask)

        mask = self.ReLu(feats_seg)
##feats_seg: torch.Size([16, 512, 32, 32])
##mask 0 : torch.Size([16, 1, 32, 32])

        # Q_feat =  feats_cls*mask      # torch.mul(feats_cls, mask)
        Q_feat =  torch.mul(feats_cls, mask)
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

        N = attn.size(-1) 
        attn_div_C = attn / N
        # attn = self.softmax(attn)
        # attn_div_C = self.dropout(attn)

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



''' make mask by SGENet '''
class Siam8V0(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam8V0, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(1)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),
            nn.MaxPool2d(3,2,1)     
            )
        self.segE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
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

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            )

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

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))

        feats_end = F.adaptive_avg_pool2d(sa_feats, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder

            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            # mask_foreg_2 = preds_2[:,1,:,:]
            # mask_backg_2 = preds_2[:,0,:,:]

            mask_foreg_2 = F.interpolate(preds_2[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(preds_2[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
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



''' 9v0: seg mask --> real/fake mask '''
''' 9v1: add maxpoolSA --> saE'''
class Siam9V1(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V1, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.ReLU(inplace=True)
        # self.BN = nn.BatchNorm2d(1)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            )
        self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)

        # self.maxpoolSA = nn.MaxPool2d(3,2,1)
        # self.maskE = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], 1, kernel_size=1, stride=1, padding=0)    
        #     )
        self.getScore = nn.Sequential(    
                nn.Linear( 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear( 512, cfg['num_classes'])
        )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        # sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))
        sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, KV_feat)))

        feats_end = F.adaptive_avg_pool2d(sa_feats, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder

            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            # mask_foreg_2 = preds_2[:,1,:,:]
            # mask_backg_2 = preds_2[:,0,:,:]

            mask_foreg_2 = F.interpolate(preds_2[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(preds_2[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
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


from torch import nn


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

''' 9v0: seg mask --> real/fake mask '''
''' 9v1: add maxpoolSA --> saE'''
''' 9v2: add concat and se'''
class Siam9V2(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V2, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(512)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            )
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)


        self.getScore = nn.Sequential(    
                nn.Linear( 1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear( 1024, cfg['num_classes'])
        )

        self.se = SELayer(1024, 16)

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        # print('##feats_cls:',feats_cls.size())

        # sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, KV_feat)))
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        # print('##sa_feats:',sa_feats.size())
        
##feats_cls: torch.Size([12, 512, 19, 19])
##sa_feats: torch.Size([12, 512, 19, 19])
##feats_end: torch.Size([12, 1024, 19, 19])


        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        # print('##feats_end:',feats_end.size())

        # exit()

        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

        # feats_seg = self.segE(feats)
        feats_cls = self.clsE(feats)
        # print('##feats_cls:',feats_cls.size(),feats_cls.size(3))

        # feed to segDecoder
        preds = self.segDecoder(feats)         ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)
        # mask_foreg = preds[:,1,:,:]
        # mask_backg = preds[:,0,:,:]

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        # print('##preds:', preds, preds.size())
        # print('##preds:', F.softmax(preds, dim=1), preds.size())
        # exit()

        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)


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

            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder

            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            predsMask_2 = F.softmax(preds_2, dim=1)
            # mask_foreg_2 = preds_2[:,1,:,:]
            # mask_backg_2 = preds_2[:,0,:,:]

            mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
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


''' 9v0: seg mask --> real/fake mask '''
''' 9v1: add maxpoolSA --> saE'''
''' 9v2: add concat and se'''
''' 9v3: add two dropout2D '''
class Siam9V3(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V3, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(512)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.2),
            )
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)


        self.getScore = nn.Sequential(    
                nn.Linear( 1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear( 1024, cfg['num_classes'])
        )

        self.se = SELayer(1024, 16)

        # self.dp = nn.Dropout2d(0.1)

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        # print('##feats_cls:',feats_cls.size())

        # sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, KV_feat)))
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        # print('##sa_feats:',sa_feats.size())
        
##feats_cls: torch.Size([12, 512, 19, 19])
##sa_feats: torch.Size([12, 512, 19, 19])
##feats_end: torch.Size([12, 1024, 19, 19])


        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        # print('##feats_end:',feats_end.size())

        # exit()

        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

        # feats_seg = self.segE(feats)
        feats_cls = self.clsE(feats)
        # print('##feats_cls:',feats_cls.size(),feats_cls.size(3))

        # feed to segDecoder
        preds = self.segDecoder(feats)         ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)
        # mask_foreg = preds[:,1,:,:]
        # mask_backg = preds[:,0,:,:]

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        # print('##preds:', preds, preds.size())
        # print('##preds:', F.softmax(preds, dim=1), preds.size())
        # exit()

        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)


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

            feats_cls_2 = self.clsE(feats_2)
            # feed to decoder

            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            predsMask_2 = F.softmax(preds_2, dim=1)
            # mask_foreg_2 = preds_2[:,1,:,:]
            # mask_backg_2 = preds_2[:,0,:,:]

            mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
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




''' 9v0: seg mask --> real/fake mask '''
''' 9v1: add maxpoolSA --> saE'''
''' 9v2: add concat and se'''
''' 9v3: add two dropout2D '''
''' 9v4: xiugai clsE '''
class Siam9V4(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V4, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(512)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)

        self.getScore = nn.Sequential(    
                nn.Linear( 1024, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear( 1024, cfg['num_classes'])
        )

        self.se = SELayer(1024, 16)

        # self.dp = nn.Dropout2d(0.1)

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        # sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, KV_feat)))
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        
##feats_cls: torch.Size([12, 512, 19, 19])
##sa_feats: torch.Size([12, 512, 19, 19])
##feats_end: torch.Size([12, 1024, 19, 19])

        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        # print('##feats_end:',feats_end.size())

        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

        feats_cls = self.clsE(feats)

        # feed to segDecoder
        preds = self.segDecoder(feats)         ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)


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

            feats_cls_2 = self.clsE(feats_2)

            # feed to decoder
            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            predsMask_2 = F.softmax(preds_2, dim=1)

            mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
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


''' 9v0: seg mask --> real/fake mask '''
''' 9v1: add maxpoolSA --> saE'''
''' 9v2: add concat and se'''
''' 9v3: add two dropout2D '''  
''' Siam9V4_1: with dropout '''         
class Siam9V4_1(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V4_1, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(512)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)

        self.getScore = nn.Sequential(    
                nn.Linear( 1024, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear( 1024, cfg['num_classes'])
        )

        self.se = SELayer(1024, 16)

        # self.dp = nn.Dropout2d(0.1)

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        # sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, KV_feat)))
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        
##feats_cls: torch.Size([12, 512, 19, 19])
##sa_feats: torch.Size([12, 512, 19, 19])
##feats_end: torch.Size([12, 1024, 19, 19])

        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        # print('##feats_end:',feats_end.size())

        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

        feats_cls = self.clsE(feats)

        # feed to segDecoder
        preds = self.segDecoder(feats)         ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)


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

            feats_cls_2 = self.clsE(feats_2)

            # feed to decoder
            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            predsMask_2 = F.softmax(preds_2, dim=1)

            mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
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

    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'auxiliary_decoder': self.auxiliary_decoder,
            'bottleneck': self.bottleneck,
            'spatial_gather_module': self.spatial_gather_module,
            'object_context_block': self.object_context_block,
            'segDecoder': self.segDecoder,
            'clsE': self.clsE,
            'getScore': self.getScore,
            'se': self.se,
        }


class Siam9V4_1_inv(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V4_1_inv, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(512)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)

        self.getScore = nn.Sequential(    
                nn.Linear( 1024, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear( 1024, cfg['num_classes'])
        )

        self.se = SELayer(1024, 16)

        # self.dp = nn.Dropout2d(0.1)

        # unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        freezeLayers = [self.backbone_net]
        # self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.clsE, self.getScore, self.Qf, self.Kf, self.Vf, self.W_z, self.se, self.BN

        self.pretrain(cfg, freezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        # sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, KV_feat)))
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        
##feats_cls: torch.Size([12, 512, 19, 19])
##sa_feats: torch.Size([12, 512, 19, 19])
##feats_end: torch.Size([12, 1024, 19, 19])

        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        # print('##feats_end:',feats_end.size())

        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

        feats_cls = self.clsE(feats)

        # feed to segDecoder
        preds = self.segDecoder(feats)         ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)


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

            feats_cls_2 = self.clsE(feats_2)

            # feed to decoder
            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            predsMask_2 = F.softmax(preds_2, dim=1)

            mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
        if self.mode == 'TRAIN' and targets!=None:
            targets['loss_inv'] =  preds

            return self.calculatelosses(
                predictions={'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds, predsLabel
        
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'auxiliary_decoder': self.auxiliary_decoder,
            'bottleneck': self.bottleneck,
            'spatial_gather_module': self.spatial_gather_module,
            'object_context_block': self.object_context_block,
            'segDecoder': self.segDecoder,
            'clsE': self.clsE,
            'getScore': self.getScore,
            'se': self.se,
        }


''' 9v0: seg mask --> real/fake mask '''
''' 9v1: add maxpoolSA --> saE'''
''' 9v2: add concat and se'''
''' 9v3: add two dropout2D '''
''' Siam9V4_1: with 2 dropout(0.1) '''
''' Siam9V4_3: with 1 dropout(0.1) '''
class Siam9V4_3(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V4_3, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(512)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            )
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)

        self.getScore = nn.Sequential(    
                nn.Linear( 1024, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear( 1024, cfg['num_classes'])
        )

        self.se = SELayer(1024, 16)

        # self.dp = nn.Dropout2d(0.1)

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        # sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, KV_feat)))
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        
##feats_cls: torch.Size([12, 512, 19, 19])
##sa_feats: torch.Size([12, 512, 19, 19])
##feats_end: torch.Size([12, 1024, 19, 19])

        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        # print('##feats_end:',feats_end.size())

        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

        feats_cls = self.clsE(feats)

        # feed to segDecoder
        preds = self.segDecoder(feats)         ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)


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

            feats_cls_2 = self.clsE(feats_2)

            # feed to decoder
            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            predsMask_2 = F.softmax(preds_2, dim=1)

            mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
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


import math
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter


''' 9v0: seg mask --> real/fake mask '''
''' 9v1: add maxpoolSA --> saE'''
''' 9v2: add concat and se'''
''' 9v3: add two dropout2D '''
''' 9v4: xiugai clsE '''
''' Siam9V4_4: add FCA block '''
class Siam9V4_4(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V4_4, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.sig = nn.Sigmoid()  
        self.ReLu = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(512)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            # nn.Dropout2d(0.1),
            )
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)

        self.getScore = nn.Sequential(    
                nn.Linear( 1024, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear( 1024, cfg['num_classes'])
        )

        self.attn = MultiSpectralAttentionLayer(channel=512, dct_h=19, dct_w=19, reduction = 16)

        self.se = SELayer(1024, 16)

        # self.dp = nn.Dropout2d(0.1)

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]

        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg

        # sa_feats = self.ReLu(self.maxpoolSA(self.SAFusion(Q_feat, KV_feat)))
        # sa_feats = self.ReLu(self.saE(self.SAFusion(Q_feat, KV_feat)))
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))

        
##feats_cls: torch.Size([12, 512, 19, 19])
##sa_feats: torch.Size([12, 512, 19, 19])
##feats_end: torch.Size([12, 1024, 19, 19])

        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        # print('##feats_end:',feats_end.size())

        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

####################################
##targets: dict_keys(['segmentation', 'edge', 'clsLabel'])

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

        feats_cls = self.clsE(feats)
        feats_cls = self.attn(feats_cls) + feats_cls
        # _b, _c, _h, _w =  feats_cls.size()

        # print('##feats_cls:',feats_cls.size())

        # exit()
        ##feats_cls: torch.Size([16, 512, 19, 19])


        # feed to segDecoder
        preds = self.segDecoder(feats)         ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)

        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)


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

            feats_cls_2 = self.clsE(feats_2)

            # feed to decoder
            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            predsMask_2 = F.softmax(preds_2, dim=1)

            mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
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
