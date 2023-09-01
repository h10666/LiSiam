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
from .ocrnetbase import OCRNetBase, OCRNetSABase, SELayer




class LiSiam(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.BN = nn.BatchNorm2d(512)
        self.se = SELayer(1024, 16)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )

        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
        preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
        feats0 = self.bottleneck(x4)                                # feed to bottleneck
        context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
        feats = self.object_context_block(feats0, context)
        feats_cls = self.clsE(feats)

        preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)

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
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux_2 = F.interpolate(preds_aux_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_aux': preds_aux, 'loss_aux_2': preds_aux_2, 'loss_cls': preds_out, 'loss_cls_2': preds_out_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
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




## unsupervised learning
## no segmentation map supervision
## only have inv
class LiSiam_AS1(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam_AS1, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.BN = nn.BatchNorm2d(512)
        self.se = SELayer(1024, 16)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )

        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
        preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
        feats0 = self.bottleneck(x4)                                # feed to bottleneck
        context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
        feats = self.object_context_block(feats0, context)
        feats_cls = self.clsE(feats)

        preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)

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



## single branch
class LiSiam_AS2(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam_AS2, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.BN = nn.BatchNorm2d(512)
        self.se = SELayer(1024, 16)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )

        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)

        if self.mode == 'TRAIN' and targets!=None:
            ## second branch: transformed image
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
        

            # targets['loss_inv'] =  preds
            # preds_out = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_out_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            # preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux_2 = F.interpolate(preds_aux_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_aux_2': preds_aux_2, 'loss_cls_2': preds_out_2, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        else:
            ## first branch:
            x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
            preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
            feats0 = self.bottleneck(x4)                                # feed to bottleneck
            context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
            feats = self.object_context_block(feats0, context)
            feats_cls = self.clsE(feats)
            preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)
            predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)
            mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
            mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
            predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)
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



## remove first branch and retain inv loss 
class LiSiam_AS3(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam_AS3, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.BN = nn.BatchNorm2d(512)
        self.se = SELayer(1024, 16)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )

        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
        preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
        feats0 = self.bottleneck(x4)                                # feed to bottleneck
        context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
        feats = self.object_context_block(feats0, context)
        feats_cls = self.clsE(feats)

        preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)


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
            # preds_out = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_out_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            # preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux_2 = F.interpolate(preds_aux_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                # predictions={'loss_aux': preds_aux, 'loss_aux_2': preds_aux_2, 'loss_cls': preds_out, 'loss_cls_2': preds_out_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                predictions={'loss_aux_2': preds_aux_2, 'loss_cls_2': preds_out_2, 'loss_inv': preds_2, 'loss_mycls_2':predsLabel_2}, 
                targets=targets, 
                losses_cfg=losses_cfg
            )

        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)
        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)
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

## remove Mask-guided transformer and 2fc
class LiSiam_AS4(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam_AS4, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        # self.BN = nn.BatchNorm2d(512)
        # self.se = SELayer(1024, 16)


        # self.clsE = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),   
        #     nn.Dropout2d(0.1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
        #     nn.BatchNorm2d(decoder_cfg['in_channels']),   
        #     nn.Dropout2d(0.1),
        #     )

        self.clsDR = nn.Sequential(
                nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=1),
            )
        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls):

        # Q_feat =  feats_cls*mask_foreg
        # KV_feat = feats_cls*mask_backg
        # sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        # feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = self.clsDR(feats_cls)
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        score = feats_end.view(feats_end.size(0), -1)
        # score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
        preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
        feats0 = self.bottleneck(x4)                                # feed to bottleneck
        context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
        feats = self.object_context_block(feats0, context)
        # feats_cls = self.clsE(feats)

        preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)
        # print('##preds:',preds.size())

        # predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)
        # mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        # mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        predsLabel = self.clsDecoder(preds)
        # print('##predsLabel:',predsLabel.size())
        # exit()

        if self.mode == 'TRAIN' and targets!=None:
            x3_2, x4_2 = self.transforminputs(self.backbone_net(images2), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to auxiliary decoder
            preds_aux_2 = self.auxiliary_decoder(x3_2)
            # feed to bottleneck
            feats0_2 = self.bottleneck(x4_2)
            # feed to ocr module
            context_2 = self.spatial_gather_module(feats0_2, preds_aux_2)
            feats_2 = self.object_context_block(feats0_2, context_2)
            # feats_cls_2 = self.clsE(feats_2)

            # feed to decoder
            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            # predsMask_2 = F.softmax(preds_2, dim=1)
            # mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            # mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(preds_2)
        
        if self.mode == 'TRAIN' and targets!=None:
            targets['loss_inv'] =  preds
            preds_out = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_out_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_aux_2 = F.interpolate(preds_aux_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_aux': preds_aux, 'loss_aux_2': preds_aux_2, 'loss_cls': preds_out, 'loss_cls_2': preds_out_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
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



## remove second branch and retain inv loss 
class LiSiam_AS5(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam_AS5, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.BN = nn.BatchNorm2d(512)
        self.se = SELayer(1024, 16)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )

        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        FreezeLayers = [self.backbone_net]
        # unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, FreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
        preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
        feats0 = self.bottleneck(x4)                                # feed to bottleneck
        context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
        feats = self.object_context_block(feats0, context)
        feats_cls = self.clsE(feats)

        preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)

        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)
        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)

        if self.mode == 'TRAIN' and targets!=None:
            x3_2, x4_2 = self.transforminputs(self.backbone_net(images2), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to auxiliary decoder
            preds_aux_2 = self.auxiliary_decoder(x3_2)
            # feed to bottleneck
            feats0_2 = self.bottleneck(x4_2)
            # feed to ocr module
            context_2 = self.spatial_gather_module(feats0_2, preds_aux_2)
            feats_2 = self.object_context_block(feats0_2, context_2)
            # feats_cls_2 = self.clsE(feats_2)

            # feed to decoder
            preds_2 = self.segDecoder(feats_2)         ## [BS, 2, 64, 64] (mask)
            # predsMask_2 = F.softmax(preds_2, dim=1)
            # mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            # mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            # predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
        if self.mode == 'TRAIN' and targets!=None:
            targets['loss_inv'] =  preds
            preds_out = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            # preds_out_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            # preds_aux_2 = F.interpolate(preds_aux_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                # predictions={'loss_aux': preds_aux, 'loss_aux_2': preds_aux_2, 'loss_cls': preds_out, 'loss_cls_2': preds_out_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                predictions={'loss_aux': preds_aux, 'loss_cls': preds_out, 'loss_inv': preds_2, 'loss_mycls':predsLabel}, 
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
        



## unsupervised learning v2
## no segmentation map supervision
## only have inv
class LiSiam_AS6(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam_AS6, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.BN = nn.BatchNorm2d(512)
        self.se = SELayer(1024, 16)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )

        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls):
        Q_feat =  feats_cls #*mask_foreg
        KV_feat = feats_cls #*mask_backg
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
        preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
        feats0 = self.bottleneck(x4)                                # feed to bottleneck
        context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
        feats = self.object_context_block(feats0, context)
        feats_cls = self.clsE(feats)

        preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        # mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        # mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        predsLabel = self.clsDecoder(feats_cls)

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

            # mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            # mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            predsLabel_2 = self.clsDecoder(feats_cls_2)
        
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



## unsupervised learning
## only have inv and only one cls, no seg
class LiSiam_AS7(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam_AS7, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.BN = nn.BatchNorm2d(512)
        self.se = SELayer(1024, 16)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )

        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
        preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
        feats0 = self.bottleneck(x4)                                # feed to bottleneck
        context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
        feats = self.object_context_block(feats0, context)
        feats_cls = self.clsE(feats)

        preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)
        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)

        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)

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
            # predsMask_2 = F.softmax(preds_2, dim=1)

            # mask_foreg_2 = F.interpolate(predsMask_2[:,1,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)
            # mask_backg_2 = F.interpolate(predsMask_2[:,0,:,:].unsqueeze(1), size=(feats_cls_2.size(2), feats_cls_2.size(3)), mode='bilinear', align_corners=self.align_corners)

            # predsLabel_2 = self.clsDecoder(feats_cls_2, mask_foreg_2, mask_backg_2)
        
        if self.mode == 'TRAIN' and targets!=None:
            targets['loss_inv'] =  preds
            return self.calculatelosses(
                predictions={'loss_inv': preds_2, 'loss_mycls':predsLabel}, 
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



## only remove second seg decoder
class LiSiam_AS9(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(LiSiam_AS9, self).__init__(cfg, **kwargs)    
        decoder_cfg = cfg['decoder']

        self.BN = nn.BatchNorm2d(512)
        self.se = SELayer(1024, 16)

        self.clsE = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),            
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(decoder_cfg['in_channels']),   
            nn.Dropout2d(0.1),
            )

        self.getScore = nn.Sequential(    
            nn.Linear( 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear( 1024, cfg['num_classes'])
            )

        unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
        self.pretrain(cfg, unFreezeLayers)

    def clsDecoder(self, feats_cls, mask_foreg, mask_backg):
        Q_feat =  feats_cls*mask_foreg
        KV_feat = feats_cls*mask_backg
        sa_feats = self.BN(self.SAFusion(Q_feat, KV_feat))
        feats_end = self.se(torch.cat((feats_cls, sa_feats),1))
        feats_end = F.adaptive_avg_pool2d(feats_end, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):

        h, w = x.size(2), x.size(3)
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))  # feed to backbone network
        preds_aux = self.auxiliary_decoder(x3)                      # feed to auxiliary decoder
        feats0 = self.bottleneck(x4)                                # feed to bottleneck
        context = self.spatial_gather_module(feats0, preds_aux)     # feed to ocr module
        feats = self.object_context_block(feats0, context)
        feats_cls = self.clsE(feats)

        preds = self.segDecoder(feats)              # feed to segDecoder  ## [BS, 2, 64, 64] (mask)

        predsMask = F.softmax(preds, dim=1)         ## [BS, 2, 64, 64] (mask)
        mask_foreg = F.interpolate(predsMask[:,1,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        mask_backg = F.interpolate(predsMask[:,0,:,:].unsqueeze(1), size=(feats_cls.size(2), feats_cls.size(3)), mode='bilinear', align_corners=self.align_corners)
        predsLabel = self.clsDecoder(feats_cls, mask_foreg, mask_backg)

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
            # preds_out_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            # preds_aux_2 = F.interpolate(preds_aux_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                # predictions={'loss_aux': preds_aux, 'loss_aux_2': preds_aux_2, 'loss_cls': preds_out, 'loss_cls_2': preds_out_2, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
                predictions={'loss_aux': preds_aux, 'loss_cls': preds_out, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
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


class Siam9V4_1_single(OCRNetSABase):
    def __init__(self, cfg, **kwargs):
        super(Siam9V4_1_single, self).__init__(cfg, **kwargs)    
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

        preds_out = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)


        if self.mode == 'TRAIN' and targets!=None:
            return self.calculatelosses(
                predictions={'loss_aux': preds_aux, 'loss_mycls':predsLabel, 'loss_cls':preds_out}, 
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


