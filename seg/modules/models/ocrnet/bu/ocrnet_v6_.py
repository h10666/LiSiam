'''
Function:
    Implementation of OCRNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import *
from ..base import BaseModel
from .objectcontext import ObjectContextBlock
from .spatialgather import SpatialGatherModule


'''OCRNet'''
class OCRNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(OCRNet, self).__init__(cfg, **kwargs)
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

################## add by wj
        # self.clsEncoder = nn.Sequential(
        #     nn.Conv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=3, stride=2, padding=1),
        #     nn.Dropout2d(decoder_cfg['dropout']),
        #     nn.Conv2d(cfg['num_classes'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        #     # nn.AdaptiveAvgPool2d((1,1))
        # )
        self.segE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1)
        self.clsE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=1, padding=1)


        self.ReLu = nn.ReLU(inplace=False)

        self.Qf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.Kf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.Vf = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.W_z = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512)
            )

        self.getScore = nn.Sequential(    
                nn.Linear( 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear( 512, cfg['num_classes'])
        )
        # self.maskDecoder = nn.Sequential(
        #     nn.Conv2d(cfg['num_classes'], 1, kernel_size=1, stride=1, padding=0)
        # )
        # self.saE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1)
        self.maskE = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=4, stride=4, padding=1)
        # nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.Dropout2d(decoder_cfg['dropout']),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=0),
        #     # nn.AdaptiveAvgPool2d((1,1))
        # )
        # self.maskFeatEncoder = nn.Sequential(
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=3, stride=2, padding=1),
        #     nn.Dropout2d(decoder_cfg['dropout']),
        #     nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['in_channels'], kernel_size=1, stride=1, padding=0),
        #     # nn.AdaptiveAvgPool2d((1,1))
        # )

        if cfg['FROZEN']==1:
            for p in self.parameters():
                p.requires_grad = False            
            unFreezeLayers = [self.segDecoder, self.auxiliary_decoder, self.object_context_block, self.spatial_gather_module, self.bottleneck, self.backbone_net]
            for unFreezeLayer in unFreezeLayers:
                self.unfreeze(unFreezeLayer)
            print('##unfreeze seg layers and freeze cls layers')
            # for p in self.parameters():
            #     p.requires_grad = False
                # print('##p:',p,'   ')
        elif cfg['FROZEN']==0:
            for p in self.parameters():
                p.requires_grad = True
            print('##unfreeze all layers')
        else:
            print('## bug in Label FROZEN:',cfg['FROZEN'])

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True
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


    def clsDecoder(self, feats, feats_seg):
        dim_bs, dim_c, dim_h, dim_w = feats.size()
        feats_cls = self.clsE(feats)
        preds_mask = self.ReLu(feats_seg)
        feat_mask = torch.mul(feats_cls, preds_mask)
        Q_feat = self.maskE(feat_mask)
        # print('##Q_feat:',Q_feat.size())

        sa_feats = self.ReLu(self.SAFusion(Q_feat, feats))  #self.saE()
        # print('##feats:',feats.size())
        # exit()

        feats_end = F.adaptive_avg_pool2d(sa_feats, (1, 1))
        feats_end = feats_end.view(feats_end.size(0), -1)
        score = self.getScore(feats_end)
        return score


##feat_mask: torch.Size([16, 512, 64, 64])
##Q_feat: torch.Size([16, 512, 16, 16])
##sa_feats: torch.Size([16, 512, 16, 16])
##score: torch.Size([16, 2])






####################################

    '''forward'''
    def forward(self, x, targets=None, images2=None, targets2=None, losses_cfg=None):
        # print('##targets:',targets['segmentation'].size())
        # print('##targets2:',targets2['segmentation'].size())

        h, w = x.size(2), x.size(3)
        # feed to backbone network
        x3, x4 = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # print('##x3:',x3.size())
        # print('##x4:',x4.size())
##x3: torch.Size([12, 728, 16, 16])
##x4: torch.Size([12, 1024, 16, 16])

        # feed to auxiliary decoder
        preds_aux = self.auxiliary_decoder(x3)
        # print('##preds_aux:',preds_aux.size())
        # exit()
## hr18:
##x3: torch.Size([12, 270, 64, 64])
##x4: torch.Size([12, 270, 64, 64])
##preds_aux: torch.Size([12, 2, 64, 64])

## hr48:
##x3: torch.Size([12, 720, 64, 64])
##x4: torch.Size([12, 720, 64, 64])
##preds_aux: torch.Size([12, 2, 64, 64])

        # feed to bottleneck
        feats0 = self.bottleneck(x4)
        # feed to ocr module
        context = self.spatial_gather_module(feats0, preds_aux)
        feats = self.object_context_block(feats0, context)

        feats_seg = self.segE(feats)

        # feed to segDecoder
        preds = self.segDecoder(feats_seg)         ## [BS, 2, 64, 64] (mask)
        predsLabel = self.clsDecoder(feats, feats_seg)


##feats: torch.Size([16, 512, 64, 64])
##feats_seg: torch.Size([16, 512, 64, 64])



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

            # feed to decoder
            preds_2 = self.segDecoder(feats_seg_2)     ## [BS, 2, 64, 64]
            predsLabel_2 = self.clsDecoder(feats_2, feats_seg_2)


##feats_2: torch.Size([8, 512, 64, 64])
##preds_2: torch.Size([8, 2, 64, 64])
##predsLabel_2: torch.Size([8, 2])

###############

        # if self.mode == 'TRAIN' and targets!=None:
            preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            preds_2 = F.interpolate(preds_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)    # add by wj
            preds_aux = F.interpolate(preds_aux, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': preds, 'loss_cls_2': preds_2, 'loss_aux': preds_aux, 'loss_inv': preds_2, 'loss_mycls':predsLabel, 'loss_mycls_2':predsLabel_2}, 
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
            'decoder': self.decoder
        }


            # print('##preds_2:',preds_2.size())
            # print('##preds_2 0:\n',preds_2[:3,0,:4,:5])
            # print('##preds_2 1:\n',preds_2[:3,1,:4,:5])
            
            # print('##preds_2 sum:\n',sum_[:3,:4,:5])
            # import scipy.misc
            # from PIL import Image

            # Image.fromarray(preds_2[0,0,:,:].cpu().data.numpy()).convert('L').save('0.png')
            # Image.fromarray(preds_2[0,1,:,:].cpu().data.numpy()).convert('L').save('1.png')


            # print('##predsLabel_2:',predsLabel_2.size())
            # exit()
            # predsLabel_2 = self.clsDecoder(preds_2).squeeze(3).squeeze(2)

            # predsLabel_2 = self.clsDecoder(preds_2).squeeze(3).squeeze(2)






        # print('##Q_feat:',Q_feat.size())
        # print('##feats_3:',feats_3.size())
        # print('##feats_end:',feats_end.size())
        
        # print('##score:',score.size())


        # return according to the mode


                # print('##preds:',preds.size())      

        # predsLabel = self.clsDecoder(preds).squeeze(3).squeeze(2)
        # print('##predsLabel:',predsLabel.size())