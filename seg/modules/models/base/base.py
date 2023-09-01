'''
Function:
    Base model for all supported models
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from ...losses import *
from ...backbones import *



'''base model'''
class BaseModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.mode = kwargs.get('mode')
        assert self.mode in ['TRAIN', 'TEST', 'VAL']
        # parse align_corners, normalization layer and activation layer cfg
        self.align_corners, self.norm_cfg, self.act_cfg = cfg['align_corners'], cfg['norm_cfg'], cfg['act_cfg']
        # build backbone
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        raise NotImplementedError('not to be implemented')
    '''transform inputs'''
    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['series'] in ['hrnet']:
                selected_indices = (0, 0, 0, 0)
            else:
                selected_indices = (0, 1, 2, 3)
        outs = []
        for idx in selected_indices:
            # print('##idx:',idx)
            outs.append(x_list[idx])
        return outs
    '''return all layers with learnable parameters'''
    def alllayers(self):
        raise NotImplementedError('not to be implemented')
    '''freeze normalization'''
    def freezenormalization(self):
        for module in self.modules():
            if type(module) in BuildNormalization(only_get_all_supported=True):
                module.eval()
    '''calculate the losses'''
    def calculatelosses(self, predictions, targets, losses_cfg):
        # parse targets
        target_seg = targets['segmentation']
        target_cls_my = targets['clsLabel']
        if 'edge' in targets:
            target_edge = targets['edge']
            num_neg_edge, num_pos_edge = torch.sum(target_edge == 0, dtype=torch.float), torch.sum(target_edge == 1, dtype=torch.float)
            weight_pos_edge, weight_neg_edge = num_neg_edge / (num_pos_edge + num_neg_edge), num_pos_edge / (num_pos_edge + num_neg_edge)
            cls_weight_edge = torch.Tensor([weight_neg_edge, weight_pos_edge]).type_as(target_edge)
        # calculate loss according to losses_cfg
        assert len(predictions) == len(losses_cfg), 'length of losses_cfg should be equal to predictions...'
        losses_log_dict = {}
        # print('##losses_cfg:',losses_cfg.items())
        
        for loss_name, loss_cfg in losses_cfg.items():
            # print('##loss_name:',loss_name)
            if 'edge' in loss_name:
                loss_cfg = copy.deepcopy(loss_cfg)
                loss_cfg_keys = loss_cfg.keys()
                for key in loss_cfg_keys: loss_cfg[key]['opts'].update({'weight': cls_weight_edge})
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=target_edge,
                    loss_cfg=loss_cfg,
                )
            elif 'my' in loss_name:
                # print('##loss_name:',loss_name)
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=target_cls_my,
                    loss_cfg=loss_cfg,
                ) 
            elif 'inv' in loss_name:    # loss_inv
                # print('##loss_name:',loss_name)
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=targets['loss_inv'], # predictions['loss_cls'],
                    loss_cfg=loss_cfg,
                )                       
            else:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=target_seg,
                    loss_cfg=loss_cfg,
                )
        loss = 0
        for key, value in losses_log_dict.items():
            value = value.mean()
            loss += value
            losses_log_dict[key] = value
        losses_log_dict.update({'total': loss})
        # convert losses_log_dict
        for key, value in losses_log_dict.items():
            if dist.is_available() and dist.is_initialized():
                value = value.data.clone()
                dist.all_reduce(value.div_(dist.get_world_size()))
                losses_log_dict[key] = value.item()
            else:
                losses_log_dict[key] = torch.Tensor([value.item()]).type_as(loss)
        
        # print('##target_cls_my:',target_cls_my.size(),target_cls_my)
        # print('##predictions[loss_name]:',predictions['loss_mycls'].size(),predictions['loss_mycls'])
        # exit()
        # return the loss and losses_log_dict
        return loss, losses_log_dict

        ##losses_log_dict: {'loss_aux': tensor([0.], device='cuda:0'), 'loss_cls': tensor([0.], device='cuda:0'), 'total': tensor([0.], device='cuda:0')}

    '''calculate the loss'''
    def calculateloss(self, prediction, target, loss_cfg):
        # print('\n\n##prediction:',prediction.size())
        # print('##target:',target.size())
        # define the supported losses
        supported_losses = {
            'celoss': CrossEntropyLoss,
            'sigmoidfocalloss': SigmoidFocalLoss,
            'binaryceloss': BinaryCrossEntropyLoss,
            'invloss': invarianceLoss,
        }
        if list(loss_cfg.keys())[0]=='invloss':
            pass
        else:
            # format prediction
            if prediction.dim() == 4:
                prediction = prediction.permute((0, 2, 3, 1)).contiguous()
            elif prediction.dim() == 3:
                prediction = prediction.permute((0, 2, 1)).contiguous()
            prediction = prediction.view(-1, prediction.size(-1))            
        # calculate the loss
        loss = 0
        # print('\n\n##loss_cfg:',list(loss_cfg.keys())[0])
        ##loss_cfg: {'celoss': {'scale_factor': 0.4, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}

        for key, value in loss_cfg.items():
            # print('\n##loss_cfg:',key,value)
            assert key in supported_losses, 'unsupport loss type %s...' % key
            if key=='invloss':
                target_iter = target
            else:
                target_iter = target.view(-1)
#            target_iter[target_iter==255] = 1
                if (key in ['binaryceloss']) and hasattr(self, 'onehot'):
                    target_iter = self.onehot(target, self.cfg['num_classes'])
            # print('##key:',key)
            # print('##prediction:',prediction.size())
            # print('##target_iter:',target_iter.size())
            # print('\n\n##prediction:',prediction.size(),prediction[:5,0].cpu().data)   #,sum(prediction)
            # print('##target_iter:',target_iter.size(),target_iter[:5].cpu().data)  #,sum(target_iter)
            
            # print('##loss:',supported_losses[key](
            #     prediction=prediction, 
            #     target=target_iter, 
            #     scale_factor=value['scale_factor'],
            #     **value['opts'] ))
##prediction: torch.Size([131072, 2])   tensor([0.0651, 0.0651, 0.0651,...])
##target_iter: torch.Size([131072])     tensor([255., 255., 255., 255., 255.,...])

            loss += supported_losses[key](
                prediction=prediction, 
                target=target_iter, 
                scale_factor=value['scale_factor'],
                **value['opts']
            )
        # return the loss
        # exit()
        return loss
