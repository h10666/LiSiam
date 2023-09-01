'''
Function:
    some utils for optimizers
Author:
    wang jian
'''
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad


'''adjust learning rate'''
def adjustLearningRate(optimizer, optimizer_cfg=None):          # policy_cfg['opts']['num_epochs']
    # parse and check the config for optimizer
    policy_cfg, selected_optim_cfg = optimizer_cfg['policy'], optimizer_cfg[optimizer_cfg['type']]
    if ('params_rules' in optimizer_cfg) and (optimizer_cfg['params_rules']):
        assert len(optimizer.param_groups) == len(optimizer_cfg['params_rules'])
    # adjust the learning rate according the policy
    if policy_cfg['type'] == 'poly':
        base_lr = selected_optim_cfg['learning_rate']
        min_lr = selected_optim_cfg.get('min_lr', base_lr * 0.01)
        num_iters, max_iters, power = policy_cfg['opts']['num_iters'], policy_cfg['opts']['max_iters'], policy_cfg['opts']['power']
        coeff = (1 - num_iters / max_iters) ** power
        target_lr = coeff * (base_lr - min_lr) + min_lr
        for param_group in optimizer.param_groups:
            if ('params_rules' in optimizer_cfg) and (optimizer_cfg['params_rules']):
                param_group['lr'] = target_lr * optimizer_cfg['params_rules'][param_group['name']]
            else:
                param_group['lr'] = target_lr
    elif policy_cfg['type'] == 'stair':
        target_lr = selected_optim_cfg['learning_rate']*0.8**(int((policy_cfg['opts']['num_epochs']-10)/3))
        # if policy_cfg['opts']['num_epochs'] % 3 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr        #param_group['lr'] * 0.8
        # target_lr = param_group['lr']
    else:
        raise ValueError('Unsupport policy %s...' % policy)
    return target_lr


'''clip gradient'''
def clipGradients(params, max_norm=35, norm_type=2):
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        clip_grad.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)