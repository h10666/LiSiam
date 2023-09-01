'''base config for deeplabv3plus'''
# config for dataset
DATASET_CFG = {
    'train': {
        'type': '',
        'set': 'train',
        'rootdir': '',
        'aug_opts': [('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
                     ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
                     ('RandomFlip', {'flip_prob': 0.5}),
                     ('PhotoMetricDistortion', {}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),
                     ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),]
    },
    'test': {
        'type': '',
        'set': 'val',
        'rootdir': '',
        'aug_opts': [('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),]
    }
}
# config for dataloader
DATALOADER_CFG = {
    'train': {
        'type': ['nondistributed', 'distributed'][0],
        'batch_size': 16,
        'num_workers': 8,
        'shuffle': True,
        'pin_memory': True,
        'drop_last': True,
    },
    'test': {
        'type': ['nondistributed', 'distributed'][0],
        'batch_size': 1,
        'num_workers': 8,
        'shuffle': False,
        'pin_memory': True,
        'drop_last': False,
    }
}
# config for optimizer
OPTIMIZER_CFG = {
    'type': 'sgd',
    'sgd': {
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    },
    'max_epochs': 0,
    'params_rules': {},
    'policy': {
        'type': 'poly',
        'opts': {'power': 0.9, 'max_iters': None, 'num_iters': None, 'num_epochs': None}
    },
    'adjust_period': ['iteration', 'epoch'][0],
}
# config for losses
LOSSES_CFG = {
    'loss_aux': {
        'celoss': {'scale_factor': 0.4, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}
    },
    'loss_cls': {
        'celoss': {'scale_factor': 1.0, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}
    },
    'loss_mycls': {
        'celoss': {'scale_factor': 1.0, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}
    },
    'loss_inv': {
        'invloss': {'scale_factor': 3.0, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}
    },    
}
# config for model
MODEL_CFG = {
    'benchmark': True,
    'num_classes': -1,
    'align_corners': False,
    'is_multi_gpus': True,
    'type': 'deeplabv3plus',
    'distributed': {'is_on': False, 'backend': 'nccl'},
#    'distributed': {'is_on': True, 'backend': 'nccl'},
    'norm_cfg': {'type': 'syncbatchnorm', 'opts': {}},
#    'norm_cfg': {'type': 'syncbatchnorm', 'opts': {}},
    'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    'backbone': {
        'type': 'resnet101',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 16,
        'use_stem': True,
        'selected_indices': (0, 1, 2, 3),
    },
    'aspp': {
        'in_channels': 2048,
        'out_channels': 512,
        'dilations': [1, 6, 12, 18],
    },
    'shortcut': {
        'in_channels': 256,
        'out_channels': 48,
    },
    'decoder': {
        'in_channels': 560,
        'out_channels': 512,
        'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 1024,
        'out_channels': 512,
        'dropout': 0.1,
    }
}
# config for inference
INFERENCE_CFG = {
    'mode': 'whole',
    'opts': {}, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': False
    }
}
# config for common
COMMON_CFG = {
    'train': {
        'backupdir': '',
        'logfilepath': '',
        'loginterval': 500,
        'saveinterval': 1
    },
    'test': {
        'backupdir': '',
        'logfilepath': '',
        'resultsavepath': ''
    }
}
