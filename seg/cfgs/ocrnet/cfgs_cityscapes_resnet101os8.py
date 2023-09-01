'''define the config file for cityscapes and resnet101os8'''
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG['train'].update(
    {
        'type': 'cityscapes',
        'rootdir': 'data/CityScapes',
        'aug_opts': [('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
                     ('RandomCrop', {'crop_size': (512, 1024), 'one_category_max_ratio': 0.75}),
                     ('RandomFlip', {'flip_prob': 0.5}),
                     ('PhotoMetricDistortion', {}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),
                     ('Padding', {'output_size': (512, 1024), 'data_type': 'tensor'}),]
    }
)
DATASET_CFG['test'].update(
    {
        'type': 'cityscapes',
        'rootdir': 'data/CityScapes',
        'aug_opts': [('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),],
    }
)
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
DATALOADER_CFG['train'].update(
    {
        'batch_size': 8,
    }
)
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update(
    {
        'max_epochs': 440
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 19,
        'backbone': {
            'type': 'resnet101',
            'series': 'resnet',
            'pretrained': True,
            'outstride': 8,
            'use_stem': True,
            'selected_indices': (2, 3),
        },
        'auxiliary': {
            'in_channels': 1024,
            'out_channels': 512,
            'dropout': 0,
        },
        'bottleneck': {
            'in_channels': 2048,
            'out_channels': 512,
        },
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'ocrnet_resnet101os8_cityscapes_train',
        'logfilepath': 'ocrnet_resnet101os8_cityscapes_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'ocrnet_resnet101os8_cityscapes_test',
        'logfilepath': 'ocrnet_resnet101os8_cityscapes_test/test.log',
        'resultsavepath': 'ocrnet_resnet101os8_cityscapes_test/ocrnet_resnet101os8_cityscapes_results.pkl'
    }
)