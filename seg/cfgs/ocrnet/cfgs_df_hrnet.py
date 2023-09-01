'''define the config file for deepfakes and hrnetv2-w48'''
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG['train'].update(
    {
        'type': 'deepfakes',
        'rootdir': 'data/deepfakes',
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
        'type': 'deepfakes',
        'rootdir': 'data/deepfakes',
        'aug_opts': [('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),],
    }
)

DATASET_CFG['gtest'].update(
    {
        'type': 'deepfakes',
        'rootdir': '/home/wj/data/FFpp/seg',
        'aug_opts': [('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
                     ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                     ('ToTensor', {}),],
    }
)
DATASET_CFG['val'].update(
    {
        'type': 'deepfakes',
        'rootdir': '/home/wj/data/FFpp/seg',
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
        'max_epochs': 30
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 2,
        'backbone': {
            'type': 'hrnetv2_w48',
            'series': 'hrnet',
            'pretrained': True,
            'selected_indices': (0, 0),
        },
        'auxiliary': {
            'in_channels': sum([48, 96, 192, 384]),
            'out_channels': 512,
            'dropout': 0,
        },
        'bottleneck': {
            'in_channels': sum([48, 96, 192, 384]),
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
        'backupdir': 'ocrnet_hrnetv2w48_cityscapes_train',
        'logfilepath': 'ocrnet_hrnetv2w48_cityscapes_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'ocrnet_hrnetv2w48_cityscapes_test',
        'logfilepath': 'ocrnet_hrnetv2w48_cityscapes_test/test.log',
        'resultsavepath': 'ocrnet_hrnetv2w48_cityscapes_test/ocrnet_hrnetv2w48_cityscapes_results.pkl'
    }
)
COMMON_CFG['val'].update(
    {
        'backupdir': 'ocrnet_hrnetv2w48_cityscapes_val',
        'logfilepath': 'ocrnet_hrnetv2w48_cityscapes_val/val.log',
        'resultsavepath': 'ocrnet_hrnetv2w48_cityscapes_val/ocrnet_hrnetv2w48_cityscapes_results.pkl'
    }
)

