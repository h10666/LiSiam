'''
Function:
    train the model
Author:
    Jian Wang
'''
import os
import copy
import torch
import warnings
import argparse
import torch.nn as nn
import torch.distributed as dist
from modules import *
from cfgs import BuildConfig
warnings.filterwarnings('ignore')
import pynvml
## new:
## add another seg loss
## add tensorboard
## Compatible with the C23/C40

## add Xception backbone
## G5: add params_rules
## G6: 1) Order of optimizer and model
##     2) change mean and std for xception
## G8  : Xtensor Xnorm
## G8.1: add avg eval
## G9.2: upgrade training type ( 270 times per epoch ) and seg mask type

try:
    import tensorflow as tf
    print('\n## Tensorboard loading...')
except ImportError:
    print("\n## Tensorflow not installed; No tensorboard logging.")
    tf = None
# tf = None
# def add_summary_value(writer, key, value, iteration):

#         tf.summary.scalar(key, value, step=iteration)
    # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=key, simple_value=value)])
    # writer.add_summary(summary, iteration)


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='sssegmentation is a general framework for our research on strongly supervised semantic segmentation')
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed training', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=1, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', default='', type=str)

## mine:  
    parser.add_argument('--net',  type=str,   default='ocrnetv2',   help='network name')
    parser.add_argument('--Dataset',  type=str,   default='C23',   help='train/test data name')

    parser.add_argument('--PATH_data',    type=str,   default='',   help='video path: /kaggle/FFpp/FFpp:/data/10104002/data/SegData')
    parser.add_argument('--onlyTest',     type=int,   default=0,   help='')
    parser.add_argument('--FROZEN',       type=int,   default=1,   help='')
    parser.add_argument('--NUM_gpus',     type=int,   default=1,   help='')
    parser.add_argument('--evalLabel',    type=int,   default=1,   help='')
    parser.add_argument('--GLabel',       type=int,   default=0,   help='')
    parser.add_argument('--with_ann',     type=int,   default=1,   help='')
    parser.add_argument('--trainBSize',   type=int,   default=16,   help='')
    parser.add_argument('--testBSize',    type=int,   default=16,   help='')
    parser.add_argument('--cropSize',     type=int,   default=256,   help='')
    parser.add_argument('--resize',       type=int,   default=300,   help='')
    parser.add_argument('--loginterval',  type=int,   default=10,   help='')
    parser.add_argument('--max_epochs',   type=int,   default=500,  help='')
    parser.add_argument('--scaleRangeL',   type=int,   default=1,  help='')
    parser.add_argument('--scaleLower',   type=float,   default=0.75,  help='')
    parser.add_argument('--scaleUpper',   type=float,   default=1.5,  help='')
    parser.add_argument('--backupdir',    type=str,   default='',   help='save path: dlv3p_res50os16_train_w1')
    parser.add_argument('--trainFile',    type=str,   default='train',   help='train data txt')
    parser.add_argument('--labelFolder',type=str,   default='labelsGTC40a',   help='path: label file')
    parser.add_argument('--trainFolder',type=str,   default='FFppDFaceC40a',   help='path: label file')
    parser.add_argument('--trainLabelFile',type=str,   default='C40a.dat',   help='path: label file')
    parser.add_argument('--segMaskFolder',type=str,   default='FFppDSegSSIM2',   help='FFppDSegSSIM2 or FFppDSegDIFF')

    parser.add_argument('--typeOPT',      type=str, default='adam',  help='OPTIMIZER')
    parser.add_argument('--LR',           type=float, default=0.001,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    parser.add_argument('--adjustLR',           type=int, default=1,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    parser.add_argument('--typeLR',       type=str,   default='poly',  metavar='W', help='poly or stair')
    parser.add_argument('--periodLR',     type=str,   default='iteration',  metavar='W', help='epoch or iteration')
    parser.add_argument('--paramsRulesL', type=int,   default=0,  help='params_rules')
    parser.add_argument('--clsWeight',    type=float, default=1.0,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    parser.add_argument('--clsWeight2',   type=float, default=2.0,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    parser.add_argument('--invWeight',    type=float, default=1.0,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    parser.add_argument('--segWeight',    type=float, default=1.0,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    parser.add_argument('--segWeight2',    type=float, default=1.0,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    parser.add_argument('--auxWeight',    type=float, default=0.4,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    parser.add_argument('--auxWeight2',    type=float, default=0.4,  metavar='W', help='lr_decay_rate (default: 5e-4)')
    
    parser.add_argument('--jpegCW',    type=float, default=0.5,  metavar='W', help='')
    parser.add_argument('--GBlurW',    type=float, default=0.5,  metavar='W', help='')
    parser.add_argument('--jpegDW',    type=float, default=0.5,  metavar='W', help='')
    parser.add_argument('--GNoiseW',    type=float, default=0.1,  metavar='W', help='')
    parser.add_argument('--colorJW',   type=float, default=0.5,  metavar='W', help='')
    parser.add_argument('--tqdm',         type=bool,  default=False,help='open tqdm or not')
    # parser.add_argument('--eval_method',type=str,   default='frame',   help='')
    args = parser.parse_args()
    return args

best_acc = {'test':0,'val':0,'Gtest1':0,'Gtest2':0,'Gdfdc':0}
best_auc = {'test':0,'val':0,'Gtest1':0,'Gtest2':0,'Gdfdc':0}
best_acc_avg = {'test':0,'val':0,'Gtest1':0,'Gtest2':0,'Gdfdc':0}
best_auc_avg = {'test':0,'val':0,'Gtest1':0,'Gtest2':0,'Gdfdc':0}

def getConfig(args, logger_handle):  # configFileName, PTrainedFile, saveFolderName
	# Create a config file
	cfg, cfg_file_path = BuildConfig(args.cfgfilepath)
# 	cfg = Config.fromfile(args.configFileName)
	cfg.MODEL_CFG['distributed']['is_on'] = False
	if args.NUM_gpus>1:
		cfg.MODEL_CFG['is_multi_gpus'] = True
	else:
		cfg.MODEL_CFG['is_multi_gpus'] = False

	# Since we use ony one GPU, BN is used instead of SyncBN
	cfg.MODEL_CFG['norm_cfg'] = dict(type='batchnorm2d', opts={})

	# modify num classes of the model in decode/auxiliary head
	cfg.MODEL_CFG['num_classes'] = 2 # 8
	cfg.MODEL_CFG['type'] = args.net # OCRNetV1
	cfg.MODEL_CFG['auxiliary']['in_channels'] = 728
	cfg.MODEL_CFG['bottleneck']['in_channels'] = 1024


	cfg.DATALOADER_CFG['train']['batch_size'] = args.trainBSize
	cfg.DATALOADER_CFG['val']['batch_size'] = args.testBSize
	cfg.DATALOADER_CFG['test']['batch_size'] = args.testBSize
	# cfg.DATALOADER_CFG['gtest']['batch_size'] = args.testBSize

	cfg.COMMON_CFG['train']['loginterval']= args.loginterval #500
	cfg.COMMON_CFG['train']['backupdir']   = args.backupdir     #'dlv3p_res50os16_train_w1'
	cfg.COMMON_CFG['train']['logfilepath'] = args.backupdir + '/train.log'    #'dlv3p_res50os16_train_w1/train.log'

	cfg.OPTIMIZER_CFG['type'] = args.typeOPT
	cfg.OPTIMIZER_CFG[cfg.OPTIMIZER_CFG['type']]['learning_rate'] = args.LR
	cfg.OPTIMIZER_CFG['max_epochs'] = args.max_epochs
	cfg.OPTIMIZER_CFG['policy']['type'] = args.typeLR   # 'poly' or 'stair'
	cfg.OPTIMIZER_CFG['adjust_period'] = args.periodLR  #  epoch or iteration
	if args.paramsRulesL == 1:
		cfg.OPTIMIZER_CFG['params_rules'] = {'backbone_net': 0.1, 'others': 1.0}
    # else:
	# 	cfg.OPTIMIZER_CFG['params_rules'] = {}

	if args.scaleRangeL == 1:
		scaleRange = (args.scaleLower, args.scaleUpper)
	elif args.scaleRangeL == 0:
		scaleRange = None

	cfg.DATASET_CFG['train']= {
		'type': 'deepfakes',
		'rootdir': args.PATH_data,
		'set': args.trainFile,
		'with_ann': args.with_ann,
		'img_dir': args.trainFolder, # trainFolder   'FFppDFaceC40a'
		'ann_dir': args.segMaskFolder,  #'FFppDSegSSIM2',
		'clsLabel': args.trainLabelFile,   # trainLabelFile 'C40a.dat'
        'labelFolder': args.labelFolder,
		'aug_opts_1': [
		             # ('Resize', {'output_size': (args.resize, args.resize), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
		            #  ('Resize', {'output_size': (args.resize, args.resize), 'keep_ratio': False, 'scale_range': None}),
		             ('Resize', {'output_size': (args.resize, args.resize), 'keep_ratio': False, 'scale_range': scaleRange}),
		             ('RandomCrop', {'crop_size': (args.cropSize, args.cropSize), 'one_category_max_ratio': 0.75}),
		             ('RandomFlip', {'flip_prob': 0.5}),
		             ('RandomShuffle', {'shuffle_prob': 0.5,'num_patch': 10}),
		            #  ('PhotoMetricDistortion', {}),
                     ],
		'aug_opts_2': [
		             ('jpegC',{'compress_prob': args.jpegCW, 'valueD':10, 'valueT':100}),    
		             ('GBlur',{'blur_prob': args.GBlurW, 'valueD':0.1, 'valueT':1.0}),       
		             ('jpegD',{'compress_prob': args.jpegDW, 'valueD':100, 'valueT':200}),   
		             ('GNoise',{'prob': args.GNoiseW}),   
                     ('colorJitter',{'prob': args.colorJW}), 
                     ],
		'aug_opts_3': [
		             ('XToTensor', {}),
		             ('XNormalize', {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}),
		             ('Padding', {'output_size': (args.cropSize, args.cropSize), 'data_type': 'tensor'}),
                     ]
	    }

	cfg.DATASET_CFG['val']={
		'type': 'deepfakes',
		'rootdir': args.PATH_data,
		'set': 'val',
		'with_ann': 0,			# args.with_ann		
		'img_dir': args.trainFolder,  # 'FFppDFaceC40a',
		'ann_dir': args.segMaskFolder,  #'FFppDSegSSIM2',
		'clsLabel': args.trainLabelFile, #  'C40a.dat', #args.clsLabel,
        'labelFolder': args.labelFolder,
		'aug_opts_1': [],
		'aug_opts_2': [],
		'aug_opts_3': [
		             ('Resize', {'output_size': (args.resize, args.resize), 'keep_ratio': False, 'scale_range': None}),
                    # ('Resize', {'output_size': (args.resize, args.resize), 'keep_ratio': True, 'scale_range': None}),
		             ('CenterCrop', {'crop_size': (args.cropSize, args.cropSize), 'one_category_max_ratio': 0.75}),
		             ('XToTensor', {}),
		             ('XNormalize', {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}),
                    ],
	    }	

# 'loss_aux_2': preds_aux_2, 'loss_cls_2': preds_out_2, 'loss_mycls_2':predsLabel_2

##weight setting
	if args.net == 'LiSiam_AS1': ## only inv and no map supervision
		del cfg.LOSSES_CFG['loss_aux']
		del cfg.LOSSES_CFG['loss_aux_2']
		del cfg.LOSSES_CFG['loss_cls']
		del cfg.LOSSES_CFG['loss_cls_2']  
		cfg.LOSSES_CFG['loss_inv']   = {'invloss': {'scale_factor': args.invWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		cfg.LOSSES_CFG['loss_mycls'] = {'celoss':  {'scale_factor': args.clsWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		cfg.LOSSES_CFG['loss_mycls_2'] = {'celoss':{'scale_factor': args.clsWeight2,'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
	elif args.net == 'LiSiam_AS2': 
		del cfg.LOSSES_CFG['loss_aux']
		del cfg.LOSSES_CFG['loss_cls']
		del cfg.LOSSES_CFG['loss_inv']
		del cfg.LOSSES_CFG['loss_mycls']
		cfg.LOSSES_CFG['loss_aux_2']   = {'celoss': {'scale_factor': args.auxWeight2, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		cfg.LOSSES_CFG['loss_cls_2']   = {'celoss': {'scale_factor': args.segWeight2, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		cfg.LOSSES_CFG['loss_mycls_2'] = {'celoss':{'scale_factor': args.clsWeight2,'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
	elif args.net == 'LiSiam_AS3':
		del cfg.LOSSES_CFG['loss_aux']
		cfg.LOSSES_CFG['loss_aux_2']   = {'celoss': {'scale_factor': args.auxWeight2, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		del cfg.LOSSES_CFG['loss_cls']
		cfg.LOSSES_CFG['loss_cls_2']   = {'celoss': {'scale_factor': args.segWeight2, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		cfg.LOSSES_CFG['loss_inv']   = {'invloss': {'scale_factor': args.invWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		del cfg.LOSSES_CFG['loss_mycls']
		cfg.LOSSES_CFG['loss_mycls_2'] = {'celoss':{'scale_factor': args.clsWeight2,'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
	elif args.net == 'LiSiam_AS5':
		cfg.LOSSES_CFG['loss_aux']   = {'celoss': {'scale_factor': args.auxWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		del cfg.LOSSES_CFG['loss_aux_2'] 
		cfg.LOSSES_CFG['loss_cls']   = {'celoss': {'scale_factor': args.segWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		del cfg.LOSSES_CFG['loss_cls_2']
		cfg.LOSSES_CFG['loss_inv']   = {'invloss': {'scale_factor': args.invWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		cfg.LOSSES_CFG['loss_mycls'] = {'celoss':  {'scale_factor': args.clsWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		del cfg.LOSSES_CFG['loss_mycls_2']
	elif args.net == 'LiSiam_AS6':
		del cfg.LOSSES_CFG['loss_aux']
		del cfg.LOSSES_CFG['loss_aux_2']
		del cfg.LOSSES_CFG['loss_cls']
		del cfg.LOSSES_CFG['loss_cls_2']  
		cfg.LOSSES_CFG['loss_inv']   = {'invloss': {'scale_factor': args.invWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		cfg.LOSSES_CFG['loss_mycls'] = {'celoss':  {'scale_factor': args.clsWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		cfg.LOSSES_CFG['loss_mycls_2'] = {'celoss':{'scale_factor': args.clsWeight2,'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
	elif args.net == 'LiSiam_AS7':
		del cfg.LOSSES_CFG['loss_aux']
		del cfg.LOSSES_CFG['loss_aux_2']
		del cfg.LOSSES_CFG['loss_cls']
		del cfg.LOSSES_CFG['loss_cls_2']  
		cfg.LOSSES_CFG['loss_inv']   = {'invloss': {'scale_factor': args.invWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		cfg.LOSSES_CFG['loss_mycls'] = {'celoss':  {'scale_factor': args.clsWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		del cfg.LOSSES_CFG['loss_mycls_2'] 
	else:
		cfg.LOSSES_CFG['loss_aux']   = {'celoss': {'scale_factor': args.auxWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		cfg.LOSSES_CFG['loss_aux_2']   = {'celoss': {'scale_factor': args.auxWeight2, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		cfg.LOSSES_CFG['loss_cls']   = {'celoss': {'scale_factor': args.segWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		cfg.LOSSES_CFG['loss_cls_2']   = {'celoss': {'scale_factor': args.segWeight2, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}    
		cfg.LOSSES_CFG['loss_inv']   = {'invloss': {'scale_factor': args.invWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		cfg.LOSSES_CFG['loss_mycls'] = {'celoss':  {'scale_factor': args.clsWeight, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}}
		cfg.LOSSES_CFG['loss_mycls_2'] = {'celoss':{'scale_factor': args.clsWeight2,'opts': {'ignore_index': 255, 'reduction': 'mean'}}}



######
	if args.FROZEN==1 or args.FROZEN==2:
		cfg.OPTIMIZER_CFG['filter_params'] = True
	elif args.FROZEN==0:
		cfg.OPTIMIZER_CFG['filter_params'] = False
	else:
		# print('## bug in Label FROZEN:',args.FROZEN)
		logger_handle.info('## bug in Label FROZEN: %s' % (args.FROZEN))


	# logger_handle.info('Model Used: %s, Backbone used: %s' % (cfg.MODEL_CFG['type'], cfg.MODEL_CFG['backbone']['type']))
	logger_handle.info('')
	logger_handle.info('##setting:')
	logger_handle.info('##model: %s' % (args.net))
	logger_handle.info('##Backbone: %s' % (cfg.MODEL_CFG['backbone']['type']))
	logger_handle.info('##jpegCW: %s, GBlurW: %s, jpegDW: %s, GNoiseW: %s, colorJW: %s' % (args.jpegCW, args.GBlurW, args.jpegDW, args.GNoiseW, args.colorJW))
	logger_handle.info('##clsW: %s, clsW2: %s, invW: %s, auxW: %s, auxW2: %s, segW: %s, segW2: %s' % (args.clsWeight, args.clsWeight2, args.invWeight, args.auxWeight, args.auxWeight2, args.segWeight, args.segWeight2))
	logger_handle.info('##opt ##opt:      %s' % (args.typeOPT))
	logger_handle.info('##opt ##LR:       %s' % (args.LR))
	logger_handle.info('##opt ##adjustLR: %s' % (args.adjustLR))
	logger_handle.info('##opt ##typeLR:   %s' % (args.typeLR))
	logger_handle.info('##opt ##periodLR: %s' % (args.periodLR))
	logger_handle.info('##opt ##pRulesL:  %s' % (args.paramsRulesL))
	logger_handle.info('##batchSize:  %s' % (args.trainBSize))
	logger_handle.info('##scaleRangeL:%s' % (args.scaleRangeL))
	logger_handle.info('##scaleLower: %s' % (args.scaleLower))
	logger_handle.info('##scaleUpper: %s' % (args.scaleUpper))
	logger_handle.info('##resize:     %s' % (args.resize))
	logger_handle.info('##cropSize:   %s' % (args.cropSize))	
	return cfg, cfg_file_path

   




def get_average(data_all, _label_='score', dataset='FFppC0'):
	if dataset == 'DFDC':
		frame_num = 10
	else:
		frame_num = 100

	new_list = []
	each_score = 0.0
	if _label_ == 'label':
		for i in range(data_all.shape[0]):
			each_score = each_score + data_all[i]
			if (i+1)%frame_num==0:
				label = np.int64( (each_score/frame_num)>0.5 )
				new_list.append(np.array([ label ]))
				each_score = 0
		new_data = np.concatenate(tuple([_ for _ in new_list]), axis = 0)
	elif  _label_ == 'score':
		for i in range(data_all.shape[0]):
			each_score = each_score + data_all[i]
			if (i+1)%frame_num==0:
				score =  each_score/frame_num 
				new_list.append(np.array([ score ]))
				each_score = 0
		new_data = np.concatenate(tuple([_ for _ in new_list]), axis = 0)
	else:
		print('##bug in get_average:',_label_)
	return new_data	



from sklearn.metrics import auc
from sklearn import metrics
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

   

'''Tester'''
class Tester():
    def __init__(self, **kwargs):
        # set attribute
        for key, value in kwargs.items(): setattr(self, key, value)
        self.use_cuda = torch.cuda.is_available()

        # init distributed testing if necessary
        distributed_cfg = self.cfg.MODEL_CFG['distributed']
        # if distributed_cfg['is_on']:
        #     dist.init_process_group(backend=distributed_cfg.get('backend', 'nccl'))

    def get_DATASET_CFG(self, args, datasetName, cfg): 
        labelFolderName = args.Dataset
        if datasetName == 'celebv1':
            setTestFile, testImgDir, clsTestLabel = 'celebv1', 'celeb_v1_test', 'celebv1.dat'
        elif datasetName == 'celebv2':
            setTestFile, testImgDir, clsTestLabel = 'celebv2', 'celeb_v2_test', 'celebv2.dat'
        elif datasetName == 'FFppC40':
            setTestFile, testImgDir, clsTestLabel, labelFolderName = 'testC40', 'FFppDFaceC40a', 'C40a.dat', 'C40'
        elif datasetName == 'FFppC23':
            setTestFile, testImgDir, clsTestLabel, labelFolderName = 'testC23', 'FFppDFaceC23a', 'C23a.dat' , 'C23'
        elif datasetName == 'FFppC0':
            setTestFile, testImgDir, clsTestLabel, labelFolderName = 'testC0', 'FFppDFaceC0a', 'C0a.dat', 'C0'
        elif datasetName == 'DFDC':
            setTestFile, testImgDir, clsTestLabel  = 'DFDC', 'DFDC', 'DFDC.dat'
        else:
            print('##please input the correct [datasetName]') 
            exit()

        cfg.DATASET_CFG['test']={
            'type': 'deepfakes',
            'rootdir': args.PATH_data,
            'with_ann': 0, #args.with_ann,

            'set': setTestFile, #'test',
            'img_dir': testImgDir, #'FFppDFaceC0',

            'ann_dir': args.segMaskFolder,  #'FFppDSegSSIM2',
            'clsLabel': clsTestLabel,
            'labelFolder': 'labelsGT{}a'.format(labelFolderName),  #args.labelFolder,
            'aug_opts_1': [],
            'aug_opts_2': [],
            'aug_opts_3': [
                            ('Resize', {'output_size': (args.resize, args.resize), 'keep_ratio': False, 'scale_range': None}),
                            # ('Resize', {'output_size': (args.resize, args.resize), 'keep_ratio': True, 'scale_range': None}),
                            ('CenterCrop', {'crop_size': (args.cropSize, args.cropSize), 'one_category_max_ratio': 0.75}),
                            # ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
                            # ('ToTensor', {}),
                            ('XToTensor', {}),
                            ('XNormalize', {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}),
                            ],
        }
        return cfg.DATASET_CFG['test']

    '''start tester'''
    def start(self, model, onlyTest=0, datasetName='celebv1'):
        cfg, logger_handle, use_cuda, cmd_args, cfg_file_path = self.cfg, self.logger_handle, self.use_cuda, self.cmd_args, self.cfg_file_path
        distributed_cfg, common_cfg = self.cfg.MODEL_CFG['distributed'], self.cfg.COMMON_CFG['train']
        # if datasetName == 'celebv1':
        cfg.DATASET_CFG['test'] = self.get_DATASET_CFG(args, datasetName, cfg)

        # instanced dataset and dataloader
        datasetTest = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        assert datasetTest.num_classes == cfg.MODEL_CFG['num_classes'], 'parsed config file %s error...' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        dataloaderTest = BuildParallelDataloader(mode='TEST', dataset=datasetTest, cfg=copy.deepcopy(dataloader_cfg))

################# delete model
        # # instanced model
        if onlyTest == 1:
            cfg.MODEL_CFG['backbone']['pretrained'] = False
            # model = BuildModel(cfg=copy.deepcopy(cfg.MODEL_CFG), mode='TEST')
            if use_cuda: 
                model = model.cuda()
                
            # load checkpoints
            checkpoints = loadcheckpoints(cmd_args.checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
            try:
                model.load_state_dict(checkpoints['model'])
            except Exception as e:
                logger_handle.warning(str(e) + '\n' + 'Try to load checkpoints by using strict=False...')
                model.load_state_dict(checkpoints['model'], strict=False)
#################

        # print config
        if cmd_args.local_rank == 0:
            logger_handle.info('Dataset used: %s, Number of images: %s' % (cfg.DATASET_CFG['train']['type'], len(datasetTest)))

        # set eval
        model.eval()
        # start to test
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        inference_cfg = copy.deepcopy(cfg.INFERENCE_CFG)
        with torch.no_grad():
            targets_list = []
            predScore_list = []
            predLabel_list = []
            # for batch_idx, samples_ in pbar:
            for batch_idx, samples_ in enumerate(tqdm(dataloaderTest, disable=(not cmd_args.tqdm))):
            # for batch_idx, samples_ in enumerate(dataloaderTest):
                samples, samples2 = samples_
                targets_clsLabel = samples['clsLabel'].type(FloatTensor)                
                imageids, images_ori= samples['id'], samples['image']  # widths, heights, gts  = samples['width'], samples['height'], samples['groundtruth']
                infer_tricks, use_probs_before_resize = inference_cfg['tricks'], inference_cfg['tricks']['use_probs_before_resize']
                align_corners = model.align_corners if hasattr(model, 'align_corners') else model.module.align_corners

                preds, outputs = self.inference(model, images_ori.type(FloatTensor), inference_cfg, datasetTest.num_classes, use_probs_before_resize)
                predScore_list.append(F.softmax(outputs)[:, 1].cpu().data.numpy())
                targets_list.append(targets_clsLabel.cpu().data.numpy())
                predLabel_list.append(outputs.cpu().data.max(1)[1].numpy() )

            acc_frame, auc_frame, acc_avg, auc_avg = self.eval_perf(predScore_list, targets_list, predLabel_list, cmd_args, datasetName)
            logger_handle.info('##test ##acc_frame:%0.6f  ##acc_avg:%0.6f' % (acc_frame, acc_avg))
            logger_handle.info('##test ##auc_frame:%0.6f  ##auc_avg:%0.6f' % (auc_frame, auc_avg))
            logger_handle.info('##File: %s done.' % (datasetName))  #  cmd_args.setTestFile

        return acc_frame, auc_frame, acc_avg, auc_avg

    def eval_perf(self, predScore_list, targets_list, predLabel_list, cmd_args, dataset):
        # label_all_numpy = np.concatenate(tuple([_ for _ in label_all]), axis=0)
        targetLabels = np.concatenate(tuple([_ for _ in targets_list]), axis=0)
        predLabels = np.concatenate(tuple([_ for _ in predLabel_list]), axis=0)
        predScores = np.concatenate(tuple([_ for _ in predScore_list]), axis=0)
        # np.save('targetLabels.npy',targetLabels)
        # np.save('predLabels.npy',predLabels)
        # np.save('predScores.npy',predScores)
        acc_frame = metrics.accuracy_score(targetLabels, predLabels)
        fpr, tpr, thresholds = metrics.roc_curve(targetLabels, predScores, pos_label=1)
        auc_frame = metrics.auc(fpr, tpr)

        # if cmd_args.eval_method == 'avg':
        targetLabels_avg = get_average(targetLabels, _label_='label', dataset=dataset)
        predLabels_avg = get_average(predLabels, _label_='label', dataset=dataset)
        predScores_avg = get_average(predScores, _label_='score', dataset=dataset)

        acc_avg = metrics.accuracy_score(targetLabels_avg, predLabels_avg)
        fpr, tpr, thresholds = metrics.roc_curve(targetLabels_avg, predScores_avg, pos_label=1)
        auc_avg = metrics.auc(fpr, tpr)
        return acc_frame, auc_frame, acc_avg, auc_avg

    '''inference'''
    def inference(self, model, images, inference_cfg, num_classes, use_probs_before_resize=False):
        assert inference_cfg['mode'] in ['whole', 'slide']
        if inference_cfg['mode'] == 'whole':
            if use_probs_before_resize: 
                outputs = F.softmax(model(images), dim=1)
            else: 
                outputs = model(images)    # model(images, targets, images2, targets2, cfg.LOSSES_CFG)
        return outputs

'''main'''
def main(args):

    # initialize logger_handle
    # logger_handle = Logger(common_cfg['logfilepath'])
    logger_handle = Logger(args.backupdir + '/train.log')

    cfg, cfg_file_path = getConfig(args, logger_handle)
    # check backup dir
    common_cfg = cfg.COMMON_CFG['train']
    checkdir(common_cfg['backupdir'])

    # number of gpus, for distribued training, only support a process for a GPU
    ngpus_per_node = torch.cuda.device_count()



    logger_handle.info('##get: %s GPUs.' % (ngpus_per_node))
    if (ngpus_per_node != args.nproc_per_node) and cfg.MODEL_CFG['distributed']['is_on']:
        if args.local_rank == 0: logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default...')
        ngpus_per_node = args.nproc_per_node

    # instanced Tester
    clientTester = Tester(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)

    if args.onlyTest == 0:
        clientValer = Valer(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)

        # instanced Trainer
        client = Trainer(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)

        client.start(clientTester, clientValer)
    elif args.onlyTest == 1:
################# only test model
        cfg.MODEL_CFG['FROZEN']=args.FROZEN
        model = BuildModel(cfg=copy.deepcopy(cfg.MODEL_CFG), mode='TEST')
        # clientTester.start(model, args.onlyTest, datasetName= 'DFDC' )
        # clientTester.start(model, args.onlyTest, datasetName= 'celebv1')
        clientTester.start(model, args.onlyTest, datasetName= 'FFppC0')
        clientTester.start(model, args.onlyTest, datasetName= 'FFppC23')
        clientTester.start(model, args.onlyTest, datasetName= 'FFppC40')


        # clientTester.start(model, args.onlyTest, datasetName= 'celebv1')
        # clientTester.start(model, args.onlyTest, datasetName= 'celebv2')
        # clientTester.start(model, args.onlyTest, datasetName= 'DFDC' )


#################
    else:
        logger_handle.info('\n##please enter the correct onlyTest value: 0/1..')
        exit()

'''debug'''
if __name__ == '__main__':
    # parse arguments
    args = parseArgs()
    # tf_summary_writer = tf and tf.compat.v1.summary.FileWriter(args.backupdir)
    tf_summary_writer = tf and tf.summary.create_file_writer(args.backupdir)
    # tf_summary_writer = tf and tf.contrib.summary.create_file_writer(args.backupdir)

    main(args)
    
