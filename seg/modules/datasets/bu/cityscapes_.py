'''
Function:
    load the CityScapes dataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
from .base import *
import pickle

'''CityScapes dataset'''
class CityScapesDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    # clsid2label = {
    #     -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    #     7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 
    #     15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 
    #     23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 
    #     31: 16, 32: 17, 33: 18, 255:1
    # }

    clsid2label = {
        0: 0, 255: 1 }
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg, **kwargs):
        super(CityScapesDataset, self).__init__(mode, logger_handle, dataset_cfg, **kwargs)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        # print('##rootdir:',rootdir)
        # import os
        # /home/wj/00code/seg/sssegmentation/ssseg/data/CityScapes
        # /home/wj/00code/seg/sssegmentation
        # print('##get pwd:',os.getcwd())
        self.image_dir = os.path.join(rootdir, 'FFppDFaceC0')	# , dataset_cfg['set']
        self.ann_dir = os.path.join(rootdir, 'FFppDSegSSIM2')	# , dataset_cfg['set']
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
        self.clsLabelFile = pickle.load(open(os.path.join(rootdir,'clsLabel.dat'),'rb')) 
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, imageid+'.png')
        annpath = os.path.join(self.ann_dir, imageid.replace('leftImg8bit', 'gtFine_labelIds')+'.png')
        sample = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', True))  # self.clsLabelFile, imageid, 
        
        # x = int(sample['segmentation'].shape[0]/2)
        # print('##data ##target 1:',sample['segmentation'].shape,sample['segmentation'][x,x:x+10])
        # print('##key:',imageid)
        # print('##value:',self.clsLabelFile[imageid])
        # exit()

        sample.update({'id': imageid})
        sample.update({'clsLabel': self.clsLabelFile[imageid]})
        # print('\n\n##sample:',sample.keys())
        # print('##sample:',sample)
        # exit()
        if self.mode == 'TRAIN':
            sample = self.synctransform(sample, 'without_totensor_normalize_pad')
            sample['edge'] = self.generateedge(sample['segmentation'].copy())
            sample = self.synctransform(sample, 'only_totensor_normalize_pad')
            # print('##data ##target 2:',sample['segmentation'].shape,sample['segmentation'][x,x:x+10])
        else:
            sample = self.synctransform(sample, 'all')
        return sample
    '''length'''
    def __len__(self):
        return len(self.imageids)
