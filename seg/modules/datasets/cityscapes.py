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
        # print('##dataset_cfg:',dataset_cfg)
        # exit()    
        # import os
        # /home/wj/00code/seg/sssegmentation/ssseg/data/CityScapes
        # /home/wj/00code/seg/sssegmentation
        # print('##get pwd:',os.getcwd())
        self.image_dir = os.path.join(rootdir, dataset_cfg['img_dir'])	# , dataset_cfg['set']      'FFppDFaceC0'
        self.ann_dir = os.path.join(rootdir, dataset_cfg['ann_dir'])	# , dataset_cfg['set']  'FFppDSegSSIM2'  
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['labelFolder'], dataset_cfg['set']+'.txt'), names=['imageids'])
        # df = pd.read_csv(os.path.join(rootdir, 'labelsGT', dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
        # print('##imageids:',self.imageids)
        # self.clsLabelFile = pickle.load(open(os.path.join(rootdir,'clsLabelAll.dat'),'rb')) 
        self.clsLabelFile = pickle.load(open(os.path.join(rootdir, dataset_cfg['labelFolder'], dataset_cfg['clsLabel']),'rb')) 
        # self.clsLabelFile = pickle.load(open(os.path.join(rootdir,'labelsGT', dataset_cfg['clsLabel']),'rb')) 
        if dataset_cfg['with_ann'] == 0:
            self.Label_with_ann = False
        else:
            self.Label_with_ann = True
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        
        imagepath = os.path.join(self.image_dir, imageid+'.png')
        # annpath = os.path.join(self.ann_dir, imageid.replace('leftImg8bit', 'gtFine_labelIds')+'.png')
        annpath = os.path.join(self.ann_dir, imageid+'.png')
        sample1 = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', self.Label_with_ann))  # self.clsLabelFile, imageid, 

        # try:
        #     sample1 = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', self.Label_with_ann))  # self.clsLabelFile, imageid, 
        # except:
        #     print('##imagepath:',imagepath)
        #     print('##annpath:',annpath)
        
        # x = int(sample['segmentation'].shape[0]/2)
        # print('##data ##target 1:',sample['segmentation'].shape,sample['segmentation'][x,x:x+10])
        # print('##key:',imageid)
        # print('##value:',self.clsLabelFile[imageid])
        # exit()



        sample1.update({'id': imageid})
        sample1.update({'clsLabel': self.clsLabelFile[imageid]})
        # print('\n\n##sample:',sample.keys())
        # print('##sample:',sample)
        # exit()

        # sample2 = sample.copy()
        if self.mode == 'TRAIN':
            # print('##mode:',self.mode)
            sample1 = self.synctransform1(sample1, 'all')
            sample1['edge'] = self.generateedge(sample1['segmentation'].copy())

            sample2 = sample1.copy()
            sample2 = self.synctransform2(sample2, 'all_random')       ## JEPG_C
            sample2 = self.synctransform3(sample2, 'all')

            sample1 = self.synctransform3(sample1, 'all')

            # sample1 = self.synctransform(sample1, 'only_totensor_normalize_pad')

            # sample = self.synctransform(sample, 'without_totensor_normalize_pad')
            # sample['edge'] = self.generateedge(sample['segmentation'].copy())
            # sample = self.synctransform(sample, 'only_totensor_normalize_pad')
            # print('##data ##target 2:',sample['segmentation'].shape,sample['segmentation'][x,x:x+10])

            # sample2 = self.synctransform2(sample2, 'without_totensor_normalize_pad')
            # sample2['edge'] = self.generateedge(sample2['segmentation'].copy())
            # sample2 = self.synctransform2(sample2, 'only_totensor_normalize_pad')
        else:
            # sample1 = self.synctransform1(sample1, 'all')
            # print('##mode:',self.mode)
            sample2 = sample1.copy()
            sample1 = self.synctransform3(sample1, 'all')
            sample2 = self.synctransform3(sample2, 'all')
            
            sample1['edge'] = sample1['segmentation']
            sample2['edge'] = sample1['edge']


            # sample2 = sample1.copy()
            # sample1 = self.synctransform3(sample1, 'all')

            # # sample2 = self.synctransform1(sample2, 'all')
            # sample2 = self.synctransform3(sample2, 'all')

            # sample1['edge'] = sample1['segmentation']
            # sample2['edge'] = sample1['segmentation']

            
        # exit()
        # print('##image       :',sample1['image'].size())
        # print('##segmentation:',sample1['segmentation'].size())
        # print('##edge        :',sample1['edge'].size())
        # print('##groundtruth :',sample1['groundtruth'].size())
        # ''
        # print('##width:',sample1['width'])
        # print('##height:',sample1['height'])


        # print('\n\n##test!!\n\n')
        return sample1, sample2
    '''length'''
    def __len__(self):
        return len(self.imageids)
