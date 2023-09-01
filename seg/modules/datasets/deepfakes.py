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

'''deepfakes dataset'''
class DFDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    # clsid2label = {
    #     -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    #     7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 
    #     15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 
    #     23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 
    #     31: 16, 32: 17, 33: 18, 255:1
    # }

    clsid2label = { 0: 0, 255: 1 }
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg, **kwargs):
        super(DFDataset, self).__init__(mode, logger_handle, dataset_cfg, **kwargs)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir'] 

        self.image_dir = os.path.join(rootdir, dataset_cfg['img_dir'])	# , dataset_cfg['set']      'FFppDFaceC0'
        self.ann_dir = os.path.join(rootdir, dataset_cfg['ann_dir'])	# , dataset_cfg['set']  'FFppDSegSSIM2'  
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['labelFolder'], dataset_cfg['set']+'.txt'), names=['imageids'])

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

        if self.mode == 'TRAIN':
            imageid = self.imageids[index]
            PATH_video = os.path.join(self.image_dir, imageid)
            img_names = os.listdir(PATH_video)
            idx = np.random.randint(0, len(img_names), dtype=np.int)
            img_name = img_names[idx]            
            
            imagepath = os.path.join(self.image_dir, imageid, img_name)
            annpath = os.path.join(self.ann_dir, imageid, img_name)
            sample1 = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', self.Label_with_ann))  # self.clsLabelFile, imageid, 

            imageid = imageid + '/' + img_name.replace('.png','')

            # print('##PATH_video:',PATH_video)
            # print('##len img_names:',len(img_names))
            # print('##img_name:',img_name)
            # print('##imagepath:',imagepath)
            # print('##annpath:',annpath)
            # # print('##idx:',idx)
            # exit()
# /home/wj/kaggle/FFpp/SegDataV3/FFppDFaceC40a/manipulated/Deepfakes/427_637/008_0
        else:
            imageid = self.imageids[index]
            imagepath = os.path.join(self.image_dir, imageid+'.png')
            annpath = os.path.join(self.ann_dir, imageid+'.png')
            sample1 = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', self.Label_with_ann))  # self.clsLabelFile, imageid, 

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
