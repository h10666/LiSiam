'''
Function:
    define the transforms for data augmentations
Author:
    Zhenchao Jin
'''
import cv2
import torch
import numpy as np
import torch.nn.functional as F


'''resize image'''
class Resize(object):
    def __init__(self, output_size, scale_range=(0.5, 2.0), **kwargs):
        # set attribute
        self.output_size = output_size
        if isinstance(output_size, int): self.output_size = (output_size, output_size)
        self.scale_range = scale_range
        self.img_interpolation = kwargs.get('img_interpolation', 'bilinear')
        self.seg_interpolation = kwargs.get('seg_interpolation', 'nearest')
        self.keep_ratio = kwargs.get('keep_ratio', True)
        # interpolation to cv2 interpolation
        self.interpolation_dict = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
    '''call'''
    def __call__(self, sample):

        # parse
        image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
        if self.scale_range is not None:
            rand_scale = np.random.random_sample() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            output_size = int(self.output_size[0] * rand_scale), int(self.output_size[1] * rand_scale)
        else:
            output_size = self.output_size[0], self.output_size[1]
        # resize image and segmentation
        if self.keep_ratio:
            scale_factor = min(max(output_size) / max(image.shape[:2]), min(output_size) / min(image.shape[:2]))
            dsize = int(image.shape[1] * scale_factor + 0.5), int(image.shape[0] * scale_factor + 0.5)
            image = cv2.resize(image, dsize=dsize, interpolation=self.interpolation_dict[self.img_interpolation])
            segmentation = cv2.resize(segmentation, dsize=dsize, interpolation=self.interpolation_dict[self.seg_interpolation])            
        else:
            if image.shape[0] > image.shape[1]:
                dsize = min(output_size), max(output_size)
            else:
                dsize = max(output_size), min(output_size)
            image = cv2.resize(image, dsize=dsize, interpolation=self.interpolation_dict[self.img_interpolation])
            segmentation = cv2.resize(segmentation, dsize=dsize, interpolation=self.interpolation_dict[self.seg_interpolation])
        # update and return sample
        sample['image'], sample['segmentation'] = image, segmentation

        return sample


'''random crop image'''
class RandomCrop(object):
    def __init__(self, crop_size, **kwargs):
        self.crop_size = crop_size
        if isinstance(crop_size, int): self.crop_size = (crop_size, crop_size)
        self.ignore_index = kwargs.get('ignore_index', 255)
        self.one_category_max_ratio = kwargs.get('one_category_max_ratio', 0.75)
    '''call'''
    def __call__(self, sample):
        # avoid the cropped image is filled by only one category
        for _ in range(10):
            # --parse
            image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
            h_ori, w_ori = image.shape[:2]
            h_out, w_out = min(self.crop_size[0], h_ori), min(self.crop_size[1], w_ori)
            # --random crop
            top, left = np.random.randint(0, h_ori - h_out + 1), np.random.randint(0, w_ori - w_out + 1)
            image = image[top: top + h_out, left: left + w_out]
            segmentation = segmentation[top: top + h_out, left: left + w_out]
            # --judge
            labels, counts = np.unique(segmentation, return_counts=True)
            counts = counts[labels != self.ignore_index]
            if len(counts) > 1 and np.max(counts) / np.sum(counts) < self.one_category_max_ratio: break
        # update and return sample
        sample['image'], sample['segmentation'] = image, segmentation
        return sample

'''center crop image'''
class CenterCrop(object):
    def __init__(self, crop_size, **kwargs):
        self.crop_size = crop_size
        if isinstance(crop_size, int): self.crop_size = (crop_size, crop_size)
        self.ignore_index = kwargs.get('ignore_index', 255)
        self.one_category_max_ratio = kwargs.get('one_category_max_ratio', 0.75)
    '''call'''
    def __call__(self, sample):
        # avoid the cropped image is filled by only one category
        for _ in range(10):
            # --parse
            image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
            h_ori, w_ori = image.shape[:2]
            h_out, w_out = min(self.crop_size[0], h_ori), min(self.crop_size[1], w_ori)
            # cv2.imwrite('image.png', image)

            # top, left = np.random.randint(0, h_ori - h_out + 1), np.random.randint(0, w_ori - w_out + 1)
            top, left = int((h_ori - h_out + 1)/2), int((w_ori - w_out + 1)/2)
            # print('##h_ori, w_ori, h_out, w_out, top, left:',h_ori, w_ori, h_out, w_out, top,left)

            image = image[top: top + h_out, left: left + w_out]
            segmentation = segmentation[top: top + h_out, left: left + w_out]

            # cv2.imwrite('image_crop.png', image)
            # exit()
            # --judge
            labels, counts = np.unique(segmentation, return_counts=True)
            counts = counts[labels != self.ignore_index]
            if len(counts) > 1 and np.max(counts) / np.sum(counts) < self.one_category_max_ratio: break
        # update and return sample
        sample['image'], sample['segmentation'] = image, segmentation
        # print('##sample image:',sample['image'].shape)
        return sample

'''random flip image'''
class RandomFlip(object):
    def __init__(self, flip_prob, fix_ann_pairs=None, **kwargs):
        self.flip_prob = flip_prob
        self.fix_ann_pairs = fix_ann_pairs
    '''call'''
    def __call__(self, sample):
        if np.random.rand() > self.flip_prob: return sample
        image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
        image, segmentation = np.flip(image, axis=1), np.flip(segmentation, axis=1)
        if self.fix_ann_pairs:
            for (pair_a, pair_b) in self.fix_ann_pairs:
                pair_a_pos = np.where(segmentation == pair_a)
                pair_b_pos = np.where(segmentation == pair_b)
                segmentation[pair_a_pos[0], pair_a_pos[1]] = pair_b
                segmentation[pair_b_pos[0], pair_b_pos[1]] = pair_a
        sample['image'], sample['segmentation'] = image, segmentation
        return sample


'''photo metric distortion'''
class PhotoMetricDistortion(object):
    def __init__(self, **kwargs):
        self.brightness_delta = kwargs.get('brightness_delta', 32)
        self.contrast_lower, self.contrast_upper = kwargs.get('contrast_range', (0.5, 1.5))
        self.saturation_lower, self.saturation_upper = kwargs.get('saturation_range', (0.5, 1.5))
        self.hue_delta = kwargs.get('hue_delta', 18)
    '''call'''
    def __call__(self, sample):
        image = sample['image'].copy()
        image = self.brightness(image)
        mode = np.random.randint(2)
        if mode == 1: image = self.contrast(image)
        image = self.saturation(image)
        image = self.hue(image)
        if mode == 0: image = self.contrast(image)
        sample['image'] = image
        return sample
    '''brightness distortion'''
    def brightness(self, image):
        if not np.random.randint(2): return image
        return self.convert(image, beta=np.random.uniform(-self.brightness_delta, self.brightness_delta))
    '''contrast distortion'''
    def contrast(self, image):
        if not np.random.randint(2): return image
        return self.convert(image, alpha=np.random.uniform(self.contrast_lower, self.contrast_upper))
    '''rgb2hsv'''
    def rgb2hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    '''hsv2rgb'''
    def hsv2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    '''saturation distortion'''
    def saturation(self, image):
        if not np.random.randint(2): return image
        image = self.rgb2hsv(image)
        image[..., 1] = self.convert(image[..., 1], alpha=np.random.uniform(self.saturation_lower, self.saturation_upper))
        image = self.hsv2rgb(image)
        return image
    '''hue distortion'''
    def hue(self, image):
        if not np.random.randint(2): return image
        image = self.rgb2hsv(image)
        image[..., 0] = (image[..., 0].astype(int) + np.random.randint(-self.hue_delta, self.hue_delta)) % 180
        image = self.hsv2rgb(image)
        return image
    '''multiple with alpha and add beat with clip'''
    def convert(self, image, alpha=1, beta=0):
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)


'''random rotate image'''
class RandomRotation(object):
    def __init__(self, **kwargs):
        # set attributes
        self.angle_upper = kwargs.get('angle_upper', 30)
        self.rotation_prob = kwargs.get('rotation_prob', 0.5)
        self.img_fill_value = kwargs.get('img_fill_value', 0)
        self.seg_fill_value = kwargs.get('seg_fill_value', 255)
        self.img_interpolation = kwargs.get('img_interpolation', 'bicubic')
        self.seg_interpolation = kwargs.get('seg_interpolation', 'nearest')
        # interpolation to cv2 interpolation
        self.interpolation_dict = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
    '''call'''
    def __call__(self, sample):
        if np.random.rand() > self.rotation_prob: return sample
        image, segmentation = sample['image'].copy(), sample['segmentation'].copy()
        h_ori, w_ori = image.shape[:2]
        rand_angle = np.random.randint(-self.angle_upper, self.angle_upper)
        matrix = cv2.getRotationMatrix2D(center=(w_ori / 2, h_ori / 2), angle=rand_angle, scale=1)
        image = cv2.warpAffine(image, matrix, (w_ori, h_ori), flags=self.interpolation_dict[self.img_interpolation], borderValue=self.img_fill_value)
        segmentation = cv2.warpAffine(segmentation, matrix, (w_ori, h_ori), flags=self.interpolation_dict[self.seg_interpolation], borderValue=self.seg_fill_value)
        sample['image'], sample['segmentation'] = image, segmentation
        return sample


'''pad image'''
class Padding(object):
    def __init__(self, output_size, data_type='numpy', **kwargs):
        self.output_size = output_size
        if isinstance(output_size, int): self.output_size = (output_size, output_size)
        assert data_type in ['numpy', 'tensor'], 'unsupport data type %s...' % data_type
        self.data_type = data_type
        self.img_fill_value = kwargs.get('img_fill_value', 0)
        self.seg_fill_value = kwargs.get('seg_fill_value', 255)
        self.output_size_auto_adaptive = kwargs.get('output_size_auto_adaptive', True)
    '''call'''
    def __call__(self, sample):
        output_size = self.output_size[0], self.output_size[1]
        if self.output_size_auto_adaptive:
            if self.data_type == 'numpy':
                h_ori, w_ori = sample['image'].shape[:2]
            else:
                h_ori, w_ori = sample['image'].shape[1:]
            h_out, w_out = output_size
            if (h_ori > w_ori and h_out < w_out) or (h_ori < w_ori and h_out > w_out):
                output_size = (w_out, h_out)
        if self.data_type == 'numpy':
            image, segmentation, edge = sample['image'].copy(), sample['segmentation'].copy(), sample['edge'].copy()
            h_ori, w_ori = image.shape[:2]
            top = (output_size[0] - h_ori) // 2
            bottom = output_size[0] - h_ori - top
            left = (output_size[1] - w_ori) // 2
            right = output_size[1] - w_ori - left
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.img_fill_value, self.img_fill_value, self.img_fill_value])
            segmentation = cv2.copyMakeBorder(segmentation, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.seg_fill_value])
            edge = cv2.copyMakeBorder(edge, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.seg_fill_value])
            sample['image'], sample['segmentation'], sample['edge'] = image, segmentation, edge
        else:
            image, segmentation, edge = sample['image'], sample['segmentation'], sample['edge']
            h_ori, w_ori = image.shape[1:]
            top = (output_size[0] - h_ori) // 2
            bottom = output_size[0] - h_ori - top
            left = (output_size[1] - w_ori) // 2
            right = output_size[1] - w_ori - left
            image = F.pad(image, pad=(left, right, top, bottom), value=self.img_fill_value)
            segmentation = F.pad(segmentation, pad=(left, right, top, bottom), value=self.seg_fill_value)
            edge = F.pad(edge, pad=(left, right, top, bottom), value=self.seg_fill_value)
            sample['image'], sample['segmentation'], sample['edge'] = image, segmentation, edge
        return sample


'''np.array to torch.Tensor'''
class ToTensor(object):
    '''call'''
    def __call__(self, sample):
        for key in sample.keys():
            if key == 'image':
                sample[key] = torch.from_numpy((sample[key].transpose((2, 0, 1))).astype(np.float32))
            elif key in ['edge', 'groundtruth', 'segmentation']:
                sample[key] = torch.from_numpy(sample[key].astype(np.float32))
        return sample

class XToTensor(object):
    '''call'''
    def __call__(self, sample):
        for key in sample.keys():
            if key == 'image':
                # print('##sample[key] 1:',sample[key])
                sample[key] = torch.from_numpy((sample[key].transpose((2, 0, 1))).astype(np.float32)).div(255)
                # print('##sample[key] 2:',sample[key])

            elif key in ['edge', 'groundtruth', 'segmentation']:
                sample[key] = torch.from_numpy(sample[key].astype(np.float32))
        return sample


'''normalize the input image'''
class XNormalize(object):
    def __init__(self, mean, std, **kwargs):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        # self.to_rgb = kwargs.get('to_rgb', True)
    '''call'''
    def __call__(self, sample):
        for key in sample.keys():
            if key == 'image':
                # image = 
                # mean = np.float64(self.mean.reshape(1, -1))
                # stdinv = 1 / np.float64(self.std.reshape(1, -1))
                # if self.to_rgb: cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                # cv2.subtract(image, mean, image)
                # cv2.multiply(image, stdinv, image)
                # sample[key] = image
                sample[key] = sample[key].sub_(self.mean[:,None,None]).div_(self.std[:,None,None])
                # print('##sample[key] 3:',sample[key])
                # exit()

        return sample

'''normalize the input image'''
class Normalize(object):
    def __init__(self, mean, std, **kwargs):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.to_rgb = kwargs.get('to_rgb', True)
    '''call'''
    def __call__(self, sample):
        for key in sample.keys():
            if key == 'image':
                image = sample[key].astype(np.float32)
                mean = np.float64(self.mean.reshape(1, -1))
                stdinv = 1 / np.float64(self.std.reshape(1, -1))
                if self.to_rgb: cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                cv2.subtract(image, mean, image)
                cv2.multiply(image, stdinv, image)
                sample[key] = image
        return sample

# import torchvision.transforms as tforms
import random

'''wrap the transforms'''
class Compose(object):
    def __init__(self, transforms, **kwargs):
        self.transforms = transforms
    '''call'''
    def __call__(self, sample, transform_type):
        # print("\n##transform0:",type(self.transforms))
        # print("\n##transform1:",)
        # exit()
        if transform_type == 'all':
            for transform in self.transforms:
                sample = transform(sample)
        elif transform_type == 'all_random':
            random.shuffle(self.transforms)
            # print('\n##transforms:',self.transforms)
            for transform in self.transforms:
                sample = transform(sample)
                # print('##transform:',transform)
        elif transform_type == 'only_totensor_normalize_pad':
            for transform in self.transforms:
                if isinstance(transform, ToTensor) or isinstance(transform, Normalize) or isinstance(transform, Padding):
                    sample = transform(sample)
        elif transform_type == 'without_totensor_normalize_pad':
            for transform in self.transforms:
                if not (isinstance(transform, ToTensor) or isinstance(transform, Normalize) or isinstance(transform, Padding)):
                    sample = transform(sample)
        else:
            raise ValueError('Unsupport transform_type %s...' % transform_type)
        return sample
        



####################################################################################
## add by wj

'''Random Shuffle patch'''
class RandomShuffle(object):
    """Random Shuffle patch.
    """
    def __init__(self, shuffle_prob=0.5, num_patch=10, **kwargs):

        self.shuffle_prob = shuffle_prob
        self.num_patch = num_patch

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list
    def swap(self, img, mask, crop):


        widthcut, highcut = img.size
        img = img.crop((10, 10, widthcut-10, highcut-10))
        mask = mask.crop((10, 10, widthcut-10, highcut-10))
        images = self.crop_image(img, crop)
        masks = self.crop_image(mask, crop)
        img_set = []
        for i in range(len(images)):
            img_set.append({'img': images[i],'mask':masks[i]})
        pro = 5
        if pro >= 5:          
            tmpx = []
            tmpy = []
            count_x = 0
            count_y = 0
            k = 1
            RAN = 2
            for i in range(crop[1] * crop[0]):
                tmpx.append(img_set[i])
                count_x += 1
                if len(tmpx) >= k:
                    tmp = tmpx[count_x - RAN:count_x]
                    random.shuffle(tmp)
                    tmpx[count_x - RAN:count_x] = tmp
                if count_x == crop[0]:
                    tmpy.append(tmpx)
                    count_x = 0
                    count_y += 1
                    tmpx = []
                if len(tmpy) >= k:
                    tmp2 = tmpy[count_y - RAN:count_y]
                    random.shuffle(tmp2)
                    tmpy[count_y - RAN:count_y] = tmp2
            random_im = []
            for line in tmpy:
                random_im.extend(line)
            
            # random.shuffle(images)
            width, high = img.size
            iw = int(width / crop[0])
            ih = int(high / crop[1])
            toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
            toMask = Image.new('P', (iw * crop[0], ih * crop[1]))
            x = 0
            y = 0
            for i in random_im:
                i_c = i['img'].resize((iw, ih), Image.ANTIALIAS)
                m_c = i['mask'].resize((iw, ih), Image.ANTIALIAS)
                toImage.paste(i_c, (x * iw, y * ih))
                toMask.paste(m_c, (x * iw, y * ih))
                x += 1
                if x == crop[0]:
                    x = 0
                    y += 1
        else:
            toImage = img
        toImage = toImage.resize((widthcut, highcut))
        toMask = toMask.resize((widthcut, highcut))
        return toImage,toMask

    def __call__(self, sample):
        # print('##sample0:',type(sample['segmentation']),sample['segmentation'].shape,sample['segmentation'].dtype)


        if self.shuffle_prob == 0:
            return  sample        
        if np.random.random() < self.shuffle_prob:

            img_new, mask_new = self.swap(Image.fromarray(sample['image']), Image.fromarray(sample['segmentation']), (self.num_patch,self.num_patch))
            sample['image'], sample['segmentation'] = np.array(img_new), np.array(mask_new)
        # print('##sample1:',type(sample['segmentation']),sample['segmentation'].shape,sample['segmentation'].dtype)
        # exit()
        return  sample 



'''jpeg compression'''
class jpegC(object):
    """jpeg compression .
    """
    # @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, compress_prob, valueD, valueT=100, **kwargs):
        self.valueD = valueD
        self.valueT = valueT
        self.compress_prob = compress_prob


    def cv2_jpg(self, img, compress_val):
        if img.ndim == 3:
            img_cv2 = img[:,:,::-1]
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
            result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            return decimg[:,:,::-1]
        elif img.ndim == 2:
            img_cv2 = img[:,:]
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
            result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            return decimg[:,:]
        else:
            print('bug in data type from transforms.py')
            exit()

    def __call__(self, sample):
        """Call function to flip bounding boxes, masks, semantic segmentation maps.
        Args:
            sample (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped sample, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        # print('##jpeg shape:',sample['img'].shape)
        if self.compress_prob == 0:
            return  sample        
        if np.random.random() < self.compress_prob:
            sample['image'] = self.cv2_jpg(sample['image'], np.random.randint(self.valueD, self.valueT)) 
#            return  Image.fromarray(img)     # Image.fromarray()    #img  #

#sample['image'], sample['segmentation'] = image, segmentation
        return  sample  

from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageOps

class GBlur(object):
    def __init__(self, blur_prob, valueD, valueT):        # GaussianBlur(0.5,[0,3])
        self.blur_sig = [valueD, valueT]   #blur_sig
        self.blur_prob = blur_prob

    def __call__(self, img):
        # print('##img:',img.shape)
        return self.data_augment(img, self.blur_prob, self.blur_sig)
        # return [self.data_augment(img, self.blur_prob, self.blur_sig) for img in img_group]

    def sample_continuous(self, s):
        if len(s) == 1:
            return s[0]
        if len(s) == 2:
            rg = s[1] - s[0]
            return np.random.random() * rg + s[0]
        raise ValueError("Length of iterable s should be 1 or 2.")

    def gaussian_blur(self, img, sigma):
        # print(img.ndim)
        if img.ndim == 3:
            gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
            gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
            gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
        elif img.ndim == 2:
            gaussian_filter(img[:, :], output=img[:, :], sigma=sigma)
        else:
            print('bug in data type from transforms.py')
            exit()
        return img

    def data_augment(self, sample, blur_prob, blur_sig):
        if np.random.random() < blur_prob:
            sig = self.sample_continuous(blur_sig)
            sample['image'] = self.gaussian_blur(sample['image'], sig)

        # if random.random() < blur_prob:  #
        #     img = cv2_jpg(img, random.randint(20, 100)) ##20,100
        return  sample     # Image.fromarray()    #img  #


class jpegD(object):
    def __init__(self, compress_prob=0.5, valueD=64, valueT=299, cropSize=256):        # jpegCompress(0.5,40)
        self.valueD = valueD
        self.valueT = valueT
        self.compress_prob = compress_prob
        self.cropSize = cropSize


    def __call__(self, img):    # img_group
        return self.data_augment(img, self.compress_prob, self.valueD, self.valueT, self.cropSize)    #[for img in img_group]

    def data_augment(self, sample, compress_prob, valueD, valueT, cropSize):
        # print('##sample:',sample['image'].shape)
        # print('##sample:',sample['image'].shape[0],sample['image'].shape[1],sample['image'].shape[2])
        cropSize1, cropSize2 = sample['image'].shape[0], sample['image'].shape[1]
        if np.random.random() < compress_prob:
            aug_size = np.random.randint(valueD, valueT)
            sample['image'] = Image.fromarray(sample['image'])
            if np.random.randint(0,1):
                sample['image'] = sample['image'].resize((aug_size, aug_size), Image.BILINEAR)
            else:
                sample['image'] = sample['image'].resize((aug_size, aug_size), Image.NEAREST)
            sample['image'] = sample['image'].resize((cropSize1, cropSize2),Image.BILINEAR)
            sample['image'] = np.array(sample['image'])
        return sample

from PIL import ImageEnhance
class colorJitter(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        img_out = Image.fromarray(sample['image'])
        if np.random.random() < self.prob:
            if np.random.random() < self.prob:
                img_out = ImageEnhance.Brightness(img_out).enhance(np.random.uniform(0.5, 2.0))
            if np.random.random() < self.prob:
                img_out = ImageEnhance.Contrast(img_out).enhance(np.random.uniform(0.1, 2.0))
            if np.random.random() < self.prob:
                img_out = ImageEnhance.Sharpness(img_out).enhance(np.random.uniform(0.1, 3.0))
            if np.random.random() < self.prob:
                img_out = ImageEnhance.Color(img_out).enhance(np.random.uniform(0.1, 3.0))
        sample['image'] = np.array(img_out)
        return sample


from skimage.util import random_noise
class GNoise(object):
    def __init__(self, prob):
        self.prob = prob
        self.mode_list = ['gaussian']   # ,'salt','pepper','s&p','speckle'

    def __call__(self, sample):

        if np.random.random() < self.prob:
            # noise_mode = random.choice(self.mode_list)
            # noise_img = random_noise(sample['image'], mode=noise_mode)  # , var=0.1**2
            var = round(random.uniform(0.001,0.01),4)
            # print(noise_mode)
            noise_img = random_noise(sample['image'], mode='gaussian', var=var)  # , var=0.1**2

            noise_img = (255*noise_img).astype(np.uint8)
            sample['image'] = noise_img
            return sample
        return sample            
####################################################################################


