import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data

#import SimpleITK as sitk
from abc import abstractmethod
from .base_dataset import BaseDataset
import os 
import random

class Cv2DenoiseDataset(BaseDataset):
    def __init__(self,args,train_dataset_name=None):
        super(Cv2DenoiseDataset,self).__init__(args,train_dataset_name)
        self.dataset_image_pathfile = os.path.join(args.dataset_path,train_dataset_name,train_dataset_name+".txt")
        self.get_path_from_txt()
        self.resize_traindata = args.resize_traindata
        
    def get_path_from_txt(self):
        """
        the image and gt filepoath will read from self.dataset_image_pathfile
        and save in self.image_path_list\self.gt_path_list
        """
        with open(self.dataset_image_pathfile,'r') as f:
            lines = f.readlines()
            for i in range(0,len(lines),2):
                self.image_path_list.append(lines[i].rstrip())
                self.gt_path_list.append(lines[i+1].rstrip())


    def crop_image(self,img):
        h,w = img.shape
        (h1,h2,w1,w2) = (174,766,449,1169)
        if h2>h :
            h2 = h
        if w2>w:
            w2 = w 
        img = img[h1:h2,w1:w2]
        return img
    
    def read_file(self,image,gt):
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image = cv2.imread(image,0)
        gt = cv2.imread(gt,0)
        return image,gt
    
    def resize_image(self,*args):
    
        def _resize(s):
            return cv2.resize(s,(self.resize_traindata,self.resize_traindata))
        ret = [_resize(a) for a in args]
        return ret

    def padding(self,image, label):
        ih, iw = image.shape[:2]
        hp = self.patch_size
        wp = self.patch_size

        def _padding(s):
            return cv2.copyMakeBorder(s, max(0, int(hp - ih)), max(0, int(hp - ih)), max(0, int(wp - iw)),
                                      max(0, int(wp - iw)), cv2.BORDER_CONSTANT, (0, 0, 0))

        if ih < hp or iw < wp:
            ret = [_padding(image),_padding(label)]
        else:
            ret = [image,label]

        return ret

    def load_file(self, image, gt):
        # print(image,gt,end='')
        image, gt = self.read_file(image, gt)
        # print(image.shape,gt.shape)
        # 返回一个numpy数组，0-255 uint8 类型
        if self.crop_traindata:
            image, gt = self.crop_image(image), self.crop_image(gt)
        if self.patch_size is not None:
            image, gt = self.padding(image, gt)
            image, gt = self.get_patch(image, gt)
            if self.argument_scale != 1:
                image, gt = self.augment_patch(image, gt)
        if self.resize_traindata is not None:
            image, gt = self.resize_image(image, gt)
            if self.argument_scale != 1:
                image, gt = self.augment_patch(image, gt)
        # print(image.shape,gt.shape)
        image = self.np2tensor(image)
        gt = self.np2tensor(gt)
        return image, gt
    
    # def np2tensor(self,img):
    #     img = np.ascontiguousarray(img)
    #     img = img.astype(float)
    #     tensor = torch.from_numpy(img)
    #     tensor = tensor.mul_(self.rgb_range/255).unsqueeze(0) # 若rgb_range = 1 则会乘上1/255 归一化到0-1 ，若是255则会乘上1，则不会归一化到255 范围 
    #     return tensor
