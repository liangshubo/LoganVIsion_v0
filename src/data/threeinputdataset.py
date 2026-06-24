import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data

import SimpleITK as sitk
from abc import abstractmethod
from .base_dataset import BaseDataset
import os 
import random  

class ThreeinputDataset(BaseDataset):
    def __init__(self,args,train_dataset_name=None):
        super(ThreeinputDataset,self).__init__(args,train_dataset_name)
        self.dataset_image_pathfile = os.path.join(args.dataset_path,train_dataset_name,train_dataset_name+".txt")
        
        self.mask_path_list=[]
        self.get_path_from_txt()
        self.resize_traindata = args.resize_traindata
    def get_path_from_txt(self):
        """
        the image and gt filepoath will read from self.dataset_image_pathfile
        and save in self.image_path_list\self.gt_path_list
        """
        #self.mask_path_list=[]
        with open(self.dataset_image_pathfile,'r') as f:
            lines = f.readlines()
            #print(len(lines))
            for i in range(0,len(lines),3):
                self.image_path_list.append(lines[i].rstrip())
                #print(f"image{i}")
                self.gt_path_list.append(lines[i+1].rstrip())
                #print(f"gt{i+1}")
                self.mask_path_list.append(lines[i+2].rstrip())
                #print(f"mask{i+2}")

    def crop_image(self,img):
        h,w = img.shape
        (h1,h2,w1,w2) = (174,766,449,1169)
        if h2>h :
            h2 = h
        if w2>w:
            w2 = w 
        img = img[h1:h2,w1:w2]
        return img
    
    def read_file(self,image,gt,mask):
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image = cv2.imread(image,0)
        gt = cv2.imread(gt,0)
        mask = cv2.imread(mask,0)
        return image,gt,mask
    def resize_image(self,*args):
    
        def _resize(s):
            return cv2.resize(s,(self.resize_traindata,self.resize_traindata))
        ret = [_resize(a) for a in args]
        return ret
    def load_file(self,image,gt,mask):
        #print(image)
        image,gt,mask = self.read_file(image,gt,mask)
        
        #print(image.shape,gt.shape,mask.shape)
        #返回一个numpy数组，0-255 uint8 类型 
        if self.crop_traindata:
            image,gt,mask = self.crop_image(image),self.crop_image(gt),self.crop_image(mask)
        if self.patch_size is not None:
            image,gt,mask = self.get_patch(image,gt,mask)
        if self.resize_traindata is not None:
            image,gt,mask = self.resize_image(image,gt,mask)
            if self.argument_scale!=1:
                image,gt,mask = self.augment_patch(image,gt,mask)
        image = self.np2tensor(image)
        gt = self.np2tensor(gt)
        mask = self.np2tensor(mask)
        return image, gt ,mask
    
    def augment_patch(self,image,gt,mask, hflip=True, rot=True):
        
        # if the patch_size = None the augment will be error because the image has filp and image 2 no flip the batch will be wrong 
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, ::-1]
            if vflip: img = img[::-1, :]
            if rot90: img = img.transpose(1, 0)
            return img

        return _augment(image),_augment(gt),_augment(mask)
    
    
    
    def __getitem__(self,idx):
        idx = idx % len(self.image_path_list)
        #print(idx)
        #print(f"len image_path_list{len(self.image_path_list)} , len mask_path_list{len(self.mask_path_list)}")
        image = self.image_path_list[idx]
        gt = self.gt_path_list[idx]
        mask  = self.mask_path_list[idx]
        tensor_image, tensor_gt ,tensor_mask = self.load_file(image,gt,mask)

        return tensor_image,tensor_gt ,tensor_mask
