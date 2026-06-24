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

class Controldata(BaseDataset):
    def __init__(self,args,train_dataset_name=None):
        super(Controldata,self).__init__(args,train_dataset_name)
        self.dataset_image_pathfile = os.path.join(args.dataset_path,train_dataset_name,train_dataset_name+".txt")
        
        self.cv1_path_list=[]
        self.cv3_path_list=[]
        self.cv5_path_list=[]
        self.get_path_from_txt()
    def get_path_from_txt(self):
        """
        the image and gt filepoath will read from self.dataset_image_pathfile
        and save in self.image_path_list\self.gt_path_list
        """
        #self.mask_path_list=[]
        with open(self.dataset_image_pathfile,'r') as f:
            lines = f.readlines()
            #print(len(lines))
            for i in range(0,len(lines),4):
                self.image_path_list.append(lines[i].rstrip())
                #print(f"image{i}")
                self.cv1_path_list.append(lines[i+1].rstrip())
                #print(f"gt{i+1}")
                self.cv3_path_list.append(lines[i+2].rstrip())
                #print(f"mask{i+2}")
                self.cv5_path_list.append(lines[i+3].rstrip())
                
                
    def crop_image(self,img):
        h,w = img.shape
        (h1,h2,w1,w2) = (180,756,424,1192)
        if h2>h :
            h2 = h
        if w2>w:
            w2 = w 
        img = img[h1:h2,w1:w2]
        return img
    
    def read_file(self,image,cv1,cv3,cv5):
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image = cv2.imread(image,0)
        cv1 = cv2.imread(cv1,0)
        cv3 = cv2.imread(cv3,0)
        cv5 = cv2.imread(cv5,0)
        
        return image,cv1,cv3,cv5
    
    
    def augment_patch(self,image,cv1,cv3,cv5, hflip=True, rot=True):
        
        # if the patch_size = None the augment will be error because the image has filp and image 2 no flip the batch will be wrong 
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, ::-1]
            if vflip: img = img[::-1, :]
            if rot90: img = img.transpose(1, 0)
            return img

        return _augment(image),_augment(cv1),_augment(cv3),_augment(cv5)
    
    def load_file(self,image,cv1,cv3,cv5):
        image,cv1,cv3,cv5 = self.read_file(image,cv1,cv3,cv5)
        #返回一个numpy数组，0-255 uint8 类型 
        if self.crop_traindata:
            image,cv1,cv3,cv5 = self.crop_image(image),self.crop_image(cv1),self.crop_image(cv3),self.crop_image(cv5)
        if self.patch_size is not None:
            image,cv1,cv3,cv5 = self.get_patch(image,cv1,cv3,cv5)
        if self.argument_scale!=1:
            image,cv1,cv3,cv5 = self.augment_patch(image,cv1,cv3,cv5)
        image = self.np2tensor(image)
        cv1 = self.np2tensor( cv1 )
        cv3 = self.np2tensor( cv3 )
        cv5 = self.np2tensor( cv5 )
       
        return image, cv1,cv3,cv5 
    
    def __getitem__(self,idx):
        idx = idx % len(self.image_path_list)
        #print(idx)
        #print(f"len image_path_list{len(self.image_path_list)} , len mask_path_list{len(self.mask_path_list)}")
        image = self.image_path_list[idx]
        cv1 = self.cv1_path_list[idx]
        cv3 = self.cv3_path_list[idx]
        cv5 = self.cv5_path_list[idx]
        
        tensor_image, tensor_cv1 ,tensor_cv3,tensor_cv5 = self.load_file(image,cv1,cv3,cv5)
        
        return tensor_image,tensor_cv1 ,tensor_cv3,tensor_cv5
