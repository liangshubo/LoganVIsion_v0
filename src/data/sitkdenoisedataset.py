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

class SitkDenoiseDataset(BaseDataset):
    def __init__(self,args,train_dataset_name=None):
        super(SitkDenoiseDataset,self).__init__(args,train_dataset_name)
        self.dataset_image_pathfile = os.path.join(args.dataset_path,train_dataset_name,train_dataset_name+".txt")
        self.get_path_from_txt()
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
        (h1,h2,w1,w2) = (174,766,449,1169)
        img = img[h1:h2,w1:w2]
        return img
    
    def read_file(self,image,gt):
        dicom_file = sitk.ReadImage(image)
        image = sitk.GetArrayFromImage(dicom_file).squeeze()[:, :, 0]
        dicom_file = sitk.ReadImage(gt)
        gt = sitk.GetArrayFromImage(dicom_file).squeeze()[:, :, 0]
        return image,gt
    
    def load_file(self,image,gt):
        image,gt = self.read_file(image,gt)
        #返回一个numpy数组，0-255 uint8 类型 
        if self.crop_traindata:
            image,gt = self.crop_image(image),self.crop_image(gt)
        if self.patch_size is not None:
            image,gt = self.get_patch(image,gt)
        if self.argument_scale!=1:
            image,gt = self.augment_patch(image,gt)
        image = self.np2tensor(image)
        gt = self.np2tensor(gt)
        return image, gt
