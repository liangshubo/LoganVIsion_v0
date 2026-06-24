import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data

import SimpleITK as sitk
from abc import abstractmethod

import os 
import random
from .base_dataset import BaseDataset

class DnBenchmark(BaseDataset):
    def __init__(self,args,test_dataset_name=None):
        super(DnBenchmark,self).__init__(args,test_dataset_name)
        self.dataset_image_pathfile = os.path.join(args.dataset_path,'benchmark',test_dataset_name,test_dataset_name+".txt")
        self.test_dataset_name = test_dataset_name
        
        #print(self.dataset_image_pathfile,"self.dataset_image_pathfile")
        self.get_path_from_txt()

    def get_path_from_txt(self):
        with open(self.dataset_image_pathfile,'r') as f:
            lines = f.readlines()
            for i in range(0,len(lines),2):
                self.image_path_list.append(lines[i].rstrip())
                self.gt_path_list.append(lines[i+1].rstrip())
    def __len__(self):
        return len(self.image_path_list)
    def __getitem__(self, idx):
        idx = idx % len(self.image_path_list)
        image = self.image_path_list[idx]
        gt = self.gt_path_list[idx]

        tensor_image, tensor_gt,namext = self.load_file(image, gt)
        return tensor_image, tensor_gt,namext
    def crop_image(self,img):
        if self.test_dataset_name.find('N20ABDO') >= 0 or self.test_dataset_name.find('N20RENA') >= 0 or self.test_dataset_name.find("urban") >=0 : # 指定数据集来选择不同的裁剪范围与尺寸 
            (h1,h2,w1,w2) = (180,756,424,1192)
            h,w = img.shape
            if h2>h :
                h2 = h
            if w2>w:
                w2 = w 
            img = img[h1:h2,w1:w2]
        else: 
            img = img
        return img
    
    def read_file(self,image,gt):

        if self.test_dataset_name.find('N20ABDO')>=0 or self.test_dataset_name.find('N20RENA')>=0:
            dicom_file = sitk.ReadImage(image)
            image = sitk.GetArrayFromImage(dicom_file).squeeze()[:, :, 0]
            dicom_file = sitk.ReadImage(gt)
            gt = sitk.GetArrayFromImage(dicom_file).squeeze()[:, :, 0]
        elif self.test_dataset_name.find("npy")>=0:
            image = np.load(image)
            gt = np.load(gt)
        else:
            image = cv2.imread(image, 0)
            gt = cv2.imread(gt, 0)
        #print("iamge ",image.shape)
        return image,gt
    
    def load_file(self,image,gt):
        path,nameext = os.path.split(image)
        image,gt = self.read_file(image,gt)
        #返回一个numpy数组，0-255 uint8 类型 
        if self.crop_testdata:
            image,gt = self.crop_image(image),self.crop_image(gt)
        # print("iamge ", self.crop_testdata ,image.shape)
        image = self.np2tensor(image)
        gt = self.np2tensor(gt)
        return image, gt ,nameext