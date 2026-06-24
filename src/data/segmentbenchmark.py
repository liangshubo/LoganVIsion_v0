from pickletools import uint8

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


class SegmentBenchmark(BaseDataset):
    def __init__(self,args,test_dataset_name=None):
        super(SegmentBenchmark,self).__init__(args,test_dataset_name)
        self.dataset_image_pathfile = os.path.join(args.dataset_path,'benchmark',test_dataset_name,test_dataset_name+".txt")
        self.get_path_from_txt()
        self.resize_traindata = args.resize_traindata

        self.num_class = args.num_class
        self.test_dataset_name = test_dataset_name

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

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        idx = idx % len(self.image_path_list)
        image = self.image_path_list[idx]
        gt = self.gt_path_list[idx]

        tensor_image,tensor_gt,nameext  = self.load_file(image,gt)
        return tensor_image, tensor_gt,nameext


    def read_file(self,image,gt):
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image_array = cv2.imread(image,0)
        gt_array = cv2.imread(gt,0)/255
        return   image_array ,gt_array

    def resize_image(self, image, mode=None):
        def _resize(s):
            if mode == "near":
                return cv2.resize(s, (self.resize_traindata, self.resize_traindata), interpolation=cv2.INTER_NEAREST)
            return cv2.resize(s, (self.resize_traindata, self.resize_traindata))

        ret = _resize(image)
        return ret

    def augment_patch(self, image,gt, hflip=True, rot=True):

        # if the patch_size = None the augment will be error because the image has filp and image 2 no flip the batch will be wrong
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        if self.light_change == 1 and random.random() < 0.5:
            light_gain = [-55, -45, -20, -10, 10, 20, 45, 55]
            light = random.choice(light_gain)
        else:
            light = 0

        def _augment(img):
            if hflip: img = img[:, ::-1]
            if vflip: img = img[::-1, :]
            if rot90: img = img.transpose(1, 0)
            if light != 0:
                imgf = img.astype(float)  # numpy  20   np.float -> float
                imgf_gain = imgf + light

                imgf_gain_high = (imgf_gain > 254)

                imgf_gain[imgf_gain_high] = 255
                img = imgf_gain.astype(int)  # numpy  20   np.int -> int
            return img

        return _augment(image)

    def single_channel_to_onehot(self,label):
        """
           将单通道类别标签转换为多通道one-hot编码
           参数:
               label_img: 单通道标签图像，像素值1-10代表不同类别
               num_classes: 类别总数（默认10）
           返回:
               onehot: one-hot编码的多通道矩阵(num_classes,H,W)，dtype=uint8
           """
        # 验证输入图像是否为单通道
        if len(label.shape) != 2:
            raise ValueError("输入图像必须是单通道")
        # 获取图像尺寸
        h, w = label.shape
        # 初始化one-hot矩阵（全0）
        onehot = np.zeros(( self.num_class,h, w,), dtype=np.uint8)
        # 为每个类别创建通道
        for class_id in range(1, self.num_class + 1):  # 类别从1开始
            # 创建当前类别的掩码
            mask = (label== class_id)
            # 在对应通道上标记
            onehot[class_id - 1,:, :] = mask.astype(np.uint8)  # 通道索引从0开始
        onehot = np.ascontiguousarray(onehot)
        onehot_tensor = torch.from_numpy(onehot).float()
        return onehot_tensor

    def load_file(self,image,gt):
        # patch 的训练方法 信息的缩放程度和原来的图大小不改变 ， 在所以不影响 测试图的大小， 而缩放的训练方法 则改变了训练的方法 因此实际上会影响 测试图的大小
        path, nameext = os.path.split(image)
        image,gt = self.read_file(image,gt)
        #返回一个numpy数组，0-255 uint8 类型 
        if self.crop_traindata:
            image,gt = self.crop_image(image),self.crop_image(gt)

        if self.resize_traindata is not None:
            image = self.resize_image(image)
            gt = self.resize_image(gt, "near")

        image_tensor = self.np2tensor(image)
        gt_tensor = torch.from_numpy(gt).long()
        #gt_tensor = self.single_channel_to_onehot(gt)

        return image_tensor, gt_tensor , nameext


if __name__ == '__main__':
    from src.option import args

    args.dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset"
    dataset = SegmentBenchmark(args, test_dataset_name="Ultrasound_Nerve_Segmention_Kaggle")
    image, label,nameext = dataset.__getitem__(104)
    print(" Dataset Raw Image shape : [", image.shape , "] Max : " ,image.max())
    print(" Dataset Raw Label(onehot) shape :  [",label.shape, "] Max :", label.max())


    def visualize_segmentation(original_img, label_img, alpha=0.5):
        """
        将语义分割结果与原始图像叠加显示

        参数:
            original_img: 原始图像（可以是灰度或彩色）
            label_img: 标签图像（单通道，像素值1-10代表不同类别）
            alpha: 标签透明度（0-1）

        返回:
            合成后的图像
        """
        # 确保原始图像是3通道
        if len(original_img.shape) == 2 :
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

        # 定义类别颜色映射（BGR格式）
        color_map = {
            1: [0, 0, 255],  # 红色
            2: [0, 255, 0],  # 绿色
            3: [255, 0, 0],  # 蓝色
            4: [0, 255, 255],  # 黄色
            5: [255, 255, 0],  # 青色
            6: [255, 0, 255],  # 品红
            7: [0, 0, 125],  # 深红
            8: [128, 128, 128],  # 灰色（示例，未指定）
            9: [255, 165, 0],  # 橙色（示例，未指定）
            10: [75, 0, 130]  # 靛蓝（示例，未指定）
        }

        # 创建彩色标签图像
        h, w = label_img.shape
        colored_label = np.zeros((h, w, 3), dtype=np.uint8)

        # 根据类别填充颜色
        for class_id, color in color_map.items():
            mask = (label_img == class_id)
            colored_label[mask] = color

        # 叠加图像（带透明度）
        overlay = cv2.addWeighted(original_img, 1 - alpha, colored_label, alpha, 0)

        return overlay

    image_array = np.array(image.squeeze(0)*255,dtype=np.uint8)
    label_array = np.array(label,dtype=np.uint8)

    print(" Np-array image shape :  [", image_array.shape, " ] type :", image_array.dtype)
    print(" Np-array label shape :  [", label_array.shape, " ] type :", label_array.dtype)

    overlay = visualize_segmentation( image_array ,label_array,alpha=0.3)
    cv2.imshow("show",overlay)
    cv2.waitKey()