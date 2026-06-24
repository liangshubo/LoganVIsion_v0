from pickletools import uint8

import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data

#import SimpleITK as sitk
from abc import abstractmethod

from labelme.utils import labelme_shapes_to_label
from pandas.io.formats.format import return_docstring
from sympy.abc import alpha

from  .cv2segmentdataset import Cv2SegmentDataset
import os 
import random

# 250901  脊骨切面识别与分割 这里是输出的上一帧的图像以及上一帧的标签。以及这一帧的图像和这一帧的标签

class Cv2SegmentDataset_lastframe(Cv2SegmentDataset):
    def __init__(self,args,train_dataset_name=None):
        super(Cv2SegmentDataset_lastframe,self).__init__(args,train_dataset_name)

    def get_path_from_txt(self):
        """
        the image and gt filepoath will read from self.dataset_image_pathfile
        and save in self.image_path_list\self.gt_path_list
        """
        self.label_list = []
        with open(self.dataset_image_pathfile,'r') as f:
            lines = f.readlines()
            for i in range(0,len(lines),2):
                self.image_path_list.append(lines[i].split(" ")[0].rstrip())
                self.label_list.append(lines[i].split(" ")[1].rstrip())
                self.gt_path_list.append(lines[i+1].rstrip())


    # change
    def read_file(self,image_t,gt_t,image_t_1, gt_t_1):
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image_t_array = cv2.imread(image_t,0)
        gt_t_array = np.load(gt_t)
        image_t_1_array = cv2.imread(image_t_1, 0)
        gt_t_1_array = np.load(gt_t_1)

        return   image_t_array ,gt_t_array,image_t_1_array,gt_t_1_array


    # change  multi_input
    @staticmethod
    def augment_image(image_t,image_t_1, rect_prob=0.5, noise_prob=0.5, blur_prob=0.5):
        """
        对单通道图像进行随机增强
        参数:
            image_t: 单通道灰度图像(二维numpy数组) 当前帧
            image_t-1 : 单通道灰度图像(二维numpy数组) 当前帧

            rect_prob: 随机遮挡的触发概率(0.0~1.0)
            noise_prob: 高斯噪声的触发概率(0.0~1.0)
            blur_prob: 高斯模糊的触发概率(0.0~1.0)
        返回:
            增强后的图像(同尺寸单通道numpy数组)
        """
        # 确保是单通道图像
        assert len(image_t.shape) == 2, "Input must be a single-channel image"

        # 操作1: 随机遮挡(添加多个黑色方块)
        if random.random() < rect_prob:
            h, w = image_t.shape
            num_rects = random.randint(80, 150)  # 随机方块数量

            for _ in range(num_rects):
                # 随机方块尺寸(2-4像素)
                rect_size = random.randint(2, 6)

                # 随机位置(确保不越界)
                x = random.randint(0, w - rect_size - 1)
                y = random.randint(0, h - rect_size - 1)

                # 将选定区域设置为0(纯黑色)   t / t-1
                image_t[y:y + rect_size, x:x + rect_size] = 0
                image_t_1[y:y + rect_size, x:x + rect_size] = 0

        # 操作2: 随机添加高斯噪声
        if random.random() < noise_prob:
            # 生成与原图相同形状的噪声
            mean = random.randint(-2, 2)
            std = random.randint(0, 20)
            noise = np.random.normal(mean, std, image_t.shape)
            image_t = np.clip(image_t + noise, 0, 255).astype(np.uint8)
            image_t_1 = np.clip(image_t_1 + noise, 0, 255).astype(np.uint8)

        # 操作3: 随机添加高斯模糊
        if random.random() < blur_prob:
            # 随机选择奇数大小的卷积核(3,5,7)
            ksize = random.choice([3, 5, 7])
            # 随机标准差(0.5~2.0)
            sigma = random.uniform(0.5, 2.0)
            image_t = cv2.GaussianBlur(image_t, (ksize, ksize), sigmaX=sigma)
            image_t_1 = cv2.GaussianBlur(image_t_1, (ksize, ksize), sigmaX=sigma)

        return image_t,image_t_1

    # change  multi_input
    def augment_patch(self, image_t, gt_t,image_t_1, gt_t_1, hflip=True, rot=True):

        # if the patch_size = None the augment will be error because the image has filp and image 2 no flip the batch will be wrong
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if self.light_change == 1 and random.random() < 0.5:
            light_gain = [-20, -10, 0, 10, 20, 30]
            light = random.choice(light_gain)
        else:
            light = 0

        def _augment(img,islabel):
            if hflip: img = img[:, ::-1]
            # if vflip: img = img[::-1, :]
            # if rot90: img = img.transpose(1, 0)
            if  not islabel :
                imgf = img.astype(float)  # numpy  20   np.float -> float
                imgf_gain = imgf + light
                imgf_gain = np.clip(imgf_gain, 0, 255)
                img = imgf_gain  # .astype(int)  # numpy  20   np.int -> int

            return img

        return _augment(image_t,islabel=False),_augment(gt_t,islabel=True),_augment(image_t_1,islabel=False),_augment(gt_t_1,islabel=True)


    # change   如果不是第一个就返回对应前一帧的效果

    def __getitem__(self, idx):
        idx = idx % len(self.image_path_list)
        image_t = self.image_path_list[idx]
        gt_t = self.gt_path_list[idx]

        if idx == 0:
            image_t_1 = image_t
            gt_t_1 = gt_t
        else:
            image_t_1 = self.image_path_list[idx-1]
            gt_t_1 = self.gt_path_list[idx-1]

        tensor_image_t, tensor_gt_t ,tensor_image_t_1, tensor_gt_t_1 =\
            self.load_file(image_t, gt_t,image_t_1, gt_t_1)

        return tensor_image_t, tensor_gt_t ,tensor_image_t_1, tensor_gt_t_1



    # change
    def load_file(self,image_t,gt_t,image_t_1, gt_t_1):
        image_t_array ,gt_t_array,image_t_1_array,gt_t_1_array = self.read_file(image_t,gt_t,image_t_1, gt_t_1)
        #返回一个numpy数组，0-255 uint8 类型 

        if self.resize_traindata is not None:
            image_t_array = self.resize_image( image_t_array)
            image_t_1_array = self.resize_image(image_t_1_array)
            gt_t_array = self.resize_image(gt_t_array,"near")
            gt_t_1_array = self.resize_image(gt_t_1_array, "near")

            if self.argument_scale!=1:
                image_t_array,gt_t_array,image_t_1_array,gt_t_1_array = self.augment_patch( image_t_array,gt_t_array,image_t_1_array,gt_t_1_array)
                image_t_array,image_t_1_array = self.augment_image( image_t_array,image_t_1_array)  # 额外的数据增广、去噪、模糊、随机遮挡

        image_t_array = np.ascontiguousarray(image_t_array)
        gt_t_array = np.ascontiguousarray(gt_t_array)

        image_t_1_array = np.ascontiguousarray(image_t_1_array)
        gt_t_1_array = np.ascontiguousarray(gt_t_1_array)

        image_t_tensor = self.np2tensor(image_t_array)
        gt_t_tensor = torch.from_numpy(gt_t_array).long()

        image_t_1_tensor = self.np2tensor(image_t_1_array)
        gt_t_1_tensor = torch.from_numpy(gt_t_1_array).long()

        return image_t_tensor, gt_t_tensor ,image_t_1_tensor, gt_t_1_tensor


if __name__ == '__main__':
    from src.option import args

    args.dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset"
    dataset = Cv2SegmentDataset(args, train_dataset_name="n6000_shoulder_segment")
    image, label = dataset.__getitem__(55)
    print(" Dataset Raw Image shape : [", image.shape , "] Max : " ,image.max())
    print(" Dataset Raw Label(onehot) shape :  [",label.shape, "] Max :", label.max())

    def one_hot_to_single(label):
        [c,h,w] = label.shape
        single_label = torch.zeros([h,w])

        for i in range(1,c+1):
            single_label += label[i-1,:,:]*i

        return single_label
    def one2single(label):
        pr = label.argmax(axis=0)
        return pr
    #labels = one_hot_to_single(label)
    labels = one2single(label)
    print(" SingleLabel shape : [",labels.shape , " Max : ", labels.max())

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
    label_array = np.array(labels,dtype=np.uint8)

    print(" Np-array image shape :  [", image_array.shape, " ] type :", image_array.dtype)
    print(" Np-array label shape :  [", label_array.shape, " ] type :", label_array.dtype)

    overlay = visualize_segmentation( image_array ,label_array,alpha=0.3)
    cv2.imshow("show",overlay)
    cv2.waitKey()