
import cv2
import numpy as np

import torch

# from .base_dataset import BaseDataset
from .segmentbenchmark import SegmentBenchmark
import os 
import random

# 250901  脊骨切面识别与分割  这里是同属带有切面的类别信息，但是只输出类别int, 这里只是改动 ，读取和输出的类型

class SegmentBenchmark_multitask_multiframe(SegmentBenchmark):
    def __init__(self,args,test_dataset_name=None):
        super(SegmentBenchmark_multitask_multiframe,self).__init__(args,test_dataset_name)

    def get_path_from_txt(self):
        """
        the image and gt filepoath will read from self.dataset_image_pathfile
        and save in self.image_path_list\self.gt_path_list
        """
        self.label_list = []
        with open(self.dataset_image_pathfile, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                self.image_path_list.append(lines[i].split(" ")[0].rstrip())
                self.label_list.append(lines[i].split(" ")[1].rstrip())
                self.gt_path_list.append(lines[i + 1].rstrip())

    # chANGE
    def __getitem__(self, idx):
        idx = idx % len(self.image_path_list)
        image = self.image_path_list[idx]
        gt = self.gt_path_list[idx]
        label = self.label_list[idx]
        tensor_image, tensor_gt ,tensor_label,nameext  = self.load_file(image, gt,label)
        return tensor_image, tensor_gt,tensor_label,nameext


    # change
    def read_file(self, image, gt, label):
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image_array = cv2.imread(image, 0)
        if os.path.isfile(gt) and int(label) != 0:
            gt_array = np.load(gt)
        else:
            gt_array = np.zeros_like(image_array)

        label_array = int(label)

        return image_array, gt_array, label_array

    # LOAD


    def load_file(self,image, gt, label):
        path, nameext = os.path.split(image)
        image_array, gt_array, label_array = self.read_file(image, gt, label)

        image_array = np.ascontiguousarray(image_array)
        gt_array = np.ascontiguousarray(gt_array)

        image_tensor = self.np2tensor(image_array)
        gt_tensor = torch.from_numpy(gt_array).long()

        label_tensor = torch.tensor(label_array)

        return image_tensor, gt_tensor ,label_tensor,nameext


if __name__ == '__main__':
    from src.option import args

    args.dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset"
    dataset = SegmentBenchmark_multitask_multiframe(args, test_dataset_name="n6000_shoulder_segment_828_cop_class")
    image,label, gt,nameext = dataset.__getitem__(104)
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
        if len(original_img.shape) == 2:
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
            10: [75, 0, 130],  # 靛蓝（示例，未指定）
            11: [0, 0, 255],  # 红色
            12: [0, 255, 0],  # 绿色
            13: [255, 0, 0],  # 蓝色
            14: [0, 255, 255],  # 黄色
            15: [255, 255, 0],  # 青色
            16: [255, 0, 255],  # 品红
            17: [0, 0, 125],  # 深红
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


    image_array = np.array(image.squeeze(0) * 255, dtype=np.uint8)
    label_array = np.array(label, dtype=np.uint8)

    print(" Np-array image shape :  [", image_array.shape, " ] type :", image_array.dtype)
    print(" Np-array label shape :  [", label_array.shape, " ] type :", label_array.dtype)

    overlay = visualize_segmentation(image_array, label_array, alpha=0.3)
    cv2.putText(overlay, str(gt.item()), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))
    cv2.imshow("show", overlay)
    cv2.waitKey()