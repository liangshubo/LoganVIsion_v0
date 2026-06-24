from pickletools import uint8

import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data

#import SimpleITK as sitk
from abc import abstractmethod

from PyQt5.sip import array

from  .cv2segmentdataset import Cv2SegmentDataset
import os 
import random
import  math
#  250815 脊骨切面识别与分割   这里是同属带有切面的类别信息，但是只输出类别int, 这里只是改动 ，读取和输出的类型

import re

class Cv2SegmentDataset_multitask_multiframe(Cv2SegmentDataset):
    def __init__(self,args,train_dataset_name=None):
        super(Cv2SegmentDataset_multitask_multiframe,self).__init__(args,train_dataset_name)
        # self.dataset_image_pathfile = os.path.join(args.dataset_path,train_dataset_name,train_dataset_name+".txt")
        # self.get_path_from_txt()
        # self.resize_traindata = args.resize_traindata
        # self.num_class = args.num_class

        # add
        self.section_to_tissue = [  [0,0,0],[0,1, 2, 3],     # 切面1的组织索引
                                    [0,3, 4, 5],     # 切面2的组织索引
                                    [0,3, 6, 7],     # 切面3的组织索引
                                    [0,3, 8, 9],     # 切面4的组织索引
                                    [0,3, 10,11,12], # 切面5的组织索引
                                    [0,3, 13, 14],   # 切面6的组织索引
                                    [0,15, 16],      # 切面7的组织索引
                                    [0,3, 16, 17],   # 切面8的组织索引
                                    [0,18, 19, 20],  # 切面10的组织索引
                                    [0,21, 22, 23],  # 切面11的组织索引
                                 ]
    # change
    @staticmethod
    def augment_image(image,image_t_1, rect_prob=0.5, noise_prob=0.5, blur_prob=0.5):
        """
        对单通道图像进行随机增强
        参数:
            image: 单通道灰度图像(二维numpy数组)
            rect_prob: 随机遮挡的触发概率(0.0~1.0)
            noise_prob: 高斯噪声的触发概率(0.0~1.0)
            blur_prob: 高斯模糊的触发概率(0.0~1.0)
        返回:
            增强后的图像(同尺寸单通道numpy数组)
        """
        # 确保是单通道图像
        assert len(image.shape) == 2, "Input must be a single-channel image"

        # 操作1: 随机遮挡(添加多个黑色方块)
        if random.random() < rect_prob:
            h, w = image.shape
            num_rects = random.randint(80, 250)  # 随机方块数量

            for _ in range(num_rects):

                # 随机方块尺寸(2-4像素)
                rect_size = random.randint(2, 8)
                rect_light = random.choice([0,50,100,30,0,0])
                # 随机位置(确保不越界)
                x = random.randint(0, w - rect_size - 1)
                y = random.randint(0, h - rect_size - 1)

                # 将选定区域设置为0(纯黑色)
                image[y:y + rect_size, x:x + rect_size] = rect_light
                image_t_1[y:y + rect_size, x:x + rect_size] = rect_light
        # 操作2: 随机添加高斯噪声
        if random.random() < noise_prob:
            # 生成与原图相同形状的噪声
            mean = random.randint(-2, 2)
            std = random.randint(0, 20)
            noise = np.random.normal(mean, std, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            image_t_1 = np.clip(image_t_1 + noise, 0, 255).astype(np.uint8)
        # 操作3: 随机添加高斯模糊
        if random.random() < blur_prob:
            # 随机选择奇数大小的卷积核(3,5,7)
            ksize = random.choice([3, 5, 7])
            # 随机标准差(0.5~2.0)
            sigma = random.uniform(0.5, 2.0)
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
            image_t_1 = cv2.GaussianBlur(image_t_1, (ksize, ksize), sigmaX=sigma)
        return image,image_t_1

    @staticmethod
    def augement_rota_translate(image,mask,  image_t_1 , mask_t_1 ):

        def _rotate_augment_cover(img: np.ndarray, mask: np.ndarray,img_t_1: np.ndarray, mask_t_1: np.ndarray,
                                 interp=cv2.INTER_LINEAR
                                 ):
            """
            旋转增广（保证覆盖原始位置，最终中心裁剪回原尺寸）。

            参数
            ----
            img : np.ndarray
                输入图像，需为正方形(H==W)，支持灰度(H,W)或彩色(H,W,C)。
            angle_deg : float
                旋转角度，期望在 [0, 45]（函数内部会clip到该范围）。
            interp : int
                OpenCV插值方式（默认双线性）。
            border_mode : int
                旋转时的边界填充模式（默认反射，避免黑边）。

            返回
            ----
            out_img : np.ndarray
                旋转+中心裁剪后的图像，尺寸与输入一致。
            info : dict
                计算信息，包括：
                - 'bbox_dim': 旋转后正方形的最小轴对齐包围框边长（int）
                - 'aabb': {'min_x','max_x','min_y','max_y','width','height'}  # 以缩放后坐标系为基准（左上为(0,0)）
                - 'extreme_points': {'top','bottom','left','right'} 四个极值点坐标 (x,y)（缩放后坐标系）
                - 'angle_deg': 实际使用的角度
            """

            angle_deg = random.randint(-20,20)

            # -------- 基本检查 --------
            if img.ndim not in (2, 3):
                raise ValueError("img 必须是灰度(H,W)或彩色(H,W,C)的numpy数组")
            H, W = img.shape[:2]
            if H != W:
                raise ValueError(f"输入需要正方形图像，但得到 H={H}, W={W}")
            N = H

            # -------- 角度与弧度 --------
            angle = float(np.clip(angle_deg, -40.0, 45.0))
            rad = math.radians(angle)

            # -------- 旋转后轴对齐包围框尺寸（正方形）--------
            # 对边长为 N 的正方形，以中心旋转 angle，其AABB宽/高均为 N*(|cos| + |sin|)
            c, s = abs(math.cos(rad)), abs(math.sin(rad))
            bbox_float = N * (c + s)
            bbox_dim = int(math.ceil(bbox_float))

            # -------- 1 \ 将原图  gt   缩放到 bbox_dim × bbox_dim --------
            #if bbox_dim != N:
            resized = cv2.resize(img, (bbox_dim, bbox_dim), interpolation=interp)

            mask = cv2.resize(mask, (bbox_dim, bbox_dim), interpolation=cv2.INTER_NEAREST)

            resized_t_1 = cv2.resize(img_t_1, (bbox_dim, bbox_dim), interpolation=interp)

            mask_t_1 = cv2.resize(mask_t_1, (bbox_dim, bbox_dim), interpolation=cv2.INTER_NEAREST)
            #else:
            #    resized = img.copy()

            # -------- 2 \ 以中心旋转 resized --------
            center = (bbox_dim * 0.5, bbox_dim * 0.5)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            rotated_full = cv2.warpAffine(
                resized, M, (bbox_dim, bbox_dim),
                flags=interp,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

            rotated_mask = cv2.warpAffine(
                mask, M, (bbox_dim, bbox_dim),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

            rotated_full_t_1 = cv2.warpAffine(
                resized_t_1, M, (bbox_dim, bbox_dim),
                flags=interp,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

            rotated_mask_t_1 = cv2.warpAffine(
                mask_t_1, M, (bbox_dim, bbox_dim),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )


            H1, W1 = rotated_full.shape[:2]
            out_img = rotated_full[int((H1 - H) / 2):int((H1 + H) / 2), int((W1 - W) / 2):int((W1 + W) / 2)]
            out_mask = rotated_mask[int((H1 - H) / 2):int((H1 + H) / 2), int((W1 - W) / 2):int((W1 + W) / 2)]
            out_img_t_1 = rotated_full_t_1[int((H1 - H) / 2):int((H1 + H) / 2), int((W1 - W) / 2):int((W1 + W) / 2)]
            out_mask_t_1 = rotated_mask_t_1[int((H1 - H) / 2):int((H1 + H) / 2), int((W1 - W) / 2):int((W1 + W) / 2)]
            return out_img, out_mask,out_img_t_1,out_mask_t_1

        def _random_translate(img: np.ndarray, mask: np.ndarray,img_t_1: np.ndarray, mask_t_1: np.ndarray ,max_shift: float,
                             interp=cv2.INTER_LINEAR):
            """
            随机方向平移数据增广

            参数
            ----
            img : np.ndarray
                输入图像，支持灰度(H,W)或彩色(H,W,C)。
            max_shift : float
                最大位移像素数（实际位移随机在 [0, max_shift]）。
            interp : int
                插值方式，默认双线性。

            返回
            ----
            out_img : np.ndarray
                平移后的图像，超出区域设为0。
            info : dict
                平移参数，包括 angle_deg, shift, dx, dy。
            """
            H, W = img.shape[:2]

            # 随机角度 (0-360度)
            angle = random.uniform(-180, 45)
            rad = math.radians(angle)

            # 随机位移大小
            shift = random.uniform(0, max_shift)
            dx = shift * math.cos(rad)
            dy = shift * math.sin(rad)

            # 仿射矩阵
            M = np.float32([[1, 0, dx],
                            [0, 1, dy]])

            # warpAffine 默认 borderValue=0 就是补零
            out_img = cv2.warpAffine(img, M, (W, H),
                                     flags=interp,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)

            out_mask = cv2.warpAffine(mask, M, (W, H),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
            out_img_t_1 = cv2.warpAffine(img_t_1, M, (W, H),
                                     flags=interp,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)

            out_mask_t_1 = cv2.warpAffine(mask_t_1, M, (W, H),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
            return out_img, out_mask,out_img_t_1,out_mask_t_1


        if random.random()<0.5:

            rotate_image ,rotate_mask,rotate_image_t_1  ,rotate_mask_t_1  = _rotate_augment_cover(image,mask,image_t_1 , mask_t_1 )

            translate_image , translate_mask,translate_image_t_1  , translate_mask_t_1  = _random_translate(rotate_image,rotate_mask,rotate_image_t_1  ,rotate_mask_t_1  ,max_shift=80)
        else:
            translate_image, translate_mask,translate_image_t_1, translate_mask_t_1 = image,mask, image_t_1 ,mask_t_1
        return  translate_image , translate_mask,translate_image_t_1, translate_mask_t_1

    def augment_patch(self, image, gt,image_t_1, gt_t_1, hflip=True, rot=True):

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

        return _augment(image,islabel=False),_augment(gt,islabel=True),_augment(image_t_1,islabel=False),_augment(gt_t_1,islabel=True)




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


    # chANGE
    def __getitem__(self, idx):
        idx = idx % len(self.image_path_list)
        image:str = self.image_path_list[idx]
        gt:str = self.gt_path_list[idx]
        label:str = self.label_list[idx]
        # 上一帧的数据 、分割标签 、分类标签   但是前提是当前帧不是 头帧
        # 判断头帧 序列
        path,nameext = os.path.split(image)
        name,ext = os.path.splitext(nameext)
        index = int(name.split("_")[-1])

        if index != 0 :
            image_t_1 = self.image_path_list[idx-1]
            gt_t_1 = self.gt_path_list[idx-1]
            label_t_1 = self.label_list[idx-1]
        else :
            image_t_1 = image
            gt_t_1 = gt
            label_t_1 =  label

        #print(image,image_t_1)

        tensor_image, tensor_gt ,tensor_label,tensor_image_t_1, tensor_gt_t_1 ,tensor_label_t_1 = self.load_file(image, gt,label, image_t_1,gt_t_1,label_t_1)
        return tensor_image, tensor_gt,tensor_label,tensor_image_t_1, tensor_gt_t_1 ,tensor_label_t_1

    # change
    def load_file(self, image, gt, label,image_t_1,gt_t_1,label_t_1 ):
        """
        用于加载对应路径下的数据、分割标签等 。这里还包括上一帧的东西
        """
        image, gt, label,image_t_1,gt_t_1,label_t_1 = self.read_file(image, gt, label,image_t_1,gt_t_1,label_t_1 ) # array [H,W ], array  [H,W ], int
        # 返回一个numpy数组，0-255 uint8 类型

        if self.resize_traindata is not None:
            image = self.resize_image(image)
            image_t_1 = self.resize_image(image_t_1)

            gt = self.resize_image(gt, "near")
            gt_t_1  = self.resize_image(gt_t_1 , "near")

            if self.argument_scale != 1:
                image, gt, image_t_1, gt_t_1  = self.augment_patch(image, gt,image_t_1, gt_t_1)

                image,image_t_1 = self.augment_image(image,image_t_1)  # 额外的数据增广、去噪、模糊、随机遮挡

                image, gt,  image_t_1 , gt_t_1  = self.augement_rota_translate(image, gt,image_t_1 , gt_t_1)


        image = np.ascontiguousarray(image)
        image_t_1 = np.ascontiguousarray(image_t_1)
        gt = np.ascontiguousarray(gt)
        gt_t_1 = np.ascontiguousarray(gt_t_1)

        image_tensor = self.np2tensor(image)
        image_t_1_tensor = self.np2tensor(image_t_1)

        gt_tensor = torch.from_numpy(gt).long()
        gt_t_1_tensor = torch.from_numpy(gt_t_1).long()

        label_tensor = torch.tensor(label)
        label_t_1_tensor = torch.tensor(label_t_1)
        return image_tensor, gt_tensor, label_tensor,image_t_1_tensor, gt_t_1_tensor, label_t_1_tensor


    @staticmethod
    def create_onehot_from_indices(indices_list, dim=24):
        """
        将组织索引列表转换为固定维度的one-hot编码
        参数:
        indices_list -- 包含组织索引的列表，例如 [3, 6, 7]
        dim -- one-hot编码的维度，默认为24
        返回:
        一个dim维的numpy数组，其中指定索引位置为1，其余为0
        """
        # 初始化全0向量
        onehot = torch.zeros(dim, dtype=float)

        # 将指定索引位置设为1
        for idx in indices_list:  #这是切面的索引 对应的组织
            if 1 <= idx <= dim: # 这个是排除0
                onehot[idx] = 1  # 索引从1开始，但数组从0开始

        return onehot

    # change
    def read_file(self,image,gt,label,image_t_1,gt_t_1,label_t_1):
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image_array = cv2.imread(image,0)
        if os.path.isfile(gt) and int(label) !=0 :
            gt_array = np.load(gt)
        else:
            gt_array = np.zeros_like(image_array)

        label_array = int(label)

        # 下一帧的效果
        image_t_1_array = cv2.imread(image_t_1, 0)
        if os.path.isfile(gt_t_1) and int(label_t_1) != 0:
            gt_t_1_array = np.load(gt_t_1)
        else:
            gt_t_1_array = np.zeros_like(image_t_1_array)

        label_t_1_array = int(label_t_1)

        return  image_array,gt_array, label_array,image_t_1_array ,gt_t_1_array,label_t_1_array


if __name__ == '__main__':
    from src.option import args

    args.light_argument = 1
    args.resize_traindata = 512
    args.argument_scale = 8

    args.dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset"
    dataset = Cv2SegmentDataset_multitask_multiframe(args, train_dataset_name="n6000_shoulder_segment_828_multitask")
    image_tensor, gt_tensor, label_tensor,image_t_1_tensor, gt_t_1_tensor, label_t_1_tensor = dataset.__getitem__(4005)
    # print(" Dataset Raw Image shape : [", image.shape, "] Max : ", image.max())
    # print(" Dataset Raw Label(onehot) shape :  [", label.shape, "] Max :", label.max())
    # print("Label " ,nameext)
    # print(torch.unique(label))

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


    image_array = np.array(image_tensor.squeeze(0) * 255, dtype=np.uint8)
    gt_array = np.array(gt_tensor, dtype=np.uint8)

    image_t_1_array = np.array(image_t_1_tensor.squeeze(0) * 255, dtype=np.uint8)
    gt_t_1_array = np.array(gt_t_1_tensor, dtype=np.uint8)

    print(" Np-array image shape :  [", image_array.shape, " ] type :", image_array.dtype)
    print(" Np-array label shape :  [", gt_array.shape, " ] type :", gt_array.dtype)

    overlay_t_1 = visualize_segmentation(image_t_1_array, gt_t_1_array, alpha=0.3)
    overlay = visualize_segmentation(image_array, gt_array, alpha=0.3)

    cv2.imshow("show", overlay)
    cv2.imshow("show_t_1 ", overlay_t_1 )
    cv2.waitKey()