# from pickletools import uint8
import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data
import math
#import SimpleITK as sitk
from abc import abstractmethod


from base_dataset import BaseDataset
import os
import random


class Cv2NPYSegmentDataset(BaseDataset):
    def __init__(self, args, train_dataset_name=None):
        super(Cv2NPYSegmentDataset, self).__init__(args, train_dataset_name)
        self.dataset_image_pathfile = os.path.join(args.dataset_path, train_dataset_name, train_dataset_name + ".txt")
        self.get_path_from_txt()
        self.resize_traindata = args.resize_traindata
        self.num_class = args.num_class

    def get_path_from_txt(self):
        """
        the image and gt filepoath will read from self.dataset_image_pathfile
        and save in self.image_path_list\self.gt_path_list
        """
        with open(self.dataset_image_pathfile, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                self.image_path_list.append(lines[i].rstrip())
                self.gt_path_list.append(lines[i + 1].rstrip())

    def crop_image(self, img):
        h, w = img.shape
        (h1, h2, w1, w2) = (174, 766, 449, 1169)
        if h2 > h:
            h2 = h
        if w2 > w:
            w2 = w
        img = img[h1:h2, w1:w2]
        return img

    def read_file(self, image, gt):
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image_array = cv2.imread(image, 0)
        gt_array = np.load(gt)
        return image_array, gt_array

    def resize_image(self, image, mode=None):
        def _resize(s):
            if mode == "near":
                return cv2.resize(s, (self.resize_traindata, self.resize_traindata), interpolation=cv2.INTER_NEAREST)
            return cv2.resize(s, (self.resize_traindata, self.resize_traindata))

        ret = _resize(image)
        return ret

    @staticmethod
    def augment_image(image, rect_prob=0.3, noise_prob=0.6, blur_prob=0.6):
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
        if random.random() < (rect_prob / 2):
            h, w = image.shape
            num_rects = random.randint(80, 250)  # 随机方块数量

            for _ in range(num_rects):
                # 随机方块尺寸(2-4像素)
                rect_size = random.randint(2, 8)

                # 随机位置(确保不越界)
                x = random.randint(0, w - rect_size - 1)
                y = random.randint(0, h - rect_size - 1)

                # 将选定区域设置为0(纯黑色)
                image[y:y + rect_size, x:x + rect_size] = random.choice([0, 50, 100, 30, 0, 0])
        # 操作2: 随机添加高斯噪声
        if random.random() < noise_prob:
            # 生成与原图相同形状的噪声
            mean = random.randint(-2, 2)
            std = random.randint(0, 20)
            noise = np.random.normal(mean, std, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # 操作3: 随机添加高斯模糊
        if random.random() < blur_prob:
            # 随机选择奇数大小的卷积核(3,5,7)
            ksize = random.choice([3, 5, 7])
            # 随机标准差(0.5~2.0)
            sigma = random.uniform(0.5, 2.0)
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)

        return image

    @staticmethod
    def augement_rota_translate(image, mask):

        def _rotate_augment_cover(img: np.ndarray, mask: np.ndarray,
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

            angle_deg = random.randint(-20, 20)

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
            if bbox_dim != N:
                resized = cv2.resize(img, (bbox_dim, bbox_dim), interpolation=interp)

                mask = cv2.resize(mask, (bbox_dim, bbox_dim), interpolation=cv2.INTER_NEAREST)
            else:
                resized = img.copy()

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

            H1, W1 = rotated_full.shape[:2]
            out_img = rotated_full[int((H1 - H) / 2):int((H1 + H) / 2), int((W1 - W) / 2):int((W1 + W) / 2)]
            out_mask = rotated_mask[int((H1 - H) / 2):int((H1 + H) / 2), int((W1 - W) / 2):int((W1 + W) / 2)]

            return out_img, out_mask

        def _random_translate(img: np.ndarray, mask: np.ndarray, max_shift: float,
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

            return out_img, out_mask

        def _random_scale(img: np.ndarray, mask: np.ndarray, scale: float,
                          interp=cv2.INTER_LINEAR):
            """
            随机缩放数据增广（保持原尺寸）

            参数
            ----
            img  : np.ndarray
                输入图像，支持灰度或彩色。
            mask : np.ndarray
                语义分割标签图。
            scale : float
                最大缩放因子，最终缩放因子会在 [1/scale , scale] 之间随机取值。
            interp : int
                图像插值方式，mask 始终用 NEAREST。

            返回
            ----
            out_img : np.ndarray
                缩放 & 裁剪/填充后的图像。
            out_mask : np.ndarray
                同样处理后的标签。
            info : dict
                包含 scale_factor。
            """

            H, W = img.shape[:2]

            # 1) 随机缩放因子
            scale_factor = random.uniform(1.0 / scale, scale)

            # 2) 计算缩放后的尺寸
            new_H = int(H * scale_factor)
            new_W = int(W * scale_factor)

            # 3) 进行缩放
            scaled_img = cv2.resize(img, (new_W, new_H), interpolation=interp)
            scaled_mask = cv2.resize(mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

            # ------------------------------
            # Case 1: 放大 → 中心裁剪到 (H, W)
            # ------------------------------
            if scale_factor > 1.0:
                # 计算中心裁剪区域起点
                start_y = (new_H - H) // 2
                start_x = (new_W - W) // 2

                out_img = scaled_img[start_y:start_y + H, start_x:start_x + W]
                out_mask = scaled_mask[start_y:start_y + H, start_x:start_x + W]

            # ------------------------------
            # Case 2: 缩小 → 置于中心后补零
            # ------------------------------
            else:
                out_img = np.zeros_like(img)
                out_mask = np.zeros_like(mask)

                # 计算中心放置起点
                start_y = (H - new_H) // 2
                start_x = (W - new_W) // 2

                out_img[start_y:start_y + new_H, start_x:start_x + new_W] = scaled_img
                out_mask[start_y:start_y + new_H, start_x:start_x + new_W] = scaled_mask

            return out_img, out_mask

        if random.random() < 0.4:
            image, mask = _random_scale(image, mask , scale = 2)
            print("scale")
        if random.random() < 0.4:
            image, mask  = _rotate_augment_cover(image, mask)
            print("rotate")
        if random.random() < 0.4:
            image, mask = _random_translate(image, mask , max_shift=80)
            print("translate")
        return image, mask

    def augment_patch(self, image, gt, hflip=True, rot=True):

        # if the patch_size = None the augment will be error because the image has filp and image 2 no flip the batch will be wrong
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if self.light_change == 1 and random.random() < 0.5:
            light_gain = [-30,-20, -10, 0, 10, 20, 30,40]
            light = random.choice(light_gain)
        else:
            light = 0

        def _augment(img, islabel):
            if hflip: img = img[:, ::-1]
            # if vflip: img = img[::-1, :]
            # if rot90: img = img.transpose(1, 0)
            if not islabel:
                imgf = img.astype(float)  # numpy  20   np.float -> float
                imgf_gain = imgf + light
                imgf_gain = np.clip(imgf_gain, 0, 255)
                img = imgf_gain  # .astype(int)  # numpy  20   np.int -> int

            return img

        return _augment(image, islabel=False), _augment(gt, islabel=True)

    def single_channel_to_onehot(self, label):
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
        onehot = np.zeros((self.num_class, h, w,), dtype=np.uint8)
        # 为每个类别创建通道
        for class_id in range(0, self.num_class):  # 类别从1开始 # 这里忽略背景类别了
            # 创建当前类别的掩码
            mask = (label == class_id)
            # 在对应通道上标记
            onehot[class_id, :, :] = mask.astype(np.uint8)  # 通道索引从0开始
        onehot = np.ascontiguousarray(onehot)
        onehot_tensor = torch.from_numpy(onehot).float()

        onehot_tensor[onehot_tensor >= 0.5] = 1  # label被resize后像素值会改变,调整像素值为原来的两类
        onehot_tensor[onehot_tensor < 0.5] = 0

        return onehot_tensor

    def load_file(self, image, gt):
        image, gt = self.read_file(image, gt)
        # 返回一个numpy数组，0-255 uint8 类型
        if self.crop_traindata:
            image, gt = self.crop_image(image), self.crop_image(gt)
        if self.patch_size is not None:
            image, gt = self.get_patch(image, gt)
            if self.argument_scale != 1:
                image, gt = self.augment_patch(image, gt)
                image = self.augment_image(image)  # 额外的数据增广、去噪、模糊、随机遮挡
                image, gt = self.augement_rota_translate(image, gt)

        if self.resize_traindata is not None:
            image = self.resize_image(image)
            gt = self.resize_image(gt, "near")
            if self.argument_scale != 1:
                image, gt = self.augment_patch(image, gt)
                image = self.augment_image(image)  # 额外的数据增广、去噪、模糊、随机遮挡
                image, gt = self.augement_rota_translate(image, gt)

        image = np.ascontiguousarray(image)
        gt = np.ascontiguousarray(gt)

        image_tensor = self.np2tensor(image)
        gt_tensor = torch.from_numpy(gt).long()
        # gt_tensor = self.single_channel_to_onehot(gt)

        return image_tensor, gt_tensor


if __name__ == '__main__':
    from src.option import args

    args.dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset"
    args.argument_scale = 2
    args.resize_traindata = None
    args.resize_traindata = 420
    dataset = Cv2NPYSegmentDataset(args, train_dataset_name="UBPB_Single")
    image, label = dataset.__getitem__(585)
    print(" Dataset Raw Image shape : [", image.shape , "] Max : " ,image.max(), "ImageName",dataset.image_path_list[585])
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
            1: [255, 0, 0],  # 红色
            2: [0, 255, 0],  # 绿色
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
    cv2.imshow("show",overlay)
    cv2.waitKey()