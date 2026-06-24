import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data

# import SimpleITK as sitk
from abc import abstractmethod
from .base_dataset import BaseDataset
# from base_dataset import BaseDataset
import os
import random


# 用于 超分辨率的 数据集加载  读取函数

class Cv2SRDataset(BaseDataset):
    def __init__(self, args, train_dataset_name=None):
        super(Cv2SRDataset, self).__init__(args, train_dataset_name)
        self.dataset_image_pathfile = os.path.join(args.dataset_path, train_dataset_name, train_dataset_name + ".txt")
        self.get_path_from_txt()
        self.resize_traindata = args.resize_traindata
        self.scale = args.scale

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
        image = cv2.imread(image, 0)
        gt = cv2.imread(gt, 0)
        return image, gt

    def resize_image(self, *args):

        def _resize(s):
            return cv2.resize(s, (self.resize_traindata, self.resize_traindata))

        ret = [_resize(a) for a in args]
        return ret

    def padding(self, image, label):
        # image: LR低分辨率图像
        # label: HR高分辨率图像
        ih, iw = image.shape[:2]  # LR尺寸
        lh, lw = label.shape[:2]  # HR尺寸
        patch_size = self.patch_size
        scale = self.scale

        # -------------------------- 计算LR(image)的padding --------------------------
        # LR目标尺寸：不小于patch_size
        target_h_lr = max(ih, patch_size)
        target_w_lr = max(iw, patch_size)
        pad_h_lr = target_h_lr - ih
        pad_w_lr = target_w_lr - iw

        # 填充策略1：上下均分、左右均分（推荐，内容居中，训练效果更好）
        pad_top_lr = pad_h_lr // 2
        pad_bottom_lr = pad_h_lr - pad_top_lr
        pad_left_lr = pad_w_lr // 2
        pad_right_lr = pad_w_lr - pad_left_lr
        # -------------------------- 计算HR(label)的padding --------------------------
        # HR目标尺寸：始终是LR目标尺寸的scale倍
        target_h_hr = target_h_lr * scale
        target_w_hr = target_w_lr * scale
        pad_h_hr = max(0, target_h_hr - lh)
        pad_w_hr = max(0, target_w_hr - lw)

        # HR填充策略与LR保持完全一致，按比例放大填充量，确保内容精确对齐
        pad_top_hr = pad_top_lr * scale
        pad_bottom_hr = pad_bottom_lr * scale
        pad_left_hr = pad_left_lr * scale
        pad_right_hr = pad_right_lr * scale

        # -------------------------- 执行padding --------------------------
        if pad_h_lr > 0 or pad_w_lr > 0:
            # 填充LR图像（通常是3通道）
            image_padded = cv2.copyMakeBorder(
                image,
                pad_top_lr, pad_bottom_lr,
                pad_left_lr, pad_right_lr,
                cv2.BORDER_CONSTANT,
                (0, 0, 0)  # 填充黑色
            )

            # 填充HR图像（单通道label填0，多通道请改为(0, 0, 0)）
            label_padded = cv2.copyMakeBorder(
                label,
                pad_top_hr, pad_bottom_hr,
                pad_left_hr, pad_right_hr,
                cv2.BORDER_CONSTANT,
                0
            )

            return [image_padded, label_padded]
        else:
            return [image, label]

    #  需要重新写基于patch 的   裁剪    这里 需要考虑label 和 image 的尺寸不一致
    def get_patch(self, img_in, img_tar):
        ih, iw = img_in.shape[:2]
        hp = self.patch_size
        wp = self.patch_size

        ix = random.randrange(0, iw - wp + 1)
        iy = random.randrange(0, ih - hp + 1)
        tx, ty = self.scale * ix, self.scale * iy

        img_in = img_in[iy:iy + hp, ix:ix + wp]
        img_tar = img_tar[ty:ty + hp * self.scale, tx:tx + wp * self.scale]

        return img_in, img_tar

    def augment_patch(self, image, mask, hflip=True, rot=True):

        # if the patch_size = None the augment will be error because the image has filp and image 2 no flip the batch will be wrong
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, ::-1]
            if vflip: img = img[::-1, :]
            if rot90: img = img.transpose(1, 0)

            return img

        return _augment(image), _augment(mask)


    def load_file(self, image, gt):
        # print(image,gt,end='')
        image, gt = self.read_file(image, gt)
        # print(image.shape,gt.shape)
        # 返回一个numpy数组，0-255 uint8 类型
        if self.crop_traindata:
            image, gt = self.crop_image(image), self.crop_image(gt)
        if self.patch_size is not None:
            image, gt = self.padding(image, gt)  # 因为是 一般做超分的patch 192 一般不会小于192 所以不需要做padding
            image, gt = self.get_patch(image, gt)
            if self.argument_scale != 1:
                image, gt = self.augment_patch(image, gt)

        image = self.np2tensor(image)
        gt = self.np2tensor(gt)
        return image, gt

    # def np2tensor(self,img):
    #     img = np.ascontiguousarray(img)
    #     img = img.astype(float)
    #     tensor = torch.from_numpy(img)
    #     tensor = tensor.mul_(self.rgb_range/255).unsqueeze(0) # 若rgb_range = 1 则会乘上1/255 归一化到0-1 ，若是255则会乘上1，则不会归一化到255 范围 
    #     return tensor


if __name__ == '__main__':
    from src.option import args

    args.dataset_path = "/home/liangshubo/Project/CTSR/dataset"

    args.argument_scale = 4
    args.scale = 4
    args.patch_size = 128
    MAX_DISPLAY_HEIGHT = 800  # 窗口最大高度（自动缩放适配）
    dataset = Cv2SRDataset(args, train_dataset_name="DIV2K4x")
    image, label = dataset.__getitem__(585)
    print(" Dataset Raw Image shape : [", image.shape, "] Max : ", image.max(), "ImageName",
          dataset.image_path_list[15])
    print(" Dataset Raw Label(onehot) shape :  [", label.shape, "] Max :", label.max())
    print(image.shape, label.shape)

    print(image.max(), label.max())
    image, label = image * 255, label * 255
    lr_np = image.squeeze().cpu().numpy().astype(np.uint8)  # [1,H,W] → [H,W]
    hr_np = label.squeeze().cpu().numpy().astype(np.uint8)

    cv2.imshow("lr_np", lr_np)
    cv2.imshow("hr_np", hr_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()