
import os
import re
import time
import cv2
import torch
import numpy as np

import torch.nn.functional as F
from skimage import measure
from scipy.ndimage import binary_fill_holes

from collections import deque
from typing import List, Tuple, Dict, Optional, Union
from torch import Tensor

from concurrent.futures import ThreadPoolExecutor

class FrameSmoother:
    def __init__(self, num_classes: int = 24, window_size: int = 3, device: str = "cuda") -> None:
        """
        Args:
            num_classes: 语义分割类别数
            window_size: 滑动窗口大小
            device: 使用设备
        """
        self.num_classes = num_classes
        self.window_size = window_size
        self.recent_probs: deque[Tensor] = deque(maxlen=window_size)
        self.device = device

    def smooth(self, logits: Tensor) -> Tensor:
        """
        输入: 模型原始输出 (B, C, H, W)
        输出: 平滑后的概率 (B, C, H, W)

        Args:
            logits: torch.Tensor, shape (B, C, H, W)

        Returns:
            probs_smooth: torch.Tensor, shape (B, C, H, W)
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        self.recent_probs.append(probs.detach())
        probs_smooth = torch.mean(torch.stack(list(self.recent_probs), dim=0), dim=0)
        return probs_smooth

    def reset(self) -> None:
        """清空缓存窗口"""
        self.recent_probs.clear()


class ArgmaxPostProcessor:
    def __init__(self, kernel_size: int = 7, min_area: int = 80,
                 background_class: int = 0, smoothing_iter: int = 3) -> None:
        """
        Args:
            kernel_size: 形态学核大小
            min_area: 最小区域阈值
            background_class: 背景类别索引
            smoothing_iter: 平滑迭代次数
        """
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.background_class = background_class
        self.smoothing_iter = smoothing_iter
        self.smoother = FrameSmoother(num_classes=24, window_size=5, device='cuda')

    def process_batch(self, pred_labels: Tensor) -> Tensor:
        """
        批量处理预测图

        Args:
            pred_labels: torch.Tensor, shape (H, W) [int64]

        Returns:
            processed: torch.Tensor, shape (H, W) [int64]
        """
        H, W = pred_labels.shape
        processed = self._process_single_image(pred_labels)
        return processed

    def _process_single_image(self, label_map: Tensor) -> Tensor:
        """
        单张图像处理

        Args:
            label_map: torch.Tensor, shape (H, W), int64

        Returns:
            processed_labels: torch.Tensor, shape (H, W), int64
        """
        labels: np.ndarray = label_map.cpu().numpy().astype(np.uint8)
        processed_labels: np.ndarray = np.zeros_like(labels)

        non_bg_classes = np.unique(labels)
        non_bg_classes = non_bg_classes[non_bg_classes != self.background_class]

        for cls in non_bg_classes:
            cls_mask = (labels == cls).astype(np.uint8)

            if cls == 1:
                optimized_mask = self._optimize_class_mask(cls_mask, NMS=False)
            else:
                optimized_mask = self._optimize_class_mask(cls_mask)

            processed_labels[optimized_mask > 0] = cls

        return torch.from_numpy(processed_labels)

    def _optimize_class_mask(self, mask: np.ndarray, NMS: bool = True) -> np.ndarray:
        """
        优化二值掩码

        Args:
            mask: np.ndarray, shape (H, W), dtype uint8
            NMS: 是否只保留最大区域

        Returns:
            smoothed_mask: np.ndarray, shape (H, W), dtype uint8
        """
        labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)
        regions = measure.regionprops(labeled_mask)

        filtered_mask = np.zeros_like(mask)
        Max_area = 0
        Max_region = None

        if NMS:
            for region in regions:
                if region.area >= Max_area:
                    Max_area = region.area
                    Max_region = region
                    coords = region.coords
            filtered_mask[coords[:, 0], coords[:, 1]] = 1
        else:
            for region in regions:
                if region.area > 100:
                    coords = region.coords
                    filtered_mask[coords[:, 0], coords[:, 1]] = 1
                    Max_region = region

        if Max_region is None:
            return filtered_mask

        minor_axis_length = Max_region.minor_axis_length
        if minor_axis_length < self.kernel_size:
            self.kernel2 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(int(minor_axis_length / 3), 5), max(int(minor_axis_length / 5), 3))
            )
            self.kernel1 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(int(minor_axis_length / 6), 5), max(int(minor_axis_length / 5), 3))
            )
        else:
            self.kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
            self.kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for _ in range(self.smoothing_iter):
            filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, self.kernel2)
            filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, self.kernel1)

        filtered_mask = binary_fill_holes(filtered_mask).astype(np.uint8) * 255
       # smoothed_mask = cv2.GaussianBlur(filtered_mask.astype(np.float32), (9, 9), 0.5)

        return filtered_mask.astype(np.uint8)


def Manual_label_adjustment(input_path: str, save_path: str,name_list:List) -> None:
    """
    对标签进行手动后处理并保存，这里namelist直接就是 一组病例的
    Args:
        input_path: str, 输入文件夹路径
        save_path: str, 输出文件夹路径
        name_list: list  ，同一病例下的顺序帧
    """

    segment_postprocess = ArgmaxPostProcessor()
    # 这里已经直接输入了 ；name_list
    # name_list: List[str] = os.listdir(input_path)
    # name_list = sorted(name_list, key=lambda x: int(re.search(r'_(\d+)\.npy$', x).group(1)))
    #
    count = 1
    for i in range(len(name_list)):
        input_data:str = os.path.join(input_path, name_list[i])
        rawinput: np.ndarray = np.load(input_data)  # shape (H, W)
        start = time.time()
        pred: np.ndarray = segment_postprocess._process_single_image(torch.from_numpy(rawinput)).numpy()
        end = time.time()
        np.save(os.path.join(save_path, name_list[i].split(".")[0] + ".npy"), pred)
        cur_time = end - start
        #print("Image[{}/{}] : {:.5f}ms".format(count, len(name_list), cur_time))
        count += 1


if __name__ == '__main__':

    '''
    这里是所有的病例都在一起的，所以这里 重新分配以及排序   
    '''
    input_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/label"
    save_path =r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/label_manual"

# 5. 运行
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    namelist = os.listdir(input_path)

    namelist = sorted(namelist, key=lambda x: (int(re.search(r'(\d+)_(\d+)_(\d+)\.npy$', x).group(1)),
                                               int(re.search(r'(\d+)_(\d+)_(\d+)\.npy$', x).group(2)),
                                               int(re.search(r'(\d+)_(\d+)_(\d+)\.npy$', x).group(3))))


    file_idx = lambda x: (int(re.search(r'(\d+)_(\d+)_(\d+)\.npy$', x).group(1)),
                          int(re.search(r'(\d+)_(\d+)_(\d+)\.npy$', x).group(2)))
    last_file_idx = None
    instance_list = []
    all_instance = []

    for i in range(len(namelist)):

        if last_file_idx is None:
            last_file_idx = file_idx(namelist[i])
            instance_list.append(namelist[i])

        else:
            if file_idx(namelist[i]) == last_file_idx:
                instance_list.append(namelist[i])
            else:
                all_instance.append(instance_list)
                instance_list =[ ]
                last_file_idx = file_idx(namelist[i])
                instance_list.append(namelist[i])
    print(all_instance)

    with ThreadPoolExecutor(max_workers=24) as executor:
        for idx in range(len(all_instance)):
            instance_list = all_instance[idx]

            #Manual_label_adjustment(input_path,save_path,instance_list)
            executor.submit(Manual_label_adjustment, input_path,save_path,instance_list)
            print(f"finish process {idx}/{len(all_instance)}")
    #
    #


        #


















