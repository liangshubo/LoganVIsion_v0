import os
import re
import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path
from functools import wraps


def log_call(func):
    """装饰器：打印函数调用信息"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = ", ".join([
            *(repr(a) for a in args),
            *(f"{k}={v!r}" for k, v in kwargs.items())
        ])
        print(f"[DEBUG] 调用 {func.__name__}({arg_str})")
        result = func(*args, **kwargs)
        print(f"[DEBUG] {func.__name__} 返回类型: {type(result).__name__}")
        return result
    return wrapper



def get_colormap(num_classes: int = 24) -> np.ndarray:
    """
    生成固定调色板
    Args:
        num_classes: 类别数量
    Returns:
        (num_classes, 3) 的 RGB 颜色表
    """
    np.random.seed(42)
    colors: np.ndarray = np.random.randint(
        0, 255, size=(num_classes, 3), dtype=np.uint8
    )
    colors[0] = [0, 0, 0]  # 背景黑色
    return colors



def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    colormap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    将 mask 叠加到图像上
    Args:
        image: 原始图像 (H, W, 3)
        mask: 分割标签 (H, W)，值为类别索引
        colormap: 调色板 (num_classes, 3)
        alpha: 透明度
    Returns:
        叠加后的图像 (H, W, 3)
    """
    color_mask: np.ndarray = colormap[mask]
    overlay: np.ndarray = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay


def play_labels_video_mem(
        image_dir: Path,
        label1_dir: Path,
        label2_dir: Path,
        name_list:List,
        alpha: float = 0.5,
        delay: int = 100  # 每帧延时 ms
):
    """将所有图像和标签加载到内存，然后连续播放"""


    colormap = get_colormap(24)

    # ---------------------------
    # 1️⃣ 读取所有图像和标签到内存
    # ---------------------------
    images: List[np.ndarray] = []
    masks1: List[np.ndarray] = []
    masks2: List[np.ndarray] = []


    image_name = lambda x:x.split(".")[0]+".png"
    print("开始加载所有图像和标签到内存...")
    for img_name in name_list:

        base_name = os.path.splitext(img_name)[0]

        # 读取图像
        img_path = image_dir / image_name(img_name)
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"图像读取失败: {img_path}")
            continue

        # 读取两个标签
        label1_path = label1_dir / f"{base_name}.npy"
        label2_path = label2_dir / f"{base_name}.npy"

        if not label1_path.exists() or not label2_path.exists():
            print(f"缺少对应标签: {base_name}")
            continue

        mask1 = np.load(label1_path)
        mask2 = np.load(label2_path)

        # 调整尺寸
        if mask1.shape != image.shape[:2]:
            mask1 = cv2.resize(mask1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask2.shape != image.shape[:2]:
            mask2 = cv2.resize(mask2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        images.append(image)
        masks1.append(mask1)
        masks2.append(mask2)

    print(f"加载完成，总帧数: {len(images)}")

    if len(images) == 0:
        print("没有有效图像可播放。")
        return

    # ---------------------------
    # 2️⃣ 创建窗口并播放
    # ---------------------------
    cv2.namedWindow("Label Comparison", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label Comparison", 1600, 1000)
    idx = 0
    paused = False
    total_frames = len(images)

    while True:
        image = images[idx]
        mask1 = masks1[idx]
        mask2 = masks2[idx]

        overlay1 = overlay_mask(image, mask1, colormap, alpha)
        overlay2 = overlay_mask(image, mask2, colormap, alpha)
        combined = np.hstack((overlay1, overlay2))

        cv2.imshow("Label Comparison", combined)

        key = cv2.waitKey(delay) & 0xFF

        if key == 27 or key == ord("q"):  # ESC 或 q
            break
        elif key == ord("p"):  # 暂停/继续
            paused = not paused
        elif key == ord("a"):  # 上一帧
            idx = max(0, idx - 1)
        elif key == ord("d"):  # 下一帧
            idx = min(total_frames - 1, idx + 1)

        if not paused:
            idx += 1
            if idx >= total_frames:  # 循环播放
                idx = 0

    cv2.destroyAllWindows()






if __name__ == '__main__':
    #  这里用于对比  神经的 前后标签

    save_path =r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/label_manual"



    image_folder = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/image"
     # 3. IMAGE-S1 这里要加载的是原始的标签
    mask_folder_raw= r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/label"

    # 4. 之后的
    mask_folder_manual = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/label_cnn"

    namelist = os.listdir(mask_folder_raw)

    namelist = sorted(namelist, key=lambda x: (int(re.search(r'(\d+)_(\d+)_(\d+)\.npy$', x).group(1)),
                                               int(re.search(r'(\d+)_(\d+)_(\d+)\.npy$', x).group(2)),
                                               int(re.search(r'(\d+)_(\d+)_(\d+)\.npy$', x).group(3))))

    # print(namelist)
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
                print(file_idx(namelist[i]))
                instance_list.append(namelist[i])
            else:
                print(file_idx(namelist[i]))
                all_instance.append(instance_list)
                instance_list = []
                print(namelist[i])
                last_file_idx = file_idx(namelist[i])
                instance_list.append(namelist[i])
    print(all_instance)




    for  j in range(len(all_instance)):
        print(all_instance[j])
        input_path = image_folder   # /home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/all_sam418_data_segment/png/test/label_cnn/S0/PATCHNE/
        mask_path_raw = mask_folder_raw
        mask_path_manual  = mask_folder_manual
    # 5. 运行

        # 示例调用
        play_labels_video_mem(Path(input_path ),
                     Path(mask_path_raw ),
                     Path(mask_path_manual ),
                     all_instance[j])
