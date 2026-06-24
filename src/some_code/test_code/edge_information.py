import cv2
import numpy as np
from PIL import Image

import cv2
import numpy as np
from PIL import Image


def get_edge_from_image(image_path):
    """
    从输入图像中提取Canny边缘。

    参数:
    - image_path: 输入图像的路径 (例如, 'path/to/image.jpg')
    - low_threshold, high_threshold: Canny算子的双阈值

    返回:
    - edge: 灰度边缘图像 (numpy array, HxW)
    """
    # 以灰度模式读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    high_threshold = image.mean()
    low_threshold = 0
    # 使用Canny算子进行边缘检测
    edges = cv2.Canny(image,  low_threshold , high_threshold)

    return edges


def get_sobel_edge_from_image(image_path, ksize=3):
    """
    从输入图像中提取Sobel边缘。

    参数:
    - image_path: 输入图像的路径 (例如, 'path/to/image.jpg')
    - ksize: Sobel算子的核大小，必须是奇数，如1, 3, 5, 7。通常为3。

    返回:
    - sobel_edge: 灰度边缘图像 (numpy array, HxW)
    """
    # 1. 以灰度模式读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 2. 计算x和y方向的梯度
    # 使用cv2.CV_64F以避免梯度值溢出，因为从白到黑的边会是负数
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # 3. 计算梯度的绝对值，并转换回uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # 4. 合并两个方向的梯度
    # 可以使用addWeighted进行加权融合，也可以直接计算幅值
    # 这里使用更精确的幅值计算方法：sqrt(grad_x^2 + grad_y^2)
    # 或者一个简单的近似：|grad_x| + |grad_y|
    sobel_edge = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return sobel_edge


import cv2
import numpy as np
from PIL import Image


def get_edge_from_label(label_path):
    """
    从P模式的标签图中提取边缘真值。

    参数:
    - label_path: P模式标签图的路径 (例如, 'path/to/label.png')

    返回:
    - edge_gt: 二值的边缘真值图 (numpy array, HxW, 值为0或1)
    """
    # 使用Pillow打开P模式图像，可以正确读取其索引值

    label_np =  np.load(label_path) # 直接转换为numpy数组，值为类别索引

    # 定义一个3x3的结构元素（kernel）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 对标签图进行膨胀操作
    dilated = cv2.dilate(label_np, kernel, iterations=1)

    # 对标签图进行腐蚀操作
    eroded = cv2.erode(label_np, kernel, iterations=1)

    # 边缘 = 膨胀结果 - 腐蚀结果
    # 在结果不为0的地方就是边缘
    edge_gt = (dilated != eroded).astype(np.uint8)

    # 可选：忽略特定类别（如背景0）的边缘
    # edge_gt[label_np == 0] = 0 # 如果背景类的边缘不需要，可以取消这行注释

    return edge_gt








# --- 使用示例 ---
label_edge_gt = get_edge_from_label('/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/sample_sam_dataset0805/png/test/label/S8/S8Pacient_20250725165242_19/S8Pacient_20250725165242_19_0.npy')

# label_edge_gt 是一个二值图，其中 1 代表边缘，0 代表非边缘
# 它可以直接作为训练时的目标 (target)

# 可视化检查（可选）
# cv2.imwrite('edge_ground_truth.png', label_edge_gt * 255)
print(f"生成的标签边缘图尺寸: {label_edge_gt.shape}")
print(f"边缘图中的唯一值: {np.unique(label_edge_gt)}")  # 应为 [0, 1]

# --- 使用示例 ---
image_edge_gt = get_sobel_edge_from_image('/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/sample_sam_dataset0805/png/test/image/S8/S8Pacient_20250725165242_19/S8Pacient_20250725165242_19_0.png')
image_edge_gt_canny = get_edge_from_image('/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/sample_sam_dataset0805/png/test/image/S8/S8Pacient_20250725165242_19/S8Pacient_20250725165242_19_0.png')
# label_edge_gt 是一个二值图，其中 1 代表边缘，0 代表非边缘
# 它可以直接作为训练时的目标 (target)
image = cv2.imread('/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/sample_sam_dataset0805/png/test/image/S8/S8Pacient_20250725165242_19/S8Pacient_20250725165242_19_0.png',0)
# 可视化检查（可选）
cv2.imshow('edge_ground_truth_sobel', image_edge_gt)
cv2.imshow("raw",image)
cv2.imshow("edge_ground_truth_canny",image_edge_gt_canny)
cv2.imshow("label",label_edge_gt*255)
cv2.waitKey(0)
print(f"生成的标签边缘图尺寸: {label_edge_gt.shape}")
print(f"边缘图中的唯一值: {np.unique(label_edge_gt)}")  # 应为 [0, 1]