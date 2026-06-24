

import cv2
import numpy as np
import math
import random

def rotate_augment_cover(img: np.ndarray,mask:np.ndarray, angle_deg: float,
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

        mask = cv2.resize(mask , (bbox_dim, bbox_dim), interpolation= cv2.INTER_NEAREST)
    else:
        resized = img.copy()

    # -------- 2 \ 以中心旋转 resized --------
    center = (bbox_dim * 0.5, bbox_dim * 0.5)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_full = cv2.warpAffine(
        resized, M, (bbox_dim, bbox_dim),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0
    )

    rotated_mask = cv2.warpAffine(
        mask, M, (bbox_dim, bbox_dim),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    H1,W1 = rotated_full.shape[:2]
    out_img = rotated_full[int((H1-H)/2):int((H1+H)/2),int((W1-W)/2):int((W1+W)/2)]
    out_mask = rotated_mask[int((H1-H)/2):int((H1+H)/2),int((W1-W)/2):int((W1+W)/2)]

    return out_img,out_mask



def random_translate(img: np.ndarray,mask: np.ndarray, max_shift: float,
                     interp=cv2.INTER_CUBIC):
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
    angle = random.uniform(-180,45 )
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



# ------------------- 示例 -------------------
if __name__ == "__main__":
    #假设 img 已读入且为正方形
    img = cv2.imread("/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/all_sam418_data_segment/png/train/image/S1/S1Pacient_20250723173943_1/S1Pacient_20250723173943_1_0.png")  # BGR
    mask = np.load("/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/all_sam418_data_segment/png/train/label/S1/S1Pacient_20250723173943_1/S1Pacient_20250723173943_1_0.npy")
    img = cv2.resize(img,(512,512))
    mask = cv2.resize(mask,(512,512))

    angle = -5
    out,mask = rotate_augment_cover(img,mask , angle)

    out_img, mask = random_translate(img,mask,50)
    


    cv2.imshow("aug_rotated.png", out)

    cv2.imshow("aug_translate.png", out_img)
    cv2.waitKey()