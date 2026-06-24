
import json
import os.path

from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Dict

import re
#import chardet

LABEL_MAP: Dict[str, int] = {
    "shenjing": 1,
    "jirouzuzhi": 2,
    "dongmai": 3,
    "jingmai": 4
}
def json_to_segmask(json_path: str,
                     save_path: Union[str, None] = None) -> np.ndarray:
    """
    读取 LabelMe 风格的 json 文件，生成单通道分割 mask。
    - json_path: json 文件路径
    - save_path: 若提供，则把生成的 mask（PNG）保存到该路径（单通道，像素值为类别索引）
    返回:
        mask: numpy.ndarray, shape (H, W), dtype=np.uint8
    """
    # 读取 json
    with open(json_path, 'rb') as f:
        raw = f.read()

    # 强制 decode，非法字符用替代符
    text = raw.decode('utf-8-sig', errors='replace')

    # 解析 JSON
    data = json.loads(text)


    # data = load_json_auto(json_path)
    # 获取图像尺寸（确保存在）
    if 'imageHeight' not in data or 'imageWidth' not in data:
        raise ValueError("json 中缺少 imageHeight 或 imageWidth 字段。")

    H = int(data['imageHeight'])
    W = int(data['imageWidth'])

    # 创建空白单通道图像（Pillow mode 'L'），初始为 0 (背景)
    mask_img = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    shapes = data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label')
        if label is None:
            continue
        # 忽略不在映射表中的标签
        if label not in LABEL_MAP:
            # 可选：打印/记录未识别标签
            # print(f"警告: 未知标签 {label}，已忽略。")
            continue

        value = LABEL_MAP[label]

        points = shape.get('points', [])
        if not points:
            continue

        # 将点列表转换为 (x,y) 的平坦序列或元组列表
        # Pillow 要求坐标为 (x, y) 且顺序为像素位置
        polygon = [(float(x), float(y)) for x, y in points]

        # 使用 ImageDraw 填充多边形，fill 指定为类别值
        draw.polygon(polygon, outline=value, fill=value)

    # 转为 numpy 数组
    mask = np.array(mask_img, dtype=np.uint8)

    # 可选保存
    if save_path:
        # 如果要保存为 PNG，确保保持单通道值
        np.save(save_path,mask)

    return mask

# -------------------------
# 示例用法
# -------------------------
if __name__ == "__main__":
    json_folder = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/ubpb/all_json"
    save_folder = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/ubpb/all_npy"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    namelist = os.listdir(json_folder)
    namelist= sorted( namelist, key=lambda x: int(re.search(r'(\d+)_(\d+)\.json$', x).group(1)))
    for i in range(len(namelist)):
        name,ext = os.path.splitext(namelist[i])
        json_path = os.path.join(json_folder,namelist[i])
        save_path  = os.path.join(save_folder,name+".npy")
        mask = json_to_segmask(json_path, save_path= save_path)

        print("mask shape:", mask.shape, "unique labels:", np.unique(mask))
        print(f"{i}/{len(namelist)} {namelist[i]}")