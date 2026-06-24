import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def visualize_segmentation(original_img, label_img, alpha=0.5):
    """
    将语义分割结果与原始图像叠加显示

    参数:
        original_img: 原始图像（可以是灰度或彩色）需要转换为np.uint8 格式
        label_img: 标签图像（单通道，像素值1-10代表不同类别） 需要转换为np.uint8
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
        4: [255, 255, 0],  # 黄色
        5: [0, 255, 255],  # 青色
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


def _draw_legend(overlay_image, pred_mask, alpha, background_class=0):
    """
    在图像上绘制图例，支持透明颜色块和中文字符。

    参数:
        overlay_image (np.array): 已经叠加了分割掩码的图像。
        pred_mask (np.array): 预测的类别索引掩码。
        alpha (float): 用于图例颜色块的透明度，应与`_visualize_segmentation_fast`中的alpha一致。
        background_class (int): 不在图例中显示的背景类别索引。

    返回:
        np.array: 绘制了图例的最终图像。
    """

    VOC_COLORMAP = [
        [0,0,0,],
       [0, 0, 255],  # 红色
    [0, 255, 0],  # 绿色
    [255, 0, 0],  # 蓝色
    [255, 255, 0],  # 青色  # 31 -32
    ]


    CLASS_NAMES = {
        1: "shenjing",2:"jirouzuzhi",3:"dongmai",4:"jingmai"}
    # 1. 筛选出需要显示的类别
    unique_labels = sorted([label for label in np.unique(pred_mask) if label != background_class])
    if not unique_labels:
        return overlay_image

    # 复制一份图像用于绘制
    overlay_with_boxes = overlay_image.copy()

    # 图例参数
    legend_start_x, legend_start_y = 10, 10
    box_height, box_width = 20, 30
    spacing, font_size = 7, 16

    # 用来存储文本信息，后续统一用Pillow绘制
    text_to_draw = []

    # --- 步骤一: 使用OpenCV绘制所有【透明】的颜色方块 ---
    for i, label in enumerate(unique_labels):
        y_pos = legend_start_y + i * (box_height + spacing)

        # 获取颜色(BGR格式)和类名
        color_bgr = VOC_COLORMAP[label] if label < len(VOC_COLORMAP) else [128, 128, 128]
        class_name = CLASS_NAMES.get(label, f"类 {label}")

        # 定义方块区域
        box_start_x, box_start_y = legend_start_x, y_pos
        box_end_x, box_end_y = legend_start_x + box_width, y_pos + box_height

        # 防止图例超出图像边界
        if box_end_y > overlay_with_boxes.shape[0] or box_end_x > overlay_with_boxes.shape[1]:
            continue

        # 核心：创建透明方块
        roi = overlay_with_boxes[box_start_y:box_end_y, box_start_x:box_end_x]  # 提取背景区域
        box_patch = np.full((box_height, box_width, 3), color_bgr, dtype=np.uint8)  # 创建纯色方块
        blended_roi = cv2.addWeighted(roi, 1 - alpha, box_patch, alpha, 0)  # 将方块与背景融合
        overlay_with_boxes[box_start_y:box_end_y, box_start_x:box_end_x] = blended_roi  # 放回原图

        # 记录文本位置和内容，供Pillow使用
        text_x = legend_start_x + box_width + 5
        text_y = y_pos + (box_height - font_size) // 2  # 垂直居中对齐
        text_to_draw.append({'text': class_name, 'pos': (text_x, text_y)})

    # --- 步骤二: 使用Pillow绘制所有【中文】文本 ---
    # 将OpenCV图像(BGR)转换为Pillow图像(RGB)
    pil_image = Image.fromarray(cv2.cvtColor(overlay_with_boxes, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # 加载支持中文的字体
    # **重要提示**: 请将 "simhei.ttf" 替换为您系统上实际存在的中文字体文件路径。
    # Windows: "C:/Windows/Fonts/simhei.ttf" (黑体) 或 "msyh.ttc" (微软雅黑)
    # macOS/Linux: 您可能需要先安装字体，然后提供其路径。
    try:
        font_path = "/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"警告: 中文字体 '{font_path}' 未找到。将使用默认字体，中文可能无法正常显示。")
        print("请提供一个有效的中文字体路径（例如 simhei.ttf）。")
        font = ImageFont.load_default()

    font_color_pil = (255, 255, 255)  # 白色

    # 统一绘制所有文本
    for item in text_to_draw:
        draw.text(item['pos'], item['text'], font=font, fill=font_color_pil)

    # --- 步骤三: 将Pillow图像转换回OpenCV图像(BGR) ---
    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return final_image


# 示例用法
if __name__ == "__main__":
    import  re
    def show(original, labels,ordername=None):
        # 检查图像是否加载成功
        if original is None or labels is None:
            print("无法加载图像，请检查路径")
            exit()

        # 可视化（透明度设为0.5）
        print(original.shape)
        print(labels.shape)
        print(original.max())
        print(labels.max())

        result = visualize_segmentation(original, labels, alpha=0.35)
        result = _draw_legend(result, labels, alpha=0.5, )
        # 显示结果
        print(ordername)
        cv2.imshow("Segmentation Visualization", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/UBPB/train/label"
    save_path=r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/UBPB/train/label_single"
    import numpy as np
    # 加载图像（替换为你的图像路径）000
    original_folder =r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/UBPB/train/image" # cv2.imread("/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/ubpb/all_image/1_111.jpg",0)  # 可以是彩色或灰度
    labels_folder =r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/UBPB/train/label_single" #np.load("/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/ubpb/all_npy/1_111.npy")  # 单通道标签图

    namelist = os.listdir(original_folder)
    # namelist = sorted(namelist, key=lambda x: (int(re.search(r'(\d+)_(\d+)_(\d+)\.png$', x).group(1)),
    #                                            int(re.search(r'(\d+)_(\d+)_(\d+)\.png$', x).group(2)),
    #                                            int(re.search(r'(\d+)_(\d+)_(\d+)\.png$', x).group(3))))
    # print(namelist)
    for i in range(len(namelist)):
        print(namelist[i])
        name,ext = os.path.splitext(namelist[i])
        original = cv2.imread(os.path.join(original_folder,namelist[i]),0)
        labels = np.load(os.path.join(labels_folder,name+".npy"))

        show(original, labels,ordername=namelist[i])
        #print(f"{i}/{len(namelist)} {namelist[i]}" )

        # 保存结果（可选）
        # cv2.imwrite("segmentation_visualization.jpg", result)

