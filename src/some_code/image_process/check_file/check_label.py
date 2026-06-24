import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import matplotlib
matplotlib.use('TkAgg')
# 1. 定义您的调色板和类别名称
# 格式: { class_index: ((R, G, B), "Class Name") }
# 注意：class_index 0 通常是背景，这里我们将其设置为黑色且在叠加时会变透明
PALETTE = {
    0: ((0, 0, 0), "Background"),
    1: ((255, 0, 0), "S1-1-肱骨大结节、肱骨小结节"),
    2: ((0, 255, 0), "S1-2-肱二头肌长头腱短轴"),
    3: ((0, 0, 255), "S1-3-三角肌"),
    4: ((255, 255, 0), "S2-4-三角肌"),
    5: ((0, 255, 255), "S2-5-肱骨头表面"),
    6: ((255, 0, 255), "S2-6-肱二头肌长头腱长轴"),
    7: ((255, 239, 213), "S3-7-肩胛下肌腱长轴"),
    8: ((0, 0, 205), "S3-8-肱骨小结节表面"),
    9: ((205, 133, 63), "S4-9-肩胛下肌腱短轴"),
    10: ((210, 180, 140), "S4-10-肱骨小结节表面"),
    11: ((102, 205, 170), "S5-11-冈上肌腱长轴"),
    12: ((0, 0, 128), "S5-12-肱骨大结节"),
    13: ((0, 139, 139), "S5-13-肩峰下囊"),
    14: ((46, 139, 87), "S6-14-肱二头肌长头肌腱短轴"),
    15: ((255, 228, 225), "S6-15-冈和冈下肌腱短轴"),
    16: ((106, 90, 205), "S6-16-三角肌"),
    17: ((221, 160, 221), "S7-17-冈下肌腱长轴"),
    18: ((233, 150, 122), "S7-18-肱骨大结节"),
    19: ((165, 42, 42), "S7-19-肱骨解剖胫"),
    20: ((255, 250, 250), "S8-20-肱骨大结节"),
    21: ((147, 112, 219), "S8-21-三角肌"),
    22: ((218, 112, 214), "S8-22-小圆肌长轴"),
    23: ((75, 0, 130), "S9-23-肱骨头表面"),
    24: ((255, 182, 193), "S9-24-后盂唇"),
    25: ((60, 179, 113), "S9-25-冈下肌腱"),
    26: ((255, 235, 205), "S9-26-关节盂"),
    27: ((255, 228, 196), "S10-27-锁骨"),
    28: ((218, 165, 32), "S10-28-肩峰"),
    29: ((0, 128, 128), "S10-29-肩锁关节囊"),
    30: ((188, 143, 143), "S11-30-肩峰"),
    31: ((255, 105, 180), "S11-31-喙突"),
    32: ((255, 218, 185), "S11-32-喙肩韧带"),
    33: ((222, 184, 135), "Label 33"),
    34: ((127, 255, 0), "Label 34")
}


def create_dummy_data(img_path, lbl_path, width=400, height=300):
    """
    创建一个虚拟的灰度图和标签npy文件用于测试。
    如果您的文件已存在，则此函数不执行任何操作。
    """
    # 1. 创建虚拟灰度图
    if not os.path.exists(img_path):
        print(f"创建虚拟灰度图: {img_path}")
        # 创建一个从左到右的渐变灰度图
        gray_data = np.zeros((height, width), dtype=np.uint8)
        gray_data[:] = np.linspace(0, 255, width)
        Image.fromarray(gray_data, 'L').save(img_path)

    # 2. 创建虚拟标签NPY文件
    if not os.path.exists(lbl_path):
        print(f"创建虚拟标签文件: {lbl_path}")
        # 创建一个包含多个类别区域的标签
        label_data = np.zeros((height, width), dtype=np.uint8)
        # 类别1的一个矩形区域
        label_data[50:150, 50:150] = 1
        # 类别2的一个矩形区域
        label_data[50:150, 250:350] = 2
        # 类别11的一个圆形区域
        cy, cx, rad = height // 2, width // 2, 80
        y, x = np.ogrid[-cy:height - cy, -cx:width - cx]
        mask = x * x + y * y <= rad * rad
        label_data[mask] = 11
        np.save(lbl_path, label_data)


def visualize_segmentation(image_path: str, label_path: str, palette: dict, alpha: float = 0.6):
    """
    加载灰度图和NPY标签文件，将它们叠加并显示。

    Args:
        image_path (str): 原始灰度图的文件路径。
        label_path (str): P模式Numpy标签 (.npy) 的文件路径。
        palette (dict): 类别索引到 (RGB元组, 类别名称) 的映射字典。
        alpha (float): 叠加标签蒙版时的透明度 (0.0 到 1.0)。
    """
    # 1. 加载原始灰度图像
    try:
        # 以 'L' 模式加载，确保是灰度图，然后转换为 'RGBA' 以便叠加
        original_image = Image.open(image_path).convert('RGBA')
    except FileNotFoundError:
        print(f"错误: 原始图像文件未找到 at '{image_path}'")
        return

    # 2. 加载标签 NPY 文件
    try:
        label_array = np.load(label_path)
    except FileNotFoundError:
        print(f"错误: 标签NPY文件未找到 at '{label_path}'")
        return

    # 3. 将P模式（单通道索引）的标签数组转换为彩色（RGBA）蒙版
    # 创建一个空的4通道（R, G, B, A）数组
    height, width = label_array.shape
    color_mask = np.zeros((height, width, 4), dtype=np.uint8)

    # 获取标签中所有唯一的类别ID
    present_classes = np.unique(label_array)
    legend_handles = []

    # 为每个类别ID填充颜色和透明度
    for class_id in present_classes:
        if class_id in palette:
            color, name = palette[class_id]
            # 找到当前类别的所有像素
            mask = label_array == class_id

            # 设置颜色 (R, G, B)
            color_mask[mask, 0:3] = color

            # 设置透明度 (Alpha)，背景(class_id=0)保持完全透明
            color_mask[mask, 3] = 0 if class_id == 0 else int(alpha * 255)

            # 为图例创建补丁
            if class_id != 0:
                legend_handles.append(mpatches.Patch(color=np.array(color) / 255.0, label=name))

    # 将numpy数组转换回Pillow图像对象
    color_mask_image = Image.fromarray(color_mask, 'RGBA')

    # 4. 将彩色蒙版叠加到原始图像上
    # 使用 alpha_composite 进行像素级混合
    composite_image = Image.alpha_composite(original_image, color_mask_image)

    # 5. 使用 Matplotlib 显示结果
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(composite_image)
    ax.set_title(f'语义分割标注检查\n(图像: {os.path.basename(image_path)}, 标签: {os.path.basename(label_path)})')
    ax.axis('off')  # 关闭坐标轴

    # 添加图例
    if legend_handles:
        fig.legend(handles=legend_handles, title="类别", loc='center', bbox_to_anchor=(0.90, 0.90))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局为图例留出空间
    else:
        plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    # --- 配置区域 ---
    # 请将下面的路径修改为您的真实文件路径
    # 如果文件不存在，脚本会创建虚拟文件用于演示
    IMAGE_PATH = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/sample_sam_dataset/png/train/image/S1/S1Pacient_20250725165242_2/S1Pacient_20250725165242_2_40.png"
    LABEL_PATH = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/sample_sam_dataset/png/train/label/S1/S1Pacient_20250725165242_2/S1Pacient_20250725165242_2_40.npy"

    # --- 运行 ---
    # 1. （可选）创建用于演示的虚拟数据
    print("--- 检查并创建虚拟数据（如果需要） ---")
    create_dummy_data(IMAGE_PATH, LABEL_PATH)
    print("--- 数据准备完毕 ---\n")

    # 2. 运行可视化函数
    print("--- 开始进行可视化 ---")
    visualize_segmentation(
        image_path=IMAGE_PATH,
        label_path=LABEL_PATH,
        palette=PALETTE,
        alpha=0.7  # 你可以在这里调整叠加的透明度，0.7表示70%不透明
    )