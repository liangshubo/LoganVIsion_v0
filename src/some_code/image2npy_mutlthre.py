import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # 导入ThreadPoolExecutor用于管理线程池


import time
# 定义处理单个图像的函数
def process_image(image_path, save_path):
    # 从图像路径中提取文件名和扩展名
    name, ext = os.path.splitext(os.path.basename(image_path))
    # 使用OpenCV加载图像
    image_array = cv2.imread(image_path, 0)  # 0表示加载为灰度图像
    # 使用NumPy保存图像数组为.npy文件
    np.save(os.path.join(save_path, name), image_array)
    # 打印处理完成的信息
    print(f"Finished processing {name}")

# 主函数，用于调度多线程处理图像
def trans(folder, save_path):
    # 如果保存路径不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    start = time.time()
    # 获取原始图像文件夹中所有文件的文件名及扩展名
    name_ext = os.listdir(folder)
    # 创建线程池，最多同时运行 16 个线程
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交每个图像处理任务到线程池
        for name in name_ext:
            image_path = os.path.join(folder, name)
            # 提交任务到线程池中，参数为图像路径和保存路径
            executor.submit(process_image, image_path, save_path)
    end = time.time()
    print(end-start)
    
# 原始图像文件夹路径
folder = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset401/label2"
# 保存处理后图像的路径
save_path = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset403/label2"

# 调用函数开始处理图像
trans(folder, save_path)
