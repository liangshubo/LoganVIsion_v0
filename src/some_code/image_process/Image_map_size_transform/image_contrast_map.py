import os

import cv2
import numpy as np  
  
def calculate_histogram(image):  
    # 计算图像的直方图  
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])  
    return hist  
  
def calculate_cdf(hist):  
    # 计算累积分布函数（CDF）  
    cdf = hist.cumsum()  
    cdf_normalized = cdf * hist.max() / cdf.max()  # 归一化  
    return cdf_normalized, cdf  
  
def match_histograms(source, reference):
    """
    左边才是参考的图像 ， 右边是等待处理的图像
    """
    # 计算源图像和参考图像的直方图  
    src_hist = calculate_histogram(source)  
    ref_hist = calculate_histogram(reference)  
  
    # 计算累积分布函数（CDF）  
    src_cdf_normalized, src_cdf = calculate_cdf(src_hist)  
    ref_cdf_normalized, ref_cdf = calculate_cdf(ref_hist)  
  
    # 创建映射表  
    mapping = np.zeros(256, dtype=np.uint8)  
    ref_idx = 0  
    for src_idx in range(256):  
        while ref_idx < 256 and ref_cdf[ref_idx] < src_cdf[src_idx]:  
            ref_idx += 1  
        mapping[src_idx] = ref_idx if ref_idx < 256 else 255  

    # 应用映射表到源图像  
    matched = cv2.LUT(source, mapping)  
    return matched  
  
# 读取图像

def dataset_process(path,ref_path,save_path):
    #首先保证的是两待处理文件夹 与 参考文件夹的内含的文件是一致的
    name_ext = os.listdir(path)
    for i in range(len(name_ext)):
        image_path = os.path.join(path,name_ext[i])
        refer_path = os.path.join(ref_path,name_ext[i])
        image_array = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        ref_array = cv2.imread(refer_path,cv2.IMREAD_GRAYSCALE)
        matched_right_image = match_histograms(   image_array,ref_array)
        save_image_path = os.path.join(save_path,name_ext[i])
        cv2.imwrite(save_image_path,matched_right_image,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish his match [{i}/{len(name_ext)}] ")

if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor

    path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/dataset_process_926/plan2/GAN_SCI_02->SAM_MSK_LV1->Down13055"
    ref_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/compose_all_cv0/remapdown13055->SRI_MSK_LV2->MapD"
    save_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/dataset_process_926/plan2/GAN_SCI_02->SAM_MSK_LV1->Down13055->contrast"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.submit(dataset_process,path,ref_path,save_path)
