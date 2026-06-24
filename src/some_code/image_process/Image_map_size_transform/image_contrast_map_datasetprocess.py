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
left_image = cv2.imread('/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/src/demoimage/Msk_Process/Msk_Process_1/SRI13_MSK_CV2/SHOULD0Q_42.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/src/demoimage/Msk_Process/Msk_Process_2/GAN_BASE_EX2_SCI->SAM_MSK_LV1/SHOULD0Q_42.png', cv2.IMREAD_GRAYSCALE)
  
# 调整右边图像的对比度以匹配左边图像  
matched_right_image = match_histograms(right_image, left_image)  
  
# 显示结果  
cv2.imshow('Left Image', left_image)  
cv2.imshow('Right Image Before Matching', right_image)  
cv2.imshow('Right Image After Matching', matched_right_image)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  
  
# 保存结果  
cv2.imwrite('matched_right_image.png', matched_right_image)