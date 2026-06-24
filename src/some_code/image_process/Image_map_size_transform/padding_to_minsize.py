"""
Data:2024/4/16
Name:liangshubo
Object:
multi process the image to padding the up\bottom\left\right 50
"""
import os

import cv2

import numpy as np

from concurrent.futures import ThreadPoolExecutor


def copyblack(floder,save_path,size=512):
    nameext = os.listdir(floder)
    for i in range(len(nameext)):
        image_path = os.path.join(floder,nameext[i])
        image = cv2.imread(image_path,0)

        h,w = image.shape
        min_h_padding = 0
        min_w_padding = 0
        if h < size:
            min_h_padding = max(0,int((size - h) / 2))
        if w < size:
            min_w_padding = max(0,int((size - w) / 2))
        image_2 = cv2.copyMakeBorder(image,min_h_padding,min_h_padding, min_w_padding, min_w_padding,cv2.BORDER_CONSTANT, value=(0,0, 0))
        cv2.imwrite(os.path.join(save_path,nameext[i]),image_2,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish {i}-{len(nameext)}")

if __name__ == '__main__':
    floder = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_denoise_yuan_equal_jin/cv0_crop_black"
    save_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_denoise_yuan_equal_jin/cv0_crop_black"
    with ThreadPoolExecutor(max_workers=16) as executor:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        executor.submit(copyblack,floder,save_path)

