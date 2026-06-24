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


def copyblack(floder,save_path):
    nameext = os.listdir(floder)
    for i in range(len(nameext)):
        image_path = os.path.join(floder,nameext[i])
        image = cv2.imread(image_path,0)
        image_2 = cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_CONSTANT, value=(0,0, 0))
        cv2.imwrite(os.path.join(save_path,nameext[i]),image_2,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish {i}-{512}")

if __name__ == '__main__':
    floder = r"G:\Work\Self-develop-SRI\Dataset\Dataset424\roi_x2\label_remap"
    save_path = r"G:\Work\Self-develop-SRI\Dataset\Dataset424\roi_x2\label_remap_padding"
    with ThreadPoolExecutor(max_workers=16) as executor:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        executor.submit(copyblack,floder,save_path)

