# -*- coding: UTF-8 -*-
"""
Create on 2024-9-20
@Author: LiangShubo
@email: liangshubo@neusoftmedical.com
@Project:MSK_PLus
@Info : 用于检查MSK各个部位的尺寸，然后裁减到一样的尺寸上
"""

import cv2
import os
import shutil

def check_size(path):
    array = cv2.imread(path,0)
    print(array.shape)

if __name__ == '__main__':
    path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3_msk/wrist/WRISTX00_17.png"

    check_size(path)

    # ankle = [700,700]
    # knee = [650,785]
    # shoulder = [700,700]
    # tknee = [650,785]
    # wrist = [650,785]

