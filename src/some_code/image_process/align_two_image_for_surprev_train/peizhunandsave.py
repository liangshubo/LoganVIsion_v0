"""
Data:2024/4/16
Name:liangshubo
Object:
USED the SCI or SHARM align the image and label ,and save in a folder 
"""
import os

import cv2
import numpy as np

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def peizhun_and_save(folder_low,folder_high,save_path):

    name_ext = os.listdir(folder_low)
    for j in range(len(name_ext)):
        image1 = cv2.imread(os.path.join(folder_low,name_ext[j]),0)
        image2 = cv2.imread(os.path.join(folder_high, name_ext[j]), 0)


        save_list = os.listdir(save_path)
        if name_ext[j] in save_list:
            print("had aligned and save ",name_ext[j])
            continue

        # 读取两幅图像
        # Initiate SIFT detector
        sift_detector = cv.SIFT_create()
        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift_detector.detectAndCompute(image1, None)
        kp2, des2 = sift_detector.detectAndCompute(image2, None)

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Filter out poor matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        matches = good_matches

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt
        if points1.shape[0]<4:
            continue
        # Find homography
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Warp image 1 to align with image 2
        img1Reg = cv2.warpPerspective(image1 , H, (image2.shape[1], image2.shape[0]))
        cv.imwrite(os.path.join(save_path,name_ext[j]), img1Reg,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish {j}/{len(name_ext)}")

if __name__ == '__main__':

    #save_path1 = r"G:\Work\line_bw_change\Dataset\Dataset_424\png\roi\lowline"
    #save_path2 = r"G:\Work\line_bw_change\Dataset\Dataset_424\png\roi\highline"
    folder_sci1 = r"F:\NEUSOFT\IMAGE\2024_08\13\__092315_sci2_png\peizhun\sci0"
    folder_sci0 = r"F:\NEUSOFT\IMAGE\2024_08\13\__092315_sci2_png\peizhun\sci1"
    save_path  = r"F:\NEUSOFT\IMAGE\2024_08\13\__092315_sci2_png\peizhun\scialign"
    if not  os.path.exists(save_path):
        os.makedirs(save_path)

    peizhun_and_save(folder_low=folder_sci1,folder_high=folder_sci0,save_path=save_path)
