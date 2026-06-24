"""
Data:2024/4/16
Name:liangshubo
Object:
single image align and save one image
"""
import cv2
import numpy as np

# 读取两幅图像
image1 = cv2.imread(r"G:\Work\line_bw_change\Code\416\abd\24_04_00_0.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r"G:\Work\line_bw_change\Code\416\abd\24_04_02_0.png", cv2.IMREAD_GRAYSCALE)
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


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

# Find homography
H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# Warp image 1 to align with image 2
img1Reg = cv2.warpPerspective(image1 , H, (image2.shape[1], image2.shape[0]))
cv.imwrite(r'G:\Work\line_bw_change\Code\416\abd\aligned_img1.jpg', img1Reg)
