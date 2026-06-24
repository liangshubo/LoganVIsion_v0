import cv2


image = cv2.imread(r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/raw_process/wrist/WRISTX0A_0.png",0)

mask = image*0
mask[40:600,60:610] = 1

image2 = image*mask

cv2.imwrite(r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/mask.png",mask,[cv2.IMWRITE_PNG_COMPRESSION,0])

cv2.imshow("mask",mask)
cv2.waitKey(0)