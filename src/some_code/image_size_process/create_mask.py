import cv2

image = cv2.imread(r"/media/ubuntu/SSD_3/Liangshubo/Thyroid_plus/dataset/rawdata/n20_thyroid_v2_pretrain/cv0_dmapdown131/CAOTID0C_231.png",0)
print(image.shape)
image2 = image*0
image2[40:450,80:450] =1
cv2.imshow("s",image2*100+image)
# #
cv2.waitKey(0)
# cv2.imwrite(r"/media/ubuntu/SSD_3/Liangshubo/Thyroid_plus/dataset/rawdata/n20_thyroid_v2_pretrain/mask.png",image2,[cv2.IMWRITE_PNG_COMPRESSION,0])
# ## N20 MASK image2[50:600,100:600]
#
#

