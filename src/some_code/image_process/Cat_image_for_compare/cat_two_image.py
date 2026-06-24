"""
Data:2024/4/15
Name:liangshubo
Object:cat two floder image that file name is same

"""
import cv2
import os
import numpy as np

def cat_image(floder1,floder2,save_path):
    name_ext = os.listdir(floder1)

    for i in range(len(name_ext)):
        image1 = os.path.join(floder1,name_ext[i])
        image2 = os.path.join(floder2,name_ext[i])
        image1_array= cv2.imread(image1,0)
        image2_array = cv2.imread(image2,0)

        light1 = image1_array.mean()
        light2 = image2_array.mean()
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(image1_array,f"light1:{light1:.2f}",(10,60),font,1,(255,255,255),2)
        cv2.putText(image2_array, f"light2:{light2:.2f}", (10, 60), font, 1, (255, 255, 255), 2)
        image = np.concatenate([image1_array,image2_array],axis=1)

        cv2.imwrite(os.path.join(save_path,name_ext[i]),image,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish {i}-{len(name_ext)}")



if __name__ == '__main__':
    path1= r"G:\Work\line_bw_change\Evluation\Line_Dense_Change\1219_CV=0"
    path2 = r"G:\Work\line_bw_change\Evluation\Line_Dense_Change\1219_CV=0-RCAN2"
    save_path = r"G:\Work\line_bw_change\Evluation\Line_Dense_Change\1219_CV=0_CAT"
    #name = ["Carotid_Heng_process","Carotid_Zong_process","Thyroid_process"]




    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cat_image(path1,path2,save_path)
