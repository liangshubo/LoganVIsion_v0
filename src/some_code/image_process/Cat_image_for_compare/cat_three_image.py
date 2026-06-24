"""
Data:2024/4/18
Name:liangshubo
Object:
A Parent folder have three sub folder ["cvo","cv4","our"] this code will cat them 
and save in a folder 
"""
import cv2
import numpy as np
import os

def cat_folder(cv0,cv,our,name1,name2,name3,savepath):
    nameext = os.listdir(cv0)
    for i in range(len(nameext)):
        cv0_image = cv2.imread(os.path.join(cv0,nameext[i]),0)
        cv_image = cv2.imread(os.path.join(cv,nameext[i]),0)
        our_image = cv2.imread(os.path.join(our,nameext[i]),0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(cv0_image, name1, (10, 60), font, 0.8, (255, 255, 255), 2)
        cv2.putText(cv_image , name2, (10, 60), font, 0.8, (255, 255, 255), 2)
        cv2.putText(our_image, name3, (10, 60), font, 0.8, (255, 255, 255), 2)

        image = np.concatenate([cv0_image, cv_image,our_image], axis=1)
        cv2.imwrite(os.path.join(savepath,nameext[i]),image,[cv2.IMWRITE_PNG_COMPRESSION,0])
    return  image

def mulprocess():
    folder = r"G:\Work\Project_SRI\Evluation\Thyroid"
    save_path = "G:\Work\Project_SRI\Evluation\Cat\Thyroid"
    cv0_path = os.path.join(folder,"cv0")
    cv_path = os.path.join(folder,"cv4")
    our_path = os.path.join(folder,"our")
    name1 = "Raw Image"
    name2 = "Clearview"
    name3 = "Our"
    cat_folder(cv0_path, cv_path, our_path, name1, name2, name3, save_path)

mulprocess()
