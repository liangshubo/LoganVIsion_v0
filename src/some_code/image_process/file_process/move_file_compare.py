import cv2
import os
import  shutil

folder = r"F:\Work\EVPevisPlus\Dataset\n20\png\human\class_crop\preset"
save_path = r"F:\Work\EVPevisPlus\Dataset\n20\png\human\class_crop\compare"

nameext = os.listdir(folder)

for i in range(len(nameext)):
    subclass = nameext[i]
    sub_folder = os.path.join(folder,subclass)
    for cv_index  in range(1,6):
        sub_folder_index = sub_folder+"\cv"+str(cv_index)
        image_name_ext = os.listdir(sub_folder_index)
        extra_name = subclass[8:]
        os.makedirs(os.path.join(save_path,"cv"+str(cv_index)),exist_ok=True)
        for a in image_name_ext:

            shutil.copy(os.path.join(sub_folder_index,a),os.path.join(save_path,"cv"+str(cv_index),a[:-4]+extra_name+".png"))
    print(f" {subclass} - {str(cv_index)}")