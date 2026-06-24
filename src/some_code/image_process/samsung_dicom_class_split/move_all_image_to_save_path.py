import cv2
import os

import shutil

def move_all_image(path,save_path):
    name_ext  = os.listdir(path)
    #print(name_ext)

    for i in range(len(name_ext)):
        image_path = os.path.join(path,name_ext[i])

        save_image_path = os.path.join(save_path,name_ext[i])
        shutil.move(image_path,save_image_path)
        print(f" finish move {i}/{len(name_ext)}")




if __name__ == '__main__':
    path = r"/media/ubuntu/SSD_3/Liangshubo/MSK_Plus/dataset/rawdata/samsung_la22_msk_png/cv5"
    FOLDER_NAME = os.listdir(path)



    for i in range(len(FOLDER_NAME)):
        folder_path = os.path.join(path,FOLDER_NAME[i])
        move_all_image( folder_path,path)





