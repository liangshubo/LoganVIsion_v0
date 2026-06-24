import os

import cv2

import shutil

def move_some_file(path,save_path):
    name_ext = os.listdir(path)

    for i in range(len(name_ext)):
            file_path = os.path.join(path,name_ext[i])
            savefile_path = os.path.join(save_path,name_ext[i])
            shutil.move(file_path,savefile_path)
            print(f"finish copy file {name_ext[i]}")


if __name__ =="__main__":
    # 将文件夹内的所有的图像都转移到外层
    # ---/train/S0-S1-S2-S3/Patient1-
    path  = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/segment/png/test/image"

    # 输入的是S切面文件外的路径 ，对于分隔来说，可以输入的是train/image 和 train/label 文件夹 以及测试的test/image 和 test/label 文件夹


    folder_list = os.listdir(path)

    for i in range(len(folder_list)):
        foldername = folder_list[i]
        Sfolder_path = os.path.join(path,foldername)
        namefolder = os.listdir( Sfolder_path )

        for j in namefolder:
            raw_path = os.path.join(Sfolder_path,j)
            move_some_file(raw_path,Sfolder_path)

       # move_some_file(path,save_path)