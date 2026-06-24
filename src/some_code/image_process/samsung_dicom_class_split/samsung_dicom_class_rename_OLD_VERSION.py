import cv2
import glob
import os
import shutil

name_list = ["01","02","03","04","05","06","07",
             "08","09","10","11","12","13","14",
             "15","16","17","18","19","20","21",
             "22","23","24","25","26","27","28"]

def rename_and_copy(path,save_path):
    real_name_list = os.listdir(path)
    real_name_list = sorted(real_name_list)
    print(real_name_list)

    for i in range(len(real_name_list)):
        real_pathnameext = os.path.join(path,real_name_list[i])
        save_pathnameext = os.path.join(save_path,real_name_list[i][:-2]+name_list[i])
        shutil.copy(real_pathnameext,save_pathnameext)
        print(f"{i}/{len(real_name_list)}")


path=r"E:\Project_Studysamsung\zigong\__101351_samsung_zigong_cv0135\cv5"

save_path = r"E:\Project_Studysamsung\zigong\__101351_samsung_zigong_cv0135\cv5_rename"
os.makedirs(save_path,exist_ok=True)
rename_and_copy(path,save_path)