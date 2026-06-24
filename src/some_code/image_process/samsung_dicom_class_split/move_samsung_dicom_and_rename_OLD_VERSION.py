import shutil

import cv2
import os

# 前面 n个 是不一样的后面都是重复的135等级
name_list = ["01","02","03","04","05","06","07",
             "08","09","10","11","12","13","14",
             "15","16","17","18","19","20","21",
             "22","23","24","25","26","27","28"]

def rename_cv0(path,save_path, n = 19):

    real_name_list = os.listdir(path)
    real_name_list = sorted(real_name_list)
    print(real_name_list)

    for i in range(len(real_name_list)):#逐个处理
        if i+1 <= n:
            # 直接移动不改变名字
            save_path_cv0 = os.path.join(save_path,"cv0")
            os.makedirs(save_path_cv0,exist_ok=True)

            shutil.move(os.path.join(path,real_name_list[i]),os.path.join(save_path_cv0,real_name_list[i]))
def move_and_rename_cv(path,save_path):
    real_name_list_cv = os.listdir(path)
    # 采集到的实际的文件的名字
    real_name_list_cv = sorted(real_name_list_cv)
    save_path_cv1 = os.path.join(save_path, "cv1")
    os.makedirs(save_path_cv1, exist_ok=True)
    save_path_cv3 = os.path.join(save_path, "cv3")
    os.makedirs(save_path_cv3, exist_ok=True)
    save_path_cv5 = os.path.join(save_path, "cv5")
    os.makedirs(save_path_cv5, exist_ok=True)
    m = 0
    for j in range(0,len(real_name_list_cv),3):
        # 如果是采集的两个等级就是 2，不然就是3
        if real_name_list_cv[j].find("I000")<0:
            break
        cv_1 = os.path.join(path, real_name_list_cv[j])
        cv_3 = os.path.join(path, real_name_list_cv[j+1])
        cv_5 = os.path.join(path, real_name_list_cv[j+2])


        shutil.move(cv_1, os.path.join(save_path_cv1, real_name_list_cv[j][:-2]+name_list[m]))
        shutil.move(cv_3, os.path.join(save_path_cv3, real_name_list_cv[j][:-2]+name_list[m]))
        shutil.move(cv_5, os.path.join(save_path_cv5, real_name_list_cv[j][:-2]+name_list[m]))
        m += 1
        print(f"{j}/{len(real_name_list_cv)}")

path =r"F:\NEUSOFT\IMAGE\2024_05\13\__162509_samsung_fubu_cv0135"
save_path = r"E:\Project_Studysamsung\fubu\162509_samsung_fubu_cv0135\dicom"
#rename_and_copy(path,save_path)

rename_cv0(path,save_path, n = 19)
move_and_rename_cv(path,save_path)