import shutil

import cv2
import os

# 前面 n个 是不一样的后面都是重复的135等级
name_list = ["001","002","003","004","005","006","007",
             "008","009","010","011","012","013","014",
             "015","016","017","018","019","020","021",
             "022","023","024","025","026","027","028","029",
             "030","031","032","033","034","035","036",
             "037","038","039","040","041","042","043",
             "044","045","046","047","048","049","050"]

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
def move_and_rename_cv_twoclass(path,save_path):
    real_name_list_cv = os.listdir(path)
    # 采集到的实际的文件的名字
    real_name_list_cv = sorted(real_name_list_cv)


    save_path_cv1 = os.path.join(save_path, "cv1")
    os.makedirs(save_path_cv1, exist_ok=True)

    save_path_cv2 = os.path.join(save_path, "cv2")
    os.makedirs(save_path_cv2, exist_ok=True)

    save_path_cv4 = os.path.join(save_path, "cv4")
    os.makedirs(save_path_cv4, exist_ok=True)

    save_path_cv3 = os.path.join(save_path, "cv3")
    os.makedirs(save_path_cv3, exist_ok=True)

    save_path_cv5 = os.path.join(save_path, "cv5")
    os.makedirs(save_path_cv5, exist_ok=True)

    m = 0
    for j in range(0,len(real_name_list_cv),5):
        # 如果是采集的两个等级就是 2，不然就是3
        if real_name_list_cv[j].find("I000")<0:
            break

        cv_1 = os.path.join(path, real_name_list_cv[j])
        cv_2 = os.path.join(path, real_name_list_cv[j + 1])
        cv_3 = os.path.join(path, real_name_list_cv[j + 2])
        cv_4 = os.path.join(path, real_name_list_cv[j + 3])
        cv_5 = os.path.join(path, real_name_list_cv[j + 4])

        shutil.move(cv_1, os.path.join(save_path_cv1, real_name_list_cv[j][:-3] + name_list[m]))
        shutil.move(cv_2, os.path.join(save_path_cv2, real_name_list_cv[j][:-3] + name_list[m]))
        shutil.move(cv_3, os.path.join(save_path_cv3, real_name_list_cv[j][:-3] + name_list[m]))
        shutil.move(cv_4, os.path.join(save_path_cv4, real_name_list_cv[j][:-3] + name_list[m]))
        shutil.move(cv_5, os.path.join(save_path_cv5, real_name_list_cv[j][:-3] + name_list[m]))

        m += 1
        print(f"{j}/{len(real_name_list_cv)}")

path =r"/media/ubuntu/LIANGDISK/NEUSOFT/IMAGE/2024_09/02/__151023_samsung_LA22M_msk_cv12345_num33"
save_path = r"/media/ubuntu/SSD_3/Liangshubo/MSK_Plus/dataset/rawdata/samsung_la22_msk"
#rename_and_copy(path,save_path)

rename_cv0(path,save_path, n =33)
move_and_rename_cv_twoclass(path,save_path)