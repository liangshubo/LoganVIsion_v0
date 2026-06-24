import os

import shutil

"移动文件夹中的label文件 ，名称带mask  "

def move_label(path,save_path):
    if not  os.path.exists(save_path):
        os.makedirs(save_path)

    name_list = os.listdir(path)

    for  i in range(len(name_list)):
        if name_list[i].find("mask") >=0:
            src_path = os.path.join(path,name_list[i])
            dst_file = os.path.join(save_path,name_list[i])
            shutil.move(src_path, dst_file )
            print(f"finish move {name_list[i]}")


if __name__ == '__main__':

    path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/USNSkaggle ultrasound-nerve-segmentation/train"
    save_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/USNSkaggle ultrasound-nerve-segmentation/train_label"
    move_label(path, save_path)