import numpy as np

import  os
from  typing import List


def multi_class_trans_single(path:str,save_path:str):
    """
    Args:
        path: 多类别的标签 路径
        save_path: 保存路径

    Returns:

    """
    name_list = os.listdir(path)

    for i in range(len(name_list)):
        raw_label = np.load(os.path.join(path,name_list[i]))
        raw_label[raw_label!=1] = 0
        save_label_path = os.path.join(save_path,name_list[i])
        np.save(save_label_path,raw_label)
        print(f"{i} / {len(name_list)}   {raw_label.max()}")


if __name__ == '__main__':

    path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/UBPB/train/label"
    save_path=r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/UBPB/train/label_single"
    os.makedirs(save_path,exist_ok=True)
    multi_class_trans_single(path, save_path)