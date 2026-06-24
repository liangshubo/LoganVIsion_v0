import os

from sympy.physics.units import percents


#
# def delet(foler,ext1):
#     name_list = os.listdir(foler)
#     for i in range(len(name_list)):
#         file_path = os.path.join(foler,name_list[i])
#         ext = file_path[-4:]
#         print(ext)
#         if ext == ".png":
#             os.remove(file_path)
#             print(f"delet{i}/{len(name_list)}")
#
# folder = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset401"
# delet(folder,"png")


def get_path_from_txt(dataset_image_pathfile):
    """
    the image and gt filepoath will read from self.dataset_image_pathfile
    and save in self.image_path_list\self.gt_path_list
    """
    image_path_list = []
    gt_path_list = []
    with open(dataset_image_pathfile, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            image_path_list.append(lines[i].rstrip().split(" ")[0])
            gt_path_list.append(lines[i].rstrip().split(" ")[1])

        for i in range(len(image_path_list)):

            print(image_path_list[i],gt_path_list[i])



path  = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/benchmark/n6000_shoulder_class/n6000_shoulder_class.txt"
get_path_from_txt(path)
