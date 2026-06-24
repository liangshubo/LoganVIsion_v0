import os
import json
import re

'''
这里用于图像分割的数据集处理与代码，其文件夹结构为 
--|train|image |IMAGE1.png
        |      |Image2.png
        |      |Image3.png
        |      |   ....
        |         
        |     
        |     
        |      
        |label |Image1.npy
               |Image2.npy
               ....
               |...
               ...

--|test|同上结构               

project_path  是项目的路径  一般来说第一次更改后续不需要更改 
dataset_name  数据集的名字 
train_path_image 是输入的图像的路径

path_label  是标签的图像的路径 
dataset_class = "open_cv2" 是数据集类型 
dataset_dict["cv2denoisedataset_list"].append(dataset_name)  是指定数据集类下的数据集名字，将用于调用对应的数据集函数 



'''

# 第一次创项目 -更改路径
project_path = r"D:\project\CTSR"
dataset_path = project_path + "\dataset"

# 创建训练以及测试文件夹 -更改名字
dataset_name = "USNS"

train_path_image = r"D:\project\CTSR\dataset\rawdata\USNSkaggle ultrasound-nerve-segmentation\train\image"
train_path_label = r"D:\project\CTSR\dataset\rawdata\USNSkaggle ultrasound-nerve-segmentation\train\label"

test_path_image = r"D:\project\CTSR\dataset\rawdata\USNSkaggle ultrasound-nerve-segmentation\test\image"
test_path_label = r"D:\project\CTSR\dataset\rawdata\USNSkaggle ultrasound-nerve-segmentation\test\label"

# 写入dataset_json -默认操作无需修改
dataset_class = "cv2segmentdataset_list"
dataset_json_path = project_path + "\\src\data\dataset.json"
dataset_json_read = open(dataset_json_path, "r")
dataset_dict = json.load(dataset_json_read)
dataset_json_read.close()

dataset_json_rewrite = open(dataset_json_path, "w", encoding='utf-8')

if dataset_dict.get(dataset_class) is None:  # 检查键是否存在（考虑默认值None的情况）
    dataset_dict[dataset_class] = []
if dataset_name not in dataset_dict[dataset_class]:
    dataset_dict[dataset_class].append(dataset_name)
json.dump(dataset_dict, dataset_json_rewrite, ensure_ascii=False)
dataset_json_rewrite.close()
# 构建数据集文件夹 - -默认操作无需修改

train_dataset_path = os.path.join(dataset_path, dataset_name)
test_dataset_path = os.path.join(dataset_path, "benchmark", dataset_name)
os.makedirs(train_dataset_path, exist_ok=True)
os.makedirs(test_dataset_path, exist_ok=True)

# 创建txt 文件 --默认操作无需修改
train_file = open(os.path.join(train_dataset_path, dataset_name + ".txt"), "w")
test_file = open(os.path.join(test_dataset_path, dataset_name + ".txt"), "w")

# -======================================进行构建一个数据集文件 ====================

train_folder_image_folder = train_path_image
train_folder_label_folder = train_path_label

train_plane_folder_namelist = os.listdir(train_folder_image_folder)  # /Pat_image1..  /Pat_image2

for i in range(len(train_plane_folder_namelist)):
    image_path = os.path.join(train_folder_image_folder,
                              train_plane_folder_namelist[i])
    label_path = os.path.join(train_folder_label_folder,
                              train_plane_folder_namelist[i])  # /test/label/s1/image1.npy ..  /test/label/s1/image2.npy
    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print("No Exist Train  File", train_plane_folder_namelist[i])

    else:

        train_file.write(image_path + "\n")
        train_file.write(label_path + "\n")

train_file.close()
# ----------------------------------------测试的数据集 文件-------------------------------------------

test_folder_image_folder = test_path_image
test_folder_label_folder = test_path_label
test_plane_folder_namelist = os.listdir(test_folder_image_folder)  # /Pat_image1..  /Pat_image2

# 这是适用于三星的

# 帧层级
for i in range(len(test_plane_folder_namelist)):
    image_path = os.path.join(test_folder_image_folder, test_plane_folder_namelist[
        i])  # /train/image/s1/Pat_image1/image1.png ..  /train/image/s1/Pat_image1/image2.png
    label_path = os.path.join(test_folder_label_folder,
                              test_plane_folder_namelist[i])  # /test/label/s1/image1.npy ..  /test/label/s1/image2.npy

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print("No Exist test  Image File", test_plane_folder_namelist[i])

    else:

        test_file.write(image_path + "\n")
        test_file.write(label_path + "\n")

train_file.close()
test_file.close()

print(f"finish create dataset named {dataset_name}")






