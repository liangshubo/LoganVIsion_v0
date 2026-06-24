
import os
import json
import re

'''
这里用于图像分割的数据集处理与代码,不同的是将会引入改切面的类别信息 ，其文件夹结构为 

--|train|image |S1 |Folder1|IMAGE1.png
        |      |            |Image2.png
        |      |            |Image3.png
        |      |   ....
         |         |Folder2 .....   
        |      |S2|....
        |      ...
        |      |S10
        |label |S1 |Image1.npy
                   |Image2.npy
                    ....
               |S2 |...
               ...
               |S10     
--|test|同上结构               

project_path  是项目的路径  一般来说第一次更改后续不需要更改 
dataset_name  数据集的名字 
train_path_image 是输入的图像的路径

path_label  是标签的图像的路径 
dataset_class = "open_cv2" 是数据集类型 
dataset_dict["cv2denoisedataset_list"].append(dataset_name)  是指定数据集类下的数据集名字，将用于调用对应的数据集函数 

test_num = 0.05 * len(name_list) # 1000 测试比例 

'''

# 第一次创项目 -更改路径
project_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/"
dataset_path = project_path+"dataset"

# 创建训练以及测试文件夹 -更改名字
dataset_name  = "n20n6000_shoulder_segment_cop_class_all_1013"

# 输入图像路径以及 标签路径 -更改路径  要指定当前的所有图像按类别子文件夹下，且共同在一个副文件夹下
#label_class_list = ["S0","S1","S2","S3","S4","S6","S7","S8","S10","S11"]    # 训练与
#label_class = {"S0":0,"S1":1,"S2":2,"S3":3,"S4":4,"S6":5,"S7":6,"S8":7,"S10":8,"S11":9}

plane_class_list = ["S0","S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S10", "S11"]    # 训练切面
label_class = {"S0":0,"S1":1,"S2":2,"S3":3,"S4":4,"S5":5,"S6":6,"S7":7,"S8":8,"S10":9,"S11":10}

# n20n6000的合并数据级别，是运行两次，一次
# train_path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/train/image"
# train_path_label = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/train/label"
#
#
# test_path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/test/image"
# test_path_label = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/test/label"

train_path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/train/image"
train_path_label = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/train/label"


test_path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/test/image"
test_path_label = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/test/label"




# 写入dataset_json -默认操作无需修改
dataset_class = "cv2segment_cop_class_dataset_list"
dataset_json_path = project_path + "src/data/dataset.json"
dataset_json_read = open(dataset_json_path, "r")
dataset_dict = json.load(dataset_json_read)
dataset_json_read.close()

dataset_json_rewrite = open(dataset_json_path,"w",encoding='utf-8')
if dataset_class == "cv2segment_cop_class_dataset_list":
    if dataset_dict.get(dataset_class) is None:  # 检查键是否存在（考虑默认值None的情况）
        dataset_dict[dataset_class] = []
    if dataset_name not  in dataset_dict[dataset_class]:
        dataset_dict["cv2segment_cop_class_dataset_list"].append(dataset_name)
    json.dump(dataset_dict, dataset_json_rewrite, ensure_ascii=False)

# 构建数据集文件夹 - -默认操作无需修改

train_dataset_path = os.path.join(dataset_path,dataset_name)
test_dataset_path = os.path.join(dataset_path,"benchmark",dataset_name)
os.makedirs(train_dataset_path,exist_ok=True)
os.makedirs(test_dataset_path,exist_ok=True)

# 创建txt 文件 --默认操作无需修改
train_file = open(os.path.join(train_dataset_path,dataset_name+".txt"),"w")
test_file = open(os.path.join(test_dataset_path,dataset_name+".txt"),"w")

# 训练测试比例 ---默认操作无需修改
test_num = 20 # 训练测试比例 1：20
# 切面层级
for s in range(len(plane_class_list)):
    plane_name = plane_class_list[s]       # "s1" 类别名称
    plane_label_idx = label_class[plane_name]  # 切面的标签索引


    # 训练的数据集 文件
    train_folder_image_folder = os.path.join(train_path_image, plane_name)    # /train/image/s1
    train_folder_label_folder = os.path.join(train_path_label, plane_name)  # /train/label/s1
    train_plane_folder_namelist = os.listdir(train_folder_image_folder)    # /Pat_image1..  /Pat_image2

    # 病例层级
    for j in range(len(train_plane_folder_namelist)):  # 对于每个切面的当前文件夹下
        train_plane_image_folder_path = os.path.join(train_folder_image_folder,
                                                     train_plane_folder_namelist[j])  # /image/s1/Pat_image1..
        train_plane_label_folder_path = os.path.join(train_folder_label_folder,
                                                     train_plane_folder_namelist[j])  # /label/s1/Pat_image1..
        train_plane_folder_image_namelist = os.listdir(train_plane_image_folder_path)
        # 按照顺序排序
        train_plane_folder_image_namelist = sorted(train_plane_folder_image_namelist,
                                                   key=lambda x: int(re.search(r'_(\d+)\.png$', x).group(1)))

        # 帧层级
        for i in range(len(train_plane_folder_image_namelist)):
            image_path  = os.path.join(train_plane_image_folder_path,
                                       train_plane_folder_image_namelist[i])  # /train/image/s1/Pat_image1/image1.png ..  /train/image/s1/Pat_image1/image2.png

            if plane_label_idx == 0 :
                label_path = "Non-Stard-Plane-Label-Clear-Mask"
            else:
                label_path = os.path.join(train_plane_label_folder_path,
                                      train_plane_folder_image_namelist[i].split(".")[0] + ".npy")  # /test/label/s1/image1.npy ..  /test/label/s1/image2.npy
            # 引入切面的类别信息    #为了兼容

            if os.path.isfile(label_path) or label_path == "Non-Stard-Plane-Label-Clear-Mask":
                train_file.write(image_path +" " + str(plane_label_idx) + "\n")
                train_file.write(label_path + "\n")

    # ============================================= 测试的数据集 ===========================================
    test_folder_image_folder = os.path.join(test_path_image, plane_name)  # /train/image/s1
    test_folder_label_folder = os.path.join(test_path_label, plane_name)  # /train/label/s1
    test_plane_folder_namelist = os.listdir(test_folder_image_folder)  # /Pat_image1..  /Pat_image2

    # 病例层级
    for j in range(len(test_plane_folder_namelist)):  # 对于每个切面的当前文件夹下
        test_plane_image_folder_path = os.path.join(test_folder_image_folder,
                                                     test_plane_folder_namelist[j])  # /image/s1/Pat_image1..
        test_plane_label_folder_path = os.path.join(test_folder_label_folder,
                                                     test_plane_folder_namelist[j])  # /label/s1/Pat_image1..
        test_plane_folder_image_namelist = os.listdir(test_plane_image_folder_path)

        test_plane_folder_image_namelist = sorted(test_plane_folder_image_namelist,
                                                   key=lambda x: int(re.search(r'_(\d+)\.png$', x).group(1)))
        # 帧层级
        for i in range(len(test_plane_folder_image_namelist)):
            image_path = os.path.join(test_plane_image_folder_path, test_plane_folder_image_namelist[i])  # /train/image/s1/Pat_image1/image1.png ..  /train/image/s1/Pat_image1/image2.png
            if plane_label_idx == 0:
                label_path = "Non-Stard-Plane-Label-Clear-Mask"
            else:
                label_path = os.path.join(test_plane_label_folder_path, test_plane_folder_image_namelist[i].split(".")[
                0] + ".npy")  # /test/label/s1/image1.npy ..  /test/label/s1/image2.npy
            if os.path.isfile(label_path):
                test_file.write(image_path + " " + str(plane_label_idx) + "\n")
                test_file.write(label_path + "\n")

train_file.close()
test_file.close()


print(f"finish create dataset named {dataset_name}")






