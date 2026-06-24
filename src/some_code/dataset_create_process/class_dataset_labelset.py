
import os
import json
import re
'''
这里用于构建图像去噪、分隔等等单输入单输出，且尺寸一致的模型的数据集的构建
project_path  是项目的路径
dataset_name  数据集的名字 
path_image  是输入的图像的路径
path_label  是标签的图像的路径 
dataset_class = "open_cv2" 是数据集类型 
dataset_dict["cv2denoisedataset_list"].append(dataset_name)  是指定数据集类下的数据集名字，将用于调用对应的数据集函数 

test_num = 0.05 * len(name_list) # 1000 测试比例 

'''

# 第一次创项目 -更改路径
project_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/"
dataset_path = project_path+"dataset"

# 创建训练以及测试文件夹 -更改名字
dataset_name  = "n20n6000_shoulder_class_all_1013"

# 输入图像路径以及 标签路径 -更改路径  要指定当前的所有图像按类别子文件夹下，且共同在一个副文件夹下
plane_class_list = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S10", "S11"]    # 训练与
label_class = {"S0":0,"S1":1,"S2":2,"S3":3,"S4":4,"S5":5,"S6":6,"S7":7,"S8":8,"S10":9,"S11":10}

# n20n6000的合并数据级别，是运行两次，一次
train_path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/train/image"
test_path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/test/image"

# train_path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/all_sam418_data_segment/png/train/image"
# test_path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/all_sam418_data_segment/png/test/image"
#


# 写入dataset_json -默认操作无需修改
dataset_class = "open_cv2_image_class"
dataset_json_path = project_path + "src/data/dataset.json"
dataset_json_read = open(dataset_json_path, "r")
dataset_dict = json.load(dataset_json_read)
dataset_json_read.close()

dataset_json_rewrite = open(dataset_json_path,"w",encoding='utf-8')
if dataset_class == "open_cv2_image_class":
    if dataset_dict.get(dataset_class) is None:  # 检查键是否存在（考虑默认值None的情况）
        dataset_dict[dataset_class] = []
    dataset_dict["cv2classdataset_list"].append(dataset_name)
    json.dump(dataset_dict, dataset_json_rewrite, ensure_ascii=False)

# 构建数据集文件夹 - -默认操作无需修改


train_dataset_path = os.path.join(dataset_path,dataset_name)
test_dataset_path = os.path.join(dataset_path,"benchmark",dataset_name)
os.makedirs(train_dataset_path,exist_ok=True)
os.makedirs(test_dataset_path,exist_ok=True)


# 创建txt 文件 --默认操作无需修改
train_file = open(os.path.join(train_dataset_path,dataset_name+".txt"),"a")
test_file = open(os.path.join(test_dataset_path,dataset_name+".txt"),"a")

# 训练测试比例 ---默认操作无需修改
test_num = 20 # 训练测试比例 1：20

### 切面层级   -- 病例层级  -- 帧层级
for s in range(len(plane_class_list)):
    plane_name = plane_class_list[s]    # "s1" 切面名称
    plane_label_idx = label_class[plane_name]    #  切面的标签索引

    # 训练的数据集一个切面内的不同文件夹文件
    train_plane_folder = os.path.join(train_path_image, plane_name)    # /image/s1
    train_plane_folder_namelist = os.listdir(train_plane_folder)    # /Pat_image1..  /Pat_image2

    # 病例层级

    for j in range(len(train_plane_folder_namelist)):  # 对于当前切面的每个文件夹
        # 获取当前切面当前文件夹内的所有文件名
        train_plane_folder_path = os.path.join(train_plane_folder,train_plane_folder_namelist[j]) #  # /image/s1/Pat_image1..
        train_plane_folder_image_namelist = os.listdir(train_plane_folder_path) # image1.png .. image2.png .. image3.png
        train_plane_folder_image_namelist = sorted(train_plane_folder_image_namelist,
                                                   key=lambda x: int(re.search(r'_(\d+)\.png$', x).group(1)))

        # 帧层级
        for i in range(len(train_plane_folder_image_namelist)):
            image_path  = os.path.join(train_plane_folder_path, train_plane_folder_image_namelist[i])  # /image/s1/Pat_image1/image1.png ..  /image/s1/Pat_image1/image2.png
            train_file.write(image_path +" " + str(plane_label_idx) + "\n")

    # ============================================= 测试的数据集 ===========================================
    test_plane_folder = os.path.join(test_path_image, plane_name)  # /image/s1
    test_plane_folder_namelist = os.listdir(test_plane_folder)  # /Pat_image1..  /Pat_image2
    # 病例层级
    for j in range(len(test_plane_folder_namelist)):  # 对于当前切面的每个文件夹
        # 获取当前切面当前文件夹内的所有文件名
        test_plane_folder_path = os.path.join(test_plane_folder, test_plane_folder_namelist[j])  # # /image/s1/Pat_image1..
        test_plane_folder_image_namelist = os.listdir(test_plane_folder_path)  # image1.png .. image2.png .. image3.png
        test_plane_folder_image_namelist = sorted(test_plane_folder_image_namelist,
                                                   key=lambda x: int(re.search(r'_(\d+)\.png$', x).group(1)))
        # 帧层级
        for i in range(len(test_plane_folder_image_namelist)):
            image_path = os.path.join(test_plane_folder_path, test_plane_folder_image_namelist[i])  # /image/s1/Pat_image1/image1.png ..  /image/s1/Pat_image1/image2.png
            test_file.write(image_path + " " + str(plane_label_idx) + "\n")

train_file.close()
test_file.close()


print(f"finish create dataset named {dataset_name}")






