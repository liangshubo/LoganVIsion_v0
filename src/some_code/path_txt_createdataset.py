
import os
import json

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
project_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/"
dataset_path = project_path+"dataset"

# 创建训练以及测试文件夹 -更改名字
dataset_name  = "n20_style1_denoise_sam_msk1"

# 输入图像路径以及 标签路径 -更改路径
path_image = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/compose_all_cv0/remapdown13055->mask"
path_label = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/compose_all_cv0/remapdown13055->NoYuanDenoise->MapD->SAM_MSK_LV1->REMapD->mask"
name_list = os.listdir(path_image)
print(name_list)


# 写入dataset_json -默认操作无需修改
dataset_class = "open_cv2"
dataset_json_path = project_path + "src/data/dataset.json"
dataset_json_read = open(dataset_json_path, "r")
dataset_dict = json.load(dataset_json_read)
dataset_json_read.close()

dataset_json_rewrite = open(dataset_json_path,"w",encoding='utf-8')
if dataset_class == "open_cv2":
    dataset_dict["cv2denoisedataset_list"].append(dataset_name)
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
test_num = 0.05 * len(name_list) # 1000 测试
for i in range(len(name_list)):
    name_ext = name_list[i]
    image_path = os.path.join(path_image,name_ext)
    label_path = os.path.join(path_label,name_ext)
    assert os.path.isfile(image_path) and os.path.isfile(label_path)
    if not os.path.isfile(image_path) or not os.path.isfile(label_path):
        continue
    if i % test_num==0 :
        test_file.write(image_path+"\n")
        test_file.write(label_path+'\n')
    else:
        train_file.write(image_path+"\n")
        train_file.write(label_path + '\n')

train_file.close()
test_file.close()

import time
print(f"finish create dataset named {dataset_name} time {time.time()}")






