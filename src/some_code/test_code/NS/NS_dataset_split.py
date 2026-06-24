
import os
import shutil
import  random



"随即划分的训练与测试数据集 ，同时处理图像以及标签  "

def random_split(path,save_path):



    src_image = os.path.join(path,"image")
    src_label = os.path.join(path,"label")

    dst_image = os.path.join(save_path, "image")
    dst_label = os.path.join(save_path, "label")

    if not os.path.exists(dst_image):
        os.makedirs(dst_image)

    if not os.path.exists(dst_label):
        os.makedirs(dst_label)


    name_list = os.listdir(src_image)
    label_name = lambda x : x #.split(".")[0]+".npy"

    dst_name_list = os.listdir(dst_image)

    while (len(name_list) / (len(dst_name_list)+1e-5) ) >=100:
        name_list = os.listdir(src_image)
        dst_name_list = os.listdir(dst_image)
        for i in range(len(src_image)):
            if random.random() < 0.2 :
                src_image_file = os.path.join(src_image,name_list[i])
                src_label_file = os.path.join(src_label,label_name(name_list[i]))

                dst_image_file = os.path.join(dst_image,name_list[i])
                dst_label_file = os.path.join(dst_label,label_name(name_list[i]))

                shutil.move(src_image_file,dst_image_file)
                shutil.move(src_label_file,dst_label_file)

                print(f"finish split")


if __name__ == '__main__':
    path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/US43d/train"
    save_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/US43d/test"
    random_split(path, save_path)



