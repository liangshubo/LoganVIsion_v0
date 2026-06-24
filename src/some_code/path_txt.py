
import os


file = open("/home/us-ubuntu/us/Liangshubo/Samsung/dataset/rawdata/133131_samsung_zigong_cv0135_sci/samsung_zigong_sci.txt","a")

folder_name = os.listdir(r"/home/us-ubuntu/us/Liangshubo/Samsung/dataset/rawdata/133131_samsung_zigong_cv0135_sci/png/cv0")
for i in folder_name:

    path_image = r"/home/us-ubuntu/us/Liangshubo/Samsung/dataset/rawdata/133131_samsung_zigong_cv0135_sci/png/cv0/"+i
    path_label = r"/home/us-ubuntu/us/Liangshubo/Samsung/dataset/rawdata/133131_samsung_zigong_cv0135_sci/png/cv3/"+i
    name_list = os.listdir(path_image)
    print(name_list)
    for i in range(len(name_list)):
        name_ext = name_list[i]
        image_path = os.path.join(path_image,name_ext)
        label_path = os.path.join(path_label,name_ext[:-4]+".png")
        assert os.path.isfile(image_path) and os.path.isfile(label_path)
        if not os.path.isfile(image_path) or not os.path.isfile(label_path):
            continue

        file.write(image_path+"\n")
        file.write(label_path+'\n')












