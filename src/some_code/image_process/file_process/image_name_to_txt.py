"""
Data:2024/4/24
Name:liangshubo
Object:

"""
import os
path = r"G:\Work\Self-develop-SRI\Dataset\Dataset_424_Contrast\label_padding_contrast"
save_path = r"G:\Work\Self-develop-SRI\Dataset\Dataset_424_Contrast\name change.txt"
name_ext = os.listdir(path)



with open(save_path,"a") as f:
    for i in range(len(name_ext)):
        f.write(name_ext[i]+"\n")
    f.close()

