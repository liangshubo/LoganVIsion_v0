"""
Data:2024/3/15
Name:liangshubo
Object:
this code is used for mask process path and save is another path
"""
import os
import cv2




def mul_mask_process(folder_path,mask_path,save_path):

    mask = cv2.imread(mask_path,0)/255

    filename_list = os.listdir(folder_path)
    for j in range(len(filename_list)):
        file_path = os.path.join(folder_path,filename_list[j])
        print(file_path)
        file_array = cv2.imread(file_path,0)
        process_array = file_array*mask
        cv2.imwrite(os.path.join(save_path,filename_list[j]),process_array,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish {j} / {len(filename_list)}")




path = r"F:\Work\Project_SRI\index0_OB_cv4_lightenhance\dataset\adjust_light_0517\cv4"

save_path = r"F:\Work\Project_SRI\index0_OB_cv4_lightenhance\dataset\adjust_light_0517\cv4_mask"
mask_path = r"F:\Work\Project_SRI\index0_OB_cv4_lightenhance\dataset\adjust_light_0517\mask.png"
if not os.path.exists(save_path):
    os.makedirs(save_path)
mul_mask_process(path,mask_path,save_path)
