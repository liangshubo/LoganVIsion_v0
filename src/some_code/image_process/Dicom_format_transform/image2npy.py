import os
import cv2

import numpy as np

def trans(folder,save_path):
    name_ext = os.listdir(folder)
    
    for i in range(len(name_ext)):
        image_path = os.path.join(folder,name_ext[i])
        name,ext = os.path.splitext(name_ext[i])
        image_array = cv2.imread(image_path,0)
        np.save(os.path.join(save_path,name),image_array)
        print(f"finish {i}/{len(name_ext)}")
        
        
folder = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset401/noise2"
save_path = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset403/noise2"
if not os.path.exists(save_path):
    os.makedirs(save_path)
trans(folder,save_path)
        