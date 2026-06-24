import cv2
import os
import numpy as np

def cat_image(path,save_path,name=None):
    class_name = ["cv1","cv2","cv3","cv4","cv5"]
    for i in range(len(class_name)):
        dengname = class_name[i]
        clearview_path = os.path.join(path,name+"_"+dengname+".BMP")
        sri_path = os.path.join(path, name + "_" + dengname + "_sri21.BMP")
        clearview_array = cv2.imread(clearview_path)
        sri_array = cv2.imread(sri_path)
        cat_image = np.concatenate([clearview_array,sri_array],axis=1)
        save_array_path = os.path.join(save_path,name+"_"+dengname+".png")
        cv2.imwrite(save_array_path,cat_image)
        print("finish cat ",name," - ",dengname)


if __name__ == '__main__':
    path = r"G:\Work\project_sri_21_evluation\evpevis_13"
    save_path = r"G:\Work\project_sri_21_evluation\cat\evpevis_13"
    name = r"image_evpelvis"

    os.makedirs(save_path,exist_ok=True)
    cat_image(path,save_path,name)