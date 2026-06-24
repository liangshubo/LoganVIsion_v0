import  cv2

import os

from src.some_code.image_process.align_two_image_for_surprev_train.select_roi2 import nameext


def check_cropsize(path):
    image = cv2.imread(path,0)
    imag2 = image.copy()
    imag2[:,150:-150] = 255
    imag2[:,152:-152] = image[:,152:-152]
    cv2.imshow("size",imag2)
    cv2.waitKey(0)

def crop_some_size_and_padding(path,save_path):
    nameext_list = os.listdir(path)
    for i in range(len(nameext_list)):
        name_ext = nameext_list[i]
        file_path = os.path.join(path,name_ext)
        array = cv2.imread(file_path,0)
        array2 = array[:,52:-52]
        array3 = cv2.copyMakeBorder(array2,30,30,100,100,cv2.BORDER_CONSTANT, value=(0,0,0))
        cv2.imwrite(os.path.join(save_path,name_ext),array3)
        print(f"finish process {i}")

# sci = 6 的时候要两边 才需要两边裁减 152
# sci = 2 的时候要两边 才需要两边裁减 52  同时上下padding 30

path = r"/home/ubuntu4090/4T_disk/liangshubo/Sci_Plus/dataset/rawdata/samsung_sci0_2/sci2"
save_path = r"/home/ubuntu4090/4T_disk/liangshubo/Sci_Plus/dataset/rawdata/samsung_sci0_2/sci2_center_crop_and_padding"

if not os.path.exists(save_path):
    os.makedirs(save_path)

crop_some_size_and_padding(path,save_path)
