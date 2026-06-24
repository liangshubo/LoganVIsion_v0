import cv2


import os
"""
image crop h

"""


def crop_height(image,save_path):
    _,name_ext= os.path.split(image)
    print(name_ext)
    image_array = cv2.imread(image,0)

    h,w = image_array.shape


    if h == 800:

        image_array = image_array[50:750,:]
        cv2.imwrite(os.path.join(save_path,name_ext),image_array,[cv2.IMWRITE_PNG_COMPRESSION,0])


def multi_process(folder,savepath):
    name_list = os.listdir(folder)
    for i in range(len(name_list)):
        image_name_ext = name_list[i]
        image = os.path.join(folder,image_name_ext)
        crop_height(image,savepath)
        print(f"finish {i}/{len(name_list)}")


if __name__ == '__main__':
    folder = r"E:\Project_Carotid\dataset\denoise\carotid_plus_denoise_dicom\label"
    multi_process(folder,folder)
