import cv2


import os
"""
image crop h

"""


def crop_size(image,save_path):
    _,name_ext= os.path.split(image)
    print(name_ext)
    image_array = cv2.imread(image,0)

    h,w = image_array.shape

    if h == 700:
        image_array = image_array[:650,:]
        print(f"h =700 size={image_array.shape}")
        cv2.imwrite(os.path.join(save_path,name_ext),image_array,[cv2.IMWRITE_PNG_COMPRESSION,0])

    elif h==650 :
        image_array = image_array[:, 40:740]
        print(f"h =650 size={image_array.shape}")
        cv2.imwrite(os.path.join(save_path, name_ext), image_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def multi_process(folder,savepath):
    name_list = os.listdir(folder)
    for i in range(len(name_list)):
        image_name_ext = name_list[i]
        image = os.path.join(folder,image_name_ext)
        crop_size(image,savepath)
        print(f"finish {i}/{len(name_list)}")


if __name__ == '__main__':
    folder = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3_msk/ankle"

    save_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/raw_process/ankle"
    if not os.path.exists(save_path):     
        os.makedirs(save_path)
    multi_process(folder,save_path)
    # ankle = [700,700]
    # knee = [650,785]
    # shoulder = [700,700]
    # tknee = [650,785]
    # wrist = [650,785]
