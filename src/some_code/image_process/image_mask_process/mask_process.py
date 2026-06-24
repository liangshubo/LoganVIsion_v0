import cv2

import os

"""
this code is used for the mask process and save in the same folder
"""
mask = cv2.imread(r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/mask/mask_down.png",0)

def mask_process(image_path,save_path):
    image = cv2.imread(image_path,0)
    _,nameext = os.path.split(image_path)
    image2 = image*mask


    cv2.imwrite(os.path.join(save_path,nameext),image2)


def multi_process(folder,savepath):
    name_list = os.listdir(folder)
    for i in range(len(name_list)):
        print(f"finish {i}/{len(name_list)}")
        image_name_ext = name_list[i]
        image = os.path.join(folder,image_name_ext)
        mask_process(image,savepath)



if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor

    folder = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/compose_all_cv0/remapdown13055->NoYuanDenoise->MapD->SAM_MSK_LV1->REMapD"

    save_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/compose_all_cv0/remapdown13055->NoYuanDenoise->MapD->SAM_MSK_LV1->REMapD->mask"
    os.makedirs(save_path,exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.submit(multi_process,folder, save_path )
