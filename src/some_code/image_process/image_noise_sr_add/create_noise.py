import random

import cv2

import os

import numpy as np


#
# noise = r'/media/liangshubo/Work/Work/AutoDn/self-develop/self-del/nopise/rawnoise/noise5.png'
# image  =  r'/media/liangshubo/Work/Work/AutoDn/self-develop/self-del/label/rawimage/image4crop.png'
#
#
# noise_array = cv2.imread(noise,0)
# image_array = cv2.imread(image,0)
#
# noise_mask_condition = (noise_array<2)
# noise_image_add = noise_array + image_array
#
# cv2.imshow('noise_add_image',noise_image_add)
# cv2.waitKey()


# one noise will add five
def hechengimage(noise_path,image_path,save_path1,save_path2):

    _,noiseext = os.path.split(noise_path)
    Q,labelext = os.path.split(image_path)



    noise_array = cv2.imread(noise_path,0)
    image_array = cv2.imread(image_path,0)
    noise_array = noise_array.astype(np.float)
    image_array = image_array.astype(np.float)

    nh,nw = noise_array.shape
    ih,iw = image_array.shape
    if ih == nh or iw ==nw:
        patch = image_array

        image_array_noise = noise_array + patch
        image_array_noise = np.clip(image_array_noise, 0, 255)
        # print(image_array_noise)

        cv2.imwrite(os.path.join(save_path1, noiseext[:-4] + "_" + labelext[:-4] + str(i) + ".png"), image_array_noise)
        cv2.imwrite(os.path.join(save_path2, noiseext[:-4] + "_" + labelext[:-4] + str(i) + ".png"), patch)
    else:
        for i in range(3):
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5
            if hflip: noise_array = noise_array[:,::-1]
            if vflip: noise_array = noise_array[::-1,:]
            if rot90: noise_array = noise_array.transpose(1,0)

            nh, nw = noise_array.shape


            py,px = random.randint(0,ih-nh-1),random.randint(0,iw-nw-1)
            patch = image_array[py:py+nh,px:px+nw]

            # mean = patch.mean()
            # neg_pos_mask = patch-mean>0
            # print(neg_pos_mask)

            # mask = np.zeros([nh,nw],dtype=float)+1-2*neg_pos_mask
            # print(mask)
            # noise_array_nepo = mask*noise_array*gama
            # print(noise_array_nepo)
            # print(noise_array)


            image_array_noise = noise_array+patch
            image_array_noise = np.clip(image_array_noise,0,255)
            #print(image_array_noise)


            cv2.imwrite(os.path.join(save_path1,noiseext[:-4]+"_"+labelext[:-4]+str(i)+".png"), image_array_noise)
            cv2.imwrite(os.path.join(save_path2,noiseext[:-4]+"_"+labelext[:-4]+str(i)+".png"),patch)
        # cv2.imshow("sss",image_array_noise)
        # cv2.imshow("label",patch)
        # cv2.waitKey()

if __name__ == '__main__':
    noise_folder = r"G:\Work\Self-develop-SRI\Dataset329\raw_noise"
    image_folder = r"G:\Work\Self-develop-SRI\Dataset329\raw_label\N6000Png"

    noise_list = os.listdir(noise_folder)
    image_list = os.listdir(image_folder)

    save_path1 = r"G:\Work\Self-develop-SRI\Dataset329\hechengdata\noise"
    save_path2 = r"G:\Work\Self-develop-SRI\Dataset329\hechengdata\label"
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)

    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
    for i in range(len(noise_list)):
        noise_path = os.path.join(noise_folder,noise_list[i])
        for j in range(len(image_list)):
            image_path = os.path.join(image_folder,image_list[j])
            hechengimage(noise_path,image_path,save_path1,save_path2)
            print(f"finish {i}-{j}/{len(noise_list)}-{len(image_list)}")



    #
    # image =r"/media/liangshubo/Work/Work/AutoDn/self-develop/self-del/label/rawimage/image1.png"
    #
    # noise = r"/media/liangshubo/Work/Work/AutoDn/self-develop/self-del/nopise/rawnoise/noise5.png"
    # #hechengimage(noise,image,gama=0.5)
    #
    #
    # hechengimage(noise,image,save_path1,save_path2)
    #
    # #


