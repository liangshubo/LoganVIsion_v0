import os

import  cv2
import  numpy as np
import random


def sr_dataset_1(image):
    scale = random.choice([1.2,1.25,1.3])
    inter_mode1 = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    inter_mode2 = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    h,w = image.shape
    image_lr = cv2.resize(image,(int(w//scale),int(h//scale)),interpolation=inter_mode1)
    image_lr_re = cv2.resize(image_lr,(w,h),interpolation=inter_mode2)
    return image_lr_re



def blur_dataset(image):
    kersize = random.choice([3,5,7])
    sigma = random.choice([0.5,0.7,0.9,1.1])

    blur_image = cv2.GaussianBlur(image,ksize=(kersize,kersize),sigmaX=sigma)
    return  blur_image


def sr_and_blur_x2(image):
    image_lr_re = sr_dataset_1(image)
    image_lr_re_blur = blur_dataset(image_lr_re)
    if random.random() < 0.6:
        image_lr_re_blur = sr_dataset_1(image_lr_re_blur)
        image_lr_re_blur = blur_dataset(image_lr_re_blur)
    return image_lr_re_blur

def sr_and_blur(image):
    image_lr_re = sr_dataset_1(image)
    image_lr_re_blur = blur_dataset(image_lr_re)

    return image_lr_re_blur




def mutil_process(folder,save_path):
    name_ext =  os.listdir(folder)
    for i in range(len(name_ext)):
        image_file = os.path.join(folder,name_ext[i])
        image = cv2.imread(image_file,0)
        image_lr_re = sr_and_blur_x2(image)

        cv2.imwrite(os.path.join(save_path,name_ext[i]),image_lr_re,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"{i}/{len(name_ext)}")


for i in range(1,6):
    folder = r"F:\Work\EVPevisPlus\Dataset\sr\label_process\folder"+str(i)
    save_path = r"F:\Work\EVPevisPlus\Dataset\sr\label_process_lr_blur_x2\folder"+str(i)
    os.makedirs(save_path,exist_ok=True)
    mutil_process(folder,save_path)


"""
# ------------ sr dataset ----------
def sr_dataset_1(image):
    scale = random.choice([1.2,1.25,1.3])
    inter_mode1 = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    inter_mode2 = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    h,w = image.shape
    image_lr = cv2.resize(image,(int(w//scale),int(h//scale)),interpolation=inter_mode1)
    image_lr_re = cv2.resize(image_lr,(w,h),interpolation=inter_mode2)
    return image_lr_re
"""