"""
Data:2024/4/16
Name:liangshubo
Object:

"""
import os

import cv2

import numpy as np


def enhance_brightness(image,label):
    # label  的亮度要与image 一样 这里使用非均匀效果
    mean_image = np.mean(image)
    image = image.astype(float)
    label = label.astype(float)

    mask = image>mean_image-10
    enhance_label = label.copy()

    mask = mask.astype(float)

    mask_blur = cv2.GaussianBlur(mask,(31,31),sigmaX=5)
    mask_blur = cv2.GaussianBlur(mask_blur,(41,41),sigmaX=9)
    mask_blur = cv2.GaussianBlur(mask_blur, (7, 7), sigmaX=13)
    print(mask_blur.mean() , mask.mean())


    cv2.imshow("mask",mask_blur)
    cv2.waitKey(0)
    #mask_blur = cv2.GaussianBlur(mask_blur, (31, 31), sigmaX=5)

    enhance_label2 = enhance_label + mask_blur*(abs(image.mean()-label.mean()))

    enhance_label3 = np.clip(enhance_label2,0,255)
    return enhance_label3



def enhance_brightness_avg(image,label):
    # label  的亮度要与image 一样 这里使用均匀效果
    mean_image = np.mean(image)
    image = image.astype(float)
    label = label.astype(float)

    #mask_blur = cv2.GaussianBlur(mask_blur, (31, 31), sigmaX=5)
    if label.mean()>image.mean():

        enhance_label2 = label +abs(image.mean()-label.mean())
    else:
        enhance_label2 = label - abs(image.mean()-label.mean())
    enhance_label3 = np.clip(enhance_label2,0,255)
    return enhance_label3




def multi_process(image_path,label_path,save_path):
    nameext = os.listdir(image_path)

    for i in range(len(nameext)):
        image_ = os.path.join(image_path,nameext[i])
        label_ = os.path.join(label_path,nameext[i])
        image = cv2.imread(image_,0)
        label = cv2.imread(label_,0)

        print(f" image {image_}  label {label_}")

        if image.mean() > label.mean():
            enhance_label = enhance_brightness_avg(image,label)
        else:
            enhance_label = label
        cv2.imshow("process",enhance_label/255)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_path,nameext[i]),enhance_label,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish {i}-{len(nameext)} ,light_image {image.mean():.2f},  light_label {label.mean():.2f} , label_enhance_label {enhance_label.mean():.2f}")


if __name__ == '__main__':
    image_path = r"G:\Work\Self-develop-SRI\Dataset\New_Dataset\HighGain\HignGainImgae"
    label_path = r"G:\Work\Self-develop-SRI\Dataset\New_Dataset\HighGain\HignGainImgae_output_gain_change"

    save_path  = r"G:\Work\Self-develop-SRI\Dataset\New_Dataset\HighGain\HignGainImgae_output_gain_change_gain"
    if not  os.path.exists(save_path):
        os.makedirs(save_path)

    multi_process(image_path,label_path,save_path)