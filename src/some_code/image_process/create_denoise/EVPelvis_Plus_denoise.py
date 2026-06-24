import os

import  cv2
import  numpy as np
import random

def noise_add_oneadd(image):
    h, w = image.shape
    noise_gama = random.choice([1.5,2])
    rayleigh_sacle = random.choice([10,11,13,15,17])

    noise_ray = np.random.rayleigh(rayleigh_sacle, (int(h // 2), int(w // 2)))

    noise_ray_re = cv2.resize(noise_ray, (w, h))
    # print(noise_ray_re)

    noise_add_image = image + noise_ray_re  # 加上锐利噪声
    return noise_add_image


def noise_add_twodiff(image):

    h,w = image.shape
    noise_gama = random.choice([2.5,3])  # dark [1.5,2] light [2.5,3]
    rayleigh_sacle = random.choice([10,11,13,15,17])

    noise_ray = np.random.rayleigh(rayleigh_sacle ,(int(h//2),int(w//2)))

    noise_ray_re = cv2.resize(noise_ray,(w,h))

    #print(noise_ray_re)

    noise_add_image = (image - noise_ray_re*np.sqrt((image/255))*noise_gama) #加上锐利噪声 如果是在正常暗处增加黑点 就用1-(image/255) 否则 可以将gama 提升
    print(f"rayleigh_sacle{rayleigh_sacle} , noise_gama { noise_gama }")
    return np.clip(noise_add_image,0,255)


def mutil_process(folder,save_path):
    name_ext =  os.listdir(folder)
    for i in range(len(name_ext)):
        image_file = os.path.join(folder,name_ext[i])
        image = cv2.imread(image_file,0)
        noise_image = noise_add_twodiff(image)

        cv2.imwrite(os.path.join(save_path,name_ext[i]),noise_image,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"{i}/{len(name_ext)}")


folder=r"F:\Work\EVPevisPlus\Dataset\denoise\label_process\folder5"
save_path = r"F:\Work\EVPevisPlus\Dataset\denoise\label_process_raylinoise_lightdiff\folder5"
os.makedirs(save_path,exist_ok=True)
mutil_process(folder,save_path)