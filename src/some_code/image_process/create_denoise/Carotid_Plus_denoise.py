import os

import  cv2
import  numpy as np
import random



mask_path = r"F:\Work\EVPevisPlus\Dataset\denoise\label\mask1.png"
image_path = r"F:\Work\EVPevisPlus\Dataset\denoise\label\samsung_folder1_image1.png"

image = cv2.imread(image_path,0)






def difference_noise(image,mask):
    h,w = image.shape
    noise_gama = random.choice([1,2.5,3])  # dark [1.5,2] light [2.5,3]
    rayleigh_sacle = random.choice([5,8,10,11,13,14])
    scale_w = random.choice([1.31,2,1.4,1.81,2.6,1.9,2.3])
    scale_h = random.choice([1.31,2,2.4,1.31,1.56,2.1,2.2])
    noise_ray = np.random.rayleigh(rayleigh_sacle,(int(h//scale_h),int(w//scale_w)))

    noise_ray_re = cv2.resize(noise_ray,(w,h))
    #print(noise_ray_re.mean())

    noise_diff_image = (image - noise_ray_re*np.sqrt((image/255))*noise_gama)*mask #加上锐利噪声 如果是在正常暗处增加黑点 就用1-(image/255) 否则 可以将gama 提升

    #noise_add_image = (noise_diff_image + noise_ray_re*np.sqrt(1-(image/255))*noise_gama)*mask
    print(noise_diff_image.max())
    noise_diff_image = np.clip(noise_diff_image,0,255)

    return noise_diff_image

def relu_x(x):
    y = x
    condition = (y<0)
    y[condition] = 0
    return y


def addgauss_noise(image,mask):
    h, w = image.shape

    noise_gama = random.choice([1,1.5,1.7])  # dark [1.5,2] light [2.5,3]
    rayleigh_sacle = random.choice([5,8,9])
    scale_w = random.choice([1.31,2,1.4,1.81,2.6,3.9,2.3,3.9])
    scale_h = random.choice([1.31,2,2.4,1.31,1.56,3.1,2.2,3.5])
    noise_ray = np.random.rayleigh(rayleigh_sacle,(int(h//scale_h),int(w//scale_w)))

    noise_ray_re = cv2.resize(noise_ray,(w,h))


    image_add_noise = image + noise_ray_re*(np.sqrt((relu_x(image.mean()-image)/(image.mean()-image).max()))*noise_gama)*mask

    print(f"noise_gama{noise_gama} rayleigh_sacle{rayleigh_sacle} scale_w{scale_w}")

    image_add_noise = np.clip(image_add_noise,0,255)
    return image_add_noise




def single_image_noise(image,mask):

    mask_iou = mask.sum() / (mask.shape[0] * mask.shape[1])
    #print(f"the value piexl is {mask_iou}")
    image_ray_noise = difference_noise(image,mask)
    print(f"rawimage light ", image.mean(),end='')
    print(f" raynoise_image mean ", image_ray_noise.mean(),end='')

    det_mean = image.mean() - image_ray_noise.mean()
    i=1
    while det_mean >= 0.1 * image.mean():
        noise_add_image = addgauss_noise(image_ray_noise, det_mean+i ,mask)
        det_mean = image.mean() - noise_add_image.mean()
        print(f" gaus_noise image mean ", noise_add_image.mean(),end='')
        print(f" det_mean {det_mean}")
        i+=1

    return np.clip(noise_add_image,0,255)


def multi_process(folder,mask_path,save_path):
    mask = cv2.imread(mask_path, 0)
    name_ext = os.listdir(folder)
    for i in range(len(name_ext)):
        image = cv2.imread(os.path.join(folder,name_ext[i]),0)
        noiseimage  = difference_noise(image,mask)
        cv2.imwrite(os.path.join(save_path,name_ext[i]),noiseimage,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"{i} / {len(name_ext)}")

def multi_process2(folder,mask_path,save_path):
    mask = cv2.imread(mask_path, 0)
    name_ext = os.listdir(folder)
    for i in range(len(name_ext)):
        image = cv2.imread(os.path.join(folder,name_ext[i]),0)
        noiseimage  = addgauss_noise(image,mask)
        cv2.imwrite(os.path.join(save_path,name_ext[i]),noiseimage,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"{i} / {len(name_ext)}")




if __name__ == '__main__':
    folder  = r"E:\Project_Carotid\dataset\denoise\carotid_plus_denoise_dicom\label"
    mask_path = r"E:\Project_Carotid\dataset\denoise\carotid_plus_denoise_dicom\mask.png"

    save_path = r"E:\Project_Carotid\dataset\denoise\carotid_plus_denoise_dicom\add_denoise"
    os.makedirs(save_path,exist_ok=True)

    multi_process2(folder,mask_path,save_path)

















#
#
#
# print(f"noise_add_image mean",noise_add_image.mean())
# gauss_noise = np.random.normal(image.mean()-noise_add_image.mean(),2,(h,w))
# #print(gauss_noise.mean())
# noise_add_image_gauss = noise_add_image + gauss_noise*mask
# print(noise_add_image_gauss.mean())
#
# gauss_noise2 = np.random.normal(image.mean()-noise_add_image_gauss.mean(),2,(h,w))
# #print(gauss_noise2.mean())
# noise_add_image_gauss2 = noise_add_image_gauss + gauss_noise2*mask
# print(noise_add_image_gauss2.mean())
#
# gauss_noise3 = np.random.normal(image.mean()-noise_add_image_gauss2.mean(),2,(h,w))
# #print(gauss_noise2.mean())
# noise_add_image_gauss3 = noise_add_image_gauss2 + gauss_noise3*mask
# print(noise_add_image_gauss3.mean())
#
# cv2.imshow("raw",image/255)
# cv2.imshow("noise_add_image",noise_add_image/255)
# cv2.imshow("noise_add_image_gauss",noise_add_image_gauss/255)
#
# cv2.imshow("noise_add_image_gauss2",noise_add_image_gauss2/255)
# cv2.imshow("noise_add_image_gauss3",noise_add_image_gauss3/255)
#
#
# cv2.waitKey(0)
