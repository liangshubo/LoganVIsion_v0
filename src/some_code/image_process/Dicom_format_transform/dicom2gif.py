"""
Data:2024/2/21
Name:liangshubo
Object:

"""
import cv2
import SimpleITK as sitk
import os
import glob
from  PIL import Image
import imageio



def resize_and_save(path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    list  = os.listdir(path)
    for imgname in list:
        img = os.path.join(path,imgname)
        dicom = sitk.ReadImage(img)
        image = sitk.GetArrayFromImage(dicom).squeeze()[:,:,0]
        image = image[180:756,424:1192]   # (180,756,424,1192)
        #image.tofile(os.path.join(save_path,imgname+".raw"))
        cv2.imwrite(os.path.join(save_path,imgname+".png"),image,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(" finish imwrite ")


def dicomloop2png(path,save_path):
    filepath,name = os.path.split(path)
    print(name)
    dicom = sitk.ReadImage(path)
   # print(dicom.shape)
    image = sitk.GetArrayFromImage(dicom)#.squeeze()[:, :, 0]
    #print(image.shape)
    for i in range(image.shape[0]):
        single_image = image[i,:,:,0]
        #print(single_image.shape)
    # image = image[180:756, 424:1192]
   # print(image.shape)
        cv2.imwrite(os.path.join(save_path, name + "_"+str(i)+".png"), single_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])





def dicomloop2gif(path,save_path):
    filepath,name = os.path.split(path)
    print(name)
    dicom = sitk.ReadImage(path)
   # print(dicom.shape)
    image = sitk.GetArrayFromImage(dicom)#.squeeze()[:, :, 0]
    #print(image.shape)
    image_list = []
    for i in range(image.shape[0]):
        single_image = image[i,:,:,0]
        #print(single_image.shape)
    # image = image[180:756, 424:1192]
   # print(image.shape)
        image_list.append(single_image)
        #cv2.imwrite(os.path.join(save_path, name + "_"+str(i)+".png"), single_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    gif_path = os.path.join(save_path,name+".gif")
    imageio.mimsave(gif_path,image_list,duration=0.08)




def mutil_process(folder,save_path):
    image_list = os.listdir(folder)
    print(folder)
    for i in range(len(image_list)):
        image_path = os.path.join(folder,image_list[i])
        save_path2 = os.path.join(save_path,image_list[i])

        if not os.path.exists(save_path2):
            os.makedirs(save_path2)
        print(image_path,"imagepath")
        #dicomloop2png(image_path,save_path2)
        dicomloop2gif(image_path, save_path2)

        print("finish process ")





if __name__ == '__main__':

    path = r"F:\AutoDenoise\Lumen\LumenEnhance\20240221\20240221\dicom\221A00"
    save = r"F:\AutoDenoise\Lumen\LumenEnhance\20240221\20240221\gif\221A00"
    if not  os.path.exists(save):
        os.makedirs(save)
    mutil_process(path,save)

