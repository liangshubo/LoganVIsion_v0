import os
from os.path import isfile

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





def png2gif(path,save_path,name,fps):
    #  在获取当前文件夹的名字
    filepath,_ = os.path.split(path)
    name_list = os.listdir(path)
    image_list = []
    for i in range(1,len(name_list)+1):
        image_path =os.path.join(path,name+"_"+str(i)+".png")   # 这个要改变
        if not isfile(image_path):
            continue
        single_image = cv2.imread(image_path)

        single_image = cv2.cvtColor( single_image, cv2.COLOR_BGR2RGB)
        #print(single_image.shape)
        image_list.append(single_image)
        #cv2.imwrite(os.path.join(save_path, name + "_"+str(i)+".png"), single_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    gif_path = os.path.join(save_path,name+".gif")
    imageio.mimsave(gif_path,image_list,fps=fps)


import os
import cv2
import numpy as np
from os.path import isfile


def png2mp4(path, save_path, name, fps=30, codec='mp4v'):
    """
    将PNG序列转换为MP4视频文件

    参数:
        path: PNG图像序列所在的文件夹路径
        save_path: MP4视频保存路径
        name: 视频文件名（不含扩展名）
        fps: 视频帧率 (默认30)
        codec: 视频编码器 (默认'mp4v')
    """
    # 获取所有PNG文件
    name_list = sorted([f for f in os.listdir(path) if f.endswith('.png')])

    if not name_list:
        print(f"错误: 在路径 {path} 中未找到PNG文件")
        return

    # 读取第一张图像获取尺寸
    first_image_path = os.path.join(path, name_list[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"错误: 无法读取图像 {first_image_path}")
        return

    height, width, layers = first_image.shape

    # 创建视频写入器
    video_path = os.path.join(save_path, f"{name}.mp4")

    # 根据操作系统选择最佳编码器
    if codec == 'auto':
        if os.name == 'nt':  # Windows系统
            codec = 'mp4v'
        else:  # Linux/Mac系统
            codec = 'avc1'

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    if not video.isOpened():
        print(f"错误: 无法创建视频文件 {video_path}")
        print("尝试使用不同的编码器，如 'XVID' 或 'MJPG'")
        return

    # 处理所有图像
    for i in range(1, len(name_list) + 1):
        img_path = os.path.join(path, name + "_" + str(i) + ".png")  # 这个要改变
        if not isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}，跳过")
            continue

        # 确保图像尺寸一致
        if img.shape != (height, width, layers):
            img = cv2.resize(img, (width, height))

        video.write(img)

        # 显示进度

        print(f"处理进度: {i + 1}/{len(name_list)}")

    # 释放视频写入器
    video.release()
    print(f"视频已保存至: {video_path}")
    print(f"总帧数: {len(name_list)}, 帧率: {fps}, 时长: {len(name_list) / fps:.2f}秒")

    return video_path

if __name__ == '__main__':
    #[ PATLSB0716_1 , S2Pacient_20250723173943_7  ,S2Pacient_20250725165242_4 ,  S3Patient_20250723123224_12 ,S3Patient_20250723123224_41 ,S4Pacient_20250725165242_9 ,  S4Patient_20250723123224_42  ,  S5Pacient_20250725165242_12 , S6Pacient_20250725165242_15 , S7Pacient_20250723173944_23,S8Patient_20250723123224_26,S10Patient_20250723123224_31,S11Patient_20250723123224_22]
    # n6000 的效果
    # name_list = [ "PATLSB0716_1",
    #               'S2Pacient_20250723173943_7',
    #               'S3Patient_20250723123224_12',
    #               'S3Patient_20250723123224_41',
    #               'S4Patient_20250723123224_42',
    #               'S5Patient_20250723123224_16',
    #               'S6Pacient_20250725165242_15',
    #               'S7Pacient_20250723173944_23',
    #               'S8Patient_20250723123224_28',
    #               'S10Patient_20250723123224_31',
    #               'S11Patient_20250723123224_22']

    name_list_n20 = ["S1WQSXXX02",
                     'S2QPXXXX04',
                     'S3MH_MSK04',
                     'S4LT_RXX0E',
                     'S5Y_MSKX0G',
                     'S6YH_MSK0I',
                     'S7Y_MSKX0O',
                     'S8WQSXXX0Q',
                     'S10SSXXXX0I',
                     'S11MH_MSK0K']

    for name in  name_list_n20:

        path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/test_inference/inference_class_tendom_segment/Png_Output1030"
        save = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/test_inference/inference_class_tendom_segment/Gif_Output1030"
        if not  os.path.exists(save):
            os.makedirs(save)
        png2gif(path,save,name,fps=15)

        png2mp4(path,save,name,fps=15)

