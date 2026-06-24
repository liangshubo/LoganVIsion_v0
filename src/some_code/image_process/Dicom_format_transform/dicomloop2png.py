"""
Data:2024/2/4
Name:liangshubo
Object:
THIS IS MULTI THERA——PROCESS dicomloop 2 png
the parent folder have a lot of sub folder
such as
folder :
        cv0
        cv1
        cv2
        ...

"""


import os
import cv2
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor


def singledicom2png(path,save_path):
    filepath,name = os.path.split(path)
    print(name)
    dicom = sitk.ReadImage(path)
    #print(dicom.shape)
    image = sitk.GetArrayFromImage(dicom).squeeze()
    print(image.shape)

    if image.ndim ==4:
        save_list = []
        for i in range(0,image.shape[0]):
            image1 = image[i,:, :,0]

            image2 = image1[200:850, 180:1100]#[150:850, 170:1150] #[180:776, 400:1216]心脏   # 三星效果子宫 [150:850,100:1200]

            cv2.imwrite(os.path.join(save_path, name + "_"+str(i)+".png"), image2, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            print(f"raw resloution is {image1.shape} ,crop resloution is H,W {image2.shape} ,num: [{i+1}/{image.shape[0]}]")
    else :
        image2 = image[170:780, 520:1100]#[150:850, 170:1150,0]   #    #[150:850,100:1200]#[150:750, 370:1270,0]  # [180:776, 400:1216]心脏
        cv2.imwrite(os.path.join(save_path, name+".png" ), image2, [cv2.IMWRITE_PNG_COMPRESSION, 0])


### n20 heart [203:779,395:1212]
### n20 乳腺 [150:850, 470:1170,0]
### n20 128 腔内 [230:780,360:1260]
### n20 MSK_ANKLE [150:850, 470:1170]
### n20 knee [150:800, 410:1195]
### n20 should
### N8000 颈动脉 [150:950, 250:1050]
### N8000 腹部 [190:870,180:1120]
### N9000 la22 MSK
### N6000 MSK  [200:850, 180:1100]
# ---------------------

def main(folder,save):

    folder_list = os.listdir(folder)

    with ThreadPoolExecutor(max_workers=24) as executor:
        for name in folder_list:
            path = os.path.join(folder,name)
            save_path = os.path.join(save,name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            os.makedirs(save_path,exist_ok=True)
            executor.submit(singledicom2png,path, save_path)

if __name__ == '__main__':

    # ---------单个实验 ------------

    #path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/sample_sam_dataset/dicom/train/image/S2/S2Pacient_20250723173943_7"
    #save_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/sample_sam_dataset/png/train/image/S2/S2Pacient_20250723173943_7"
    #os.makedirs(save_path,exist_ok=True)
    #singledicom2png(path, save_path)

    # -----------肌骨数据集处理 ---------
    train_dicom_dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/dicom/train/image"
    train_png_dataset_savepath = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/image"

    test_dicom_dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/dicom/test/image"
    test_png_dataset_savepath = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/test/image"

    #S_List = ["S0"]
    # 每个切面下的文


    # 训练的每一个切面的文件夹     以及 保存的切面位置
    train_dicom_image_folder = train_dicom_dataset_path    # dicom/train/image/folder1
    train_png_image_savefolder = train_png_dataset_savepath    # png/train/image/folder1


    # 训练数据集每一个切面文件夹下的每一个个文件处理
    train_name = os.listdir(train_dicom_image_folder)
    with ThreadPoolExecutor(max_workers=24) as executor:
        for i in range(len(train_name)):
            imag_path = os.path.join(train_dicom_image_folder,train_name[i])
            save_path = train_png_image_savefolder

            os.makedirs(save_path, exist_ok=True)
            executor.submit(singledicom2png, imag_path, save_path)


    # -----------------------------------------------------------------------

    # 测试的每一个切面的文件夹     以及 保存的切面位置


    test_dicom_image_folder = test_dicom_dataset_path
    test_png_image_savefolder = test_png_dataset_savepath
    # 测试数据集每一个切面的文件夹处理
    test_name = os.listdir(test_dicom_image_folder)
    with ThreadPoolExecutor(max_workers=24) as executor:
        for j in range(len(test_name)):
            imag_path = os.path.join(test_dicom_image_folder,test_name[j])
            save_path = test_png_image_savefolder

            os.makedirs(save_path, exist_ok=True)
            # singledicom2png(imag_path, save_path)
            executor.submit(singledicom2png, imag_path, save_path)
    #
    #


# path = r"G:\Work\Data_Accquire\Lumen\20240204\020401\dicomloop\0204FR06"
# save_path = r"G:\Work\Data_Accquire\Lumen\20240204\020401\png\0204FR06"
