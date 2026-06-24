"""
Data:2024/2/4
Name:liangshubo
Object:

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
            print(image1.shape)
            image2 = image1 [150:950, 250:1050] #[180:776, 400:1216]心脏   # 三星效果子宫 [150:850,100:1200]
            cv2.imwrite(os.path.join(save_path, name + "_"+str(i)+".png"), image2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else :
        image2 = image[150:950, 250:1050,0]   #    #[150:850,100:1200]#[150:750, 370:1270,0]  # [180:776, 400:1216]心脏
        cv2.imwrite(os.path.join(save_path, name+".png" ), image2, [cv2.IMWRITE_PNG_COMPRESSION, 0])



###  n20 乳腺 [150:850, 470:1170,0]
### n20 128 腔内 [230:780,360:1260]
### N8000 颈动脉 [150:950, 250:1050]


def main():
    folder = (r"E:\Project_Studysam"
              r"zzsung\zigong\142215_samsung_zigong_cv0135_sharm\dicom\cv0")
    save = r"E:\Project_Studysamsung\zigong\240530_samsung_zigong_cv024_scisharm\png"
    folder_list = os.listdir(folder)

    with ThreadPoolExecutor(max_workers=16) as executor:
        for name in folder_list:
            path = os.path.join(folder,name)
            save_path = os.path.join(save,name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            os.makedirs(save_path,exist_ok=True)
            executor.submit(singledicom2png,path, save_path)

if __name__ == '__main__':
    #main(

    folder = r"E:\Project_Thyroid\dataset\dicom\cv5"
    save = r"E:\Project_Thyroid\dataset\png\cv5"
    name = os.listdir(folder)

    for i in range(len(name)):
        save_path = os.path.join(save,name[i])
        imag_path = os.path.join(folder,name[i])

        os.makedirs(save_path, exist_ok=True)
        singledicom2png(imag_path, save_path)
# path = r"G:\Work\Data_Accquire\Lumen\20240204\020401\dicomloop\0204FR06"
# save_path = r"G:\Work\Data_Accquire\Lumen\20240204\020401\png\0204FR06"
