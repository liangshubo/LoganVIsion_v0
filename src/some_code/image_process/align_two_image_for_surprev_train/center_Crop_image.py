"""
Data:2024/4/18
Name:liangshubo
Object:
center crop image

"""
import  cv2
import os

def crop_size(folder,save_path):
    nameext = os.listdir(folder)
    for i in range(len(nameext)):
        image = os.path.join(folder,nameext[i])
        image_array = cv2.imread(image,0)
        image_array_crop = image_array[180:768, 527:1089]
        cv2.imwrite(os.path.join(save_path,nameext[i]),image_array_crop)

if __name__ == '__main__':
    folder = r"F:\Work\Ultrasound_imaging\SCI\png\240821_sciplus_test_self_scanandsci_thyroid\sci0"
    save_path = r"F:\Work\Ultrasound_imaging\SCI\png\240821_sciplus_test_self_scanandsci_thyroid\sci0_crop"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    crop_size(folder,save_path)
