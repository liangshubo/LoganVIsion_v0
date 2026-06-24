"""
Data:2024/4/16
Name:liangshubo
Object:this file is used two folder tha the name is same and after the align 
 we will choose a region this will crop the region
always used the sci 0_1 train  

"""
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# namelist = ["Breast_dmap","Carotid_Heng","Carotid_Zong","Thyroid"]
# 文件夹路径c
image_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_denoise_yuan_equal_jin/cv0_image"   # low line

label_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_denoise_yuan_equal_jin/cv0_image_copy"   #  high line
# 裁剪后保存的路径
save_path1 = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_denoise_yuan_equal_jin/cv0_crop_tiss"
save_path2 = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_denoise_yuan_equal_jin/msk_tiss_crop"
if not os.path.exists(save_path1):
    os.makedirs(save_path1)

if not  os.path.exists(save_path2):
    os.makedirs(save_path2)




# 获取文件夹中所有图像文件的路径
image_files = [os.path.join(image_path , f) for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
label_files = [os.path.join(label_path , f) for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 交互式选择矩形区域的回调函数

def select_roi(event, x, y, flags, param):
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        if x<refPt[0][0] or y<refPt[0][1] :
            refPt.append((refPt[0][0]+100 , refPt[0][1]+100))
        else:
            refPt.append((x, y))
        cropping = False

        # 绘制矩形框
        cv2.rectangle(image_copy, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image_copy)

        cv2.rectangle(label_copy, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("label", label_copy)


    # 遍历所有图像文件
for i in range(len(image_files)):
    # 读取图像
    image_file = image_files[i]
    label_file = label_files[i]
    print(i,"_",image_file)
    # test
    save_path_list = os.listdir(save_path1)
    print(f"save path have {len(save_path_list)} image")

    _,nameext = os.path.split(image_file)
    if nameext in save_path_list:
        print(f"this image had save will pass")
        continue


    image = cv2.imread(image_file)

    label = cv2.imread(label_file)

    if  image is None or label is None:
        print("无法读取图像文件:", image_file, label_file)
        continue

    image_copy = image.copy()
    label_copy = label.copy()

    # 创建一个窗口显示图像



    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_roi)

    # 等待用户交互选择矩形区域
    while True:
        cv2.imshow("label", label_copy)
        cv2.imshow("image", image_copy)

        key = cv2.waitKey(1) & 0xFF

        # 保存选定的矩形区域并关闭图像窗口
        if key == ord("c"):
            if len(refPt) == 2:
                roi1 = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]


                roi2 = label[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                if roi1 is None or roi2 is None:
                    print(f"roi is wrong",image_file)
                    continue
                cv2.imwrite(os.path.join(save_path1, os.path.basename(image_file)), roi1)

                cv2.imwrite(os.path.join(save_path2, os.path.basename(image_file)), roi2)
                break

        # 取消选择
        elif key == ord("r"):
            image_copy = image.copy()

    # 清除选择框
    refPt = []

    # 关闭图像窗口
    cv2.destroyAllWindows()

