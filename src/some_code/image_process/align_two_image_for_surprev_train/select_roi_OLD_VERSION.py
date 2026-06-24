"""
Data:2024/4/16
Name:liangshubo
Object:

"""
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 文件夹路径
folder_path = r"G:\Work\Self-develop-SRI\Dataset\Dataset416\Choice_Data"
# 裁剪后保存的路径
save_path = r"G:\Work\Self-develop-SRI\Dataset\Dataset416\Select_Roi"

# 获取文件夹中所有图像文件的路径
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 交互式选择矩形区域的回调函数
def select_roi(event, x, y, flags, param):
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        # 绘制矩形框
        cv2.rectangle(image_copy, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image_copy)

# 遍历所有图像文件
for image_file in image_files:
    # 读取图像
    image = cv2.imread(image_file)
    image_copy = image.copy()

    # 创建一个窗口显示图像
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_roi)

    # 等待用户交互选择矩形区域
    while True:
        cv2.imshow("image", image_copy)
        key = cv2.waitKey(1) & 0xFF

        # 保存选定的矩形区域并关闭图像窗口
        if key == ord("c"):
            if len(refPt) == 2:
                roi = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                cv2.imwrite(os.path.join(save_path, os.path.basename(image_file)), roi)
                break

        # 取消选择
        elif key == ord("r"):
            image_copy = image.copy()

    # 清除选择框
    refPt = []

    # 关闭图像窗口
    cv2.destroyAllWindows()
