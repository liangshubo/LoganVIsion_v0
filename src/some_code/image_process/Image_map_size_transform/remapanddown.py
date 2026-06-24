"""
Data:2024/4/24
Name:liangshubo
Object:

"""
"""
Data:2024/3/15
Name:liangshubo
Object:

"""
# -*- coding: UTF-8 -*-
"""
Create on 2023-12-18
@Author: LiangShubo
@email: liangshubo@neusoftmedical.com

"""
import os

import cv2
import numpy as np
from numpy import array
import random
from scipy.interpolate import interp1d


def reverse_Dmap():
    mapping_points_un = [
        (0, 0), (1, 8), (2, 12), (3, 16), (5, 20), (7, 24), (9, 28), (12, 32),
        (14, 36), (17, 40), (19, 44), (22, 48), (24, 52), (27, 56), (30, 60), (33, 64),
        (36, 68), (39, 72), (42, 76), (45, 80), (48, 84), (51, 88), (54, 92), (58, 96),
        (61, 100), (64, 104), (68, 108), (72, 112), (76, 116), (80, 120), (84, 124), (88, 128),
        (92, 132), (96, 136), (100, 140), (104, 144), (108, 148), (112, 152), (117, 156), (121, 160),
        (126, 164), (130, 168), (135, 172), (139, 176), (144, 180), (149, 184), (154, 188), (159, 192),
        (164, 196), (169, 200), (174, 204), (179, 208), (185, 212), (190, 216), (196, 220), (202, 224),
        (208, 228), (214, 232), (221, 236), (228, 240), (236, 244), (243, 248), (249, 252), (255, 256)
    ]

    # mapping_points_un = [
    #     (0, 4), (1, 8), (2, 12), (3, 16), (5, 20), (7, 24), (9, 28), (12, 32),
    #     (14, 36), (17, 40), (19, 44), (22, 48), (24, 52), (27, 56), (30, 60), (33, 64),
    #     (36, 68), (39, 72), (42, 76), (45, 80), (48, 84), (51, 88), (54, 92), (58, 96),
    #     (61, 100), (64, 104), (68, 108), (72, 112), (76, 116), (80, 120), (84, 124), (88, 128),
    #     (92, 132), (96, 136), (100, 140), (104, 144), (108, 148), (112, 152), (117, 156), (121, 160),
    #     (126, 164), (130, 168), (135, 172), (139, 176), (144, 180), (149, 184), (154, 188), (159, 192),
    #     (164, 196), (169, 200), (174, 204), (179, 208), (185, 212), (190, 216), (196, 220), (202, 224),
    #     (208, 228), (214, 232), (221, 236), (228, 240), (236, 244), (243, 248), (249, 252), (255, 256)
    # ]
    x_coords_un, y_coords_un = zip(*mapping_points_un)
    interp_func_un = interp1d(x_coords_un, y_coords_un, kind='cubic', fill_value="extrapolate")
    return interp_func_un


def reverse_map(map="D"):
    # D 实际是说就是remap 而 RD是机器上的正向map抑制效果
    mapping_points_un = {
        "D": [
            (0, 4), (1, 8), (2, 12), (3, 16), (5, 20), (7, 24), (9, 28), (12, 32),
            (14, 36), (17, 40), (19, 44), (22, 48), (24, 52), (27, 56), (30, 60), (33, 64),
            (36, 68), (39, 72), (42, 76), (45, 80), (48, 84), (51, 88), (54, 92), (58, 96),
            (61, 100), (64, 104), (68, 108), (72, 112), (76, 116), (80, 120), (84, 124), (88, 128),
            (92, 132), (96, 136), (100, 140), (104, 144), (108, 148), (112, 152), (117, 156), (121, 160),
            (126, 164), (130, 168), (135, 172), (139, 176), (144, 180), (149, 184), (154, 188), (159, 192),
            (164, 196), (169, 200), (174, 204), (179, 208), (185, 212), (190, 216), (196, 220), (202, 224),
            (208, 228), (214, 232), (221, 236), (228, 240), (236, 244), (243, 248), (249, 252), (255, 256)
        ],"RD": [
            (4,0), (8,1), ( 12,2), (16,3), ( 20,5), (24,7), ( 28,9), ( 32,12),
            (36,14), (40,17), ( 44,19), (48,22), ( 52,24), ( 56,27), (60,30), ( 64,33),
            ( 68,36), (72,39), ( 76,42), (80,45), ( 84,48), (88,51), (92,54), ( 96,58),
            (100,61), ( 104,64), (108,68), ( 112,72), (116,76), (120,80), (124,84), (128,88),
            (132,92), (136,96), (140,100), (144,104), (148,108), ( 152,112), (156,117), (160,121),
            (164,126), (168,130), (172,135), ( 176,139), ( 180,144), ( 184,149), ( 188,154), ( 192,159),
            (196,164), (200,169), (204,174), (208,179), (212,185), (216,190), (220,196), (224,202),
            ( 228,208), ( 232,214), ( 236,221), (240,228), ( 244,236), (248,243), ( 252,249), ( 256,255)
        ],"RC":[(4, 0), (8, 1), (12, 2), (16, 3), (20, 5), (24, 8), (28, 10), (32, 13),
                (36, 15), (40, 18), (44, 21), (48, 24), (52, 27), (56, 30), (60, 33), (64, 37),
                (68, 40), (72, 43), (76, 47), (80, 51), (84, 55), (88, 59), (92, 63), (96, 67),
                (100, 72), (104, 76), (108, 81), (112, 85), (116, 90), (120, 95), (124, 100), (128, 105),
                (132, 110), (136, 117), (140, 123), (144, 128), (148, 133), (152, 138), (156, 143), (160, 148),
                (164, 153), (168, 158), (172, 163), (176, 167), (180, 172), (184, 177), (188, 181), (192, 186),
                (196, 191), (200, 195), (204, 200), (208, 204), (212, 209), (216, 213), (220, 218), (224, 222),
                (228, 227), (232, 231), (236, 236), (240, 240), (244, 245), (248, 249), (252, 252), (256, 255)
        ],"C":[(0, 0), (1, 8), (2, 12), (3, 16), (5, 20), (8, 24), (10, 28), (13, 32),
                (15, 36), (18, 40), (21, 44), (24, 48), (27, 52), (30, 56), (33, 60), (37, 64),
                (40, 68), (43, 72), (47, 76), (51, 80), (55, 84), (59, 88), (63, 92), (67, 96),
                (72, 100), (76, 104), (81, 108), (85, 112), (90, 116), (95, 120), (100, 124), (105, 128),
                (110, 132), (117, 136), (123, 140), (128, 144), (133, 148), (138, 152), (143, 156), (148, 160),
                (153, 164), (158, 168), (163, 172), (167, 176), (172, 180), (177, 184), (181, 188), (186, 192),
                (191, 196), (195, 200), (200, 204), (204, 208), (209, 212), (213, 216), (218, 220), (222, 224),
                (227, 228), (231, 232), (236, 236), (240, 240), (245, 244), (249, 248), (252, 252), (255, 256)
        ] # c 原来是0-4 ，我改成0-0
        , "F": [(0, 4), (0.5, 8), (1, 12), (2, 16), (3, 20), (4, 24), (5, 28), (6, 32),
                 (8, 36), (9, 40), (11, 44), (12, 48), (14, 52), (15, 56), (17, 60), (19, 64),
                 (21, 68), (23, 72), (25, 76), (27, 80), (30, 84), (32, 88), (35, 92), (37, 96),
                 (40, 100), (42, 104), (45, 108), (47, 112), (50, 116), (52, 120), (55, 124), (58, 128),
                 (61, 132), (64, 136), (67, 140), (70, 144), (74, 148), (78, 152), (82, 156), (86, 160),
                 (90, 164), (95, 168), (99, 172), (104, 176), (109, 180), (114, 184), (119, 188), (124, 192),
                 (129, 196), (135, 200), (141, 204), (147, 208), (153, 212), (160, 216), (167, 220), (175, 224),
                 (183, 228), (192, 232), (201, 236), (210, 240), (220, 244), (232, 248), (245, 252), (255, 256)
                 ], "K": [(0, 4), (1, 8), (2, 12), (3, 16), (4, 20), (5, 24), (6, 28), (8, 32),
                          (9, 36), (11, 40), (13, 44), (15, 48), (17, 52), (19, 56), (21, 60), (23, 64),
                          (26, 68), (28, 72), (31, 76), (33, 80), (36, 84), (39, 88), (42, 92), (45, 96),
                          (48, 100), (51, 104), (54, 108), (57, 112), (60, 116), (63, 120), (66, 124), (70, 128),
                          (73, 132), (77, 136), (81, 140), (85, 144), (89, 148), (93, 152), (98, 156), (103, 160),
                          (107, 164), (112, 168), (117, 172), (122, 176), (127, 180), (132, 184), (137, 188),
                          (143, 192),
                          (148, 196), (154, 200), (160, 204), (166, 208), (172, 212), (178, 216), (184, 220),
                          (191, 224),
                          (197, 228), (204, 232), (210, 236), (217, 240), (224, 244), (231, 248), (238, 252), (245, 256)
                          ]}

    x_coords_un, y_coords_un = zip(*mapping_points_un[map])
    interp_func_un = interp1d(x_coords_un, y_coords_un, kind='cubic', fill_value="extrapolate")
    return interp_func_un


def map_process(image: array, reverse_map) -> array:
    return reverse_map(image)


def down_sample(image: array, mode="linear", scale=1.26):
    h, w = image.shape
    if mode == 'nearest':
        inter_mode = cv2.INTER_NEAREST
    elif mode == "linear":
        inter_mode = cv2.INTER_LINEAR
    elif mode == "cubic":
        inter_mode = cv2.INTER_CUBIC
    elif mode == "area":
        inter_mode = cv2.INTER_AREA
    elif mode == "lanz":
        inter_mode = cv2.INTER_LANCZOS4
    elif mode == "random":
        inter_mode = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])

    resize = cv2.resize(image, (int(w // scale), int(h // scale)), inter_mode)
    return resize


def multi_process(path, save_path, map=None, mode=None, scale=None):
    image_path_list = os.listdir(path)

    for image in image_path_list:
        image_path = os.path.join(path, image)
        # if dicom please use sitk read
        image_array = cv2.imread(image_path, 0)
        assert map is not None or (mode is not None and scale is not None), "no process"

        if mode is not None and scale is not None:
            image_array = down_sample(image_array, mode, scale)

        #  调整下顺序
        if map:
            map_func = reverse_map(map)
            image_array = map_func(image_array)

        save_array_path = os.path.join(save_path, image)
        cv2.imwrite(save_array_path, image_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"finish {image}")
    return


if __name__ == '__main__':
    inter_mode = ['nearest', "linear", "cubic", "area", "random", "lanz"]
    scale = 1.26

    #  -----add by liangshubo 240205  for more folder
    name_list1 = ["cv0", "label/cv1", "label/cv2", "label/cv3", "label/cv4", "label/cv5"]
    name_list2 = ["cv0", "label_refine/cv1", "label_refine/cv2", "label_refine/cv3", "label_refine/cv4",
                  "label_refine/cv5"]
    name_list3 = ["cv0", "cv1", "cv2", "cv3", "cv4", "cv5"]
    name_list4= [""]
    for name in name_list4:
        path = r"G:\Work\Self-develop-SRI\Dataset\Dataset424\roi_x1\label"  # /label_refine/
        save_path = r"G:\Work\Self-develop-SRI\Dataset\Dataset424\roi_x1\label_remap"
        #path = r"G:\Work\Project_SRI\SRI_4_8\Breast\Dataset\N20Dicom\png\cv0"  # /label_refine/
        #save_path = r"G:\Work\Project_SRI\SRI_4_8\Breast\Dataset\N20Dicom\png\cv0_remapdown"
        #  -----add by liangshubo 240205  for more folder

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        multi_process(path, save_path, map="D", mode=None,scale=None)

# ---20240308 ---------


# -------------20240205 -------------------- 进行正确尺寸下得三个数据  ， 第一个是只remap 不down  第二个只down 不map  第三个 map 加上down  注意 LEArt F map  LEVE   Dmap   UE kmap
# path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/Carotid_0205IMT_B/cv0"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/Carotid_0205IMT_B_remapdown/cv0"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map="D",mode=inter_mode[4],scale=1.30555)

#  multi_process(path,save_path,map="K",mode=inter_mode[4],scale=1.30555)
#  multi_process(path,save_path,map="K",mode=inter_mode[4],scale=1.30555)
# --------------------20240205  - --------------------------------以上数据只map 不down  ------------------
# 添加了多文件夹处理程序
# name_list1 = ["cv0","label/cv1","label/cv2","label/cv3","label/cv4","label/cv5"]

# for name in name_list1:
#     path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/Carotid_0205IMT_B/"+name # /label_refine/
#     save_path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/Carotid_0205IMT_B_remap/"+name
# #  -----add by liangshubo 240205  for more folder

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     multi_process(path,save_path,map="D",mode=None,scale=None)

#     for name in name_list1:
#     path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/Carotid_0205IMT_B/"+name # /label_refine/
#     save_path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/Carotid_0205IMT_B_remap/"+name
# #  -----add by liangshubo 240205  for more folder

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     multi_process(path,save_path,map="D",mode=None,scale=None)

# ---------------------------------
#     for name in name_list2:
#     path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/LEUEArtVein/UEArtVe/"+name # /label_refine/
#     save_path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/LEUEArtVein/UEArtVe_remap/"+name
# #  -----add by liangshubo 240205  for more folder

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     multi_process(path,save_path,map="K",mode=None,scale=None)


# -------------20240201 --------------------
# 对第一版就是错误尺寸下的内中膜数据集进行反map ，这里Carotid_IMT_A\B 不下采样仅进行反map  Carotid_C 下采样1.30444加上map 另外管腔的两个Carotid_A\B 也进行下采样和反map
# path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/Carotid_IMT_A/label/cv1"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/dataset/rawdata/Carotid_IMT_A_remap/label/cv1"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map="D",mode=None,scale=None)


# ----------------------------------------------------

# inter_mode = ['nearest',"linear","cubic","area","random"]
# scale = 1.26
# path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/noise_black_light"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/noise_black_light_remapdown"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map=True,mode=inter_mode[4],scale=1.263)


# inter_mode = ['nearest',"linear","cubic","area","random"]
# scale = 1.26
# path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv1_black_light_deblur"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv1_black_light_deblur_down"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map=False,mode=inter_mode[3],scale=1.263)

#  path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv3_black_light_deblur"
#     save_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv3_black_light_deblur_down"
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     multi_process(path,save_path,map=False,mode=inter_mode[3],scale=1.263)

# path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv5_black_light_deblur"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv5_black_light_deblur_down"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map=False,mode=inter_mode[3],scale=1.263)


######## remap-down
# inter_mode = ['nearest',"linear","cubic","area","random"]
# scale = 1.26
# path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv1_black_light_deblur"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv1_black_light_deblur_remapdown"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map=True,mode=inter_mode[3],scale=1.263)

# -----------------------------------
#     inter_mode = ['nearest',"linear","cubic","area","random","lanz"]
# scale = 1.26
# path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv5_black_light"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1204_size128/cv5_black_light_remapdown_cubic"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map=True,mode=inter_mode[4],scale=1.263)

# ---------------------------------------------------------

# inter_mode = ['nearest',"linear","cubic","area","random","lanz"]
# scale = 1.26
# path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1215_remap/cv2_black_light"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1215_remap/cv2_black_light_remapdown"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map=True,mode=inter_mode[4],scale=1.26)

# ---------------------------------------------------------
# inter_mode = ['nearest',"linear","cubic","area","random","lanz"]
# scale = 1.26
# path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1215_remap/cv4_black_light"
# save_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/dataset/Rawdata/Patch_1215_remap/cv4_black_light_remapdown"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# multi_process(path,save_path,map=True,mode=inter_mode[4],scale=1.26)
