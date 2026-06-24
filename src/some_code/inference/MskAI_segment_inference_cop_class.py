'''

这一个代码是用于 联合分割以及输入分类信息和组织权重的推理  ，

'''

import numpy as np
import torch
import cv2
import torchvision
import argparse
import src.model
from importlib import import_module
import os
import time


from src.train.utility import utility
import glob
import re
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from PIL import Image, ImageDraw, ImageFont
from numpy import uint8
import torch.nn as nn
project_path  = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
print(project_path)

def get_args():
    parser = argparse.ArgumentParser(description='Thermal and Rail SR')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--model', default='Unet',
                        help='model name')
    parser.add_argument('--pre_train', type=str, default=None,
                        help='pre-trained model directory')
    parser.add_argument('--iterates', type=int, default=30,
                        help='iterates number ')
    parser.add_argument('--input_data', type=str, default=None,
                        help='single image  directory')
    parser.add_argument('--input_path', type=str, default=None,
                        help='more image directory')
    parser.add_argument('--project_path', type=str, default=project_path,
                        help='project_path ')
    parser.add_argument('--save_path', type=str, default=None,
                        help='save_path ')
    parser.add_argument('--sw_mode', type=str, default="S",
                        help='save_path ')
    parser.add_argument('--resume', type=int, default=0,
                        help='control the load model is best or lastest or other ')
    parser.add_argument('--input_size', type=int, default=512,
                        help='control the input data crop patch (592,720), if patch_size != None it will be crop shuffle 作为正方形')
    parser.add_argument('--num_class', type=int, default=24,
                        help='segment  num_class ')
    parser.add_argument('--class_label', type=int, default=0,
                        help='plane class ')
    parser.add_argument('--extra_application', type=int, default=1,
                        help='extra_application ')
    args = parser.parse_args()
    return args


def tensor2array(tensor, rgb_range):
    tensor = tensor.squeeze(0)
    array = tensor.squeeze(0).cpu().numpy() * (255 / rgb_range)
    return array


def control_pretrain(resume: int) -> str:
    if resume == 0:
        return "best"
    elif resume == -1:
        return "latest"
    else:
        return resume

def create_onehot_from_indices(indices_list, dim=24):
    """
    将组织索引列表转换为固定维度的one-hot编码
    参数:
    indices_list -- 包含组织索引的列表，例如 [3, 6, 7]
    dim -- one-hot编码的维度，默认为24
    返回:
    一个dim维的numpy数组，其中指定索引位置为1，其余为0
    """
    # 初始化全0向量
    onehot = torch.zeros(dim, dtype=float)

    # 将指定索引位置设为1
    for idx in indices_list:
        if 1 <= idx <= dim:
            onehot[idx] = 1  # 索引从1开始，但数组从0开始

    return onehot

def inference_multi_image(args,info,class_label):
    segment_postprocess = utility.ArgmaxPostProcessor()
    section_to_tissue = [[0, 0, 0], [0, 1, 2, 3],  # 切面1的组织索引
                         [0, 3, 4, 5],  # 切面2的组织索引
                         [0, 3, 6, 7],  # 切面3的组织索引
                         [0, 3, 8, 9],  # 切面4的组织索引
                         [0, 3, 10, 11, 12],  # 切面5的组织索引
                         [0, 3, 13, 14],  # 切面6的组织索引
                         [0, 15, 16],  # 切面7的组织索引
                         [0, 3, 16, 17],  # 切面8的组织索引
                         [0, 18, 19, 20],  # 切面10的组织索引
                         [0, 21, 22, 23],  # 切面11的组织索引
                         ]
    if args.cpu:
        # ----加载模型 #导入模型 ，所以这里要注意的是都是小写的
        module = import_module('src.model.' + args.model.lower())  # 导入模型 ，所以这里要注意的是都是小写的
        model = module.make_model(args).cpu()
        # ----预训练参数路径
        if args.pre_train:
            pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train, "model")
            model.load_state_dict(
                torch.load(pre_train_model_path + f"/model_{control_pretrain(args.resume)}.pt", map_location='cpu'),
                strict=False)
        model = model.cpu()
        print("[----> Model inference in ", next(model.parameters()).device, "<----]")

        name_list = os.listdir(args.input_path)
        count = 1
        for nameext in name_list:
            # -------输入数据路径与文件名分离
            input_data = os.path.join(args.input_path, nameext)

            # -------输入数据路径与文件名分离

            # -------数据读取与预处理
            input = torch.tensor(cv2.imread(input_data, 0)).unsqueeze(0).unsqueeze(0) / 255
            INPUT = input.cpu()

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # -------模型前向处理---------
            avg_time = 0
            with torch.no_grad():
                # starter.record()
                start = time.time()
                #    model input idx

                output = model(INPUT)

                # ------结果保存-----
                output = tensor2array(output, 1)
                cv2.imwrite(os.path.join(args.save_path, nameext), output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # ender.record()
                end = time.time()
                # cur_time = starter.elapsed_time(ender)
                cur_time = (end - start) * 1000
                avg_time += cur_time
                print("Image[{}/{}] : {:.5f}ms".format(count, len(name_list), cur_time))
            count += 1
    else:
        # ----加载模型 #导入模型 ，所以这里要注意的是都是小写的
        module = import_module('src.model.' + args.model.lower())
        model = module.make_model(args)
        # ----预训练参数路径
        if args.pre_train:
            pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train, "model")
            model.load_state_dict(torch.load(pre_train_model_path + f"/model_{control_pretrain(args.resume)}.pt"),
                                  strict=False)
        model = model.cuda().eval()
        print("[----> Model inference in ", next(model.parameters()).device, "<----]")

        # ------加载文件夹内输入数据 并排序  ----

        name_list = os.listdir(args.input_path)
        name_list = sorted(name_list,key=lambda x: int(re.search(r'_(\d+)\.png$', x).group(1)))

        count = 1
        for nameext in name_list:
            # -------输入数据路径与文件名分离
            input_data = os.path.join(args.input_path, nameext)
            # -------数据读取与预处理
            rawinput = cv2.imread(input_data, 0)   # [0-255 ] [H,W]  array   cpu
            h,w = rawinput.shape    # [0-255 ] [H,W]
            resizeinput = cv2.resize(rawinput,(args.input_size,args.input_size),interpolation=cv2.INTER_CUBIC)    # [0-255 ] [args.inputsize  ,args.inputsize ]   array   cpu
            # ------------ 引入 切面 信息     -----------------
            label_array = np.ones_like(resizeinput) * int(class_label)
            label_tensor = torch.from_numpy( label_array).float().unsqueeze(0).unsqueeze(0).cuda()
            plane_channel_idx = section_to_tissue[int(class_label)]
            plane_channel_idx_tensor = create_onehot_from_indices(plane_channel_idx, args.num_class).cuda().unsqueeze(0)

            input = torch.tensor( resizeinput).unsqueeze(0).unsqueeze(0) / 255 # [0-1 ] [1,1 , args.inputsize  ,args.inputsize ]   tensor   cpu
            # cat
            input = input.cuda()
            image_cat_plane = torch.cat([input , label_tensor  / 10], dim=1)

            # [0-1 ] [1,1 , args.inputsize  ,args.inputsize ]   tensor  gpu
            listinput = [image_cat_plane, plane_channel_idx_tensor]

            # print(INPUT.shape, "INPUT ")
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # -------模型前向处理---------
            avg_time = 0
            prev_logits_ema = None
            with torch.no_grad():

                starter.record()
                # 　－－－进行推理 这里要看看模型的输出情况，有可能是多输出  －－－－  这里默认的输出应该是 【1,24,512,512】
                output = model(listinput)

                #output = output[3]

                if prev_logits_ema is None:
                    logits_smooth = output
                else:
                    logits_smooth = segment_postprocess.temporal_smoothing_v2(prev_logits_ema,output,args.num_class)
                prev_logits_ema = output

                # ------结果保存-----
                ender.record()
                torch.cuda.synchronize()
                # 后处理阶段的
                pred = torch.argmax(logits_smooth.squeeze(0), dim=0).cpu() # [512,512] ,[0-24 ]  cpu

                pred = segment_postprocess.process_batch(pred).numpy()

                # 保存阶段 恢复
                pred = cv2.resize(pred,(w,h),interpolation=cv2.INTER_NEAREST)
                #print(pred.shape)

                # 保存 npy
                #np.save(os.path.join(args.save_path, nameext.split(".")[0]+".npy"), pred )

                # 输出没有分类 只有单输出
                # outputlist = [output[0], output[1], output[2], pred ]

                # ------------- 附加功能 --------------
                if args.extra_application:
                    save_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/Png_inference2/"+args.pre_train
                    os.makedirs(save_path,exist_ok=True)
                    # 要改
                    save_png_output(save_path, nameext,  rawinput ,class_label,  pred)

                # cv2.imwrite(os.path.join(args.save_path,nameext), mask_array,[cv2.IMWRITE_PNG_COMPRESSION,0])

                cur_time = starter.elapsed_time(ender)
                avg_time += cur_time
                print(info+" Image[{}/{}]  Name {} : {:.5f}ms".format(count, len(name_list),nameext, cur_time))
            count += 1
    return model


def save_png_output(save_path, name, image_array, label_tensor, output):
    segment_class_names = {0: "背景",
                                1: "S1肱骨大结节、肱骨小结节", 2: "S1肱二头肌长头腱短轴", 3: "三角肌", 4: "S2肱骨头表面",
                                5: "S2肱二头肌长头腱长轴", 6: "S3肩胛下肌腱长轴", 7: "S3肱骨小结节表面",
                                8: "S4肩胛下肌腱短轴", 9: "S4肱骨小结节表面", 10: "S5冈上肌腱长轴", 11: "S5肱骨大结节",
                                12: "S5肩峰下囊", 13: "S6肱二头肌长头肌腱短轴", 14: "S6冈和冈下肌腱短轴",
                                15: "S7冈下肌腱长轴", 16: "肱骨大结节", 17: "S8小圆肌长轴",
                                18: "S10锁骨", 19: "S10肩峰", 20: "S10肩锁关节囊", 21: "S11肩峰",
                                22: "S11喙突", 23: "S11-23-喙肩韧带"
                                }
    plane_class_name = ["S0:非标准切面", "S1:肱二头肌长头腱短轴切面", "S2:肱二头肌长头腱长轴切面",
                             "S3:肩胛下肌腱长轴切面", "S4:肩胛下肌腱短轴切面", "S5:冈上肌腱长轴切面",
                             "S6:冈上肌腱短轴切面", "S7:冈下肌腱长轴切面", "S8:小圆肌长轴切面", "S10:肩锁关节切面",
                             "S11:喙肩韧带切面"]

    # ----------------------分割结果 ----------------
    def _mask_to_color(mask):  # BGR
        VOC_COLORMAP = [
            [0, 0, 0], [0, 0, 255], [0, 255, 0], [128, 0, 0],  # 0-3
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],  # 4-7
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],  # 8-11
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],  # 12 -15
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],  # 16 -20
            [46, 139, 87], [210, 180, 140], [102, 205, 170], [233, 150, 22], [106, 90, 205],  # 21 -25
            [165, 42, 42], [147, 112, 219], [60, 179, 113], [218, 165, 32], [188, 143, 143],  # 26 -30
            [255, 105, 180], [255, 218, 185], [102, 205, 170]  # 31 -32
        ]
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)  # MASK 的尺寸是  H，W   这里面是H，W，3 的构建一个
        for label in np.unique(mask):
            if label < len(VOC_COLORMAP):
                # 对于每一个标签 的类 索引 ，如果对于输入的标签预测图，如果在预测索引是当前的标签的时候，将预测图的三通道设置为调色板的颜色
                color_mask[mask == label] = VOC_COLORMAP[label]

        return color_mask

    # ------------- 以上是局部函数自定义 -----------------------
    def _visualize_segmentation_fast(original_img, pred_mask, colored_label, alpha=0.5, background_class=0):
        """
        高效版本的语义分割可视化

        参数:
            original_img: 原始图像 (H,W,3) 或 (H,W)
            pred_mask : 预测标签 P模式
            colored_label: 彩色标签图 (H,W,3)
            alpha: 叠加透明度 (0-1)
            background_class: 背景类别索引 (默认为0)

        返回:
            overlay: 融合后的图像
        """
        # 确保原始图像是3通道
        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

        # 创建背景掩码（背景类为True，非背景为False）
        # 通过检查每个像素是否为背景类来创建掩码
        background_mask = np.all(colored_label == [0, 0, 0], axis=-1)  # 用单通道会导致 显示亮度降低
        # background_mask = np.all(pred_mask==0)

        if background_mask.all() == True:
            return original_img

        # 出图像创建输
        overlay = original_img.copy()

        # 只对非背景区域应用透明度混合
        # 使用掩码索引只处理前景区域
        try:
            overlay[~background_mask] = cv2.addWeighted(
                original_img[~background_mask],
                1 - alpha,
                colored_label[~background_mask],
                alpha,
                0
            )
        except TypeError:
            print(background_mask)
            print(original_img[~background_mask])
            print(colored_label[~background_mask])

        return overlay

    def _draw_legend(overlay_image, pred_mask, alpha, background_class=0):
        """
        在图像上绘制图例，支持透明颜色块和中文字符。

        参数:
            overlay_image (np.array): 已经叠加了分割掩码对应色调板叠加的图像。
            pred_mask (np.array): 预测的类别索引掩码。
            alpha (float): 用于图例颜色块的透明度，应与`_visualize_segmentation_fast`中的alpha一致。
            background_class (int): 不在图例中显示的背景类别索引。

        返回:
            np.array: 绘制了图例的最终图像。
        """

        VOC_COLORMAP = [
            [0, 0, 0], [0, 0, 255], [0, 255, 0], [128, 0, 0],  # 0-3
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],  # 4-7
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],  # 8-11
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],  # 12 -15
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],  # 16 -20
            [46, 139, 87], [210, 180, 140], [102, 205, 170], [233, 150, 22], [106, 90, 205],  # 21 -25
            [165, 42, 42], [147, 112, 219], [60, 179, 113], [218, 165, 32], [188, 143, 143],  # 26 -30
            [255, 105, 180], [255, 218, 185], [102, 205, 170]  # 31 -32
        ]

        # 1. 筛选出需要显示的类别
        unique_labels = sorted([label for label in np.unique(pred_mask) if label != background_class])
        if not unique_labels:
            return overlay_image

        # 复制一份图像用于绘制
        overlay_with_boxes = overlay_image.copy()

        # 图例参数
        legend_start_x, legend_start_y = 10, 50
        box_height, box_width = 20, 30
        spacing, font_size = 7, 16

        # 用来存储文本信息，后续统一用Pillow绘制
        text_to_draw = []

        # --- 步骤一: 使用OpenCV绘制所有【透明】的颜色方块 ---
        for i, label in enumerate(unique_labels):
            y_pos = legend_start_y + i * (box_height + spacing)

            # 获取颜色(BGR格式)和类名
            color_bgr = VOC_COLORMAP[label] if label < len(VOC_COLORMAP) else [128, 128, 128]
            class_name = segment_class_names.get(label, f"类 {label}")

            # 定义方块区域
            box_start_x, box_start_y = legend_start_x, y_pos
            box_end_x, box_end_y = legend_start_x + box_width, y_pos + box_height

            # 防止图例超出图像边界
            if box_end_y > overlay_with_boxes.shape[0] or box_end_x > overlay_with_boxes.shape[1]:
                continue

            # 核心：创建透明方块
            roi = overlay_with_boxes[box_start_y:box_end_y, box_start_x:box_end_x]  # 提取背景区域
            box_patch = np.full((box_height, box_width, 3), color_bgr, dtype=np.uint8)  # 创建纯色方块
            blended_roi = cv2.addWeighted(roi, 1 - alpha, box_patch, alpha, 0)  # 将方块与背景融合
            overlay_with_boxes[box_start_y:box_end_y, box_start_x:box_end_x] = blended_roi  # 放回原图

            # 记录文本位置和内容，供Pillow使用
            text_x = legend_start_x + box_width + 5
            text_y = y_pos + (box_height - font_size) // 2  # 垂直居中对齐
            text_to_draw.append({'text': class_name, 'pos': (text_x, text_y)})

        # --- 步骤二: 使用Pillow绘制所有【中文】文本 ---
        # 将OpenCV图像(BGR)转换为Pillow图像(RGB)
        pil_image = Image.fromarray(cv2.cvtColor(overlay_with_boxes, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # 加载支持中文的字体
        # **重要提示**: 请将 "simhei.ttf" 替换为您系统上实际存在的中文字体文件路径。
        # Windows: "C:/Windows/Fonts/simhei.ttf" (黑体) 或 "msyh.ttc" (微软雅黑)
        # macOS/Linux: 您可能需要先安装字体，然后提供其路径。
        try:
            font_path = "/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"警告: 中文字体 '{font_path}' 未找到。将使用默认字体，中文可能无法正常显示。")
            print("请提供一个有效的中文字体路径（例如 simhei.ttf）。")
            font = ImageFont.load_default()

        font_color_pil = (255, 255, 255)  # 白色

        # 统一绘制所有文本
        for item in text_to_draw:
            draw.text(item['pos'], item['text'], font=font, fill=font_color_pil)

        # --- 步骤三: 将Pillow图像转换回OpenCV图像(BGR) ---
        final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return final_image



    def _draw_legend2(overlay_image, pred_mask, alpha, background_class=0,
                     segment_class_names=None, min_component_area=50):
        """
        在每个分割区域中心显示类别文字（支持中文），不再绘制左上角图例。

        参数:
            overlay_image (np.array): 已经叠加了分割掩码对应色调板叠加的图像 (BGR uint8)。
            pred_mask (np.array): 预测的类别索引掩码 (H, W)。
            alpha (float): 保留以兼容接口（未在此函数中用于绘制方块）。
            background_class (int): 不显示的背景类别索引。
            segment_class_names (dict): 可选，类别索引->名称 映射。若为 None，则用 "类 {idx}"。
            min_component_area (int): 小连通区域的面积阈值，低于该值将被忽略（默认 50）。
        返回:
            np.array: 绘制了类名的最终图像 (BGR uint8)。
        """

        if segment_class_names is None:
            segment_class_names = {}

        h, w = pred_mask.shape[:2]
        # 复制，避免修改原图
        out_img = overlay_image.copy()

        # 获取需要显示的类别列表（去掉 background）
        unique_labels = sorted([int(label) for label in np.unique(pred_mask) if int(label) != background_class])
        if not unique_labels:
            return out_img

        # 为了支持中文渲染，用 Pillow 在 BGR->RGB 图像上绘制文字（支持描边）
        pil_img = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载中文字体（请根据实际路径调整）
        # Windows 示例: "C:/Windows/Fonts/simhei.ttf"
        # Linux/Mac: 填写你系统上存在的中文字体路径
        font_size = max(14, int(min(h, w) * 0.02))  # 字体大小按图像尺寸自适应，至少 14
        try:
            font_path = "/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            try:
                # 尝试常见 Windows 字体路径作为备选
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
            except Exception:
                # 兜底：默认字体（可能无法显示中文）
                print(f"警告: 未找到指定中文字体，使用默认字体，中文可能无法正确显示。")
                font = ImageFont.load_default()

        # 文本样式
        text_fill = (255, 255, 255)  # 白色文字
        stroke_fill = (0, 0, 0)  # 黑色描边
        stroke_width = max(1, font_size // 12)  # 描边宽度自适应

        # 对每个类别，查找连通组件并在每个组件中心写字
        for label in unique_labels:
            # 二值化该类别掩码
            mask = (pred_mask == label).astype(np.uint8)

            if mask.sum() == 0:
                continue

            # 使用 connectedComponentsWithStats 获取每个连通区域的质心和面积
            num_cc, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            # stats: (num_cc, 5) -> [x, y, width, height, area]
            # centroids: (num_cc, 2) -> (cx, cy)
            for cid in range(1, num_cc):  # 跳过背景 cc id = 0
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area < min_component_area:
                    continue

                cx, cy = centroids[cid]
                # 转为整数像素坐标
                text_x = int(cx)
                text_y = int(cy)

                # 获取类名
                class_name = segment_class_names.get(label, f"类 {label}")

                # 测量文本尺寸以便中心对齐
                text_size = draw.textsize(class_name, font=font)
                text_w, text_h = text_size

                # 计算左上角坐标，使文字以质心为中心
                draw_x = text_x - text_w // 2
                draw_y = text_y - text_h // 2

                # 防止越界
                draw_x = max(0, min(draw_x, w - text_w))
                draw_y = max(0, min(draw_y, h - text_h))

                # 绘制文本（带描边以提高可读性）
                # Pillow >= 5.2 支持 stroke_width/stroke_fill
                try:
                    draw.text((draw_x, draw_y), class_name, font=font,
                              fill=text_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
                except TypeError:
                    # 如果 Pillow 版本不支持 stroke 参数，先画黑字作为描边，然后白字覆盖
                    # 四周偏移作为简单描边替代
                    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for ox, oy in offsets:
                        draw.text((draw_x + ox, draw_y + oy), class_name, font=font, fill=stroke_fill)
                    draw.text((draw_x, draw_y), class_name, font=font, fill=text_fill)

        # 转回 OpenCV BGR
        final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return final_img

    # ---------------------------------------------------------
    # onehot  2 P mode label
    # pred = torch.argmax(output[3].squeeze(), dim=0).cpu()
    # 后处理
    pred = output
    # 类型转换
    image_array = np.array(image_array, dtype=uint8)
    # P模式 转换为 3通道的调色板图

    mask_color = _mask_to_color(pred)
    # 叠加到原图
    overlay = _visualize_segmentation_fast(image_array, pred, mask_color, alpha=0.5)
    # 添加 图例
    #overlay = _draw_legend(overlay, pred, alpha=0.5, background_class=0)
    overlay = _draw_legend(overlay, pred, alpha=0.5, background_class=0, segment_class_names=segment_class_names ) # 文字在 中心
    # ------------------------- save --------------
    png_save_path = os.path.join(save_path, "Png_Output")
    if not os.path.exists(png_save_path):
        os.makedirs(png_save_path)
    if name.find(".png") < 0:
        cv2.imwrite(png_save_path + '/' + name + '.png',  overlay , [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if name.find(".png") > 0:
        cv2.imwrite(png_save_path + '/' + name,  overlay , [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    import glob

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # 1. 指定模型
    args.model = "deeplabv3_channel"
    args.resume = 0
    args.cpu = False

    args.pre_train = "[ResNet50DeepLabV3-EX11-planeInput_yueshuattention]-[n20_shoulder_segment_1010_cop_class]-[2025-10-11-15-05]"  # "[NAFNet-BlEX12]-[Test_Dataset]-[2023-11-20-07-33]"#"[NAFNet-TEX10]-[Patch_1102]-[2023-11-17-09-34]"


    input_path_list = ["S1","S2","S3","S4","S5","S6","S7","S8","S10","S11"]
    #save_path_list = []
    # 切面层
    for i in range(len(input_path_list)):
        class_label = torch.tensor(i+1)
        # 3.，IMAGE-S1 里面还有病例文件夹层
        input_folder= r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/trainall/image/"+input_path_list[i]

        # 4. 指定保存路径
        # " # 串联模型的顺序从左到右侧，左就是数据的来源，右就是模型
        save_folder = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/trainall/label_cnn2/"+input_path_list[i]

        path_name = os.listdir(input_folder)

        # 病例层
        for  j in range(len(path_name)):

            args.input_path = os.path.join(input_folder,path_name[j])
            args.save_path = os.path.join(save_folder,path_name[j])

        # 5. 运行
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            info = f"plane {i+1} / {len(input_path_list)} case {j+1}/ {len(path_name)}"

            model = inference_multi_image(args,info,class_label)  # 这里一定是 一个病例的  可以放心的用

        #


















