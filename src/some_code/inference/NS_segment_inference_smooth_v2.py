'''

这一个代码是用于 单输入的推理 ，个用来分割
尝试先 上采样 == 帧间平滑 == 形态学
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
import torch.nn.functional as F

from src.train.utility import utility
import glob
import re
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from PIL import Image, ImageDraw, ImageFont
from numpy import uint8
import torch.nn as nn

from skimage.metrics import peak_signal_noise_ratio #as psnr

import  warnings
warnings.filterwarnings('ignore')


project_path  = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
print(project_path)

def get_args():
    parser = argparse.ArgumentParser(description='Thermal and Rail SR')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    #    ---------------------   分类模型  -----------------------------------
    parser.add_argument('--model_class', default='Unet',
                        help='model name')
    parser.add_argument('--pre_train_class', type=str, default=None,
                        help='pre-trained model directory')
    #  ---------------------   分割 模型  -----------------------------------
    parser.add_argument('--model_segment', default='Unet',
                        help='model name')
    parser.add_argument('--pre_train_segment', type=str, default=None,
                        help='pre-trained model directory')
    # -=-----------------------------------------------
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

    parser.add_argument('--inference_path', type=str, default=None,
                        help='inference result save  directory')
    parser.add_argument('--sw_mode', type=str, default="S",
                        help='save_path ')
    parser.add_argument('--resume', type=int, default=0,
                        help='control the load model is best or lastest or other ')
    parser.add_argument('--input_size', type=int, default=512,
                        help='control the input data crop patch (592,720), if patch_size != None it will be crop shuffle 作为正方形')
    parser.add_argument('--num_class', type=int, default=24,
                        help='segment  num_class ')

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

def upsample_pred_onehot_torch(pred_idx, num_classes, out_size, device='cpu', mode='bicubic'):
    """
    将类别索引 -> one-hot -> 在概率空间上插值 -> argmax -> 返回上采样后的类别图
    输入:
        pred_idx: np.ndarray or torch.Tensor, shape (H0, W0), dtype int (0..C-1)
        num_classes: int
        out_size: (H, W) 目标尺寸
        device: 'cpu' 或 'cuda'（如果有 GPU）
        mode: 'bilinear' 或 'bicubic' (注意 bicubic 可能需要 torch>=1.8)
    返回:
        pred_up: np.ndarray, shape (H, W), dtype=np.uint8
    """
    import torch
    import torch.nn.functional as F

    # 转为 tensor
    if isinstance(pred_idx, np.ndarray):
        pred_t = torch.from_numpy(pred_idx).long().to(device)
    else:
        pred_t = pred_idx.long().to(device)

    H_out, W_out = out_size

    # one-hot: (H0, W0) -> (C, H0, W0)
    one_hot = F.one_hot(pred_t, num_classes=num_classes)  # (H0, W0, C)
    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H0, W0)

    # 插值（在概率/响应图上）
    # align_corners: 对于 bilinear，一般使用 align_corners=False
    try:
        one_hot_up = F.interpolate(one_hot, size=(H_out, W_out), mode=mode, align_corners=False)
    except TypeError:
        # 某些 mode/bicubic 对 align_corners 参数不支持，直接不传
        one_hot_up = F.interpolate(one_hot, size=(H_out, W_out), mode=mode)

    # argmax 得到离散标签
    pred_up = one_hot_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_up

import numpy as np
import cv2

def upsample_pred_onehot_torch_with_morph(
    pred_idx, num_classes, out_size, device='cpu', mode='bicubic',
    morph_kernel=3, morph_iterations=1, min_component_area=50,
    merge_small=True
):
    """
    将类别索引 -> one-hot -> 在概率空间上插值 -> argmax -> 形态学开闭 + 小连通域合并/移除 -> 返回上采样后的类别图

    参数:
        pred_idx: np.ndarray or torch.Tensor, shape (H0, W0), dtype int (0..C-1)
        num_classes: int
        out_size: (H, W) 目标尺寸
        device: 'cpu' 或 'cuda'
        mode: 'bilinear' or 'bicubic'
        morph_kernel: int, 形态学内核尺寸（奇数），较大值平滑更多
        morph_iterations: int, 形态学迭代次数
        min_component_area: int, 面积低于该值的小连通域将被合并或移除
        merge_small: bool, True -> 将小连通域合并到邻接最多的类别；False -> 设为背景(0)
    返回:
        pred_final: np.ndarray, shape (H, W), dtype=np.uint8
    """

    # ---- 上采样部分 ----
    import torch
    import torch.nn.functional as F

    # 转为 tensor
    if isinstance(pred_idx, np.ndarray):
        pred_t = torch.from_numpy(pred_idx).long().to(device)
    else:
        pred_t = pred_idx.long().to(device)

    H_out, W_out = out_size

    # one-hot: (H0, W0) -> (C, H0, W0)
    one_hot = F.one_hot(pred_t, num_classes=num_classes)  # (H0, W0, C)
    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H0, W0)

    # 插值（在概率/响应图上）
    try:
        one_hot_up = F.interpolate(one_hot, size=(H_out, W_out), mode=mode, align_corners=False)
    except TypeError:
        one_hot_up = F.interpolate(one_hot, size=(H_out, W_out), mode=mode)

    # argmax 得到离散标签
    pred_up = one_hot_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H_out, W_out)

    # ---- 形态学平滑（按类进行 opening -> closing） ----
    kernel_size = max(1, int(morph_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    H, W = pred_up.shape
    smoothed = np.zeros_like(pred_up, dtype=np.uint8)

    # 对每个类别分别进行形态学处理，然后在最后合并（保证互斥）
    for c in range(num_classes):
        mask_c = (pred_up == c).astype(np.uint8) * 255  # 0/255
        if mask_c.sum() == 0:
            continue

        # opening -> closing
        if morph_iterations > 0:
            m = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        else:
            m = mask_c

        # 转回 0/1
        m = (m > 127).astype(np.uint8)
        # 把该类写回 smoothed（后面会处理重叠/空隙）
        smoothed[m == 1] = c

    # 注意：单独处理每个类可能会造成边界上短暂的空白或小重叠（不大可能因二值化），
    # 但我们已经以类为单位重写，最后 smoothed 覆盖所有被保留像素。

    # ---- 处理小连通域：移除或合并到邻近类 ----
    final = smoothed.copy()

    # 对每个类别处理小连通域
    for c in range(num_classes):
        mask_c = (final == c).astype(np.uint8)
        if mask_c.sum() == 0:
            continue

        n_cc, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(mask_c, connectivity=8)
        for cid in range(1, n_cc):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area >= min_component_area:
                continue  # 合格，保留

            # 小区域：找其像素坐标
            comp_mask = (labels_cc == cid)
            # 默认行为：合并到邻接最多的类别（在 merge_small=True 时）
            if merge_small:
                # 扩张该小区域一个膨胀半径以包含邻域像素
                # 使用膨胀再投票的方式选择邻近类别
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                comp_mask_uint8 = (comp_mask.astype(np.uint8) * 255)
                comp_dilated = cv2.dilate(comp_mask_uint8, dilate_kernel, iterations=3)  # iterations 可调
                comp_dilated = (comp_dilated > 0)

                # 在膨胀区域内统计最终标签分布（排除自身像素）
                neigh_region = final.copy()
                neigh_region[comp_mask] = 255  # 临时标记自身为 255 （排除）
                # 搜索膨胀区域内的像素，统计频率（排除255）
                neigh_vals = neigh_region[comp_dilated & (~comp_mask)]
                # 如果没有邻居（极少见），直接设为背景 0
                if neigh_vals.size == 0:
                    replacement_label = 0
                else:
                    # 统计最常见的标签作为合并目标
                    replacement_label = int(np.bincount(neigh_vals[neigh_vals != 255]).argmax())

                # 替换小连通域为 replacement_label
                final[comp_mask] = replacement_label
            else:
                # 直接设为背景（0）
                final[comp_mask] = 0

    # ---- 可选：再次进行一次轻微的形态学平滑，消除合并后的小锯齿 ----
    # 这里做一次轻度 close 操作，防止合并后出现单像素缝隙
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_masks = np.zeros_like(final, dtype=np.uint8)
    for c in range(num_classes):
        mask_c = (final == c).astype(np.uint8) * 255
        if mask_c.sum() == 0:
            continue
        m = cv2.morphologyEx(mask_c, cv2.MORPH_CLOSE, kernel2, iterations=1)
        final_masks[(m > 127)] = c

    return final_masks


def inference_multi_image(args,info,model_segment):
    segment_postprocess = utility.ArgmaxPostProcessor()

    # ============================ 加载分割模型   =============================
    if model_segment is None:
        module = import_module('src.model.' + args.model_segment.lower())
        args.num_class = 2
        model_segment = module.make_model(args)
        # -----------------------------  预训练参数路径  -------------------------
        if args.pre_train_segment:
            pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train_segment, "model")
            model_segment.load_state_dict(torch.load(pre_train_model_path + f"/model_{control_pretrain(args.resume)}.pt"),
                                        strict=False)
        model_segment = model_segment.cuda().eval()
        print("[----> Model_segment inference in ", next(model_segment.parameters()).device, "<----]")

    # =================== 加载文件夹内输入数据 并排序 ==============================
    name_list = os.listdir(args.input_path)

    # --------------------三星采集  的 按照 这个-----------------------------
    namelist = sorted(name_list, key=lambda x: (int(re.search(r'(\d+)_(\d+)_(\d+)\.png$', x).group(1)),
                                               int(re.search(r'(\d+)_(\d+)_(\d+)\.png$', x).group(2)),
                                               int(re.search(r'(\d+)_(\d+)_(\d+)\.png$', x).group(3))))

    # ----- -------------------划分 实例 ----------------------------------
    file_idx = lambda x: (int(re.search(r'(\d+)_(\d+)_(\d+)\.png$', x).group(1)),
                          int(re.search(r'(\d+)_(\d+)_(\d+)\.png$', x).group(2)))
    last_file_idx = None
    instance_list = []
    all_instance = []

    for i in range(len(namelist)):

        if last_file_idx is None:
            last_file_idx = file_idx(namelist[i])
            instance_list.append(namelist[i])

        else:
            if file_idx(namelist[i]) == last_file_idx:
                instance_list.append(namelist[i])
            else:
                all_instance.append(instance_list)
                instance_list =[ ]
                last_file_idx = file_idx(namelist[i])
                instance_list.append(namelist[i])
    print(all_instance)
    # ----- -------------------划分 实例 ----------------------------------
    count = 1
    # ---------------
    last_input = None
    last_output = None

    layer = nn.Softmax()
    # 如果是直接的训推理 则不能在进行这个了
    for d in range(len(all_instance)):
        for nameext in all_instance[d]:
            # -----------------输入数据路径与文件名分离-----------------------
            input_data = os.path.join(args.input_path, nameext)
            # -------------------数据读取与预处理--------------------
            rawinput = cv2.imread(input_data, 0)   # [0-255 ] [H,W]  array   cpu
            h,w = rawinput.shape    # [0-255 ] [H,W]
            resizeinput = cv2.resize(rawinput,(args.input_size,args.input_size),interpolation=cv2.INTER_CUBIC)    # [0-255 ] [args.inputsize  ,args.inputsize ]   array   cpu
            input = torch.tensor( resizeinput).unsqueeze(0).unsqueeze(0) / 255 # [0-1 ] [1,1 , args.inputsize  ,args.inputsize ]   tensor   cpu
            INPUT = input.cuda()   # [0-1 ] [1,1 , args.inputsize  ,args.inputsize ]   tensor  gpu

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # ====================模型前向处理-===================
            avg_time = 0
            prev_logits_ema = None
            with (torch.no_grad()):
                starter.record()

                segment_output = model_segment(INPUT)
                # -----------------1、 概率空间上采样 -------------
                segment_output = torch.softmax(segment_output, dim=1)
                resize_output = F.interpolate(segment_output, size=(h, w), mode='bicubic', align_corners=False)

                # ---------------- 切面多帧处理 -------------------

                if prev_logits_ema is None:
                    logits_smooth = resize_output
                else:
                    logits_smooth = segment_postprocess.temporal_smoothing_v2(prev_logits_ema,resize_output,args.num_class)
                # prev_logits_ema = segment_output
                prev_logits_ema = resize_output

                # ======================== 结果保存 -==========================
                ender.record()
                torch.cuda.synchronize()
                pred = torch.argmax(logits_smooth.squeeze(0), dim=0).cpu() # [512,512] ,[0-24 ]  cpu
                pred = segment_postprocess.process_batch(pred).numpy()

                # ------------------保存阶段 为标签 ------------------------
                np.save(os.path.join(args.save_path, nameext.split(".")[0]+".npy"), pred ) # 不保存

                outputlist = pred

                # ------------- 附加功能 可视化结果  --------------
                if args.extra_application:
                    save_path = args.inference_path
                    os.makedirs(save_path,exist_ok=True)
                    save_png_output(save_path, nameext,  rawinput , outputlist)

                # cv2.imwrite(os.path.join(args.save_path,nameext), mask_array,[cv2.IMWRITE_PNG_COMPRESSION,0])

                cur_time = starter.elapsed_time(ender)
                avg_time += cur_time
                print(info+" Image[{}/{}] : {:.5f}ms".format(count, len(name_list), cur_time))
            count += 1
    return model_segment


def save_png_output(save_path, name, image_array, output):
    def _mask_to_color(mask):
        VOC_COLORMAP = [[0, 0, 0], [0, 255, 255], [0, 0, 255], [255, 0, 0], [128, 128, 0], [255, 128, 128]]
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
            overlay_image (np.array): 已经叠加了分割掩码的图像。
            pred_mask (np.array): 预测的类别索引掩码。
            alpha (float): 用于图例颜色块的透明度，应与`_visualize_segmentation_fast`中的alpha一致。
            background_class (int): 不在图例中显示的背景类别索引。

        返回:
            np.array: 绘制了图例的最终图像。
        """

        VOC_COLORMAP = [[0, 0, 0], [0, 255, 255], [0, 0, 255], [255, 0, 0], [128, 128, 0], [255, 128, 128]]
        CLASS_NAMES = {0: "Background",
                       1: "Nerve",
                       2: "Muscle",
                       3: "Artery",
                       4: "Vein"}

        # 1. 筛选出需要显示的类别
        unique_labels = sorted([label for label in np.unique(pred_mask) if label != background_class])
        if not unique_labels:
            return overlay_image

        # 复制一份图像用于绘制
        overlay_with_boxes = overlay_image.copy()

        # 图例参数
        legend_start_x, legend_start_y = 10, 10
        box_height, box_width = 20, 30
        spacing, font_size = 7, 16

        # 用来存储文本信息，后续统一用Pillow绘制
        text_to_draw = []

        # --- 步骤一: 使用OpenCV绘制所有【透明】的颜色方块 ---
        for i, label in enumerate(unique_labels):
            y_pos = legend_start_y + i * (box_height + spacing)

            # 获取颜色(BGR格式)和类名
            color_bgr = VOC_COLORMAP[label] if label < len(VOC_COLORMAP) else [128, 128, 128]
            class_name = CLASS_NAMES.get(label, f"类 {label}")

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

    def _draw_bounding_boxes(overlay_image, pred_mask, min_area=500, thickness=2,
                             draw_only_max_box=False, background_class=0):
        """
        在 overlay_image 上为每个预测类别绘制边界框（并用 Pillow 写中文类名）。
        参数:
            overlay_image: BGR numpy array
            pred_mask: 2D numpy array (H, W) 的标签图  （0,num_class ）
            min_area: 忽略小于该像素面积的连通域（噪声）
            thickness: 框线粗细
            draw_only_max_box: 对每个类别只绘制面积最大的bbox（True）或全部（False）
            background_class: 背景标签索引
        """

        if overlay_image.ndim == 2:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2BGR)

        img_out = overlay_image.copy()
        # 定义颜色表 & 类名（BGR）
        VOC_COLORMAP = [[0, 0, 0], (0, 255, 255), (0, 0, 255), (255, 0, 0), (128, 128, 0), (255, 128, 128)]
        CLASS_NAMES = {0: "Background",
                       1: "Nerve",
                       2: "Muscle",
                       3: "Artery",
                       4: "Vein"}

        unique_labels = sorted(
            [int(l) for l in np.unique(pred_mask) if int(l) != background_class])  # 分析mask ，分析出所有的类的索引 ， 而且排除背景
        if not unique_labels:  # 如果是空 ，也就是 0 只有背景类
            return img_out

        text_tasks = []
        legend_line_height = 25  # 每行文字的高度

        for label in unique_labels:
            bin_mask = (pred_mask == label).astype(np.uint8) * 255  # 所有的这个区域等于对应标签之后 ，放在255
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 返回
            if not contours:
                continue

            # # 2. 筛选面积并排序 (关键步骤：按面积从大到小排序)
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    valid_contours.append((cnt, area))

            # 按面积降序排列: list of (cnt, area)
            valid_contours.sort(key=lambda x: x[1], reverse=True)

            # 如果只画最大的，就切片取第一个
            if draw_only_max_box and valid_contours:
                valid_contours = valid_contours[:1]

            if not valid_contours:
                continue

            # 获取颜色
            color_bgr = VOC_COLORMAP[label] if label < len(VOC_COLORMAP) else [128, 128, 128]

            # print(color_bgr)
            color_rgb = tuple(color_bgr[::-1])  # PIL 需要 RGB

            boxes = []
            for idx, (cnt, area) in enumerate(valid_contours):
                instance_id = idx + 1  # 1, 2, 3...
                class_name = CLASS_NAMES.get(label, f"Class{label}")
                instance_name = f"{class_name}_{instance_id}"  # 例如 Nerve_1

                # 计算边界框  以及 绘制边界框
                x, y, w, h = cv2.boundingRect(cnt)  # 返回了轮廓的左上角的边界以及轮廓的宽和高
                cv2.rectangle(img_out, (x, y), (x + w, y + h), color_bgr, thickness)
                # --- 收集文字任务：局部标签 (轮廓旁) ---
                x, y, w, h = cv2.boundingRect(cnt)
                text_tasks.append({
                    'type': 'local',
                    'text': instance_name,
                    'pos': (x, max(0, y - 20)),  # 放在轮廓上方
                    'color': color_rgb,  # 局部标签用白色字
                })

            if not boxes:
                continue

        #  文本绘制
        pil_img = Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 字体设置
        try:
            font_path = "/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf"
            font_local = ImageFont.truetype(font_path, 14)  # 局部标签字体
            font_legend = ImageFont.truetype(font_path, 16)  # 列表字体稍大
        except IOError:
            font_local = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        # 为了防止左上角列表看不清，可以先在左上角画一个半透明黑底背景 (可选)
        # 计算列表区域的总高度
        # if legend_start_y > 10:
        #    draw.rectangle([(5, 5), (250, legend_start_y + 5)], fill=(0, 0, 0, 120))  # RGBA, A=120半透明

        for task in text_tasks:
            text = task['text']
            x, y = task['pos']
            color = task['color']

            if task['type'] == 'local':
                # 绘制局部标签 (带背景框)
                # 获取文字大小
                try:
                    bbox = draw.textbbox((0, 0), text, font=font_local)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    w, h = draw.textsize(text, font=font_local)

                # 画文字
                draw.text((x + 2, y + 2), text, font=font_local, fill=color)

            elif task['type'] == 'legend':
                # 绘制左上角列表 (无背景框，或者上面已经统一画了大背景)
                # 这里让字体颜色 = 类别颜色，方便区分
                draw.text((x, y), text, font=font_legend, fill=color)
        # 转回 OpenCV
        img_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return img_out

    def _draw_contours_with_ordered_legend(overlay_image, pred_mask, min_area=500, thickness=2,
                                           draw_only_max_box=False, background_class=0):
        """
        1. 在 overlay_image 上绘制预测 mask 的轮廓。
        2. 对同类别的多个实例按面积从大到小排序并编号 (如 Nerve_1, Nerve_2)。
        3. 图片左上角显示详细信息列表 (Nerve_1 : Area 1000 px)。
        4. Mask 轮廓左上角仅标注编号名称 (Nerve_1)。
        """
        piex2mm = 0.05

        img_out = overlay_image.copy()

        if img_out.ndim == 2:
            img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
            # print(img_out.shape)

        # 颜色表 & 类名
        VOC_COLORMAP = [(0, 0, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0), (128, 128, 0), (255, 128, 128)]
        CLASS_NAMES = {0: "Background",
                       1: "Nerve",
                       2: "Muscle",
                       3: "Artery",
                       4: "Vein"}

        unique_labels = sorted([int(l) for l in np.unique(pred_mask) if int(l) != background_class])

        if not unique_labels:
            return img_out

        # 用于存储后续需要绘制的文字任务，避免频繁转换 PIL/OpenCV
        # 格式: {'text': str, 'pos': (x, y), 'color': (r, g, b), 'type': 'local'/'legend'}
        text_tasks = []

        # 左上角列表的起始 Y 坐标
        legend_start_y = 10
        legend_line_height = 25  # 每行文字的高度

        for label in unique_labels:
            # 1. 提取 Mask 并找轮廓
            bin_mask = (pred_mask == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            # 2. 筛选面积并排序 (关键步骤：按面积从大到小排序)
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    valid_contours.append((cnt, area))

            # 按面积降序排列: list of (cnt, area)
            valid_contours.sort(key=lambda x: x[1], reverse=True)

            # 如果只画最大的，就切片取第一个
            if draw_only_max_box and valid_contours:
                valid_contours = valid_contours[:1]

            if not valid_contours:
                continue

            # 获取颜色
            color_bgr = VOC_COLORMAP[label] if label < len(VOC_COLORMAP) else [128, 128, 128]

            # print(color_bgr)
            color_rgb = tuple(color_bgr[::-1])  # PIL 需要 RGB

            # 3. 遍历该类的所有实例
            for idx, (cnt, area) in enumerate(valid_contours):
                instance_id = idx + 1  # 1, 2, 3...
                class_name = CLASS_NAMES.get(label, f"Class{label}")
                instance_name = f"{class_name}_{instance_id}"  # 例如 Nerve_1

                # --- OpenCV 绘制轮廓 ---
                cv2.drawContours(img_out, [cnt], -1, color=color_bgr, thickness=thickness)

                # --- 收集文字任务：局部标签 (轮廓旁) ---
                x, y, w, h = cv2.boundingRect(cnt)
                text_tasks.append({
                    'type': 'local',
                    'text': instance_name,
                    'pos': (x, max(0, y - 20)),  # 放在轮廓上方
                    'color': color_rgb,  # 局部标签用白色字
                })

                # --- 收集文字任务：全局列表 (左上角) ---

                legend_text = f"{instance_name} : Area {area * piex2mm * piex2mm:.4f} mm²"
                text_tasks.append({
                    'type': 'legend',
                    'text': legend_text,
                    'pos': (10, legend_start_y),
                    'color': color_rgb,  # 列表文字颜色与Mask颜色一致，方便对应
                })

                # 更新列表的 Y 坐标，为下一行腾出空间
                legend_start_y += legend_line_height

        # --- 4. 统一转 PIL 绘制所有文字 ---
        pil_img = Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 字体设置
        try:
            font_path = "/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf"
            font_local = ImageFont.truetype(font_path, 14)  # 局部标签字体
            font_legend = ImageFont.truetype(font_path, 16)  # 列表字体稍大
        except IOError:
            font_local = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        # 为了防止左上角列表看不清，可以先在左上角画一个半透明黑底背景 (可选)
        # 计算列表区域的总高度
        # if legend_start_y > 10:
        #    draw.rectangle([(5, 5), (250, legend_start_y + 5)], fill=(0, 0, 0, 120))  # RGBA, A=120半透明

        for task in text_tasks:
            text = task['text']
            x, y = task['pos']
            color = task['color']

            if task['type'] == 'local':
                # 绘制局部标签 (带背景框)
                # 获取文字大小
                try:
                    bbox = draw.textbbox((0, 0), text, font=font_local)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    w, h = draw.textsize(text, font=font_local)

                # 画文字背景框
                # draw.rectangle([(x, y), (x + w + 4, y + h + 4)], fill=(0, 0, 0))
                # 画文字
                draw.text((x + 2, y + 2), text, font=font_local, fill=color)

            elif task['type'] == 'legend':
                # 绘制左上角列表 (无背景框，或者上面已经统一画了大背景)
                # 这里让字体颜色 = 类别颜色，方便区分
                draw.text((x, y), text, font=font_legend, fill=color)

        # 转回 OpenCV
        img_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return img_out

    def _draw_ellipse_axes(overlay_image, pred_mask, min_area_for_ellipse=300,
                           ellipse_only_max_contour=False, background_class=0, max_rows_in_legend=10, thickness=2):
        """
        overlay_image ： 将用来绘制的图像
        对每个类别的连通区域（或每类最大连通域）进行椭圆拟合，绘制椭圆与长短轴线段，
        并在图左上角绘制每个类/区域的 major_px, minor_px, ratio。
        返回： (图像（BGR）, 用于左上角文本的行列表)
        """

        def draw_dashed_line(img, p1, p2, color, thickness=1, gap=5):
            pt1 = np.array(p1)
            pt2 = np.array(p2)
            dist = np.linalg.norm(pt2 - pt1)
            if dist < gap:
                cv2.line(img, tuple(p1), tuple(p2), color, thickness)
                return
            segments = int(dist // gap)
            for i in range(segments):
                # 简单的虚线算法：画一段，空一段
                start_t = i / segments
                end_t = (i + 0.5) / segments
                curr_p1 = pt1 + (pt2 - pt1) * start_t
                curr_p2 = pt1 + (pt2 - pt1) * end_t

                cp1 = (int(round(curr_p1[0])), int(round(curr_p1[1])))
                cp2 = (int(round(curr_p2[0])), int(round(curr_p2[1])))
                cv2.line(img, cp1, cp2, color, thickness)

        piex2mm = 0.05

        img_out = overlay_image.copy()
        if img_out.ndim == 2:  # 这里注意 ，判断维度用的 的len(img.shape) == 2 或者 img.dim ==2 ，但是不能用 len(img)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

        H, W = img_out.shape[:2]

        VOC_COLORMAP = [[0, 0, 0], (0, 255, 255), (0, 0, 255), (255, 0, 0), (128, 128, 0), (255, 128, 128)]
        CLASS_NAMES = {0: "Background",
                       1: "Nerve",
                       2: "Muscle",
                       3: "Artery",
                       4: "Vein"}

        text_tasks = []  # 用来在左上角绘制的文本行

        unique_labels = sorted([int(l) for l in np.unique(pred_mask) if int(l) != background_class])
        if not unique_labels:
            return img_out

        legend_start_y = 10
        legend_line_height = 25  # 每行文字的高度

        # 轮廓
        for label in unique_labels:
            # 1、计算 轮廓
            bin_mask = (pred_mask == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # 2. 筛选面积并排序 (关键步骤：按面积从大到小排序)
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area_for_ellipse:
                    valid_contours.append((cnt, area))

            # 按面积降序排列: list of (cnt, area)
            valid_contours.sort(key=lambda x: x[1], reverse=True)

            # 如果只画最大的，就切片取第一个
            if ellipse_only_max_contour and valid_contours:
                valid_contours = valid_contours[:1]

            if not valid_contours:
                continue

            # 获取颜色
            color_bgr = VOC_COLORMAP[label] if label < len(VOC_COLORMAP) else [128, 128, 128]
            color_rgb = color_bgr[::-1]  # PIL 需要 RGB

            # 对每一个 contour 尝试拟合椭圆（但需过滤面积）
            for idx, (cnt, area) in enumerate(valid_contours):

                instance_id = idx + 1  # 1, 2, 3...
                class_name = CLASS_NAMES.get(label, f"Class{label}")
                instance_name = f"{class_name}_{instance_id}"  # 例如 Nerve_1

                # 尝试用 fitEllipse（需要至少 5 个点）
                fitted = False
                axis_w = axis_h = angle_deg = None
                try:
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)  # ((cx,cy),(axis_w,axis_h),angle)
                        (cx, cy), (axis_w, axis_h), angle_deg = ellipse
                        fitted = True
                    else:
                        fitted = False
                except Exception:
                    fitted = False

                # 如果 fitEllipse 失败，退回用 minAreaRect 或 boundingRect 来估计
                if not fitted:
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (w_rect, h_rect), angle_rect = rect
                    axis_w, axis_h, angle_deg = w_rect, h_rect, angle_rect
                    # 若 minAreaRect 返回 0，退回 boundingRect
                    if (axis_w == 0 or axis_h == 0) or np.isnan(axis_w) or np.isnan(axis_h):
                        x, y, w_b, h_b = cv2.boundingRect(cnt)
                        axis_w, axis_h = w_b, h_b
                        cx = x + w_b / 2.0
                        cy = y + h_b / 2.0
                        angle_deg = 0.0

                # 确保数值是浮点
                cx = float(cx)
                cy = float(cy)
                axis_w = float(axis_w)
                axis_h = float(axis_h)
                angle_deg = float(angle_deg)

                # 绘制椭圆（注意：cv2.ellipse 接受半轴长度）
                half_w = axis_w / 2.0
                half_h = axis_h / 2.0
                # color_bgr = VOC_COLORMAP[label] if label < len(VOC_COLORMAP) else [128, 128, 128]

                # cv2.ellipse(
                #     img_out,
                #     (int(round(cx)), int(round(cy))),
                #     (int(round(half_w)), int(round(half_h))),
                #     angle_deg, 0, 360,
                #     color_bgr,
                #     2
                # )
                # 计算两个轴端点（基于 OpenCV 的语义：angle 对应 axis_w 的方向）
                theta = np.deg2rad(angle_deg)

                # axis_w 对应方向向量 A（从中心到正端点）
                dx_a = np.cos(theta) * half_w
                dy_a = np.sin(theta) * half_w

                # axis_h 对应方向向量 B（为 A 旋转 +90deg）
                dx_b = np.cos(theta + np.pi / 2) * half_h
                dy_b = np.sin(theta + np.pi / 2) * half_h

                A_p1 = (int(round(cx - dx_a)), int(round(cy - dy_a)))
                A_p2 = (int(round(cx + dx_a)), int(round(cy + dy_a)))
                B_p1 = (int(round(cx - dx_b)), int(round(cy - dy_b)))
                B_p2 = (int(round(cx + dx_b)), int(round(cy + dy_b)))

                # 根据 axis_w/axis_h 大小决定哪条是长轴（major）哪条是短轴（minor）
                if axis_w >= axis_h:
                    major_p1, major_p2 = A_p1, A_p2
                    minor_p1, minor_p2 = B_p1, B_p2
                    major_len = axis_w
                    minor_len = axis_h
                else:
                    major_p1, major_p2 = B_p1, B_p2
                    minor_p1, minor_p2 = A_p1, A_p2
                    major_len = axis_h
                    minor_len = axis_w

                # 绘制：长轴 红色，短轴 绿色，中心点 标记为白色小圆

                # draw_dashed_line(img_out, major_p1[0], major_p1[1], (0, 0, 255), 2, 5)  # 红虚线
                # draw_dashed_line(img_out, minor_p1[0], minor_p1[1], (0, 255, 0), 2, 5)  # 绿虚线

                cv2.line(img_out, major_p1, major_p2, color_bgr, thickness=thickness)
                cv2.line(img_out, minor_p1, minor_p2, color_bgr, thickness=thickness)
                cv2.circle(img_out, (int(round(cx)), int(round(cy))), 3, (255, 255, 255), -1)

                # 绘制 长短轴两端绘制 "×"
                cv2.drawMarker(img_out, major_p1, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, thickness=thickness)
                cv2.drawMarker(img_out, major_p2, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, thickness=thickness)

                cv2.drawMarker(img_out, minor_p1, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 8, thickness=thickness)
                cv2.drawMarker(img_out, minor_p2, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 8, thickness=thickness)

                # 将该 contour 的信息加入左上角文本行
                x, y, w, h = cv2.boundingRect(cnt)

                text_tasks.append({
                    'type': 'local',
                    'text': instance_name,
                    'pos': (x, max(0, y - 20)),  # 放在轮廓上方
                    'color': color_rgb,  # 局部标签用白色字
                })

                major_px = float(major_len) * piex2mm
                minor_px = float(minor_len) * piex2mm if minor_len > 0 else 1.0
                ratio = major_px / minor_px if minor_px != 0 else float('inf')

                # --- 收集文字任务：全局列表 (左上角) ---
                legend_text = f"{instance_name} : major={major_px:.4f} mm  minor={minor_px:.4f}mm  ratio={ratio:.4f}  angle={angle_deg:.1f}°"
                text_tasks.append({
                    'type': 'legend',
                    'text': legend_text,
                    'pos': (10, legend_start_y),
                    'color': color_rgb,  # 列表文字颜色与Mask颜色一致，方便对应
                })
                legend_start_y += legend_line_height

        # 在图像左上角绘制这些行（Pillow 绘图，支持中文）
        pil_img = Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 字体设置
        try:
            font_path = "/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf"
            font_local = ImageFont.truetype(font_path, 14)  # 局部标签字体
            font_legend = ImageFont.truetype(font_path, 16)  # 列表字体稍大
        except IOError:
            font_local = ImageFont.load_default()
            font_legend = ImageFont.load_default()

        # 为了防止左上角列表看不清，可以先在左上角画一个半透明黑底背景 (可选)
        # 计算列表区域的总高度
        # if legend_start_y > 10:
        #    draw.rectangle([(5, 5), (250, legend_start_y + 5)], fill=(0, 0, 0, 120))  # RGBA, A=120半透明

        for task in text_tasks:
            text = task['text']
            x, y = task['pos']
            color = task['color']

            if task['type'] == 'local':
                # 绘制局部标签 (带背景框)
                # 获取文字大小
                try:
                    bbox = draw.textbbox((0, 0), text, font=font_local)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    w, h = draw.textsize(text, font=font_local)

                # 画文字背景框
                # draw.rectangle([(x, y), (x + w + 4, y + h + 4)], fill=(0, 0, 0))
                # 画文字
                draw.text((x + 2, y + 2), text, font=font_local, fill=color)

            elif task['type'] == 'legend':
                # 绘制左上角列表 (无背景框，或者上面已经统一画了大背景)
                # 这里让字体颜色 = 类别颜色，方便区分
                draw.text((x, y), text, font=font_legend, fill=color)

        # 转回 OpenCV
        img_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return img_out

        # -------------------------以上用来可视化  --------------------------------

    # onehot  2 P mode label

    pred = output

    # 类型转换
    image_array = np.array(image_array, dtype=uint8)
    # P模式 转换为 3通道的调色板图
    mask_color = _mask_to_color(pred)
    # 叠加到原图  mask + image
    overlay1 = _visualize_segmentation_fast(image_array, pred, mask_color, alpha=0.5)
    # add new  bounding box
    overlay2 = _draw_bounding_boxes(image_array, pred, min_area=100, thickness=2,
                                    draw_only_max_box=False, background_class=0)
    overlay3 = _draw_ellipse_axes(image_array, pred, min_area_for_ellipse=100, ellipse_only_max_contour=False,
                                  background_class=0, max_rows_in_legend=10)
    # 绘制节段轮廓
    overlay4 = _draw_contours_with_ordered_legend(image_array, pred, min_area=100, thickness=2)

    # 添加 图例 左上角的
    # overlay1 = _draw_legend( overlay1 ,pred,alpha=0.5,background_class=0)
    # baocun

    png_save_path = os.path.join(save_path, "Png_Output")
    bbox_png_save_path =  os.path.join(png_save_path,"bbox")
    contours_png_save_path = os.path.join(png_save_path, "contours")
    ellipse_png_save_path = os.path.join(png_save_path, "ellipse")

    file_exist = lambda  x : os.makedirs(x,exist_ok=True)

    file_exist(bbox_png_save_path )
    file_exist(contours_png_save_path )
    file_exist(ellipse_png_save_path )

    if name.find(".png") > 0:
        # cv2.imwrite(png_save_path + '/' + name + '.png', overlay1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(bbox_png_save_path  + '/' + name , overlay2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(ellipse_png_save_path + '/' + name , overlay3, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(contours_png_save_path + '/' + name , overlay4, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    import glob

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # 1. 指定模型
    # args.model_class = "convnext"#"convnext"   # "resnet"
    args.model_segment = "deeplabv3"#"deeplabv3_channel"   # "segxnet"
    args.resume = 0
    args.cpu = False

    #deeplabv3_channel
    args.pre_train_segment = "[Deeplab-EX4-Finetune]-[N6000_1125]-[2025-11-28-09-07]"  # "[ResNet50DeepLabV3-EX11-planeInput_yueshuattention]-[n6000_shoulder_segment_924_cop_class]-[2025-09-25-09-06]"

    # model_class=None
    model_segment=None

    # class_label = torch.tensor(i)
    # 3.，IMAGE-S1 里面还有病例文件夹层
    input_folder= r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/image/"

    # 4. 指定保存路径
    # " # 串联模型的顺序从左到右侧，左就是数据的来源，右就是模型
    save_folder = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/label_cnn/"


    args.input_path = input_folder # 输入路径
    args.save_path = save_folder  # 这是 保存的标签的 效果 pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train_segment, "model")
    args.inference_path = os.path.join(args.project_path, 'experiment', args.pre_train_segment, "Inference")  # 这是 用来 保存推理结果

# 5. 运行
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    info = f"Ns"

    model_segment = inference_multi_image(args,info,model_segment)  # 这里一定是 一个病例的  可以放心的用

#














