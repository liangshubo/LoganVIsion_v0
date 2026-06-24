'''

这一个代码是用于 单输入的推理 ，其中有两个模型 一个用来分类一个用来分割  ,增加了类别的前后帧的异常判断后处理
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


def inference_multi_image(args,info,class_label,model_class,model_segment):
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
        # ==============================  加载分类   ====================
        if model_class is None:
            module = import_module('src.model.' + args.model_class.lower())
            args.num_class = 11
            model_class = module.make_model(args)
            # --------------------------预训练参数路径  -----------------------
            if args.pre_train_class:
                pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train_class, "model")
                model_class.load_state_dict(torch.load(pre_train_model_path + f"/model_{control_pretrain(args.resume)}.pt"),
                                      strict=False)
            model_class = model_class.cuda().eval()
            print("[----> Model_class inference in ", next(model_class.parameters()).device, "<----]")

        # ============================ 加载分割模型   =============================
        if model_segment is None:
            module = import_module('src.model.' + args.model_segment.lower())
            args.num_class = 24
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
        name_list = sorted(name_list,key=lambda x: int(re.search(r'_(\d+)\.png$', x).group(1)))

        count = 1
        # ---------------
        last_input = None
        last_output = None

        layer = nn.Softmax()

        for nameext in name_list:
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
                #===================分类推理==================

                class_output = model_class(INPUT)


                # -----------------分类帧间平滑参考 ---------
                # 当前分类结果和上一帧不一致
                if last_output is not  None and np.argmax(np.array(class_output.cpu())) != np.argmax(np.array(last_output.cpu())):
                    if peak_signal_noise_ratio(last_input,rawinput) >=20 and layer(class_output)[:, np.argmax(np.array(class_output.cpu()))] < 0.80: # 相似度高则 而且对当前的结果有疑虑 用上一帧代替
                        class_output = last_output

                # ----------------分类上一帧信息更新-------------
                last_input = rawinput
                last_output = class_output

                # ===================分割推理 ===============
                # ------------------切面信息引入 ----------------
                pred_class = np.argmax(np.array(class_output.cpu()))  # 这里可控制
                label_array = np.ones_like(resizeinput) * int(pred_class)
                label_tensor = torch.from_numpy(label_array).float().unsqueeze(0).unsqueeze(0).cuda()
                plane_channel_idx = section_to_tissue[int(pred_class)]
                plane_channel_idx_tensor = create_onehot_from_indices(plane_channel_idx,
                                                                      args.num_class).cuda().unsqueeze(0)
                image_cat_plane = torch.cat([INPUT, label_tensor / 10], dim=1)

                listinput = [image_cat_plane, plane_channel_idx_tensor]
                segment_output = model_segment(listinput)
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

                # ---------------------------上采样 1:直接上采样 快 但是有锯齿   2：索引上采样 折中  3：索引上采样 + 形态学  最平滑 最慢 --------------------
                # pred = cv2.resize(pred,(w,h),interpolation=cv2.INTER_LINEAR)
                # pred = upsample_pred_onehot_torch(pred, args.num_class,(h,w), device='cuda', mode='bicubic')
                # pred = upsample_pred_onehot_torch_with_morph(
                #     pred, args.num_class, (h,w), device='cuda', mode='bicubic',
                #     morph_kernel=7, morph_iterations=2, min_component_area=300,
                #     merge_small=False)

                # ------------------保存阶段 为标签 ------------------------
                # np.save(os.path.join(args.save_path, nameext.split(".")[0]+".npy"), pred ) # 不保存

                outputlist = [class_output, pred]

                # ------------- 附加功能 可视化结果  --------------
                if args.extra_application:
                    save_path = args.inference_path
                    os.makedirs(save_path,exist_ok=True)
                    save_png_output(save_path, nameext,  rawinput ,class_label, outputlist)

                # cv2.imwrite(os.path.join(args.save_path,nameext), mask_array,[cv2.IMWRITE_PNG_COMPRESSION,0])

                cur_time = starter.elapsed_time(ender)
                avg_time += cur_time
                print(info+" Image[{}/{}] : {:.5f}ms".format(count, len(name_list), cur_time))
            count += 1
    return model_class,model_segment


def save_png_output(save_path, name, image_array, label_tensor, output):
    segment_class_names = {0: "背景",
                                1: "肱骨结节", 2: "肱二头肌长头腱", 3: "三角肌", 4: "肱骨头表面",
                                5: "肱二头肌长头腱", 6: "肩胛下肌腱", 7: "肱骨小结节表面",
                                8: "肩胛下肌腱", 9: "肱骨小结节", 10: "冈上肌腱", 11: "肱骨大结节",
                                12: "肩峰下囊", 13: "肱二头肌长头肌腱", 14: "冈上冈下肌腱",
                                15: "冈下肌腱", 16: "肱骨大结节", 17: "小圆肌",
                                18: "锁骨", 19: "肩峰", 20: "肩锁关节囊", 21: "肩峰",
                                22: "喙突", 23: "喙肩韧带"
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
            overlay_image (np.array): 已经叠加了分割掩码的图像。
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
                     segment_class_names=None, min_component_area=200):
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
        #  判断segment_classname
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
        font_size = max(16, int(min(h, w) * 0.02))  # 字体大小按图像尺寸自适应，至少 14
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

        # 对每个类别，查找连通组件并在每个组件中心写字  对于第一个类
        for label in unique_labels:
            # 二值化该类别掩码
            mask = (pred_mask == label).astype(np.uint8)

            if mask.sum() == 0:
                continue

            # 使用 connectedComponentsWithStats 获取每个连通区域的质心和面积   这里面是 有第一个是 背景
            num_cc, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            # stats: (num_cc, 5) -> [x, y, width, height, area]
            # centroids: (num_cc, 2) -> (cx, cy)
            for cid in range(1, num_cc):  # 跳过背景 cc id = 0
                area = int(stats[cid, cv2.CC_STAT_AREA])  # 直接索引 area = stats[i, 4]
                if area < min_component_area:
                    continue

                cx, cy = centroids[cid]
                # 转为整数像素坐标  质心坐标
                centro_x = int(cx)
                centro_y = int(cy)

                # 获取类名
                class_name = segment_class_names.get(label, f"类 {label}")

                # 测量文本尺寸以便中心对齐
                text_size = draw.textsize(class_name, font=font)
                text_w, text_h = text_size

                # 计算左上角坐标，使文字以质心为中心
                draw_x = centro_x - text_w // 2
                draw_y = centro_y - text_h // 2

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

    # ------------------------------ 分割处理 ---------------------------
    # onehot  2 P mode label
    # pred = torch.argmax(output[3].squeeze(), dim=0).cpu()
    # 后处理
    pred = output[1]
    # 类型转换
    image_array = np.array(image_array, dtype=uint8)
    # P模式 转换为 3通道的调色板图

    mask_color = _mask_to_color(pred)
    # 叠加到原图
    overlay = _visualize_segmentation_fast(image_array, pred, mask_color, alpha=0.3)
    # 添加 图例
    overlay = _draw_legend2(overlay, pred, alpha=0.5, background_class=0,segment_class_names=segment_class_names)

    # ----------------------------------分类结果 -------------------------
    true_class = label_tensor.cpu().item()
    pred_class = np.argmax(np.array(output[0].cpu())) # 这里可控制
    layer = nn.Softmax()
    confidence = layer(output[0])[:, pred_class]    # 这里也要控制
    true_class_name = plane_class_name[true_class]
    pred_class_name = plane_class_name[pred_class]

    true_text = f"Label: {true_class_name}"
    pred_text = f"Pred: {pred_class_name} ({confidence.item():.2f})"
    # 绘制
    pil_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_path = "/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf"
    spacing, font_size = 7, 16
    font = ImageFont.truetype(font_path, font_size)
    font_color_pil = (255, 255, 255)  # 白色

    if true_class == pred_class:
        draw.text((10, 10), true_text, font=font, fill=font_color_pil)
        draw.text((10, 29), pred_text, font=font, fill=font_color_pil)
    else:

        pred_text = f"Pred: {pred_class_name} ({confidence.item():.2f})" + " Prediction Incorrect (!)"
        draw.text((10, 10), true_text, font=font, fill=font_color_pil)
        draw.text((10, 29), pred_text, font=font, fill=(255, 0, 0))

    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # ------------------------- save --------------
    png_save_path = os.path.join(save_path, "Png_Output")
    if not os.path.exists(png_save_path):
        os.makedirs(png_save_path)
    if name.find(".png") < 0:
        cv2.imwrite(png_save_path + '/' + name + '.png', final_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if name.find(".png") > 0:
        cv2.imwrite(png_save_path + '/' + name, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    import glob

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # 1. 指定模型
    args.model_class = "convnext"#"convnext"   # "resnet"
    args.model_segment = "deeplabv3_channel"#"deeplabv3_channel"   # "segxnet"
    args.resume = 0
    args.cpu = False
    #resnet
    #args.pre_train_class = "[AttentionResNet50-n20-n6000-EX1]-[n20n6000_shoulder_class_all_1013]-[2025-10-13-17-37]"  #
    args.pre_train_class = "[Convnext-n20-n6000-EX1]-[n20n6000_shoulder_class_all_1013]-[2025-10-20-14-25]"  # "

    #deeplabv3_channel
    args.pre_train_segment = "[ResNet50DeepLabV3-EX11-planeInput_yueshuattention]-[n20_shoulder_segment_1010_cop_class]-[2025-10-11-15-05]"  # "[ResNet50DeepLabV3-EX11-planeInput_yueshuattention]-[n6000_shoulder_segment_924_cop_class]-[2025-09-25-09-06]"
    # [ResNet50DeepLabV3-EX11-planeInput_yueshuattention]-[n6000_shoulder_segment_924_cop_class]-[2025-09-25-09-06]
    # [Segnext-EX11-planeInput_segment]-[n20n6000_shoulder_segment_cop_class_all_1013]-[2025-10-22-16-50]
    # [AttentionResNet50-n20-n6000-EX1]-[n20n6000_shoulder_class_all_1013]-[2025-10-13-17-37]
    # [ResNet50DeepLabV3-EX11-planeInput_yueshuattention]-[n20_shoulder_segment_1010_cop_class]-[2025-10-11-15-05]
    #args.pre_train_segment = "[Segnext-EX11-planeInput_segment]-[n20n6000_shoulder_segment_cop_class_all_1013]-[2025-10-22-16-50]"

    input_path_list = ["S0","S1","S2","S3","S4","S5","S6","S7","S8","S10","S11"]
    #input_path_list = ["S5"]
    #save_path_list = []
    model_class=None
    model_segment=None
    for i in range(len(input_path_list)):
        class_label = torch.tensor(i)
        # 3.，IMAGE-S1 里面还有病例文件夹层
        input_folder= r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/test/image/"+input_path_list[i]

        # 4. 指定保存路径
        # " # 串联模型的顺序从左到右侧，左就是数据的来源，右就是模型
        save_folder = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/trainall/label_cnn/"+input_path_list[i]
        args.inference_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/test_inference/inference_class_tendom_segment"
        path_name = os.listdir(input_folder)

        for  j in range(len(path_name)):

            args.input_path = os.path.join(input_folder,path_name[j])
            args.save_path = os.path.join(save_folder,path_name[j])

        # 5. 运行
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            info = f"plane {i+1} / {len(input_path_list)} case {j+1}/ {len(path_name)}"

            model_class,model_segment = inference_multi_image(args,info,class_label,model_class,model_segment)  # 这里一定是 一个病例的  可以放心的用

        #


















