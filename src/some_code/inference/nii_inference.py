import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image

from tqdm import tqdm
from typing import Dict, Optional, List, Tuple, Any



import os
import sys
import time
import glob
import re
import platform
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from importlib import import_module

import numpy as np
import torch
import cv2
import psutil
import torch.nn as nn
import warnings

# 定义常用的CT窗宽窗位预设
WINDOW_PRESETS: Dict[str, Optional[Tuple[int, int]]] = {
    'nonewin': None,  # 不加窗，使用全局最小值和最大值
    'xiongwin': (1600, -600),  # 胸窗(肺窗)：窗宽1600HU，窗位-600HU
    'zonggewin': (350, 40),  # 纵隔窗：窗宽350HU，窗位40HU
    'guwin': (1500, 300)  # 骨窗：窗宽1500HU，窗位300HU
}

# 定义三个平面的名称和对应的维度
PLANES: Dict[str, int] = {
    'zhou': 0,  # 轴向(横断面)：z轴方向
    'guan': 1,  # 冠状面：y轴方向
    'shi': 2  # 矢状面：x轴方向
}
project_path: str = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
print(f"项目根路径: {project_path}")

def apply_window(image_array: np.ndarray,
                 window_width: Optional[int] = None,
                 window_center: Optional[int] = None) -> np.ndarray:
    """
    对CT图像应用窗宽窗位调整

    参数:
        image_array: numpy数组，原始CT图像数据(HU值)
        window_width: 窗宽，None表示使用全局范围
        window_center: 窗位，None表示使用全局范围

    返回:
        numpy数组，调整后的图像数据(0-255)
    """
    if window_width is None or window_center is None:
        # 不加窗，使用全局最小值和最大值
        min_val: float = np.min(image_array)
        max_val: float = np.max(image_array)
    else:
        # 计算窗的上下限
        min_val = window_center - 0.5 * window_width
        max_val = window_center + 0.5 * window_width

    # 将值限制在窗范围内
    windowed_array: np.ndarray = np.clip(image_array, min_val, max_val)

    # 归一化到0-255
    if max_val != min_val:
        windowed_array = (windowed_array - min_val) / (max_val - min_val) * 255.0
    else:
        windowed_array = np.zeros_like(windowed_array)

    return windowed_array.astype(np.uint8)



import torch
from typing import Union, Tuple

# 环境变量必须在导入torch前设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")


# ==================== 数据类定义（静态类型核心） ====================


def _to_2tuple(input_size: Union[int, Tuple[int, int]]):
    """
    支持 args.input_size = 256 或 args.input_size = (256, 256)
    """
    if isinstance(input_size, int):
        return input_size, input_size
    if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        return int(input_size[0]), int(input_size[1])
    raise ValueError(f"input_size 格式错误: {input_size}")

@torch.no_grad()
def forward_chop(
        model_sr: torch.nn.Module,
        input_tensor: torch.Tensor,
        input_size: Union[int, Tuple[int, int]],
        scale: int,
        shave: int = 16,
        output_on_cpu: bool = False,
) -> torch.Tensor:
    """
    带上下文扩展 + 中心裁剪的 chop 推理。

    核心逻辑：
        1. 每次真正要写回的是 input_size × input_size 的 core 区域；
        2. 实际送入模型的是 core 区域四周额外扩展 shave 个像素；
        3. 模型输出后，把扩展区域对应的 SR 结果裁掉；
        4. 只把中心 core 区域写回完整输出。

    Args:
        model_sr:
            超分模型。

        input_tensor:
            输入张量，shape = [B, C, H, W]。

        input_size:
            每个核心 patch 的大小，例如 128 或 256。
            注意：这个是最终有效写回区域，不包含额外上下文。

        scale:
            超分倍率，例如 4。

        shave:
            patch 四周额外扩展的上下文像素数。
            例如 input_size=128, shave=16，则中间区域是 128，
            实际送入模型的最大 patch 大小约为 160×160。

        output_on_cpu:
            True：完整输出放 CPU，省 GPU 显存。
            False：完整输出放 GPU，速度略快但更吃显存。

    Returns:
        拼接后的 SR 输出，shape = [B, C, H*scale, W*scale]
    """

    if input_tensor.dim() != 4:
        raise ValueError(
            f"input_tensor 必须是 4D: [B, C, H, W]，当前 shape={input_tensor.shape}"
        )

    tile_h, tile_w = _to_2tuple(input_size)

    b, c, h, w = input_tensor.shape
    device = next(model_sr.parameters()).device

    input_tensor = input_tensor.to(device=device, dtype=torch.float32)

    if output_on_cpu:
        output = torch.empty(
            size=(b, c, h * scale, w * scale),
            dtype=torch.float32,
            device="cpu"
        )
    else:
        output = input_tensor.new_empty((b, c, h * scale, w * scale))

    print(
        f"Will inference with context chop: "
        f"input={tuple(input_tensor.shape)}, "
        f"core_tile=({tile_h}, {tile_w}), "
        f"shave={shave}, "
        f"scale={scale}, "
        f"output_on_cpu={output_on_cpu}"
    )

    for top in range(0, h, tile_h):
        for left in range(0, w, tile_w):

            # -----------------------------
            # 1. 当前真正要写回的 core 区域
            # -----------------------------
            bottom = min(top + tile_h, h)
            right = min(left + tile_w, w)

            core_h = bottom - top
            core_w = right - left

            # -----------------------------
            # 2. 在 core 基础上向外扩展 shave
            # -----------------------------
            ext_top = max(top - shave, 0)
            ext_left = max(left - shave, 0)
            ext_bottom = min(bottom + shave, h)
            ext_right = min(right + shave, w)

            input_patch = input_tensor[:, :, ext_top:ext_bottom, ext_left:ext_right].contiguous()

            # -----------------------------
            # 3. 推理扩展 patch
            # -----------------------------
            output_patch = model_sr(input_patch)

            if output_patch.dim() != 4:
                raise ValueError(
                    f"模型输出必须是 4D: [B, C, H, W]，当前 shape={output_patch.shape}"
                )

            # -----------------------------
            # 4. 计算 core 在扩展 patch 输出里的位置
            # -----------------------------
            crop_top = (top - ext_top) * scale
            crop_left = (left - ext_left) * scale
            crop_bottom = crop_top + core_h * scale
            crop_right = crop_left + core_w * scale

            output_core = output_patch[
                :,
                :,
                crop_top:crop_bottom,
                crop_left:crop_right
            ]

            # -----------------------------
            # 5. 写回完整 SR 输出对应位置
            # -----------------------------
            out_top = top * scale
            out_left = left * scale
            out_bottom = bottom * scale
            out_right = right * scale

            if output_on_cpu:
                output[:, :, out_top:out_bottom, out_left:out_right] = output_core.detach().cpu()
            else:
                output[:, :, out_top:out_bottom, out_left:out_right] = output_core

            del input_patch
            del output_patch
            del output_core

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output



def init_model(
        args,
        info: str,
        model_sr: Optional[nn.Module] = None
) -> Tuple[Optional[nn.Module]]:
    """
    批量推理SR模型并测量完整性能指标

    Args:
        args: 配置参数
        info: 日志前缀
        model_sr: 预加载的模型(可选，用于多次调用)

    Returns:
        Tuple[模型实例, 推理性能指标]
    """
    # ============================ 加载SR模型 =============================
    if model_sr is None:
        module = import_module(f"src.model.{args.model_sr.lower()}")
        model_sr = module.make_model(args)

        # 加载预训练参数
        if args.pre_train_sr_model:
            pre_train_model_path: str = os.path.join(
                args.project_path, "experiment", args.pre_train_sr_model, "model"
            )
            model_path: str = os.path.join(
                pre_train_model_path, f"model_{control_pretrain(args.resume)}.pt"
            )
            print(f"加载预训练模型: {model_path}")

            checkpoint: Dict[str, Any] = torch.load(model_path, map_location="cpu")
            model_sr.load_state_dict(checkpoint, strict=False)

        if not args.cpu and torch.cuda.is_available():
            model_sr = model_sr.cuda()

        model_sr.eval()
        print(f"[----> SR模型运行在: {next(model_sr.parameters()).device} <----]")
    return model_sr, info


def extract_and_save_slices(args,nii_path: str,
                            output_dir: str,
                            planes: Optional[List[str]] = None,
                           ) -> None:
    """
    从NIfTI文件中提取指定平面和窗位的切片并保存为PNG

    参数:
        nii_path: NIfTI文件路径
        output_dir: 输出目录
        planes: 要提取的平面列表，默认为所有三个平面
        windows: 要应用的窗位列表，默认为所有预设窗位
        start_slice: 起始切片索引，默认为0
        end_slice: 结束切片索引，默认为最后一张
    """
    # -------------------------------读取NIfTI文件----------------------------------


    try:
        image: sitk.Image = sitk.ReadImage(nii_path)
        image_array: np.ndarray = sitk.GetArrayFromImage(image)  # 形状为(z, y, x)
        print(image_array.dtype)   #
        # 防止 移除
        rmin_val: float = np.min(image_array)
        rmax_val: float = np.max(image_array)
        if rmax_val > 8000:
            image_array = np.clip(image_array, -5000, 8000).astype(np.int16)
        # ------------------ get spacing ----------------
        original_spacing = image.GetSpacing()
        original_origin = image.GetOrigin()
        original_direction = image.GetDirection()

    except Exception as e:
        print(f"错误：无法读取文件 {nii_path}，原因：{e}")

    # -------------------------------init   model ----------------------------------

    model_sr, info = init_model(args,info="SR MODE")

    # 遍历每个平面
    for plane_name in planes:
        if plane_name not in PLANES:
            print(f"警告：未知的平面名称 {plane_name}，跳过")
            continue

        axis: int = PLANES[plane_name]
        num_slices: int = image_array.shape[axis]
        z, y, x = image_array.shape
        sx, sy, sz = original_spacing
    # --------------output init ------------
        if plane_name == "zhou":
            # 横断面：放大 y, x，不放大 z
            output_volume = np.zeros((z, y * args.scale, x * args.scale), dtype=np.int16)
            spacing = (sx / args.scale, sy / args.scale, sz)

        elif plane_name == "guan":
            # 冠状面：切片是 z-x 平面，放大 z, x，不放大 y
            output_volume = np.zeros((z * args.scale, y, x * args.scale), dtype=np.int16)
            spacing = (sx / args.scale, sy, sz / args.scale)

        elif plane_name == "shi":
            # 矢状面：切片是 z-y 平面，放大 z, y，不放大 x
            output_volume = np.zeros((z * args.scale, y * args.scale, x), dtype=np.int16)
            spacing = (sx, sy / args.scale, sz / args.scale)
    # --------------------------------------------
        # 设置切片范围   没有指定切片范围就从头到尾部全部都要
        # 确保切片范围有效
        current_start = 0
        current_end = num_slices

        if current_start >= current_end:
            print(f"警告：平面 {plane_name} 的切片范围无效，跳过")
            continue

        print(
            f"正在处理 {plane_name} 平面，切片范围：{current_start}-{current_end - 1}，共 {current_end - current_start} 张")
        device = next(model_sr.parameters()).device
        # 遍历每个切片
        for slice_idx in tqdm(range(current_start, current_end), desc=f"{plane_name} 平面"):
            # 提取切片
            slice_data: np.ndarray
            if axis == 0:  # 轴向
                slice_data = image_array[slice_idx, :, :]
            elif axis == 1:  # 冠状面
                slice_data = image_array[:, slice_idx, :]
            else:  # 矢状面
                slice_data = image_array[:, :, slice_idx]

            #  只有一个窗   就是没有
            min_val: float = np.min(slice_data)
            max_val: float = np.max(slice_data)

            windowed_array = (slice_data - min_val) / (max_val - min_val)

            windowed_array = windowed_array.astype(np.float32)
            input_tensor: torch.Tensor = torch.tensor(windowed_array ).unsqueeze(0).unsqueeze(0).to(device=device,dtype=torch.float32)
#
            print(f"will process datasize {input_tensor.shape}")

            with torch.no_grad():

                if args.chop:
                    sr_output: torch.Tensor =forward_chop(model_sr=model_sr,input_tensor=input_tensor,input_size=args.input_size,
                                                          scale=args.scale,shave=16, output_on_cpu=False)
                    torch.cuda.synchronize()
                else:

                    sr_output: torch.Tensor = model_sr(input_tensor)

                    torch.cuda.synchronize()

                output_array: np.ndarray = sr_output.cpu().squeeze(0).squeeze(0).numpy().astype(np.float32)

                hu_array = output_array*(max_val - min_val) + min_val

                hu_array = hu_array.astype(np.int16)

            if plane_name == "zhou":
                output_volume[slice_idx, :, :] = hu_array

            elif plane_name == "guan":
                output_volume[:, slice_idx, :] = hu_array

            elif plane_name == "shi":
                output_volume[:, :, slice_idx] = hu_array
            # ----------------------tositk
        #  实时保存呢
            if slice_idx % 300 == 0 or slice_idx == 0 or slice_idx == num_slices - 1:
                image_sitk = sitk.GetImageFromArray(output_volume)
                image_sitk.SetSpacing(spacing)
                image_sitk.SetOrigin(original_origin)
                image_sitk.SetDirection(original_direction)
                sitk.WriteImage(image_sitk, output_dir)

    return 0
                # 旋转图像以符合医学影像显示习惯
                # 轴向：逆时针旋转90度
                # 冠状面：逆时针旋转90度
                # 矢状面：逆时针旋转90度并左右翻转

def batch_process(args,input_dir: str,
                  output_dir: str,
                  planes: Optional[List[str]] = None
                  ) -> None:
    """
    批量处理目录下的所有NIfTI文件

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        planes: 要提取的平面列表
        windows: 要应用的窗位列表
        start_slice: 起始切片索引
        end_slice: 结束切片索引
    """
    # 查找所有.nii和.nii.gz文件
    nii_files: List[str] = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))

    if not nii_files:
        print(f"在目录 {input_dir} 中未找到任何NIfTI文件")
        return

    print(f"找到 {len(nii_files)} 个NIfTI文件")

    # 处理每个文件
    for i, nii_file in enumerate(nii_files):
        print(f"\n处理第 {i + 1}/{len(nii_files)} 个文件：{nii_file}")

        # 为每个文件创建单独的输出目录
        file_name: str = os.path.basename(nii_file)
        base_name: str = os.path.splitext(file_name)[0]
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]
        print(file_name )
        file_output_dir=  os.path.join( output_dir,file_name )
        extract_and_save_slices(args,nii_file, file_output_dir, planes)


def tensor2array(tensor: torch.Tensor, rgb_range: float = 1.0) -> np.ndarray:
    """将PyTorch张量转换为numpy数组"""
    tensor = tensor.squeeze(0).squeeze(0)
    array = tensor.cpu().numpy() * (255.0 / rgb_range)
    return array

def control_pretrain(resume: int) -> str:
    """控制预训练模型加载类型"""
    if resume == 0:
        return "best"
    elif resume == -1:
        return "latest"
    else:
        return str(resume)

def get_args() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='从NIfTI CT数据中提取轴向、矢状面、冠状面切片并 进行推理 然后 拼接 '
    )
    parser.add_argument("--cpu", action="store_true", help="use cpu only")
    parser.add_argument("--model_sr", default="Unet", help="model name")
    parser.add_argument("--pre_train_sr_model", type=str, default=None, help="pre-trained model directory")
    parser.add_argument("--iterates", type=int, default=30, help="iterates number")
    parser.add_argument("--project_path", type=str, default=project_path, help="project root path")
    parser.add_argument("--resume", type=int, default=0, help="0: best, -1: latest, other: epoch number")
    parser.add_argument("--input_size", type=int, default=256, help="input image size")
    parser.add_argument("--scale", type=int, default=4, help="SR scale factor")
    parser.add_argument('--input',  default=r'E:\project\Dataset\TEST_VIW_UPSAMPLE\NII_istrop', help='输入NIfTI文件路径或目录')
    parser.add_argument('--output',default=r'E:\project\Dataset\TEST_VIW_UPSAMPLE\TEST_IMAGE_istrop\shi\allwin',  help='输出目录')
    parser.add_argument('--planes', nargs='+', default=['guan'],   #  'zhou' , 'guan', 'shi'
                        help='要提取的平面，可选值：zhou(轴向), guan(冠状面), shi(矢状面)')
    parser.add_argument('--windows', nargs='+', default=['xiongwin', 'zonggewin', 'guwin'],
                        help='要应用的窗位，可选值：nonewin(不加窗), xiongwin(胸窗), zonggewin(纵隔窗), guwin(骨窗)')
    parser.add_argument('--start', default=0,type=int, help='起始切片索引')
    parser.add_argument('--end',default=None, type=int, help='结束切片索引')
    parser.add_argument('--batch',default=1, type=int,help='批量处理目录下的所有NIfTI文件')
    parser.add_argument("--chop", type=int, default=0, help="SR inference mode with/ot chop patch for big image")

    args: argparse.Namespace = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_args()

    args.model_sr = "rcan"
    args.resume = 0
    args.cpu = False
    args.pre_train_sr_model = "[RCAN-EX4]-[LDIC_Fix_cfg2_AreaDownX4]-[2026-06-12-11-17]"
    args.warmup_runs = 8
    args.chop = 0
    torch.backends.cudnn.benchmark = False
    args.batch = 1
    # 验证输入
    args.input = r"/home/liangshubo/Project/CTSR/dataset/rawdata/CT_Compare_Image/Elite_Part001/nii/"
    output = r"/home/liangshubo/Project/CTSR/dataset/rawdata/CT_Compare_Image/Elite_Part001/nii_inference"
    args.planes = ['zhou', 'guan', 'shi']

    for plane in args.planes:

        args.output = os.path.join(output, plane)

        if not os.path.exists(args.output):
            os.makedirs(args.output)
        now_plane = [plane]   # must use this because we use is split the zhou \guan\shi\ and need to save one b one
        if args.batch:
            if not os.path.isdir(args.input):
                print(f"错误：批量模式下输入必须是目录")
                sys.exit(1)
            batch_process(args, args.input, args.output,  now_plane)
        else:
            if not os.path.isfile(args.input):
                print(f"错误：文件不存在 {args.input}")
                sys.exit(1)
            extract_and_save_slices(args.input, args.output,  now_plane)

        print("\n处理完成！")