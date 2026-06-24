from __future__ import annotations

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

# 环境变量必须在导入torch前设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")


# ==================== 数据类定义（静态类型核心） ====================
@dataclass
class SystemInfo:
    """系统硬件信息数据类"""
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    cpu_freq: float  # GHz
    gpu_model: str
    gpu_total_memory: float  # GB
    torch_version: str
    cuda_version: str
    system_os: str


@dataclass
class GPUMemoryMetrics:
    """GPU显存指标数据类"""
    allocated: float  # MB
    reserved: float  # MB
    max_allocated: float  # MB
    max_reserved: float  # MB


@dataclass
class SystemMemoryMetrics:
    """系统内存指标数据类"""
    process_used: float  # MB
    system_total: float  # GB
    system_available: float  # GB
    system_used_percent: float  # %


@dataclass
class InferenceMetrics:
    """推理综合性能指标数据类"""
    total_images: int
    model_size: float  # MB
    avg_inference_time: float  # ms
    min_inference_time: float  # ms
    max_inference_time: float  # ms
    fps: float
    avg_gpu_memory: float  # MB
    peak_gpu_memory: float  # MB
    avg_system_memory: float  # MB


# ==================== 工具函数 ====================
def get_system_info() -> SystemInfo:
    """获取完整的系统硬件和软件信息  与 模型和输入都无关  """
    # CPU信息
    cpu_model: str = platform.processor()

    cpu_cores: int = psutil.cpu_count(logical=False) or 0
    cpu_threads: int = psutil.cpu_count(logical=True) or 0
    cpu_freq: float = (psutil.cpu_freq().max if psutil.cpu_freq() else 0) / 1000.0

    # GPU信息
    gpu_model: str = "N/A"
    gpu_total_memory: float = 0.0
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    # 软件信息
    torch_version: str = torch.__version__
    cuda_version: str = torch.version.cuda or "N/A"
    system_os: str = f"{platform.system()} {platform.release()}"

    return SystemInfo(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        cpu_freq=cpu_freq,
        gpu_model=gpu_model,
        gpu_total_memory=gpu_total_memory,
        torch_version=torch_version,
        cuda_version=cuda_version,
        system_os=system_os
    )


def clear_gpu_cache() -> None:
    """清空GPU缓存并重置峰值显存统计"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory_usage() -> GPUMemoryMetrics:
    """获取当前GPU显存使用情况"""
    if not torch.cuda.is_available():
        return GPUMemoryMetrics(0.0, 0.0, 0.0, 0.0)

    return GPUMemoryMetrics(
        allocated=torch.cuda.memory_allocated() / 1024 ** 2,
        reserved=torch.cuda.memory_reserved() / 1024 ** 2,
        max_allocated=torch.cuda.max_memory_allocated() / 1024 ** 2,
        max_reserved=torch.cuda.max_memory_reserved() / 1024 ** 2
    )


def get_system_memory_usage() -> SystemMemoryMetrics:
    """获取当前系统内存使用情况"""
    process = psutil.Process(os.getpid())
    process_used: float = process.memory_info().rss / 1024 ** 2

    system_memory = psutil.virtual_memory()
    return SystemMemoryMetrics(
        process_used=process_used,
        system_total=system_memory.total / 1024 ** 3,
        system_available=system_memory.available / 1024 ** 3,
        system_used_percent=system_memory.percent
    )


def measure_model_size(model: nn.Module) -> float:
    """测量模型参数和缓冲区占用的显存大小(MB)"""
    param_size: int = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size: int = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024 ** 2


file_exist = lambda x: os.makedirs(x, exist_ok=True)

# ==================== 核心业务函数 ====================
project_path: str = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
print(f"项目根路径: {project_path}")


def get_args() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Thermal and Rail SR Inference")
    parser.add_argument("--cpu", action="store_true", help="use cpu only")
    parser.add_argument("--model_sr", default="Unet", help="model name")
    parser.add_argument("--pre_train_sr_model", type=str, default=None, help="pre-trained model directory")
    parser.add_argument("--iterates", type=int, default=30, help="iterates number")
    parser.add_argument("--input_data", type=str, default=None, help="single image directory")
    parser.add_argument("--input_path", type=str, default=None, help="multiple images directory")
    parser.add_argument("--project_path", type=str, default=project_path, help="project root path")
    parser.add_argument("--save_path", type=str, default=None, help="save path")
    parser.add_argument("--inference_path", type=str, default=None, help="inference result save directory")
    parser.add_argument("--sw_mode", type=str, default="S", help="save mode")
    parser.add_argument("--resume", type=int, default=0, help="0: best, -1: latest, other: epoch number")
    parser.add_argument("--input_size", type=int, default=512, help="input image size")
    parser.add_argument("--patch_size", type=int, default=512, help="input image size just for Transformer")
    parser.add_argument("--scale", type=int, default=4, help="SR scale factor")
    parser.add_argument("--chop", type=int, default=0, help="SR inference mode with/ot chop patch for big image")
    parser.add_argument("--extra_application", type=int, default=1, help="enable result saving")
    parser.add_argument("--warmup_runs", type=int, default=5, help="model warmup runs")
    args = parser.parse_args()
    return args


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


def save_png_output(inference_path: str, name: str, output: torch.Tensor) -> None:
    """保存SR输出为PNG图片"""
    # 类型转换和归一化
    output_array: np.ndarray = output.cpu().squeeze(0).squeeze(0).numpy() * 255.0
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)

    # 创建保存目录
    png_save_path: str = os.path.join(inference_path, "Png_Output")
    file_exist(png_save_path)

    # 保存图片（仅保存到子目录，修复原代码重复保存问题）
    output_save_path: str = os.path.join(png_save_path, name)
    cv2.imwrite(output_save_path, output_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])





import torch
from typing import Union, Tuple


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
def forward_chop_wo_overlap(
        model_sr: torch.nn.Module,
        input_tensor: torch.Tensor,
        input_size: Union[int, Tuple[int, int]],
        scale: int,
) -> torch.Tensor:
    """
    对超分模型做 chop 推理。

    Args:
        model_sr: SR 模型
        input_tensor: 输入张量，shape = [B, C, H, W]
        input_size: 每个 patch 的输入尺寸，例如 256 或 (256, 256)

    Returns:
        拼接后的 SR 输出
    """
    print(f" Will Inference with chop mode in  size {input_tensor.shape} ")
    if input_tensor.dim() != 4:
        raise ValueError(f"input_tensor 必须是 4D: [B, C, H, W]，当前 shape={input_tensor.shape}")

    tile_h, tile_w = _to_2tuple(input_size)

    b, c, h, w = input_tensor.shape
    output = input_tensor.new_empty((b, c, h*scale, w*scale))

    for top in range(0, h, tile_h):
        for left in range(0, w, tile_w):

            bottom = min(top + tile_h, h)
            right = min(left + tile_w, w)

            # 当前 patch，最后不足 input_size 的部分就直接取剩余区域
            input_patch = input_tensor[:, :, top:bottom, left:right].contiguous()

            # patch 推理
            output_patch = model_sr(input_patch)

            if output_patch.dim() != 4:
                raise ValueError(
                    f"模型输出必须是 4D: [B, C, H, W]，当前 shape={output_patch.shape}"
                )
            # 计算输出 patch 应该放置的位置
            out_top = top * scale
            out_left = left * scale
            out_bottom = bottom * scale
            out_right = right * scale

            output[:, :, out_top:out_bottom, out_left:out_right] = output_patch

            # 释放中间变量，降低显存峰值
            del input_patch
            del output_patch

    return output

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





def inference_multi_image(
        args,
        info: str,
        model_sr: Optional[nn.Module] = None
) -> Tuple[Optional[nn.Module], InferenceMetrics]:
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

    # 测量模型本身占用的显存和内存
    clear_gpu_cache()
    model_size: float = measure_model_size(model_sr)
    initial_gpu_mem: GPUMemoryMetrics = get_gpu_memory_usage()
    initial_sys_mem: SystemMemoryMetrics = get_system_memory_usage()
    print(f"模型参数大小: {model_size:.2f} MB")
    print(f"模型加载后GPU显存: {initial_gpu_mem.allocated:.2f} MB")
    print(f"模型加载后系统内存: {initial_sys_mem.process_used:.2f} MB")

    # =================== 加载输入数据 ==============================
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"输入路径不存在: {args.input_path}")

    name_list: List[str] = sorted(os.listdir(args.input_path))
    total_images: int = len(name_list)

    if total_images == 0:
        print("警告: 输入文件夹中没有找到任何图片")
        return model_sr, InferenceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    print(f"\n找到 {total_images} 张图片待处理")

    # ==================== 性能统计初始化 ===================
    timings: List[float] = []
    gpu_memory_list: List[float] = []
    system_memory_list: List[float] = []
    peak_gpu_memory: float = 0.0

    # 创建CUDA事件用于高精度计时
    starter: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
    ender: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

    # ==================== 模型预热 ===================
    print(f"\n正在进行模型预热({args.warmup_runs}次)...")
    with torch.no_grad():
        # 读取第一张图片进行预热

        input_tensor: torch.Tensor = torch.randn([1,1,args.input_size,args.input_size])

        if not args.cpu and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # 预热运行
        for _ in range(args.warmup_runs):
            _ = model_sr(input_tensor)

    clear_gpu_cache()
    print("预热完成，开始正式推理...\n")

    # ==================== 正式推理循环 ===================
    for count, nameext in enumerate(name_list, 1):
        input_data: str = os.path.join(args.input_path, nameext)

        # 数据读取与预处理
        rawinput: Optional[np.ndarray] = cv2.imread(input_data, 0)
        if rawinput is None:
            print(f"警告: 无法读取图片 {input_data}，跳过")
            continue

        input_tensor: torch.Tensor = torch.tensor(rawinput).unsqueeze(0).unsqueeze(0) / 255.0

        if not args.cpu and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # 清空缓存确保测量准确
        clear_gpu_cache()

        # 模型前向推理
        with torch.no_grad():

            if args.chop :
                starter.record()
                sr_output: torch.Tensor = forward_chop(model_sr,input_tensor,args.input_size,args.scale)
                ender.record()
                torch.cuda.synchronize()
            else:
                starter.record()
                sr_output: torch.Tensor = model_sr(input_tensor)
                ender.record()
                torch.cuda.synchronize()

        # 计算时间和内存指标
        cur_time: float = starter.elapsed_time(ender)
        cur_gpu_mem: GPUMemoryMetrics = get_gpu_memory_usage()
        cur_sys_mem: SystemMemoryMetrics = get_system_memory_usage()

        # 更新统计数据
        timings.append(cur_time)
        gpu_memory_list.append(cur_gpu_mem.allocated)
        system_memory_list.append(cur_sys_mem.process_used)

        if cur_gpu_mem.max_allocated > peak_gpu_memory:
            peak_gpu_memory = cur_gpu_mem.max_allocated

        # 保存结果
        if args.extra_application:
            save_png_output(args.inference_path, nameext, sr_output)

        # 打印单张图片信息
        print(
            f"{info} Image[{count}/{total_images}] : {cur_time:.2f}ms | "
            f"GPU: {cur_gpu_mem.allocated:.2f}MB | 峰值: {cur_gpu_mem.max_allocated:.2f}MB | "
            f"内存: {cur_sys_mem.process_used:.2f}MB"
        )

    # ==================== 计算综合指标 ===================
    avg_inference_time: float = float(np.mean(timings))
    min_inference_time: float = float(np.min(timings))
    max_inference_time: float = float(np.max(timings))
    fps: float = 1000.0 / avg_inference_time
    avg_gpu_memory: float = float(np.mean(gpu_memory_list))
    avg_system_memory: float = float(np.mean(system_memory_list))

    metrics: InferenceMetrics = InferenceMetrics(
        total_images=total_images,
        model_size=model_size,
        avg_inference_time=avg_inference_time,
        min_inference_time=min_inference_time,
        max_inference_time=max_inference_time,
        fps=fps,
        avg_gpu_memory=avg_gpu_memory,
        peak_gpu_memory=peak_gpu_memory,
        avg_system_memory=avg_system_memory
    )

    return model_sr, metrics


def print_system_info(info: SystemInfo) -> None:
    """打印系统信息报告"""
    print("\n" + "=" * 80)
    print("系统硬件与软件信息")
    print("=" * 80)
    print(f"操作系统: {info.system_os}")
    print(f"PyTorch版本: {info.torch_version}")
    print(f"CUDA版本: {info.cuda_version}")
    print(f"\nCPU信息:")
    print(f"  型号: {info.cpu_model}")
    print(f"  物理核心: {info.cpu_cores} | 逻辑线程: {info.cpu_threads}")
    print(f"  最大频率: {info.cpu_freq:.2f} GHz")
    print(f"\nGPU信息:")
    print(f"  型号: {info.gpu_model}")
    print(f"  总显存: {info.gpu_total_memory:.2f} GB")
    print("=" * 80 + "\n")


def print_performance_report(metrics: InferenceMetrics) -> None:
    """打印详细的推理性能报告"""
    print("\n" + "=" * 80)
    print("SR推理性能综合报告")
    print("=" * 80)
    print(f"处理图片总数: {metrics.total_images}")
    print(f"模型参数大小: {metrics.model_size:.2f} MB")
    print(f"\n⏱️  时间指标:")
    print(f"  平均推理时间: {metrics.avg_inference_time:.2f} ms")
    print(f"  最小推理时间: {metrics.min_inference_time:.2f} ms")
    print(f"  最大推理时间: {metrics.max_inference_time:.2f} ms")
    print(f"  推理FPS: {metrics.fps:.2f}")
    print(f"\n💾 GPU显存指标:")
    print(f"  平均推理显存: {metrics.avg_gpu_memory:.2f} MB")
    print(f"  全局峰值显存: {metrics.peak_gpu_memory:.2f} MB")
    print(f"\n🖥️  系统内存指标:")
    print(f"  平均进程内存: {metrics.avg_system_memory:.2f} MB")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # 检查依赖
    try:
        import psutil
    except ImportError:
        print("错误: 缺少psutil依赖，请执行: pip install psutil")
        sys.exit(1)

    # 获取系统信息
    system_info: SystemInfo = get_system_info()
    print_system_info(system_info)

    # 获取配置参数
    args = get_args()

    # 1. 模型配置
    args.model_sr = "rcan"
    args.resume = 0
    args.cpu = True
    args.pre_train_sr_model = "[RCAN-EX4]-[LDIC_Fix_cfg2_AreaDownX4]-[2026-06-12-11-17]"
    args.warmup_runs = 0
    args.chop = 1
    torch.backends.cudnn.benchmark = False

    args.input_size = 384
    # 2. 输入输出路径
    input_folder: str = r"/home/liangshubo/Project/CTSR/dataset/rawdata/inference_image/"
    args.input_path = input_folder

    # 3. 推理结果保存路径
    # args.inference_path = os.path.join(
    #      args.project_path, "experiment", args.pre_train_sr_model, "Inference"
    #  )

    args.inference_path = os.path.join(r"/home/liangshubo/Project/CTSR/dataset/rawdata/inference_output/",
                                       args.pre_train_sr_model, "Inference")

    file_exist(args.inference_path)
    print(f"推理结果将保存到: {args.inference_path}")

    # 4. 开始推理
    info: str = "SR"
    model_sr, metrics = inference_multi_image(args, info)

    # 5. 打印性能报告
    print_performance_report(metrics)