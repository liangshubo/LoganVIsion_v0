import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple, Any
import concurrent.futures  # 多线程核心库
import threading            # 线程锁

# 线程锁：防止多线程同时写文件导致图片损坏
lock = threading.Lock()

# 定义常用的CT窗宽窗位预设
WINDOW_PRESETS: Dict[str, Optional[Tuple[int, int]]] = {
    'nonewin': None,
    'xiongwin': (1600, -600),
    'zonggewin': (350, 40),
    'guwin': (1500, 300)
}

PLANES: Dict[str, int] = {
    'zhou': 0,
    'guan': 1,
    'shi': 2
}


def apply_window(image_array: np.ndarray,
                 window_width: Optional[int] = None,
                 window_center: Optional[int] = None) -> np.ndarray:
    if window_width is None or window_center is None:
        min_val: float = np.min(image_array)
        max_val: float = np.max(image_array)
    else:
        min_val = window_center - 0.5 * window_width
        max_val = window_center + 0.5 * window_width

    windowed_array: np.ndarray = np.clip(image_array, min_val, max_val)

    if max_val != min_val:
        windowed_array = (windowed_array - min_val) / (max_val - min_val) * 255.0
    else:
        windowed_array = np.zeros_like(windowed_array)

    return windowed_array.astype(np.uint8)


def process_single_slice(args_tuple):
    """
    【多线程函数】处理单张切片 + 单种窗位
    """
    slice_data, plane_name, window_name, slice_idx, base_name, output_dir = args_tuple

    # 窗处理
    window_params = WINDOW_PRESETS[window_name]
    if window_params is None:
        windowed_slice = apply_window(slice_data)
    else:
        windowed_slice = apply_window(slice_data, window_params[0], window_params[1])

    # 构建文件名
    output_name: str = f"{base_name}_{plane_name}_{window_name}_{slice_idx}.png"
    output_path: str = os.path.join(output_dir, output_name)

    # 加锁保存（防止多线程写冲突）
    with lock:
        try:
            img = Image.fromarray(windowed_slice)
            img.save(output_path)
        except Exception as e:
            print(f"保存失败 {output_path}: {str(e)}")


def extract_and_save_slices(nii_path: str,
                            output_dir: str,
                            planes: Optional[List[str]] = None,
                            windows: Optional[List[str]] = None,
                            start_slice: Optional[int] = None,
                            end_slice: Optional[int] = None,
                            max_workers: int = 8) -> None:
    try:
        image: sitk.Image = sitk.ReadImage(nii_path)
        image_array: np.ndarray = sitk.GetArrayFromImage(image)
    except Exception as e:
        print(f"读取失败 {nii_path}: {e}")
        return

    file_name: str = os.path.basename(nii_path)
    base_name: str = os.path.splitext(file_name)[0]
    if base_name.endswith('.nii'):
        base_name = os.path.splitext(base_name)[0]

    os.makedirs(output_dir, exist_ok=True)

    if planes is None:
        planes = list(PLANES.keys())
    if windows is None:
        windows = list(WINDOW_PRESETS.keys())

    # 收集所有需要处理的切片任务
    tasks = []
    for plane_name in planes:
        if plane_name not in PLANES:
            continue
        axis = PLANES[plane_name]
        num_slices = image_array.shape[axis]

        current_start = start_slice if start_slice is not None else 0
        current_end = end_slice if end_slice is not None else num_slices
        current_start = max(0, current_start)
        current_end = min(num_slices, current_end)

        if current_start >= current_end:
            continue

        for slice_idx in range(current_start, current_end):
            # 提取切片
            if axis == 0:
                slice_data = image_array[slice_idx, :, :]
            elif axis == 1:
                slice_data = image_array[:, slice_idx, :]
            else:
                slice_data = image_array[:, :, slice_idx]

            # 每个窗位生成一个任务
            for wname in windows:
                if wname not in WINDOW_PRESETS:
                    continue
                tasks.append((
                    slice_data.copy(),
                    plane_name,
                    wname,
                    slice_idx,
                    base_name,
                    output_dir
                ))

    # 多线程执行所有切片
    if tasks:
        print(f"文件 {base_name} 总任务数：{len(tasks)}，启用 {max_workers} 线程")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(process_single_slice, tasks),
                total=len(tasks),
                desc=f"🚀 {base_name} 多线程处理中"
            ))


def batch_process(input_dir: str,
                  output_dir: str,
                  planes: Optional[List[str]] = None,
                  windows: Optional[List[str]] = None,
                  start_slice: Optional[int] = None,
                  end_slice: Optional[int] = None,
                  file_workers: int = 4,    # 同时处理多少个 .nii 文件
                  slice_workers: int = 8    # 每个文件用多少线程切切片
                  ) -> None:

    nii_files: List[str] = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))

    if not nii_files:
        print(f"未找到任何 NIfTI 文件")
        return

    print(f"✅ 找到 {len(nii_files)} 个文件，文件级并行：{file_workers} 线程")

    # 多线程处理多个文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=file_workers) as executor:
        futures = [
            executor.submit(
                extract_and_save_slices,
                file,
                output_dir,
                planes,
                windows,
                start_slice,
                end_slice,
                slice_workers
            )
            for file in nii_files
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="📁 总文件处理进度"):
            try:
                future.result()
            except Exception as e:
                print(f"文件处理异常: {e}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='【多线程加速版】CT NIfTI 切片提取工具'
    )
    parser.add_argument('--input', default=r'/home/liangshubo/Project/CTSR/dataset/rawdata/Elite_dataset/all_isotropic/',
                        help='输入路径')
    parser.add_argument('--output', default=r'/home/liangshubo/Project/CTSR/dataset/rawdata/Elite_dataset/slice/',
                        help='输出目录')
    parser.add_argument('--planes', nargs='+', default=['shi'],
                        help='zhou, guan, shi')
    parser.add_argument('--windows', nargs='+', default=[ 'nonewin','xiongwin', 'zonggewin', 'guwin'],
                        help='窗位')
    parser.add_argument('--start', type=int, default=0, help='起始切片')
    parser.add_argument('--end', type=int, default=None, help='结束切片')
    parser.add_argument('--batch', type=int, default=1, help='批量模式')
    # 多线程参数（Ubuntu 服务器可拉满）
    parser.add_argument('--file_workers', type=int, default=1, help='同时处理几个文件')
    parser.add_argument('--slice_workers', type=int, default=24, help='每个文件多少线程')

    args: argparse.Namespace = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.input):
            print("❌ 批量模式需要输入文件夹")
            sys.exit(1)
        batch_process(
            args.input, args.output, args.planes, args.windows,
            args.start, args.end, args.file_workers, args.slice_workers
        )
    else:
        if not os.path.isfile(args.input):
            print(f"❌ 文件不存在")
            sys.exit(1)
        extract_and_save_slices(
            args.input, args.output, args.planes, args.windows,
            args.start, args.end, args.slice_workers
        )

    print("\n🎉 全部处理完成！")


if __name__ == "__main__":
    main()