import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple, Any

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


def extract_and_save_slices(nii_path: str,
                            output_dir: str,
                            planes: Optional[List[str]] = None,
                            windows: Optional[List[str]] = None,
                            start_slice: Optional[int] = None,
                            end_slice: Optional[int] = None) -> None:
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
    # 读取NIfTI文件
    try:
        image: sitk.Image = sitk.ReadImage(nii_path)
        image_array: np.ndarray = sitk.GetArrayFromImage(image)  # 形状为(z, y, x)
    except Exception as e:
        print(f"错误：无法读取文件 {nii_path}，原因：{e}")
        return

    # 获取文件名(不含扩展名)
    file_name: str = os.path.basename(nii_path)
    base_name: str = os.path.splitext(file_name)[0]
    # 处理.nii.gz扩展名
    if base_name.endswith('.nii'):
        base_name = os.path.splitext(base_name)[0]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置默认值  如果没有指定 平面和 窗的 选项 将会全部都进行提取
    if planes is None:
        planes = list(PLANES.keys())
    if windows is None:
        windows = list(WINDOW_PRESETS.keys())

    # 遍历每个平面
    for plane_name in planes:
        if plane_name not in PLANES:
            print(f"警告：未知的平面名称 {plane_name}，跳过")
            continue

        axis: int = PLANES[plane_name]
        num_slices: int = image_array.shape[axis]

        # 设置切片范围   没有指定切片范围就从头到尾部全部都要
        current_start: int = start_slice if start_slice is not None else 0
        current_end: int = end_slice if end_slice is not None else num_slices

        # 确保切片范围有效
        current_start = max(0, current_start)
        current_end = min(num_slices, current_end)

        if current_start >= current_end:
            print(f"警告：平面 {plane_name} 的切片范围无效，跳过")
            continue

        print(
            f"正在处理 {plane_name} 平面，切片范围：{current_start}-{current_end - 1}，共 {current_end - current_start} 张")

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

            # 遍历每个窗位
            for window_name in windows:
                if window_name not in WINDOW_PRESETS:
                    print(f"警告：未知的窗位名称 {window_name}，跳过")
                    continue

                window_params: Optional[Tuple[int, int]] = WINDOW_PRESETS[window_name]

                # 应用窗宽窗位
                windowed_slice: np.ndarray
                if window_params is None:
                    windowed_slice = apply_window(slice_data)
                else:
                    windowed_slice = apply_window(slice_data, window_params[0], window_params[1])

                # 旋转图像以符合医学影像显示习惯
                # 轴向：逆时针旋转90度
                # 冠状面：逆时针旋转90度
                # 矢状面：逆时针旋转90度并左右翻转


                # 构建输出文件名
                output_name: str = f"{base_name}_{plane_name}_{window_name}_{slice_idx}.png"
                output_path: str = os.path.join(output_dir, output_name)

                # 保存为PNG
                try:
                    img: Image.Image = Image.fromarray(windowed_slice)
                    img.save(output_path)
                except Exception as e:
                    print(f"错误：无法保存文件 {output_path}，原因：{e}")
                    continue


def batch_process(input_dir: str,
                  output_dir: str,
                  planes: Optional[List[str]] = None,
                  windows: Optional[List[str]] = None,
                  start_slice: Optional[int] = None,
                  end_slice: Optional[int] = None) -> None:
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

        file_output_dir=output_dir
        extract_and_save_slices(nii_file, file_output_dir, planes, windows, start_slice, end_slice)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='从NIfTI CT数据中提取轴向、矢状面、冠状面切片并应用不同窗宽窗位'
    )
    parser.add_argument('--input',  default=r'/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_NII_RESPACE_Z', help='输入NIfTI文件路径或目录')
    parser.add_argument('--output',default=r'/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_SLIPTSLICE_SIMPLE/Zhou',  help='输出目录')
    parser.add_argument('--planes', nargs='+', default=['zhou'],   #  'guan', 'shi'
                        help='要提取的平面，可选值：zhou(轴向), guan(冠状面), shi(矢状面)')
    parser.add_argument('--windows', nargs='+', default=['nonewin', 'xiongwin', 'zonggewin', 'guwin'],
                        help='要应用的窗位，可选值：nonewin(不加窗), xiongwin(胸窗), zonggewin(纵隔窗), guwin(骨窗)')
    parser.add_argument('--start', default=0,type=int, help='起始切片索引')
    parser.add_argument('--end',default=None, type=int, help='结束切片索引')
    parser.add_argument('--batch',default=1, type=int,help='批量处理目录下的所有NIfTI文件')

    args: argparse.Namespace = parser.parse_args()

    # 验证输入
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"错误：批量模式下输入必须是目录")
            sys.exit(1)
        batch_process(args.input, args.output, args.planes, args.windows, args.start, args.end)
    else:
        if not os.path.isfile(args.input):
            print(f"错误：文件不存在 {args.input}")
            sys.exit(1)
        extract_and_save_slices(args.input, args.output, args.planes, args.windows, args.start, args.end)

    print("\n处理完成！")


if __name__ == "__main__":
    main()