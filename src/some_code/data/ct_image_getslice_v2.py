import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tensorboard.compat.tensorflow_stub.dtypes import double
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


def rotate_3d_image(image: sitk.Image, plane_name: str, angle_deg: float,default_pixel_value: float) -> sitk.Image:
    """
    【新增】3D医学图像中心旋转，生成指定平面的倾斜切面
    :param image: 原始sitk图像
    :param plane_name: 平面名称(zhou/guan/shi)
    :param angle_deg: 倾斜角度(度数)
    :return: 旋转后的sitk图像
    """
    if angle_deg == 0:
        return image  # 0度不旋转，直接返回，提升速度

    # 角度转弧度（SimpleITK使用弧度）
    angle_rad = np.deg2rad(angle_deg)
    # 获取图像中心（物理空间坐标）
    center = image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0)
    # 3D旋转变换（中心旋转，无平移）
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    print("will transformer ")
    # 关键：按平面指定旋转轴，生成医学标准斜切面
    if plane_name == 'zhou':    # 轴向：绕Y轴旋转
        transform.SetRotation(0, angle_rad, 0)
    elif plane_name == 'guan':  # 冠状面：绕X轴旋转
        transform.SetRotation(angle_rad, 0, 0)
    elif plane_name == 'shi':   # 矢状面：绕Z轴旋转
        transform.SetRotation(0, 0, angle_rad)

    # 重采样（线性插值，保证图像质量）
    print("will resampler  with bsline3  ")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkBSpline3)
    resampler.SetDefaultPixelValue(float(default_pixel_value))

    return resampler.Execute(image)


def rotate_3d_image_v2(image: sitk.Image, plane_name: str, angle_deg: float, default_pixel_value: float) -> sitk.Image:
    """
    【修复版】3D医学图像中心旋转，自动调整输出大小以完整显示旋转后的图像
    :param image: 原始sitk图像
    :param plane_name: 平面名称(zhou/guan/shi)
    :param angle_deg: 倾斜角度(度数)
    :param default_pixel_value: 超出原始图像区域的填充值
    :return: 旋转后的sitk图像（尺寸自动扩大，无截断）
    """
    if angle_deg == 0:
        return image  # 0度不旋转，直接返回，提升速度

    # 角度转弧度（SimpleITK使用弧度）
    angle_rad = np.deg2rad(angle_deg)
    # 获取图像中心（物理空间坐标）
    center = image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0)
    # 3D旋转变换（中心旋转，无平移）
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)

    # 关键：按平面指定旋转轴，生成医学标准斜切面
    if plane_name == 'zhou':    # 轴向：绕Y轴旋转
        transform.SetRotation(0, angle_rad, 0)
    elif plane_name == 'guan':  # 冠状面：绕X轴旋转
        transform.SetRotation(angle_rad, 0, 0)
    elif plane_name == 'shi':   # 矢状面：绕Z轴旋转
        transform.SetRotation(0, 0, angle_rad)

    # ===================== 【核心修复1：计算旋转后的边界框】 =====================
    # 获取原始图像的8个角点（物理空间坐标）
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    # 生成8个角点的索引坐标
    corners_idx = [
        (0, 0, 0), (size[0], 0, 0), (0, size[1], 0), (size[0], size[1], 0),
        (0, 0, size[2]), (size[0], 0, size[2]), (0, size[1], size[2]), (size[0], size[1], size[2])
    ]

    # 将索引坐标转换为物理空间坐标，并应用旋转变换
    transformed_corners = []
    for idx in corners_idx:
        physical_point = image.TransformIndexToPhysicalPoint(idx)
        transformed_point = transform.TransformPoint(physical_point)
        transformed_corners.append(transformed_point)

    # 计算变换后的最小和最大物理坐标
    transformed_corners_np = np.array(transformed_corners)
    min_physical = np.min(transformed_corners_np, axis=0)
    max_physical = np.max(transformed_corners_np, axis=0)

    # 计算新的输出大小（保持原始间距不变）
    new_size = [
        int(np.ceil((max_physical[0] - min_physical[0]) / spacing[0])),
        int(np.ceil((max_physical[1] - min_physical[1]) / spacing[1])),
        int(np.ceil((max_physical[2] - min_physical[2]) / spacing[2]))
    ]

    # ===================== 【核心修复2：配置重采样参数】 =====================
    resampler = sitk.ResampleImageFilter()
    # 不再使用原始图像作为参考，手动设置所有输出参数
    resampler.SetOutputOrigin(min_physical.tolist())  # 新原点为变换后的最小坐标
    resampler.SetOutputSpacing(spacing)               # 保持原始像素间距
    resampler.SetOutputDirection(direction)           # 保持原始方向
    resampler.SetSize(new_size)                       # 使用计算出的新大小
    resampler.SetTransform(transform.GetInverse())    # 关键：使用逆变换（输出→输入）
    resampler.SetInterpolator(sitk.sitkBSpline)       # B样条插值，质量更高
    resampler.SetDefaultPixelValue(float(default_pixel_value))

    return resampler.Execute(image)


import numpy as np
import SimpleITK as sitk


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    绕任意单位向量 axis 旋转 angle_rad 的 3x3 矩阵
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=np.float64)


def resample_oblique_volume(
        image: sitk.Image,
        plane_name: str = "zhou",
        angle_deg: float = 25.0,
        default_pixel_value: float = -1000.0,
        interpolator=sitk.sitkBSpline3
) -> sitk.Image:
    """
    直接从原始3D CT中生成倾斜MPR体数据。
    输出仍是3D，但它的 z 方向已经是倾斜方向。
    后面再 GetArrayFromImage 后按 z/y/x 切片即可。
    """

    size = np.array(image.GetSize(), dtype=np.int64)        # x, y, z
    spacing = np.array(image.GetSpacing(), dtype=np.float64)

    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    # 原图物理坐标系下的 x/y/z 方向单位向量
    ex = direction[:, 0]
    ey = direction[:, 1]
    ez = direction[:, 2]

    angle_rad = np.deg2rad(angle_deg)

    # 标准三个平面的输出坐标轴
    # u: 输出图像横向
    # v: 输出图像纵向
    # n: 切片推进方向
    if plane_name == "zhou":
        # 轴向面：标准为 x-y 平面，法向为 z
        u, v, n = ex.copy(), ey.copy(), ez.copy()

        # 你的原代码中 zhou 是绕 y 轴旋转
        R = rotation_matrix(ey, angle_rad)

    elif plane_name == "guan":
        # 冠状面：标准为 x-z 平面，法向为 y
        u, v, n = ex.copy(), ez.copy(), ey.copy()

        # 你的原代码中 guan 是绕 x 轴旋转
        R = rotation_matrix(ex, angle_rad)

    elif plane_name == "shi":
        # 矢状面：标准为 y-z 平面，法向为 x
        u, v, n = ey.copy(), ez.copy(), ex.copy()

        # 你的原代码中 shi 是绕 z 轴旋转
        R = rotation_matrix(ez, angle_rad)

    else:
        raise ValueError("plane_name 必须是 'zhou'、'guan' 或 'shi'")

    # 旋转输出平面的三个方向向量
    u = R @ u
    v = R @ v
    n = R @ n

    # 输出方向矩阵，SimpleITK 按列解释 x/y/z 三个输出轴方向
    out_direction = np.stack([u, v, n], axis=1)

    # 输出尺寸和 spacing
    if plane_name == "zhou":
        out_size = [int(size[0]), int(size[1]), int(size[2])]
        out_spacing = [float(spacing[0]), float(spacing[1]), float(spacing[2])]
    elif plane_name == "guan":
        out_size = [int(size[0]), int(size[2]), int(size[1])]
        out_spacing = [float(spacing[0]), float(spacing[2]), float(spacing[1])]
    else:  # shi
        out_size = [int(size[1]), int(size[2]), int(size[0])]
        out_spacing = [float(spacing[1]), float(spacing[2]), float(spacing[0])]

    out_size_np = np.array(out_size, dtype=np.float64)
    out_spacing_np = np.array(out_spacing, dtype=np.float64)

    # 原始图像中心，物理坐标
    center_index = (size - 1) / 2.0
    center_physical = np.array(
        image.TransformContinuousIndexToPhysicalPoint(center_index.tolist()),
        dtype=np.float64
    )

    # 让输出MPR体数据的中心和原始体数据中心对齐
    half_extent = (out_size_np - 1) * out_spacing_np / 2.0
    out_origin = (
        center_physical
        - half_extent[0] * u
        - half_extent[1] * v
        - half_extent[2] * n
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(x) for x in out_size])
    resampler.SetOutputSpacing([float(x) for x in out_spacing])
    resampler.SetOutputOrigin(out_origin.tolist())
    resampler.SetOutputDirection(out_direction.reshape(-1).tolist())

    # 关键：这里用 identity
    # 因为输出空间本身已经被定义到了原图物理空间里
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))

    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_pixel_value))

    return resampler.Execute(image)



def extract_and_save_slices(nii_path: str,
                            output_dir: str,
                            planes: Optional[List[str]] = None,
                            windows: Optional[List[str]] = None,
                            degrees: Optional[List[int]] = None,  # 【新增】角度参数
                            start_slice: Optional[int] = None,
                            end_slice: Optional[int] = None) -> None:
    """
    从NIfTI文件中提取【标准+倾斜】切片并保存为PNG
    """
    # 读取NIfTI文件
    try:
        print("read nii ")
        image: sitk.Image = sitk.ReadImage(nii_path)
        image_array: np.ndarray = sitk.GetArrayFromImage(image)
        global_min: float = float(np.min(image_array).item())
    except Exception as e:
        print(f"错误：无法读取文件 {nii_path}，原因：{e}")
        return

    # 获取文件名(不含扩展名)
    file_name: str = os.path.basename(nii_path)
    base_name: str = os.path.splitext(file_name)[0]
    if base_name.endswith('.nii'):
        base_name = os.path.splitext(base_name)[0]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 默认值
    if planes is None:
        planes = list(PLANES.keys())
    if windows is None:
        windows = list(WINDOW_PRESETS.keys())
    if degrees is None:
        degrees = [0]  # 【新增】默认0度（标准切面）

    # 遍历：平面 → 角度 → 切片 → 窗位
    for plane_name in planes:
        if plane_name not in PLANES:
            print(f"警告：未知平面 {plane_name}，跳过")
            continue

        axis: int = PLANES[plane_name]

        # ===================== 【新增】遍历倾斜角度 =====================
        for angle in degrees:
            print(f"\n=== 处理 {plane_name} 平面，倾斜角度 {angle}° ===")
            # 生成旋转后的3D图像（0度=原图）
            #rotated_img = rotate_3d_image_v2(image, plane_name, angle,global_min)
            rotated_img = resample_oblique_volume(image, plane_name, angle,global_min)
            img_array = sitk.GetArrayFromImage(rotated_img)

            # 新方法下，切片序列统一在第0维
            num_slices = img_array.shape[0]

            current_start = start_slice if start_slice is not None else 0
            current_end = end_slice if end_slice is not None else num_slices
            current_start = max(0, current_start)
            current_end = min(num_slices, current_end)

            for slice_idx in tqdm(range(current_start, current_end), desc=f"{plane_name}-{angle}°"):
                slice_data = img_array[slice_idx, :, :]

                # 遍历窗位
                for window_name in windows:
                    if window_name not in WINDOW_PRESETS:
                        print(f"警告：未知窗位 {window_name}，跳过")
                        continue

                    window_params = WINDOW_PRESETS[window_name]
                    windowed_slice = apply_window(slice_data, *window_params) if window_params else apply_window(
                        slice_data)

                    # ===================== 【修改】文件名：添加角度 =====================
                    plane_suffix = f"{plane_name}_degree_{angle}"
                    output_name: str = f"{base_name}_{plane_suffix}_{window_name}_{slice_idx}.png"
                    output_path: str = os.path.join(output_dir, output_name)

                    # 保存
                    try:
                        Image.fromarray(windowed_slice).save(output_path)
                    except Exception as e:
                        print(f"错误：保存失败 {output_path}，原因：{e}")


def batch_process(input_dir: str,
                  output_dir: str,
                  planes: Optional[List[str]] = None,
                  windows: Optional[List[str]] = None,
                  degrees: Optional[List[int]] = None,  # 【新增】角度
                  start_slice: Optional[int] = None,
                  end_slice: Optional[int] = None) -> None:
    """批量处理所有NIfTI文件"""
    nii_files: List[str] = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))

    if not nii_files:
        print(f"未找到NIfTI文件")
        return

    print(f"找到 {len(nii_files)} 个文件")
    for i, nii_file in enumerate(nii_files):
        print(f"\n处理第 {i + 1}/{len(nii_files)} 个：{nii_file}")
        base_name = os.path.splitext(os.path.basename(nii_file))[0]
        base_name = os.path.splitext(base_name)[0] if base_name.endswith('.nii') else base_name
        file_output_dir = output_dir#os.path.join(output_dir, base_name)
        extract_and_save_slices(nii_file, file_output_dir, planes, windows, degrees, start_slice, end_slice)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='NIfTI CT切片提取：支持标准切面 + 倾斜角度切面'
    )
    parser.add_argument('--input', default=r'/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/Test',
                        help='输入文件/目录')
    parser.add_argument('--output', default=r'/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/TestSLICEv3',
                        help='输出目录')
    parser.add_argument('--planes', nargs='+', default=['shi'],   #  'zhou', 'guan', 'shi'
                        help='平面：zhou(轴), guan(冠), shi(矢)')
    parser.add_argument('--windows', nargs='+', default=['guwin'],   # 'nonewin', 'xiongwin', 'zonggewin', 'guwin'
                        help='窗位')
    # ===================== 【新增】角度参数 =====================
    parser.add_argument('--degrees', nargs='+', type=float, default=[0,15,45],
                        help='倾斜角度(度数)，默认0°(标准切面)，例：--degrees 0 6 10')
    parser.add_argument('--start', type=int, default=0, help='起始切片')
    parser.add_argument('--end', type=int,  default=None, help='结束切片')
    parser.add_argument('--batch', type=int, default=1, help='批量处理=1，单个文件=0')

    args: argparse.Namespace = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.input):
            print("错误：批量模式需输入目录")
            sys.exit(1)
        batch_process(args.input, args.output, args.planes, args.windows, args.degrees, args.start, args.end)
    else:
        if not os.path.isfile(args.input):
            print(f"错误：文件不存在")
            sys.exit(1)
        extract_and_save_slices(args.input, args.output, args.planes, args.windows, args.degrees, args.start, args.end)

    print("\n✅ 处理完成！")


if __name__ == "__main__":
    main()