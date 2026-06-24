import os
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional


def advanced_resample_thick_ct(
        image: sitk.Image,
        z_target_spacing: float = 1.0,
        z_mode_equalxy:bool = False,
        xy_target_spacing_rate: float = 1,
        z_interpolator: int = sitk.sitkBSpline,
        xy_interpolator: int = sitk.sitkBSpline,
        pre_smooth_sigma: float = 0.3,
        post_sharpen_alpha: float = 0.25,
        default_pixel_value: float = -1000.0
) -> sitk.Image:
    """
    厚层CT各向同性重采样终极方案，四步法消除z轴伪影

    步骤：
    1. z轴预平滑（反混叠）
    2. 分轴重采样（z轴用Lanczos，x/y轴用B样条）
    3. 边缘增强后处理
    4. 伪影抑制滤波

    参数:
        image: 原始CT图像
        z_target_spacing: 目标体素间距(mm)
        z_interpolator: z轴插值器，推荐sitkLanczosWindowedSinc
        xy_interpolator: x/y轴插值器，推荐sitkBSpline
        pre_smooth_sigma: z轴预平滑sigma，0.2-0.5之间
        post_sharpen_alpha: 后锐化强度，0.1-0.4之间
        default_pixel_value: 边界填充值

    返回:
        重采样后的各向同性图像
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()

    # 计算z轴上采样倍数

    if z_mode_equalxy:
        z_target_spacing = original_spacing[0]
    z_upsample_factor = original_spacing[2] / z_target_spacing

    # 如果上采样倍数小于2，直接使用普通重采样
    if z_upsample_factor < 2.0:
        return resample_to_isotropic(image, z_target_spacing, xy_interpolator)

    print(f"检测到z轴上采样倍数: {z_upsample_factor:.1f}x，启用高级抗伪影模式")

    # ====================== 步骤1: z轴预平滑（反混叠）======================
    # 这是最关键的一步！大多数人都跳过了这一步
    # 对z轴进行轻微高斯平滑，消除混叠频率
    if pre_smooth_sigma > 0:
        print("步骤1/4: 执行z轴反混叠预平滑...")
        smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        # 修复：ITK不允许sigma为0，使用极小值1e-6代替
        # 这样x/y轴几乎不会被平滑，只在z轴方向进行平滑
        smooth_filter.SetSigma((1e-6, 1e-6, pre_smooth_sigma))
        image = smooth_filter.Execute(image)

    # ====================== 步骤2: 分轴重采样 ======================
    print("步骤2/4: 执行分轴重采样...")

    # 第一步：只重采样z轴
    z_target_size = int(np.round(original_size[2] * original_spacing[2] / z_target_spacing))

    resampler_z = sitk.ResampleImageFilter()
    resampler_z.SetOutputSpacing((original_spacing[0], original_spacing[1], z_target_spacing))
    resampler_z.SetOutputDirection(original_direction)
    resampler_z.SetOutputOrigin(original_origin)
    resampler_z.SetSize((original_size[0], original_size[1], z_target_size))
    resampler_z.SetDefaultPixelValue(default_pixel_value)
    resampler_z.SetOutputPixelType(image.GetPixelID())
    resampler_z.SetInterpolator(z_interpolator)

    image_z = resampler_z.Execute(image)

    # 第二步：重采样x和y轴

    if xy_target_spacing_rate == 1 :
        image_resampled =  image_z
    else:
        xy_target_spacing = original_spacing[0]*xy_target_spacing_rate
        target_size_xy = (
            int(np.round(original_size[0] * original_spacing[0] / xy_target_spacing )),
            int(np.round(original_size[1] * original_spacing[1] /xy_target_spacing )),
            z_target_size
        )

        resampler_xy = sitk.ResampleImageFilter()
        resampler_xy.SetOutputSpacing((xy_target_spacing, xy_target_spacing, z_target_spacing))
        resampler_xy.SetOutputDirection(original_direction)
        resampler_xy.SetOutputOrigin(original_origin)
        resampler_xy.SetSize(target_size_xy)
        resampler_xy.SetDefaultPixelValue(default_pixel_value)
        resampler_xy.SetOutputPixelType(image.GetPixelID())
        resampler_xy.SetInterpolator(xy_interpolator)

        image_resampled = resampler_xy.Execute(image_z)

    # ====================== 步骤3: 边缘增强后处理 ======================
    # ====================== 步骤3: 边缘增强后处理 ======================
    if post_sharpen_alpha > 0:
        print("步骤3/4: 执行边缘增强后处理...")
        # 使用非锐化掩模增强边缘，只在x/y轴平滑，z轴保持不变
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma((0.8, 0.8, 1e-6))  # 只在x/y轴平滑
        smoothed = gaussian.Execute(image_resampled)

        # 简单但有效的非锐化掩模公式
        # sharpened = original + alpha * (original - smoothed)
        sharpened = sitk.Add(
            image_resampled,
            sitk.Multiply(
                sitk.Subtract(image_resampled, smoothed),
                post_sharpen_alpha
            )
        )
    else:
        sharpened = image_resampled

    # ====================== 步骤4: 伪影抑制滤波 ======================
    print("步骤4/4: 执行z轴伪影抑制滤波...")
    # 使用中值滤波抑制z轴方向的条纹伪影
    median_filter = sitk.MedianImageFilter()
    # 只在z轴方向使用3x1x1的核
    median_filter.SetRadius((0, 0, 1))
    final_image = median_filter.Execute(sharpened)

    return final_image


def resample_to_isotropic(
        image: sitk.Image,
        target_spacing: float = 1.0,
        interpolator: int = sitk.sitkBSpline,
        default_pixel_value: float = -1000.0
) -> sitk.Image:
    """普通各向同性重采样，用于上采样倍数较小的情况"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    target_size = (
        int(np.round(original_size[0] * original_spacing[0] / target_spacing)),
        int(np.round(original_size[1] * original_spacing[1] / target_spacing)),
        int(np.round(original_size[2] * original_spacing[2] / target_spacing))
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((target_spacing, target_spacing, target_spacing))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetSize(target_size)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetOutputPixelType(image.GetPixelID())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def resample_nii_file(
        input_path: str,
        output_path: str,
        z_target_spacing: float = 1.0,
        z_mode_equalxy=False,
xy_target_spacing_rate=1,
        use_advanced_mode: bool = True
) -> None:
    """重采样单个NIfTI文件"""
    try:
        print(f"正在读取文件: {input_path}")
        image = sitk.ReadImage(input_path)

        original_spacing = image.GetSpacing()
        print(f"原始间距: {original_spacing[0]:.2f} x {original_spacing[1]:.2f} x {original_spacing[2]:.2f} mm")

        # 检查是否已经是各向同性
        if (abs(original_spacing[0] - original_spacing[1]) < 0.01 and
                abs(original_spacing[1] - original_spacing[2]) < 0.01):
            print("图像已经是各向同性，跳过重采样")
            sitk.WriteImage(image, output_path)
            return

        if use_advanced_mode:
            print("使用高级抗伪影重采样模式")
            resampled_image = advanced_resample_thick_ct(
                image,
                z_target_spacing=z_target_spacing,
                z_mode_equalxy= z_mode_equalxy,
                xy_target_spacing_rate=xy_target_spacing_rate,
                z_interpolator=sitk.sitkLanczosWindowedSinc,
                xy_interpolator=sitk.sitkBSpline,
                pre_smooth_sigma=0.3,
                post_sharpen_alpha=0.25
            )
        else:
            print("使用普通重采样模式")
            resampled_image = resample_to_isotropic(
                image,
                target_spacing=z_target_spacing,
                interpolator=sitk.sitkBSpline
            )

        new_spacing = resampled_image.GetSpacing()
        new_size = resampled_image.GetSize()
        print(f"重采样完成，新间距: {new_spacing[0]:.2f} x {new_spacing[1]:.2f} x {new_spacing[2]:.2f} mm")
        print(f"新尺寸: {new_size[0]} x {new_size[1]} x {new_size[2]}")

        sitk.WriteImage(resampled_image, output_path)
        print(f"已保存到: {output_path}")

    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {e}")
        import traceback
        traceback.print_exc()


# 使用示例
if __name__ == "__main__":
    # 这是针对5mm厚层CT重采样到1mm的最佳配置
    resample_nii_file(
        input_path=r"E:\project\Dataset\CT-0\TEST\study_0001.nii.gz",
        output_path=r"E:\project\Dataset\CT-0\TEST\study_0001_resampled.nii.gz",
        z_target_spacing= 3.0,
        z_mode_equalxy = False,
        xy_target_spacing_rate = 1,
        use_advanced_mode=True
    )