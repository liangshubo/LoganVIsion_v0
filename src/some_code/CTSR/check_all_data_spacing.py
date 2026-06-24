import os
import SimpleITK as sitk


def print_nii_spacing_dimension(folder_path: str, recursive: bool = False):
    """
    输出文件夹内所有 nii / nii.gz 文件的 spacing 和 dimension(size)

    Args:
        folder_path: nii 文件所在文件夹
        recursive: 是否递归搜索子文件夹
    """

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"文件夹不存在: {folder_path}")

    nii_files = []

    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    nii_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nii_files.append(os.path.join(folder_path, file))

    if len(nii_files) == 0:
        print("没有找到 .nii 或 .nii.gz 文件")
        return

    print(f"{'FileName':<50} {'Spacing(x,y,z)':<35} {'Dimension/Size(x,y,z)':<30}")
    print("-" * 120)

    for nii_path in sorted(nii_files):
        try:
            image = sitk.ReadImage(nii_path)

            spacing = image.GetSpacing()   # 物理 spacing，顺序是 x, y, z
            size = image.GetSize()         # 图像尺寸，顺序是 x, y, z
            dim = image.GetDimension()     # 维度数量，一般 3D nii 是 3

            filename = os.path.basename(nii_path)

            print(
                f"{filename:<50} "
                f"{str(tuple(round(s, 6) for s in spacing)):<35} "
                f"{str(size) + '  dim=' + str(dim):<30}"
            )

        except Exception as e:
            print(f"读取失败: {nii_path}")
            print(f"错误信息: {e}")


if __name__ == "__main__":
    folder_path = r"/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_NII_RESPACE_Z"  # 修改成你的 nii 文件夹路径

    print_nii_spacing_dimension(folder_path, recursive=False)