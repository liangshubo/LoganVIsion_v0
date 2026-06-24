import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from sympy.codegen import Print


def parse_lidc_filename(file_path: str | Path) -> Tuple[int, str, int, str]:
    """
    解析 LIDC 图像文件名。

    示例文件名：
        LIDC_IDRI_0001_NA_03192_guan_xiongwin_172.png

    解析结果：
        instance_id = 1
        window_name = xiongwin
        frame_idx = 172
        filename = 原始文件名，用于排序兜底

    返回：
        instance_id: 病例实例编号，例如 0001 -> 1
        window_name: 窗类型，例如 xiongwin / guwin / zonggewin
        frame_idx: 帧索引，例如 172
        filename: 原始文件名
    """
    file_path = Path(file_path)
    stem = file_path.stem
    parts = stem.split("_")

    # 示例：
    # ['LIDC', 'IDRI', '0001', 'NA', '03192', 'guan', 'xiongwin', '172']

    if len(parts) < 8:
        raise ValueError(f"文件名格式不符合要求: {file_path.name}")

    # LIDC_IDRI_0001 中的 0001
    instance_id = int(parts[2])

    # png 前面的数字是帧索引
    frame_idx = int(parts[-1])

    # 帧索引前面一个字段是窗类型
    window_name = parts[-2]

    return instance_id, window_name, frame_idx, file_path.name


def sort_lidc_images(
    folder: str | Path,
    window_order: Optional[List[str]] = None,
    recursive: bool = False,
    suffixes: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> List[Path]:
    """
    按照 实例编号 -> 窗类型 -> 帧索引 对图像文件排序。

    参数：
        folder:
            图像所在文件夹。

        window_order:
            窗类型排序顺序。
            例如：
                ["xiongwin", "guwin", "zonggewin", "nonewin"]

            如果遇到不在 window_order 里的窗类型，会排在后面。

        recursive:
            是否递归读取子文件夹。
            False: 只读取当前文件夹。
            True : 递归读取所有子文件夹。

        suffixes:
            支持的图像后缀。

    返回：
        排序后的 Path 列表。
    """
    folder = Path(folder)

    if window_order is None:
        # 这里可以按照你的需求调整顺序
        window_order = [
            "xiongwin",    # 胸窗
            "guwin",       # 骨窗
            "zonggewin",   # 纵隔窗
            "nonewin",     # 无窗
        ]

    window_rank: Dict[str, int] = {
        name: idx for idx, name in enumerate(window_order)
    }

    # 读取图像文件
    if recursive:
        files = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in suffixes
        ]
    else:
        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in suffixes
        ]

    def sort_key(p: Path):
        instance_id, window_name, frame_idx, filename = parse_lidc_filename(p)

        # 不在 window_order 中的窗类型排到后面
        win_rank = window_rank.get(window_name, 9999)

        return (
            instance_id,       # 先按病例编号排序
            win_rank,          # 再按窗类型顺序排序
            window_name,       # 未知窗类型按名字排序，防止顺序不稳定
            frame_idx,         # 最后按帧索引排序
            filename,          # 兜底
        )

    sorted_files = sorted(files, key=sort_key)

    return sorted_files



from pathlib import Path
from typing import List, Tuple
import shutil


def copy_files_by_interval(

    file_names: List[str],
    target_dir: str | Path,
    interval: int,
    start_index: int = 0,
) -> Tuple[List[Path], List[Path]]:
    """
    按照指定间隔，从文件名列表中抽取文件，并从源文件夹复制到目标文件夹。

    参数：
        source_dir:
            源文件夹路径。

        file_names:
            文件名列表，例如：
            [
                "LIDC_IDRI_0001_NA_03192_guan_xiongwin_1.png",
                "LIDC_IDRI_0001_NA_03192_guan_xiongwin_2.png",
                ...
            ]

        target_dir:
            目标文件夹路径。

        interval:
            间隔数量。
            例如 interval=5，表示每隔5个取一个：
            file_names[0], file_names[5], file_names[10] ...

        start_index:
            从第几个文件开始取，默认从0开始。

    返回：
        copied_files:
            成功复制的目标文件路径列表。

        missing_files:
            源路径中不存在的文件路径列表。
    """

    target_dir = Path(target_dir)
    #print(source_dir)

    # print(target_dir)
    if interval <= 0:
        raise ValueError("interval 必须大于 0")

    if start_index < 0:
        raise ValueError("start_index 不能小于 0")

    # 创建目标文件夹
    if not os.path.exists(target_dir):
       os.mkdir(target_dir)

    copied_files: List[Path] = []
    missing_files: List[Path] = []

    # 按间隔抽取文件名
    selected_file_names = file_names[start_index::interval]

    for file_name in selected_file_names:
        src_path = file_name
        #print(target_dir)
        dst_path = os.path.join(target_dir  ,file_name.name)
        #print(src_path,"src" , dst_path,"dst")
        if not src_path.exists():
            print(f"文件不存在，跳过: {src_path}")
            missing_files.append(src_path)
            continue

        # 如果目标路径的父文件夹不存在，则创建
       # dst_path.parent.mkdir(parents=True, exist_ok=True)

        # copy2 会尽量保留文件的修改时间等元信息

        #print(src_path , dst_path)
        shutil.copy2(src_path, dst_path)

        copied_files.append(dst_path)
        print(f"复制成功: {src_path} -> {dst_path}")

    print(f"\n复制完成")
    print(f"成功复制数量: {len(copied_files)}")
    print(f"缺失文件数量: {len(missing_files)}")

    return copied_files, missing_files





if __name__ == "__main__":
    all_data = 0
    size_list = ["0kb_50kb","50kb_100kb","100kb_150kb","150kb_200kb","200kb_250kb"]
    for i in range(0,len(size_list)):
        folder = r"/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_SLICE/Zhou/Zhou_"+size_list[i]

        target_dir = r"/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_SLIPTSLICE/Zhou/Zhou_"+size_list[i]

        os.makedirs(target_dir, exist_ok=True)
        sorted_images = sort_lidc_images(
            folder,
            window_order=["xiongwin", "guwin", "zonggewin", "nonewin"],
            recursive=False,
        )

        for p in sorted_images[:-1]:

            print(p.name)
        print(len(sorted_images))


        # ----------------

        print(target_dir)

        copied_files, missing_files = copy_files_by_interval(

            file_names=sorted_images,
            target_dir=target_dir,
            interval=10,
            start_index=0,
        )
        all_data += len(copied_files)
    print(f"all data:{all_data}")

