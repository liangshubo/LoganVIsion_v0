'''

this code used to check the all file size in folder and will move the file to the subfolder which
the size
'''


import shutil
from pathlib import Path
from typing import List, Tuple
import os

def get_size_bin(
        file_path: Path,
        bin_size_kb: int = 100
) -> Tuple[int, int]:
    """
    根据文件大小计算所属区间。

    例如：
    0KB   ~ 99.9KB  -> 0kb_100kb
    100KB ~ 199.9KB -> 100kb_200kb
    """

    size_bytes = file_path.stat().st_size
    size_kb = size_bytes / 1024

    bin_start = int(size_kb // bin_size_kb) * bin_size_kb
    bin_end = bin_start + bin_size_kb

    return bin_start, bin_end


def get_unique_target_path(target_path: Path) -> Path:
    """
    防止目标文件夹中出现重名文件。

    例如：
    image.png 已存在，则自动改为 image_1.png
    image_1.png 已存在，则自动改为 image_2.png
    """

    if not target_path.exists():
        return target_path

    parent = target_path.parent
    stem = target_path.stem
    suffix = target_path.suffix

    index = 1
    while True:
        new_path = parent / f"{stem}_{index}{suffix}"
        if not new_path.exists():
            return new_path
        index += 1


def move_files_by_size_distribution(
        folder_path: str,
        bin_size_kb: int = 100,
        recursive: bool = True,
        dry_run: bool = False
) -> None:
    """
    按文件大小区间创建子文件夹，并将文件移动到对应区间文件夹中。

    参数
    ----
    folder_path : str
        原始文件夹路径，例如 /data/zhou

    bin_size_kb : int
        每个区间的大小，默认 100KB

    recursive : bool
        是否递归处理子文件夹中的文件。
        True  表示处理所有子文件夹中的文件。
        False 表示只处理当前文件夹下的文件。

    dry_run : bool
        是否只预览，不真正移动文件。
        True  只打印移动计划。
        False 真正移动文件。
    """

    source_dir = Path(folder_path).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"文件夹不存在：{source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{source_dir}")

    parent_dir = source_dir.parent
    source_name = source_dir.name

    if recursive:
        files: List[Path] = [p for p in source_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in source_dir.iterdir() if p.is_file()]

    print(f"待处理文件数量：{len(files)}")
    print("-" * 60)

    for file_path in files:
        try:
            bin_start, bin_end = get_size_bin(
                file_path=file_path,
                bin_size_kb=bin_size_kb
            )

            target_dir_name = f"{source_name}_{bin_start}kb_{bin_end}kb"
            target_dir = Path(os.path.join(parent_dir , target_dir_name))

            target_path = Path(os.path.join(target_dir , file_path.name))
            target_path = get_unique_target_path(target_path)

            if dry_run:
                print(f"[预览] {file_path}  ->  {target_path}")
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(target_path))
                print(f"[移动] {file_path}  ->  {target_path}")

        except OSError as e:
            print(f"[失败] {file_path}，原因：{e}")

    print("-" * 60)

    if dry_run:
        print("预览完成：当前没有真正移动文件。")
    else:
        print("文件移动完成。")


if __name__ == "__main__":
    folder_path = r"/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_SLICE/Zhou/"

    move_files_by_size_distribution(
        folder_path=folder_path,
        bin_size_kb=50,
        recursive=False,
        dry_run=False
    )