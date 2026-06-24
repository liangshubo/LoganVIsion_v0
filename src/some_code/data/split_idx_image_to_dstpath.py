"""
功能：
    读取指定文件夹中的所有图像文件，文件名格式类似：
        name_12.png
        nameu_122.png
        abc_test_3.jpg

    程序会提取文件名最后的整数索引，例如：
        name_12.png      -> 12
        nameu_122.png    -> 122

    然后把索引号在指定集合中的图像文件，移动到目标文件夹中。

注意：
    1. 只提取文件名最后连续的整数。
    2. 目标文件夹不存在会自动创建。
    3. 如果目标文件夹中已经存在同名文件，默认会报错，避免误覆盖。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def get_last_index(file_path: Path) -> int | None:
    """
    从文件名末尾提取整数索引。

    例如：
        name_12.png -> 12
        image003.png -> 3
    """
    match = re.search(r"(\d+)$", file_path.stem)

    if match is None:
        return None

    return int(match.group(1))


def move_images_by_indices(
        src_dir: str | Path,
        dst_dir: str | Path,
        target_indices: Iterable[int],
) -> None:
    """
    将 src_dir 中索引号属于 target_indices 的图像文件移动到 dst_dir。
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    target_index_set = set(target_indices)

    if not src_path.exists():
        raise FileNotFoundError(f"源文件夹不存在: {src_path}")

    if not src_path.is_dir():
        raise NotADirectoryError(f"源路径不是文件夹: {src_path}")

    dst_path.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for file_path in src_path.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        index = get_last_index(file_path)

        if index is None:
            print(f"跳过：文件名末尾没有整数索引 -> {file_path.name}")
            continue

        if index not in target_index_set:
            continue

        target_path = dst_path / file_path.name

        if target_path.exists():
            raise FileExistsError(f"目标文件已存在，停止移动: {target_path}")

        shutil.move(str(file_path), str(target_path))
        moved_count += 1

        print(f"移动: {file_path.name} -> {target_path}")

    print(f"\n完成，共移动 {moved_count} 个文件。")


if __name__ == "__main__":
    src_dir = r"/home/liangshubo/Project/CTSR/dataset/rawdata/Elite_dataset/slice/"
    dst_dir = r"/home/liangshubo/Project/CTSR/dataset/rawdata/Elite_dataset/Test_HR"

    # 需要移动的索引号
    target_indices = [100, 300, 500,800]

    move_images_by_indices(
        src_dir=src_dir,
        dst_dir=dst_dir,
        target_indices=target_indices,
    )