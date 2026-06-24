import os
from collections import defaultdict
from typing import Dict, Tuple


def count_file_size_distribution(
        folder_path: str,
        bin_size_kb: int = 100
) -> Dict[Tuple[int, int], int]:
    """
    统计文件夹内所有文件的大小分布。

    参数
    ----
    folder_path : str
        要统计的文件夹路径
    bin_size_kb : int
        分布区间大小，默认每 100 KB 一个区间

    返回
    ----
    Dict[Tuple[int, int], int]
        key 为大小区间，例如 (0, 100)、(100, 200)
        value 为该区间内的文件数量
    """

    distribution: Dict[Tuple[int, int], int] = defaultdict(int)

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            try:
                # 文件大小，单位：Byte
                size_bytes = os.path.getsize(file_path)

                # 转换为 KB
                size_kb = size_bytes / 1024

                # 计算属于哪个 100KB 区间
                bin_start = int(size_kb // bin_size_kb) * bin_size_kb
                bin_end = bin_start + bin_size_kb

                distribution[(bin_start, bin_end)] += 1

            except OSError as e:
                print(f"无法读取文件：{file_path}，原因：{e}")

    return dict(distribution)


def print_distribution(distribution: Dict[Tuple[int, int], int]) -> None:
    """
    打印文件大小分布结果
    """

    print("文件大小分布统计：")
    print("-" * 30)

    for size_range in sorted(distribution.keys()):
        start, end = size_range
        count = distribution[size_range]
        print(f"{start:>6} KB - {end:>6} KB : {count} 个文件")


if __name__ == "__main__":
    folder_path = r"/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_SLICE/Guan"  # 修改为你的文件夹路径

    distribution = count_file_size_distribution(
        folder_path=folder_path,
        bin_size_kb=50
    )

    print_distribution(distribution)