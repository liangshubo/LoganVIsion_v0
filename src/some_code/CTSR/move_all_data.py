
import os
import shutil

def copy_subfolder_contents(source_dir, target_dir):
    """
    把源文件夹下所有一级子文件夹的内容，复制到目标文件夹
    :param source_dir: 源根文件夹
    :param target_dir: 目标文件夹（不存在会自动创建）
    """
    # 检查源文件夹是否存在
    if not os.path.isdir(source_dir):
        print(f"错误：源文件夹不存在 → {source_dir}")
        return

    # 自动创建目标文件夹（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    print(f"目标文件夹已准备好 → {target_dir}\n")

    # 遍历源文件夹下的所有一级子文件夹
    for folder_name in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, folder_name)

        # 只处理文件夹，跳过文件
        if not os.path.isdir(subfolder_path):
            continue

        print(f"正在处理子文件夹 → {folder_name}")

        # 遍历子文件夹里的所有内容
        for item_name in os.listdir(subfolder_path):
            src_item = os.path.join(subfolder_path, item_name)
            dest_item = os.path.join(target_dir, item_name)

            try:
                if os.path.isdir(src_item):
                    # 如果是子文件夹，递归复制整个目录
                    shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
                else:
                    # 如果是文件，直接复制
                    shutil.copy2(src_item, dest_item)
                print(f"  已复制 → {item_name}")
            except Exception as e:
                print(f"  复制失败 {item_name}：{str(e)}")

    print("\n✅ 所有子文件夹内容复制完成！")



# ====================== 在这里修改你的路径 ======================
SOURCE_FOLDER = r"/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_SLIPTSLICE/Zhou"  # 要扫描的根目录
TARGET_FOLDER = r"/home/liangshubo/Project/CTSR/dataset/rawdata/LDIC/LDIC_SLIPTSLICE/HR_all"    # 所有文件要复制到这里
# =================================================================
if not os.path.exists(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)
if __name__ == "__main__":
    copy_subfolder_contents(SOURCE_FOLDER, TARGET_FOLDER)