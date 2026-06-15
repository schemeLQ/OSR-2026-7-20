import os
import requests
import tarfile
from tqdm import tqdm
import shutil

# ================= 配置区 =================
# 存放 OOD 数据的根目录 (与 osr_main.py 对应)
DATA_ROOT = './data/datasets/OSR_External'

# 数据集下载链接 (来自 ODIN/ARPL 官方仓库)
URLS = {
    'imagenet_crop': 'https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz?dl=1',
    'imagenet_resize': 'https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz?dl=1',
    'lsun_crop': 'https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz?dl=1',
    'lsun_resize': 'https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?dl=1'
}

# 压缩包内的原始文件夹名称 -> 目标文件夹名称的映射
# (因为 tar 包解压出来的名字通常是 Imagenet/LSUN，我们需要重命名以区分 crop/resize)
FOLDER_MAP = {
    'Imagenet': 'imagenet_crop',  # Imagenet.tar.gz 解压后是 Imagenet
    'Imagenet_resize': 'imagenet_resize',
    'LSUN': 'lsun_crop',  # LSUN.tar.gz 解压后是 LSUN
    'LSUN_resize': 'lsun_resize'
}


# ================= 核心逻辑 =================
def download_file(url, dest_path):
    """带进度条的下载函数"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1KB

    print(f"📥 Downloading to {dest_path}...")
    with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def extract_and_rename(tar_path, extract_root, target_name):
    """解压并重命名文件夹"""
    print(f"📦 Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # 获取压缩包里的顶级目录名 (例如 "Imagenet")
            top_dir = os.path.commonprefix(tar.getnames())
            tar.extractall(path=extract_root)

        # 重命名: extract_root/Imagenet -> extract_root/imagenet_crop
        src_dir = os.path.join(extract_root, top_dir)
        dst_dir = os.path.join(extract_root, target_name)

        if os.path.exists(dst_dir):
            print(f"⚠️ Target directory {dst_dir} already exists. Merging/Skipping rename.")
        else:
            os.rename(src_dir, dst_dir)
            print(f"✅ Renamed {src_dir} -> {dst_dir}")

        # 删除压缩包以节省空间 (可选)
        os.remove(tar_path)
        print("🗑️ Cleaned up tar file.")

    except Exception as e:
        print(f"❌ Error during extraction: {e}")


def main():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
        print(f"Ccrated directory: {DATA_ROOT}")

    print(f"🚀 Start downloading {len(URLS)} datasets to {DATA_ROOT}...\n")

    # 1. ImageNet-Crop (即 Tiny-ImageNet Crop)
    # 对应 URLS['imagenet_crop'] -> 也就是 Imagenet.tar.gz
    # 解压后名字是 'Imagenet', 我们要把它改成 'imagenet_crop'

    # 遍历下载列表
    tasks = [
        ('imagenet_crop', 'Imagenet.tar.gz', 'Imagenet'),
        ('imagenet_resize', 'Imagenet_resize.tar.gz', 'Imagenet_resize'),
        ('lsun_crop', 'LSUN.tar.gz', 'LSUN'),
        ('lsun_resize', 'LSUN_resize.tar.gz', 'LSUN_resize')
    ]

    for target_folder, filename, original_folder_name in tasks:
        target_path = os.path.join(DATA_ROOT, target_folder)
        if os.path.exists(target_path):
            print(f"⏭️ {target_folder} already exists. Skipping.")
            continue

        url = URLS[target_folder]
        tar_path = os.path.join(DATA_ROOT, filename)

        # 下载
        download_file(url, tar_path)

        # 解压并重命名
        extract_and_rename(tar_path, DATA_ROOT, target_folder)
        print("-" * 50)

    print("\n🎉 All datasets are ready!")
    print(f"📂 Location: {os.path.abspath(DATA_ROOT)}")


if __name__ == "__main__":
    main()