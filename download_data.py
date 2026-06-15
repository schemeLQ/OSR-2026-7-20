import os
import requests
import zipfile
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_and_extract_tiny_imagenet(target_dir='./data'):
    # 1. 确保 data 目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    save_path = os.path.join(target_dir, filename)
    extract_path = os.path.join(target_dir, "tiny-imagenet-200")

    # 2. 检查是否已经存在
    if os.path.exists(extract_path):
        print("✅ Tiny-ImageNet 似乎已经存在于:", extract_path)
        print("跳过下载。如果文件损坏，请手动删除 data 文件夹重试。")
        return

    # 3. 下载
    print(f"正在下载 Tiny-ImageNet (约 237MB) ...")
    response = requests.get(url, stream=True, verify=False)
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    # 4. 解压
    print("正在解压...")
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # 5. 清理压缩包
    os.remove(save_path)
    print(f"✅ 完成！数据集已准备好：{extract_path}")


if __name__ == "__main__":
    # 需要安装 requests 和 tqdm
    # pip install requests tqdm
    try:
        download_and_extract_tiny_imagenet()
    except ImportError:
        print("缺少库，请先安装: pip install requests tqdm")
    except Exception as e:
        print(f"发生错误: {e}")