import torch
import os

# 你需要抢救的两个文件路径
files_to_rescue = [
    'ckpt/osr/cifar10/0407_2126/medaf_backbone_converted.pth',
    'ckpt/osr/cifar10/0407_2126/epoch_150.pth'
]

print("=== 开始执行 PyTorch 权重抢救程序 ===")

for file_path in files_to_rescue:
    print(f"\n正在分析文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 找不到该文件，请检查路径是否正确。")
        continue

    # 生成抢救成功后的新文件名
    salvaged_path = file_path.replace('.pth', '_salvaged.pth')

    try:
        # 核心抢救逻辑：强制映射到 CPU，并且限制只加载权重字典
        # 提示：如果你的 PyTorch 版本较低报错不支持 weights_only，可以删掉这个参数
        weights = torch.load(file_path, map_location='cpu', weights_only=True)

        print(f"✅ 提取成功！底层张量未损坏。")
        print(f"💾 正在另存为: {salvaged_path}")
        torch.save(weights, salvaged_path)

    except Exception as e:
        print(f"❌ 抢救失败。文件的二进制结构已严重损坏。")
        print(f"报错详情: {e}")

print("\n=== 抢救程序执行完毕 ===")