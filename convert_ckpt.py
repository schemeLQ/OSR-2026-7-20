import torch
import os

# 1. 定义路径
source_path = r'C:\Users\10943\PycharmProjects\PythonProject\MEDAF\ckpt\osr\cifar10\0513_2201\model_best.pth'
target_path = r'C:\Users\10943\PycharmProjects\PythonProject\MEDAF\ckpt\osr\cifar10\0513_2201\medaf_backbone_converted.pth'


def convert_osr_to_ncd(src, dst):
    # 2. 加载原始权重
    print(f"正在加载: {src}")
    checkpoint = torch.load(src, map_location='cpu')

    # 3. 获取真正的 state_dict
    # 因为 3.03/3月3日版本是将权重包裹在 'state_dict' 键值对里的
    if 'state_dict' in checkpoint:
        full_sd = checkpoint['sta=te_dict']
    else:
        full_sd = checkpoint

    # 4. 核心过滤逻辑
    backbone_dict = {}
    for k, v in full_sd.items():
        # 去除 DataParallel 产生的 'module.' 前缀
        new_k = k.replace('module.', '')

        # 排除包含 'fc' 或 'classifier' 的分类头参数
        # 这样下游任务加载时会重新随机初始化 DINOHead，避免旧类偏见
        if 'fc' not in new_k and 'classifier' not in new_k:
            backbone_dict[new_k] = v

    # 5. 保存纯净的骨干网络权重
    torch.save(backbone_dict, dst)
    print(f"✅ 转换完成！已保存至: {dst}")
    print(f"   保留参数量: {len(backbone_dict)} 项")


if __name__ == '__main__':
    convert_osr_to_ncd(source_path, target_path)