import os
import sys
import traceback

# ==============================================================================
# 🛡️ 1. 环境防御层 (防止 DLL 冲突导致的闪退)
# ==============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 重定向错误输出到文件，防止控制台关闭后看不到
log_file = open("crash_log.txt", "w", encoding='utf-8')


def log(message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


log("🚀 [Step 1] 初始化环境...")
log(f"   Current Working Directory: {os.getcwd()}")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torch.nn.utils.parametrizations import weight_norm
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    log(f"   Torch Version: {torch.__version__}")
    log(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"   GPU: {torch.cuda.get_device_name(0)}")

except Exception as e:
    log("\n❌ [Fatal Error] 依赖库导入失败！")
    traceback.print_exc(file=log_file)
    print("详细错误已写入 crash_log.txt")
    input("按回车键退出...")
    sys.exit(1)

# ==============================================================================
# 🛡️ 2. 模型与逻辑定义 (直接内嵌，排除 Import 路径错误)
# ==============================================================================
try:
    log("🚀 [Step 2] 加载 net.py...")
    # 尝试导入你的 net.py
    sys.path.append(os.getcwd())  # 确保当前目录在 path 中
    from net import MultiBranchNet

    log("   ✅ 成功导入 MultiBranchNet")
except ImportError as e:
    log(f"\n❌ [Fatal Error] 找不到 net.py 或其依赖！")
    log(f"   请确保 debug_run.py 和 net.py 都在: {os.getcwd()}")
    traceback.print_exc(file=log_file)
    input("按回车键退出...")
    sys.exit(1)


# 定义 Projector
class GCDProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            weight_norm(nn.Linear(512, out_dim))
        )

    def forward(self, x): return self.mlp(x)


class Wrapper(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        # 初始化 OSR 模型
        self.backbone = MultiBranchNet(backbone='resnet18', num_classes=6)

        # 加载权重
        if os.path.exists(ckpt_path):
            log(f"   正在加载权重: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint.get('net', checkpoint.get('state_dict', checkpoint))
            new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(new_state, strict=False)
            log("   ✅ 权重加载完成")
        else:
            log(f"   ⚠️ 警告：找不到权重文件 {ckpt_path}，使用随机初始化测试！")

        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        self.projector = GCDProjector(512, 10)  # 假设输出是 512

    def forward(self, x):
        with torch.no_grad():
            out = self.backbone(x)
            feat = out['fts']
        return self.projector(feat)


# ==============================================================================
# 🛡️ 3. 数据完整性检查
# ==============================================================================
log("🚀 [Step 3] 检查数据路径...")
data_root = r'C:\Users\10943\PycharmProjects\PythonProject\MEDAF\data'
ckpt_path = r'C:\Users\10943\PycharmProjects\PythonProject\MEDAF\ckpt\osr\cifar10\0113_1204\epoch_150.pth'

if not os.path.exists(data_root):
    log(f"   ❌ 数据目录不存在: {data_root}")
    # 这里不退出，尝试自动下载
else:
    log("   ✅ 数据目录存在")


# ==============================================================================
# 🛡️ 4. 试运行 (Safe Run)
# ==============================================================================
def safe_run():
    log("🚀 [Step 4] 开始安全模式试运行...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = Wrapper(ckpt_path).to(device)

    # 准备极小数据 (Batch Size = 2, Num Workers = 0)
    # 这能彻底排除 显存溢出 和 多进程死锁
    log("   正在初始化 Dataset (如果卡住，说明是下载数据或文件IO问题)...")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # 使用 CIFAR10 官方类测试
    dataset = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)  # 👈 强制单线程

    log("   ✅ Dataset 初始化成功")

    # 模拟一次迭代
    log("   正在尝试 Forward Pass (显存检查)...")
    optimizer = SGD(model.projector.parameters(), lr=0.01)

    iterator = iter(loader)
    images, targets = next(iterator)
    images = images.to(device)

    # Forward
    logits = model(images)
    log(f"   ✅ Forward 成功. Output shape: {logits.shape}")

    # Backward
    loss = logits.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    log("   ✅ Backward 成功")

    log("\n🎉🎉🎉 诊断完成！系统环境正常，没有发生闪退。🎉🎉🎉")
    log("您可以放心地运行正式训练代码了（记得保持 num_workers=0）。")


if __name__ == '__main__':
    try:
        safe_run()
    except Exception as e:
        log("\n❌❌❌ [CRASH DETECTED] 程序崩溃了！❌❌❌")
        log(f"错误类型: {type(e).__name__}")
        log(f"错误信息: {str(e)}")
        traceback.print_exc(file=log_file)
        print("\n详细报错信息已保存到 crash_log.txt")

    log_file.close()
    print("\n--------------------------------------------------")
    input("👉 按回车键退出 (Press Enter to exit)...")