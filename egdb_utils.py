"""
egdb_utils.py
新增的工具类，配合 EGDB-Aug 使用。
可以直接 append 到 ncd_utils.py 末尾，或作为独立文件导入。
"""

from torchvision import transforms


class ThreeCropTransform:
    """
    生成三个视图：
      - view1: 弱增强（与原 TwoCropTransform 相同的增强强度）
      - view2: 弱增强（与 view1 同级，用于基础 SimCLR）
      - view3: 强增强（RandAugment + 更强的 ColorJitter），用于 EGDB-Aug

    使用方式（替换 TwoCropTransform）：
        ncd_transform = ThreeCropTransform(weak_transform, strong_transform)

    在训练循环中接收：
        for u_images, _, _ in loader:
            view1, view2, view3 = u_images[0], u_images[1], u_images[2]
    """

    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        return [
            self.weak_transform(x),   # view1: 弱增强
            self.weak_transform(x),   # view2: 弱增强（与 view1 独立采样）
            self.strong_transform(x), # view3: 强增强
        ]


def build_transforms_cifar(mean, std, use_egdb=False):
    """
    统一构建 CIFAR 系列数据集的数据增强。

    Args:
        mean, std : 数据集的均值和标准差
        use_egdb  : 是否开启三视图模式（EGDB-Aug），False 则退回原始 TwoCropTransform

    Returns:
        train_transform  : 有标签数据的单视图增强（用于 SupCon + CE）
        ncd_transform    : 无标签数据的多视图增强
        test_transform   : 测试集标准预处理
    """
    from ncd_utils import TwoCropTransform  # 原始工具类

    # -------------------------------------------------------
    # 1. 弱增强（与原 ncd_transform 完全相同，保证实验可对比）
    # -------------------------------------------------------
    weak_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # -------------------------------------------------------
    # 2. 强增强（EGDB-Aug 的强增强视图）
    #    关键设计：在弱增强基础上叠加 RandAugment + 更强的 ColorJitter
    #    RandAugment(num_ops=2, magnitude=9) 是经典的强增强配置
    #    注意：强度不要太高，否则即便已知类专家也会不一致（冷启动更严重）
    # -------------------------------------------------------
    strong_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # 更强的颜色抖动
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # RandAugment：2 个操作，强度 9（范围 0-30，9 是中等偏强）
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # -------------------------------------------------------
    # 3. 有标签数据增强（保持原始逻辑，不变）
    # -------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # -------------------------------------------------------
    # 4. 测试集预处理
    # -------------------------------------------------------
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # -------------------------------------------------------
    # 5. 根据 use_egdb 决定返回两视图还是三视图
    # -------------------------------------------------------
    if use_egdb:
        ncd_transform = ThreeCropTransform(weak_aug, strong_aug)
    else:
        ncd_transform = TwoCropTransform(weak_aug)

    return train_transform, ncd_transform, test_transform
