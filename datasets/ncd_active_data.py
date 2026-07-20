import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ActiveDataHandler(Dataset):
    """
    基础数据包装器：支持返回 (图片, 标签, 原始索引)
    """

    def __init__(self, data, targets, transform=None, original_indices=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        # 记录样本在原始大池子里的绝对索引，以便 Query 时能找回它
        self.original_indices = original_indices

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        # 兼容 CIFAR10 (H,W,C) 和 SVHN (C,H,W) 的 numpy 格式
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if isinstance(x, np.ndarray):
            # 如果发现通道数跑到最前面去了 (SVHN格式)，就把它翻转回 (H,W,C)
            if x.ndim == 3 and x.shape[0] == 3:
                x = np.transpose(x, (1, 2, 0))
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 255).astype(np.uint8)
            x = Image.fromarray(x)

        if self.transform:
            x = self.transform(x)

        # 返回 (图, 标, 绝对索引)
        idx_ret = self.original_indices[index] if self.original_indices is not None else index
        return x, y, idx_ret

    def __len__(self):
        return len(self.data)


class ActiveDataset:
    """
    数据池管理器：维护全局 labeled_mask
    """

    def __init__(self, base_dataset):
        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)
        self.n_pool = len(self.data)

        # 核心状态：False=未标(Unlabeled), True=已标(Labeled)
        self.labeled_mask = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, initial_indices):
        """初始化已知类样本为 Labeled"""
        self.labeled_mask[initial_indices] = True

    def update_labels(self, new_indices):
        """[Active Update] 将新查询的样本标记为 Labeled"""
        self.labeled_mask[new_indices] = True

    def get_labeled_dataset(self, transform):
        """获取当前已标注数据集 (用于 SupCon 训练)"""
        idxs = np.where(self.labeled_mask)[0]
        return ActiveDataHandler(self.data[idxs], self.targets[idxs], transform, idxs)

    def get_unlabeled_dataset(self, transform):
        """获取当前未标注数据集 (用于 SimGCD 训练)"""
        idxs = np.where(~self.labeled_mask)[0]
        return ActiveDataHandler(self.data[idxs], self.targets[idxs], transform, idxs)

    def get_subset_by_idxs(self, idxs):
        """获取指定索引的数据子集 (用于 Strategy 计算分数)"""
        # 查询阶段只做基础归一化；Strategy 会按数据集覆盖 transform。
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])
        return ActiveDataHandler(self.data[idxs], self.targets[idxs], base_transform, idxs)
