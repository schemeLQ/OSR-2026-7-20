# code in this file is adpated from
# https://github.com/iCGY96/ARPL
# https://github.com/wjun0830/Difficulty-Aware-Simulator

import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from .tools import *
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder # 确保导入这个
# 改成当前目录下的 data 文件夹
DATA_PATH = './data'
# 这一行也改简单点，直接指向 DATA_PATH 即可
TINYIMAGENET_PATH = DATA_PATH


class CIFAR10_Filter(CIFAR10):
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(
            np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        augment = options.get('augment', 'randaugment') if options else 'randaugment'
        train_transform = predata(img_size, augment=augment)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory
        )

        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)        
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CIFAR100_Filter(CIFAR100):
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(
            np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR100_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = options.get('unknown', list(set(list(range(0, 100))) - set(known))) if options else list(set(list(range(0, 100))) - set(known))
        print('Selected Labels: ', known)
        print('Unknown Labels: ', self.unknown)

        augment = options.get('augment', 'randaugment') if options else 'randaugment'
        train_transform = predata(img_size, augment=augment)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory
        )

        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """

    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)


class SVHN_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        augment = options.get('augment', 'randaugment') if options else 'randaugment'
        train_transform = predata(img_size, augment=augment)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train',
                               download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Tiny_ImageNet_Filter(ImageFolder):
    """
    【最终修复版】将图片打包成单一 Tensor 存入内存。
    彻底解决 Windows 下 10万个 PIL 对象导致的 GC 卡顿问题。
    """

    def __init__(self, root, transform=None, memory_data=None, memory_targets=None, fast_tensor=False):
        self.fast_tensor = fast_tensor
        self.tensor_mean = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(3, 1, 1)
        self.tensor_std = torch.tensor((0.25, 0.25, 0.25), dtype=torch.float32).view(3, 1, 1)

        if memory_data is not None and memory_targets is not None:
            self.root = root
            self.transform = transform
            self.target_transform = None
            self.memory_data = memory_data
            self.memory_targets = memory_targets.long()
            self.samples = self.memory_data
            return

        super(Tiny_ImageNet_Filter, self).__init__(root, transform)

        images_list = []
        targets_list = []

        print(f"🚀 [系统优化] 正在将 {len(self.imgs)} 张图片打包进内存 (Tensor加速版)...")
        print("    (这个过程需要约 1-2 分钟，请耐心等待，这是解决时间激增的唯一办法！)")

        for path, target in tqdm(self.imgs):
            with open(path, 'rb') as f:
                # 1. 读取图片并转 RGB
                img = Image.open(f).convert('RGB')
                # 2. 确保大小一致 (TinyImageNet 都是 64x64)
                if img.size != (64, 64):
                    img = img.resize((64, 64))
                # 3. 转为 numpy 数组 (uint8 节省内存)
                img_np = np.array(img, dtype=np.uint8)
                # 4. 转为 Tensor 并暂存
                images_list.append(torch.from_numpy(img_np))
                targets_list.append(target)

        # 🔥 关键一步：将 10万个 Tensor 堆叠成一个大 Tensor (N, 64, 64, 3)
        # 这样 Python 只需要管理 1 个大对象，而不是 10 万个小对象，彻底杜绝 GC 卡顿
        self.memory_data = torch.stack(images_list)
        self.memory_targets = torch.tensor(targets_list).long()

        print(f"✅ 预加载完成！数据形状: {self.memory_data.shape}，内存已规整化。")

    def __Filter__(self, known):
        # 使用掩码 (Mask) 快速筛选，避免重建列表
        known_tensor = torch.tensor(known)
        # 找到所有属于 known 类别的样本索引
        mask = torch.isin(self.memory_targets, known_tensor)

        # 筛选数据 (Tensor 切片非常快)
        self.memory_data = self.memory_data[mask]

        # 重新映射标签 (Target Remapping)
        old_targets = self.memory_targets[mask]
        new_targets = torch.zeros_like(old_targets)
        for i, k in enumerate(known):
            new_targets[old_targets == k] = i

        self.memory_targets = new_targets
        self.samples = self.memory_data  # 兼容性接口

    def __getitem__(self, index):
        # 1. 直接从大 Tensor 切片读取 (极快，无系统开销)
        img_tensor = self.memory_data[index]  # (64, 64, 3)
        target = self.memory_targets[index].item()

        if self.fast_tensor:
            img = img_tensor.permute(2, 0, 1).float().div(255.0)
            img = (img - self.tensor_mean) / self.tensor_std
            return img, target

        # 2. 转回 PIL 图片以适配训练增强
        img = Image.fromarray(img_tensor.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.memory_data)

class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot=TINYIMAGENET_PATH, use_gpu=True, batch_size=128, img_size=64, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)

        augment = options.get('augment', 'randaugment') if options else 'randaugment'
        train_transform = predata(img_size, augment=augment)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False
        num_workers = int(options.get('num_workers', 0)) if options else 0
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers > 0:
            loader_kwargs.update({'persistent_workers': True, 'prefetch_factor': 2})

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)        
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs

        )

        fast_eval = bool(options.get('tiny_fast_eval_tensor', True)) if options else True

        val_root = os.path.join(dataroot, 'tiny-imagenet-200', 'val')
        val_memory = Tiny_ImageNet_Filter(val_root, None, fast_tensor=fast_eval)

        testset = Tiny_ImageNet_Filter(
            val_root, None if fast_eval else transform,
            memory_data=val_memory.memory_data,
            memory_targets=val_memory.memory_targets,
            fast_tensor=fast_eval)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, **loader_kwargs
        )

        outset = Tiny_ImageNet_Filter(
            val_root, None if fast_eval else transform,
            memory_data=val_memory.memory_data,
            memory_targets=val_memory.memory_targets,
            fast_tensor=fast_eval)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, drop_last=True, **loader_kwargs
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

