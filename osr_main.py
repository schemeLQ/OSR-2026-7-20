import os
import torch
import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
import gc
import torch.nn.functional as F
import argparse
import numpy as np  # 必须导入
from torchvision import datasets, transforms

# =================================================================
# 导入核心组件
# =================================================================
from core import *

try:
    from core.train import train
    from core.test import evaluation
except ImportError:
    pass
from misc import *
# 必须包含 CIFAR100_OSR
from datasets.osr_loader import CIFAR10_OSR, SVHN_OSR, Tiny_ImageNet_OSR, CIFAR100_OSR

import warnings

warnings.filterwarnings('ignore')

# =================================================================
# 1. 划分定义 (Split Definitions)
# =================================================================
splits_AUROC = {
    'cifar10': [[0, 1, 2, 4, 5, 9]],
    'svhn': [[2, 3, 4, 5, 6, 7]],
    'tiny_imagenet': [list(range(20))],

    # CIFAR+N 实验的 5 次随机划分
    'cifar_plus': [
        [0, 1, 2, 3],  # Split 1
        [4, 5, 6, 7],  # Split 2
        [2, 3, 4, 5],  # Split 3
        [6, 7, 8, 9],  # Split 4
        [1, 3, 5, 7]  # Split 5
    ]
}


# =================================================================
# 2. 数据加载器
# =================================================================
def getLoader(options):
    print(f"📥 Preparing Loader for {options['dataset']}...")

    # ---------------------------------------------------
    # 模式 C: CIFAR+N 实验 (CIFAR10 + CIFAR100)
    # ---------------------------------------------------
    if options['dataset'] == 'cifar_plus':
        print(f"🚀 Mode: CIFAR+{options['plus_num']}")
        options['img_size'] = 32

        # 1. 加载已知类 (CIFAR-10 的 4 个类)
        Data = CIFAR10_OSR(known=options['known'], batch_size=options['batch_size'], img_size=32, options=options)

        # 2. 加载未知类 (从 CIFAR-100 随机取 N 个)
        all_c100 = np.arange(100)
        np.random.seed(options['item'] + 42)  # 用 split 序号做种子
        np.random.shuffle(all_c100)
        unknown_classes = all_c100[:options['plus_num']].tolist()

        print(f"   -> Known (C10): {options['known']}")
        print(f"   -> Unknown (C100): {unknown_classes}")

        # 用 CIFAR100_OSR 加载这些类，并取其 test_loader 作为 OOD 数据
        OutData = CIFAR100_OSR(known=unknown_classes, batch_size=options['batch_size'], img_size=32, options=options)

        options['num_known'] = 4
        # ⚠️ 注意缩进：return 必须在 if 块内部或函数结尾
        return Data.train_loader, Data.test_loader, OutData.test_loader

    # ---------------------------------------------------
    # 预处理定义 (用于 Cross-Dataset)
    # ---------------------------------------------------
    if options['dataset'] == 'svhn':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ---------------------------------------------------
    # 模式 A: 跨数据集 (Cross-Dataset)
    # ---------------------------------------------------
    if options.get('out_dataset') is not None:
        print(f"🚀 Mode: Cross-Dataset OSR")
        options['img_size'] = 32

        # A.1 加载已知类
        if options['dataset'] == 'cifar10':
            options['known'] = list(range(10))
            Data = CIFAR10_OSR(known=list(range(10)), batch_size=options['batch_size'], img_size=32, options=options)
        elif options['dataset'] == 'svhn':
            options['known'] = list(range(10))
            Data = SVHN_OSR(known=list(range(10)), batch_size=options['batch_size'], img_size=32, options=options)
        else:
            raise NotImplementedError(f"Cross-Dataset known dataset {options['dataset']} not supported yet.")

        # A.2 加载未知类
        print(f"📦 Loading {options['out_dataset']} as External OOD data...")
        if options['out_dataset'] == 'cifar10':
            out_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        elif options['out_dataset'] == 'svhn':
            out_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        else:
            out_root = os.path.join('./data/datasets/OSR_External', options['out_dataset'])
            if not os.path.exists(out_root):
                print(f"❌ Error: External dataset not found at {out_root}")
                exit(1)
            out_set = datasets.ImageFolder(root=out_root, transform=transform_test)

        out_loader = torch.utils.data.DataLoader(out_set, batch_size=options['batch_size'], shuffle=False,
                                                 num_workers=options.get('num_workers', 0))
        options['num_known'] = 10
        return Data.train_loader, Data.test_loader, out_loader

    # ---------------------------------------------------
    # 模式 B: 单数据集分裂 (Split-Dataset)
    # ---------------------------------------------------
    if 'cifar10' == options['dataset']:
        options['img_size'] = 32
        Data = CIFAR10_OSR(known=options['known'], batch_size=options['batch_size'], img_size=32, options=options)
    elif 'svhn' in options['dataset']:
        options['img_size'] = 32
        Data = SVHN_OSR(known=options['known'], batch_size=options['batch_size'], img_size=32, options=options)
    elif 'tiny_imagenet' in options['dataset']:
        options['img_size'] = 64
        Data = Tiny_ImageNet_OSR(known=options['known'], batch_size=options['batch_size'], img_size=64, options=options)
    else:
        raise ValueError(f"Unknown dataset: {options['dataset']}")

    options['num_known'] = Data.num_known
    return Data.train_loader, Data.test_loader, Data.out_loader


# =================================================================
# 3. 主逻辑
# =================================================================
def main(options):
    options['loss_keys'] = ['b1', 'b2', 'b3', 'gate', 'divAttn', 'odl', 'orth', 'total']
    options['acc_keys'] = ['acc1', 'acc2', 'acc3', 'accGate']

    splits = splits_AUROC
    now_time = datetime.datetime.now().strftime("%m%d_%H:%M")
    log_path = os.path.join('logs', 'osr', options['dataset'])
    ensure_dir(log_path)

    suffix = options['out_dataset'] if options['out_dataset'] else "Split"
    # 如果是 CIFAR+N，文件名加上 plus_num
    if options['dataset'] == 'cifar_plus':
        suffix = f"Plus{options['plus_num']}"

    stats_log = open(os.path.join(log_path, f"MEDAF_{suffix}_{now_time.replace(':', '')}.txt"), 'w')

    # --- 分支 C: CIFAR+N 实验 ---
    if options['dataset'] == 'cifar_plus':
        current_splits = splits['cifar_plus']
        for i in range(len(current_splits)):
            known = current_splits[i]
            options['item'] = i
            options['known'] = known
            options['unknown'] = []  # 动态生成占位

            print(f"\n{'=' * 30}\nRunning CIFAR+{options['plus_num']} Split {i + 1}/{len(current_splits)}\n{'=' * 30}")

            res = trainLoop(options)
            stats_log.write(f"CIFAR+{options['plus_num']}_SPLIT[{i + 1}] => Acc: {res[0]:.3f}, AUROC: {res[1]:.3f}\n")
            stats_log.flush()

            # 调试时跑完一个就 break
            # break

    # --- 分支 A: 跨数据集 ---
    elif options['out_dataset'] is not None:
        print(f"\n{'=' * 40}\nRunning Cross-Dataset: {options['dataset']} vs {options['out_dataset']}\n{'=' * 40}")
        options['item'] = 'cross'
        if options['dataset'] == 'cifar10':
            options['known'] = list(range(10))
        elif options['dataset'] == 'svhn':
            options['known'] = list(range(10))

        res = trainLoop(options)
        stats_log.write(f"Cross[{options['out_dataset']}] => Acc: {res[0]:.3f}, AUROC: {res[1]:.3f}\n")

    # --- 分支 B: Split 实验 ---
    else:
        current_splits = splits.get(options['dataset'], [])
        for i in range(len(current_splits)):
            known = current_splits[i]
            options['item'] = i
            unknown = list(set(range(10)) - set(known))
            if 'tiny' in options['dataset']: unknown = list(set(range(200)) - set(known))

            options.update({'known': known, 'unknown': unknown})
            print(f"\n{'=' * 30}\nRunning Split {i + 1}/{len(current_splits)}\n{'=' * 30}")

            res = trainLoop(options)
            stats_log.write(f"SPLIT[{i + 1}] => Acc: {res[0]:.3f}, AUROC: {res[1]:.3f}\n")
            stats_log.flush()
            # break

    stats_log.close()


# =================================================================
# 4. 训练/测试循环
# =================================================================
def trainLoop(options):
    train_loader, test_loader, out_loader = getLoader(options)

    model = get_model(options)
    model = nn.DataParallel(model).cuda()

    if options['resume'] and options['ckpt']:
        print(f"🔄 Loading weights from: {options['ckpt']}")
        checkpoint = torch.load(options['ckpt'])
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.module.load_state_dict(state_dict)

    if options.get('test_only', False):
        print("🚀 [TEST ONLY] Evaluating...")
        model.eval()
        with torch.no_grad():
            res = evaluation(model, test_loader, out_loader, **options)
        print(f"📊 Results: AUROC={res[1]:.4f}")
        return res

    optimizer = torch.optim.SGD(model.parameters(), lr=options['lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options['milestones'], gamma=0.1)
    criterion = {'entropy': nn.CrossEntropyLoss().cuda()}

    best_auroc = 0.0
    for epoch in range(options['epoch_num']):
        options['current_epoch'] = epoch + 1
        print(f"Epoch [{epoch + 1}/{options['epoch_num']}]")
        train(train_loader, model, criterion, optimizer, args=options)

        if (epoch + 1) % options['test_step'] == 0:
            res = evaluation(model, test_loader, out_loader, **options)
            if res[1] > best_auroc:
                best_auroc = res[1]
                save_dir = f"ckpt/osr/{options['dataset']}"
                if options['dataset'] == 'cifar_plus':
                    save_dir += f"_plus{options['plus_num']}"
                ensure_dir(save_dir)

                print(f"📦 Exporting pure backbone for NCD/GCD stage...")
                full_sd = model.module.state_dict()
                # 过滤掉全连接层，只保留骨干网络权重
                backbone_dict = {k: v for k, v in full_sd.items() if 'fc' not in k and 'classifier' not in k}
                torch.save(backbone_dict, os.path.join(save_dir, 'medaf_backbone_converted.pth'))
                print(f"📦 Exporting pure backbone for NCD/GCD stage (Best AUROC: {best_auroc:.4f})...")
                full_sd = model.module.state_dict()

                # 核心过滤逻辑：剔除所有包含 'fc' 或 'classifier' 的分类头参数
                backbone_dict = {k: v for k, v in full_sd.items() if 'fc' not in k and 'classifier' not in k}

                # 保存为 medaf_backbone_converted.pth
                torch.save(backbone_dict, os.path.join(save_dir, 'medaf_backbone_converted.pth'))

            scheduler.step()

    return evaluation(model, test_loader, out_loader, **options)


# =================================================================
# 5. 入口函数
# =================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_pos', type=str, nargs='?', default=None, help='Dataset name')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--out_dataset', type=str, default=None)
    parser.add_argument('--plus_num', type=int, default=10, help='For CIFAR+N: 10 or 50')  # 新增参数

    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test_only', action='store_true')

    args = parser.parse_args()
    options = get_config('osr')

    if args.dataset_pos:
        options['dataset'] = args.dataset_pos
    elif args.dataset:
        options['dataset'] = args.dataset

    options.update({
        'out_dataset': args.out_dataset,
        'plus_num': args.plus_num,
        'gpu_ids': args.gpu_ids,
        'batch_size': args.batch_size,
        'resume': args.resume,
        'ckpt': args.ckpt,
        'test_only': args.test_only
    })

    os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu_ids']
    set_seeding(options['seed'])

    if not options['dataset']:
        print("❌ Error: Please specify a dataset.")
        exit(1)

    main(options)