import os
import torch
import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
import gc
import torch.nn.functional as F
import argparse
import numpy as np  # 必须导入
import sys
from torchvision import datasets, transforms

# =================================================================
# 导入核心组件
# =================================================================
from core import *

try:
    from core.train import train
    from core.test import (evaluation, compute_react_thresholds,
                           set_react_thresholds, clear_react_thresholds)
except ImportError:
    pass
from misc import *
from misc.util import splits_AUROC as MEDAF_ORIG_SPLITS_AUROC
# 必须包含 CIFAR100_OSR
from datasets.osr_loader import CIFAR10_OSR, SVHN_OSR, Tiny_ImageNet_OSR, CIFAR100_OSR
from datasets.cub200 import CUB200_OSR, PROTOCOL_NOTE

import warnings

warnings.filterwarnings('ignore')

# =================================================================
# 1. 划分定义 (Split Definitions)
# =================================================================
splits_AUROC = {
    'cifar10': [[0, 1, 2, 4, 5, 9]],
    'svhn': [[2, 3, 4, 5, 6, 7]],
    'tiny_imagenet': [
        [2, 3, 13, 30, 44, 45, 64, 66, 76, 101, 111, 121, 128, 130, 136, 158, 167, 170, 187, 193],
        [4, 11, 32, 42, 51, 53, 67, 84, 87, 104, 116, 140, 144, 145, 148, 149, 155, 168, 185, 193],
        [3, 9, 10, 20, 23, 28, 29, 45, 54, 74, 133, 143, 146, 147, 156, 159, 161, 170, 184, 195],
        [1, 15, 17, 31, 36, 44, 66, 69, 84, 89, 102, 137, 154, 160, 170, 177, 182, 185, 195, 197],
        [4, 14, 16, 33, 34, 39, 59, 69, 77, 92, 101, 103, 130, 133, 147, 161, 166, 168, 172, 173]
    ],
    # MEDAF original CIFAR+N protocol: CIFAR10 known classes.
    'cifar100': [
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9]
    ],
    # MEDAF original CIFAR+10 protocol: CIFAR100 unknown classes.
    'cifar100-10': [
        [3, 15, 19, 21, 42, 46, 66, 72, 78, 98],
        [26, 31, 34, 44, 45, 63, 65, 77, 93, 98],
        [7, 11, 66, 75, 77, 93, 95, 97, 98, 99],
        [2, 11, 15, 24, 32, 34, 63, 88, 93, 95],
        [1, 11, 38, 42, 44, 45, 63, 64, 66, 67]
    ],
    # MEDAF original CIFAR+50 protocol: CIFAR100 unknown classes.
    'cifar100-50': [
        [1, 2, 7, 9, 10, 12, 15, 18, 21, 23, 26, 30, 32, 33, 34,
         36, 37, 39, 40, 42, 44, 45, 46, 47, 49, 50, 51, 52, 55,
         56, 59, 60, 61, 63, 65, 66, 70, 72, 73, 74, 76, 78, 80,
         83, 87, 91, 92, 96, 98, 99],
        [0, 2, 4, 5, 9, 12, 14, 17, 18, 20, 21, 23, 24, 25, 31,
         32, 33, 35, 39, 43, 45, 49, 50, 51, 52, 54, 55, 56, 60,
         64, 65, 66, 68, 70, 71, 73, 74, 77, 78, 79, 80, 82, 83,
         86, 91, 93, 94, 96, 97, 98],
        [0, 4, 10, 11, 12, 14, 15, 17, 18, 21, 23, 26, 27, 28, 29,
         31, 32, 33, 36, 39, 40, 42, 43, 46, 47, 51, 53, 56, 57,
         59, 60, 64, 66, 71, 73, 74, 75, 76, 78, 79, 80, 83, 87,
         91, 92, 93, 94, 95, 96, 99],
        [0, 2, 5, 6, 9, 10, 11, 12, 14, 16, 18, 19, 21, 22, 23,
         26, 27, 28, 29, 31, 33, 35, 36, 37, 38, 39, 40, 43, 45,
         49, 52, 56, 59, 61, 62, 63, 64, 65, 71, 74, 75, 78, 80,
         82, 86, 87, 91, 93, 94, 96],
        [0, 1, 4, 6, 7, 12, 15, 16, 17, 19, 20, 21, 22, 23, 26,
         27, 28, 32, 39, 40, 42, 43, 44, 47, 49, 50, 52, 53, 54,
         55, 56, 59, 61, 62, 63, 65, 66, 67, 68, 73, 74, 77, 82,
         83, 86, 87, 93, 94, 97, 98]
    ],
    'cifar_plus': None
}
splits_AUROC['cifar_plus'] = splits_AUROC['cifar100']


# =================================================================
# 2. 数据加载器
# =================================================================
def getLoader(options):
    print(f"📥 Preparing Loader for {options['dataset']}...")

    if options['dataset'] == 'cub':
        print("Custom CUB-10/10 Easy-OSR protocol.")
        print("This is not an official SSB benchmark split.")
        options['img_size'] = int(options.get('image_size', 224))
        options['stem_type'] = options.get('stem_type', 'imagenet')
        split_idx = int(options.get('split_idx', 0))
        split_seed = int(options.get('split_seed', split_idx))
        split_dir = options.get('cub_split_dir', os.path.join(options.get('data_root', './data'), 'open_set_splits', 'cub_10_10_easy'))
        split_json = os.path.join(split_dir, f'split_{split_idx}.json')
        if not os.path.exists(split_json):
            raise FileNotFoundError(f'CUB split file not found: {split_json}. Run: python tools/generate_cub_10_10_splits.py --data_root {options.get("data_root", "./data")}')
        print(f'training_seed={options.get("seed")}')
        print(f'split_seed={split_seed}')
        print(f'split_idx={split_idx}')
        Data = CUB200_OSR(split_json=split_json, data_root=options.get('data_root', './data'), batch_size=options['batch_size'],
                         num_workers=options.get('num_workers', 0), image_size=int(options.get('image_size', 224)),
                         resize_size=int(options.get('resize_size', 256)), val_ratio=float(options.get('val_ratio', 0.2)),
                         data_split_seed=int(options.get('data_split_seed', 123)), use_bbox=bool(options.get('cub_use_bbox', False)),
                         pin_memory=torch.cuda.is_available())
        options['known'] = list(Data.known)
        options['unknown'] = list(Data.unknown)
        options['num_known'] = Data.num_known
        options['cub_val_loader'] = Data.val_loader
        options['cub_data_obj'] = Data
        print('Known classes:', options['known'])
        print('Unknown classes:', options['unknown'])
        print('Train Num:', len(Data.train_dataset), 'Val Num:', len(Data.val_dataset), 'Known Test Num:', len(Data.test_dataset), 'Unknown Test Num:', len(Data.out_dataset))
        return Data.train_loader, Data.test_loader, Data.out_loader

    # ---------------------------------------------------
    # 模式 C: CIFAR+N 实验 (CIFAR10 + CIFAR100)
    # ---------------------------------------------------

    # --- 分支 C: CIFAR+N 实验 ---
    if options['dataset'] == 'cifar_plus':
        print(f"🚀 Mode: CIFAR+{options['plus_num']}")
        options['img_size'] = 32

        # 1. 加载已知类 (CIFAR-10 的 4 个类)
        Data = CIFAR10_OSR(known=options['known'], batch_size=options['batch_size'], img_size=32, options=options)

        # 2. 加载未知类。MEDAF 原始协议使用固定 CIFAR100-10 / CIFAR100-50 splits；
        #    plus_num 不是 10/50 时，才退回可复现随机采样。
        c100_key = f"cifar100-{options['plus_num']}"
        if c100_key in splits_AUROC:
            unknown_classes = splits_AUROC[c100_key][options['item']]
        else:
            all_c100 = np.arange(100)
            np.random.seed(options['item'] + 42)
            np.random.shuffle(all_c100)
            unknown_classes = all_c100[:options['plus_num']].tolist()

        print(f"   -> Known (C10): {options['known']}")
        print(f"   -> Unknown (C100): {unknown_classes}")
        options['unknown'] = list(unknown_classes)

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
    elif 'cifar100' == options['dataset']:
        options['img_size'] = 32
        Data = CIFAR100_OSR(known=options['known'], batch_size=options['batch_size'], img_size=32, options=options)
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
    options['loss_keys'] = ['b1', 'b2', 'b3', 'gate', 'divAttn', 'odl', 'orth', 'decorr',
                            'cbdo', 'center_align', 'baed_anchor', 'baed_div', 'total']
    options['acc_keys'] = ['acc1', 'acc2', 'acc3', 'accGate']

    splits = splits_AUROC
    now_time = datetime.datetime.now().strftime("%m%d_%H%M")
    options['run_time'] = now_time          # passed to trainLoop for ckpt naming
    log_path = os.path.join('logs', 'osr', options['dataset'])
    ensure_dir(log_path)

    suffix = options['out_dataset'] if options['out_dataset'] else "Split"
    # 如果是 CIFAR+N，文件名加上 plus_num
    if options['dataset'] == 'cifar_plus':
        suffix = f"Plus{options['plus_num']}"

    stats_log = open(os.path.join(log_path, f"MEDAF_{suffix}_{now_time.replace(':', '')}.txt"), 'w')

    # --- CUB custom 10/10 Easy-OSR ---
    if options['dataset'] == 'cub':
        print("Custom CUB-10/10 Easy-OSR protocol.")
        print("This is not an official SSB benchmark split.")
        split_ids = list(range(5)) if options.get('run_five_splits', False) else [int(options.get('split_idx', 0))]
        cub_results = []
        for sid in split_ids:
            options['split_idx'] = sid
            options['split_seed'] = sid if options.get('run_five_splits', False) or options.get('split_seed') is None else int(options.get('split_seed'))
            options['split_tag'] = f'split_{sid:02d}'
            print("\n" + "=" * 80)
            print(f'Running CUB split_idx={sid} | training_seed={options.get("seed")} | split_seed={options.get("split_seed")}')
            print("=" * 80)
            res = trainLoop(options)
            meta = res[-1] if len(res) > 7 and isinstance(res[-1], dict) else {}
            cub_results.append({'split': sid, 'acc': float(res[0]), 'auroc': float(res[1]), 'best_auroc': meta.get('best_auroc', float(res[1])), 'save_dir': meta.get('save_dir', ''), 'known': list(options.get('known', [])), 'unknown': list(options.get('unknown', []))})
            stats_log.write(f'CUB_SPLIT[{sid}] => Acc: {res[0]:.3f}, AUROC: {res[1]:.3f}, BestValAcc: {cub_results[-1]["best_auroc"]:.3f}, SaveDir: {cub_results[-1]["save_dir"]}\n')
            stats_log.flush()
        if cub_results:
            print("\n================ CUB-10/10 EASY-OSR SUMMARY ================")
            print("Protocol: Custom CUB-10/10 Easy-OSR, not official SSB")
            print(f"Training seed: {options.get('seed')}")
            print("Split seeds: [0, 1, 2, 3, 4]" if options.get('run_five_splits', False) else f"Split seed: {options.get('split_seed')}")
            print("Split | Known Acc | AUROC Energy | Best ValAcc | SaveDir")
            for r in cub_results:
                print(f"{r['split']:>5} | {r['acc']:>9.2f} | {r['auroc']:>12.4f} | {r['best_auroc']:>10.4f} | {r['save_dir']}")
            vals = np.asarray([r['auroc'] for r in cub_results], dtype=float)
            if len(vals) > 1:
                print(f"Mean AUROC={vals.mean():.4f}, Std={vals.std(ddof=1):.4f}")
            print("=============================================================")
        stats_log.close()
        return


    # --- 分支 C: CIFAR+N 实验 ---
    if options['dataset'] == 'cifar_plus':
        current_splits = splits['cifar_plus']
        split_results = []
        for i in range(len(current_splits)):
            known = current_splits[i]
            options['item'] = i
            options['split_tag'] = f"split_{i + 1:02d}"
            options['known'] = known
            options['unknown'] = []  # 动态生成占位

            print(f"\n{'=' * 30}\nRunning CIFAR+{options['plus_num']} Split {i + 1}/{len(current_splits)}\n{'=' * 30}")

            res = trainLoop(options)
            meta = res[-1] if len(res) > 7 and isinstance(res[-1], dict) else {}
            split_results.append({
                'split': i + 1,
                'known': list(known),
                'unknown': list(options.get('unknown', [])),
                'acc': float(res[0]),
                'auroc': float(res[1]),
                'save_dir': meta.get('save_dir', ''),
                'best_auroc': meta.get('best_auroc', float(res[1])),
            })
            stats_log.write(
                f"CIFAR+{options['plus_num']}_SPLIT[{i + 1}] => "
                f"Acc: {res[0]:.3f}, AUROC: {res[1]:.3f}, "
                f"BestValAcc: {split_results[-1]['best_auroc']:.3f}, "
                f"SaveDir: {split_results[-1]['save_dir']}\n"
            )
            stats_log.flush()

            print("\n" + "-" * 96)
            print(f"CIFAR+{options['plus_num']} Split Progress ({len(split_results)}/{len(current_splits)})")
            print(f"{'Split':^7} | {'ACC(%)':^8} | {'AUROC':^8} | {'Best':^8} | {'Backbone Path'}")
            print("-" * 96)
            for row in split_results:
                bb_path = os.path.join(row['save_dir'], 'medaf_backbone_converted.pth') if row['save_dir'] else ''
                print(f"{row['split']:^7} | {row['acc']:^8.2f} | {row['auroc']:^8.4f} | {row['best_auroc']:^8.4f} | {bb_path}")
            print("-" * 96)

            # 调试时跑完一个就 break
            # break

        if split_results:
            print("\n" + "=" * 96)
            print(f"CIFAR+{options['plus_num']} 5-Split Summary")
            print("-" * 96)
            print(f"{'Split':^7} | {'ACC(%)':^8} | {'AUROC':^8} | {'Best':^8} | {'Backbone Path'}")
            print("-" * 96)
            for row in split_results:
                bb_path = os.path.join(row['save_dir'], 'medaf_backbone_converted.pth') if row['save_dir'] else ''
                print(f"{row['split']:^7} | {row['acc']:^8.2f} | {row['auroc']:^8.4f} | {row['best_auroc']:^8.4f} | {bb_path}")
            best_row = max(split_results, key=lambda r: r['best_auroc'])
            print("-" * 96)
            print(f"BEST Split {best_row['split']} | Best AUROC={best_row['best_auroc']:.4f} | SaveDir={best_row['save_dir']}")
            print("=" * 96)
            stats_log.write("\nCIFAR+{} 5-Split Summary\n".format(options['plus_num']))
            for row in split_results:
                stats_log.write(
                    f"Split {row['split']}: ACC={row['acc']:.3f}, AUROC={row['auroc']:.4f}, "
                    f"BestAUROC={row['best_auroc']:.4f}, SaveDir={row['save_dir']}\n"
                )
            stats_log.write(f"BEST Split {best_row['split']} | BestAUROC={best_row['best_auroc']:.4f} | SaveDir={best_row['save_dir']}\n")
            stats_log.flush()

    # --- Branch A: Cross-Dataset ---
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
        if len(current_splits) == 0:
            raise ValueError(
                f"No split is defined for dataset '{options['dataset']}'. "
                f"Available split datasets: {list(splits.keys())}"
            )
        split_plan = list(enumerate(current_splits))
        if 'tiny' in options['dataset'] and not options.get('run_five_splits', False):
            sid = int(options.get('split_idx', 0))
            if sid < 0 or sid >= len(current_splits):
                raise ValueError(f"split_idx={sid} is out of range for {options['dataset']} ({len(current_splits)} splits).")
            split_plan = [(sid, current_splits[sid])]
            print(f"   [Tiny-ImageNet] Running split_idx={sid}. Use --run_five_splits to run all 5 splits.")

        for i, split_def in split_plan:
            options['item'] = i
            total_classes = 100 if options['dataset'] == 'cifar100' else 10
            if 'tiny' in options['dataset']:
                total_classes = 200

            if options['dataset'] == 'cifar100' and split_def == 'random_10_10':
                rng = np.random.default_rng(int(options.get('seed', 0)) + i)
                selected = rng.choice(total_classes, size=20, replace=False).tolist()
                known = sorted(selected[:10])
                unknown = sorted(selected[10:])
                print(f"   [CIFAR100 Split] Random 10 known / 10 unknown with seed={options.get('seed', 0)}")
                print(f"   -> Known classes:   {known}")
                print(f"   -> Unknown classes: {unknown}")
            else:
                known = split_def
                unknown = list(set(range(total_classes)) - set(known))

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options['device'] = device
    print(f"Using device: {device}")

    if options['resume'] and options['ckpt'] and os.path.exists(options['ckpt']):
        ckpt_probe = torch.load(options['ckpt'], map_location='cpu')
        probe_sd = ckpt_probe['state_dict'] if 'state_dict' in ckpt_probe else ckpt_probe
        has_bacl = any(k.startswith('bacl') for k in probe_sd)
        has_se = any(k.startswith('se') for k in probe_sd)
        if has_bacl and not has_se and not options.get('use_bacl', False):
            options['use_bacl'] = True
            print("🔧 [CKPT Compat] Detected BACL checkpoint; enabling use_bacl=True for evaluation.")
        elif has_se and not has_bacl and options.get('use_bacl', False):
            options['use_bacl'] = False
            print("🔧 [CKPT Compat] Detected SE checkpoint; enabling use_bacl=False for evaluation.")

    model = get_model(options)
    model = model.to(device)
    if device.type == 'cuda':
        model = nn.DataParallel(model)

    if options['resume'] and options['ckpt']:
        print(f"🔄 Loading weights from: {options['ckpt']}")
        checkpoint = torch.load(options['ckpt'], map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        target_model = model.module if isinstance(model, nn.DataParallel) else model
        missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Missing keys (random init): {missing}")
        if unexpected:
            print(f"⚠️  Unexpected keys (ignored): {unexpected}")

    def _prepare_react_for_eval():
        if not options.get('use_react', False):
            clear_react_thresholds(model)
            return
        pct = float(options.get('react_percentile', 90.0))
        print(f"🧪 [ReAct] Estimating per-branch thresholds from training activations (p={pct:g})...")
        thresholds = compute_react_thresholds(
            model, train_loader, percentile=pct, device=device,
            max_values_per_branch=int(options.get('react_max_values', 2000000)))
        set_react_thresholds(model, thresholds)
        print("   [ReAct] Branch thresholds: "
              + ", ".join([f"B{i + 1}={t:.4f}" for i, t in enumerate(thresholds)]))

    if options.get('test_only', False):
        print("🚀 [TEST ONLY] Evaluating...")
        model.eval()
        with torch.no_grad():
            _prepare_react_for_eval()
            res = evaluation(model, test_loader, out_loader, **options)
        print(f"📊 Results: AUROC={res[1]:.4f}")
        return res

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=options['lr'],
        momentum=0.9,
        weight_decay=float(options.get('weight_decay', 0.0)))
    ls = options.get('label_smoothing', 0.0)
    criterion = {'entropy': nn.CrossEntropyLoss(label_smoothing=ls).to(device)}
    if options.get('lr_scheduler', 'cosine') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=options['epoch_num'], eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=options['milestones'], gamma=options.get('gamma', 0.1))

    run_time  = options.get('run_time', datetime.datetime.now().strftime("%m%d_%H%M"))
    save_root = f"ckpt/osr/{options['dataset']}"
    if options['dataset'] == 'cifar_plus':
        save_root += f"_plus{options['plus_num']}"
    split_tag = options.get('split_tag', None)
    save_dir = os.path.join(save_root, run_time, split_tag) if split_tag else os.path.join(save_root, run_time)
    ensure_dir(save_dir)
    if options.get('dataset') == 'cub' and options.get('cub_data_obj') is not None:
        options['cub_data_obj'].save_manifests(save_dir)

    _BACKBONE_PREFIXES = (
        'shared_l3', 'branch1_l4', 'branch1_l5',
        'branch2_l4', 'branch2_l5', 'branch3_l4', 'branch3_l5',
        'gate_l3', 'gate_l4', 'gate_l5', 'gate_cls',
        'se1', 'se2', 'se3', 'bacl1', 'bacl2', 'bacl3',
    )

    def _checkpoint_payload(epoch, auroc_val):
        tgt = model.module if isinstance(model, nn.DataParallel) else model
        return {
            'epoch': epoch,
            'auroc': auroc_val,
            'state_dict': tgt.state_dict(),
            'options': dict(options),
        }

    def _save_epoch_ckpt(epoch, auroc_val):
        payload = _checkpoint_payload(epoch, auroc_val)
        torch.save(payload, os.path.join(save_dir, f'epoch_{epoch}.pth'))
        if options.get('dataset') == 'cub':
            torch.save(payload, os.path.join(save_dir, 'model_last.pth'))
        metric_name = 'ValAcc' if options.get('dataset') == 'cub' else 'AUROC'
        print(f"Saved epoch_{epoch}.pth  (best {metric_name}={auroc_val:.4f})")

    def _save_best_ckpt(epoch, auroc_val):
        payload = _checkpoint_payload(epoch, auroc_val)
        sd = payload['state_dict']
        torch.save(payload, os.path.join(save_dir, 'model_best.pth'))
        bb_dict = {k: v for k, v in sd.items()
                   if any(k.startswith(p) for p in _BACKBONE_PREFIXES)}
        torch.save(bb_dict, os.path.join(save_dir, 'medaf_backbone_converted.pth'))
        metric_name = 'ValAcc' if options.get('dataset') == 'cub' else 'AUROC'
        print(f"Saved model_best.pth + medaf_backbone_converted.pth  ({metric_name}={auroc_val:.4f})")

    def _eval_known_accuracy(loader, max_batches=None):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if max_batches is not None and bi >= int(max_batches):
                    break
                data, labels = batch[0].to(device), batch[1].to(device)
                od = model(data)
                pred = od['logits'][-1].argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
        model.train()
        return 100.0 * correct / max(total, 1)

    best_auroc = 0.0
    for epoch in range(options['epoch_num']):
        options['current_epoch'] = epoch + 1
        print(f"Epoch [{epoch + 1}/{options['epoch_num']}]")
        clear_react_thresholds(model)
        train(train_loader, model, criterion, optimizer, args=options)

        if (epoch + 1) % options['test_step'] == 0:
            _prepare_react_for_eval()
            res = evaluation(model, test_loader, out_loader, **options)
            if options.get('dataset') == 'cub':
                val_loader = options.get('cub_val_loader')
                val_acc = _eval_known_accuracy(val_loader, options.get('max_eval_batches')) if val_loader is not None else res[0]
                print(f"[CUB] Checkpoint selection metric: known validation accuracy={val_acc:.3f}. Unknown test samples are not used.")
                if val_acc > best_auroc:
                    best_auroc = val_acc
                    _save_best_ckpt(epoch + 1, best_auroc)
            elif res[1] > best_auroc:
                best_auroc = res[1]
                _save_best_ckpt(epoch + 1, best_auroc)

        scheduler.step()

    _save_epoch_ckpt(options['epoch_num'], best_auroc)

    # 最终评估前按需计算原型（每轮eval不算，只在最终结果上计算）
    if options.get('use_proto_score', False):
        print("Computing branch prototypes from training data...")
        from core.test import compute_branch_prototypes
        _eval_model = model.module if isinstance(model, nn.DataParallel) else model
        options['prototypes'] = compute_branch_prototypes(
            _eval_model, train_loader, options['num_known'], device)

    _prepare_react_for_eval()
    final_res = evaluation(model, test_loader, out_loader, **options)
    return final_res + [{'save_dir': save_dir, 'best_auroc': best_auroc}]


# =================================================================
# 5. 入口函数
# =================================================================
if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_pos', type=str, nargs='?', default=None, help='Dataset name')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--out_dataset', type=str, default=None)
    parser.add_argument('--plus_num', type=int, default=10, help='For CIFAR+N: 10 or 50')  # 新增参数
    parser.add_argument('--split_idx', type=int, default=0, help='Split index for CUB/Tiny-ImageNet: 0..4')
    parser.add_argument('--split_seed', type=int, default=None, help='Independent CUB class split seed; does not change training seed')
    parser.add_argument('--run_five_splits', action='store_true', help='Run all 5 splits for CUB/Tiny-ImageNet')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=256)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--data_split_seed', type=int, default=123)
    parser.add_argument('--cub_use_bbox', action='store_true')
    parser.add_argument('--stem_type', type=str, default=None, choices=['cifar', 'imagenet'])
    parser.add_argument('--epochs', type=int, default=None, help='Alias for epoch_num, useful for CUB smoke test')
    parser.add_argument('--max_train_batches', type=int, default=None)
    parser.add_argument('--max_eval_batches', type=int, default=None)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--use_react', action='store_true',
                        help='Enable test-time ReAct clipping on each MoE branch.')
    parser.add_argument('--react_percentile', type=float, default=None,
                        help='Percentile for ReAct thresholds, typically 90-95.')

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
        'seed': args.seed if args.seed is not None else options['seed'],
        'resume': args.resume,
        'ckpt': args.ckpt,
        'test_only': args.test_only,
        'use_react': bool(args.use_react or options.get('use_react', False)),
        'split_idx': args.split_idx,
        'split_seed': args.split_seed,
        'run_five_splits': args.run_five_splits,
        'data_root': args.data_root,
        'image_size': args.image_size,
        'resize_size': args.resize_size,
        'val_ratio': args.val_ratio,
        'data_split_seed': args.data_split_seed,
        'cub_use_bbox': args.cub_use_bbox,
        'max_train_batches': args.max_train_batches,
        'max_eval_batches': args.max_eval_batches,
        'debug': args.debug,
    })
    if args.stem_type is not None:
        options['stem_type'] = args.stem_type
    if options.get('dataset') == 'cub':
        options['stem_type'] = options.get('stem_type', 'imagenet')
    if args.epochs is not None:
        options['epoch_num'] = args.epochs
    if args.react_percentile is not None:
        options['react_percentile'] = args.react_percentile

    os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu_ids']
    set_seeding(options['seed'])

    if not options['dataset']:
        print("❌ Error: Please specify a dataset.")
        exit(1)

    main(options)








