import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from misc import get_config, set_seeding
from osr_main import getLoader
from core import get_model


SPLITS_AUROC = {
    'cifar10': [[0, 1, 2, 4, 5, 9]],
    'svhn': [[2, 3, 4, 5, 6, 7]],
    'tiny_imagenet': [list(range(20))],
}


def _unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _probe_ckpt_options(options, ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    has_bacl = any(k.startswith('bacl') for k in state_dict)
    has_se = any(k.startswith('se') for k in state_dict)
    if has_bacl and not has_se:
        options['use_bacl'] = True
        options['use_se'] = False
        print('[CKPT Compat] BACL checkpoint detected; use_bacl=True, use_se=False')
    elif has_se and not has_bacl:
        options['use_bacl'] = False
        options['use_se'] = True
        print('[CKPT Compat] SE checkpoint detected; use_bacl=False, use_se=True')


def _load_model(options, ckpt_path, device):
    model = get_model(options).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_missing = [k for k in missing if not k.startswith('se')]
    if ignored_missing:
        print(f'[Load] Missing keys: {ignored_missing}')
    if unexpected:
        print(f'[Load] Unexpected keys: {unexpected}')
    model.eval()
    return model


def branch_js(logits):
    probs = [F.softmax(logits[i].float(), dim=1) for i in range(3)]
    mean_prob = (probs[0] + probs[1] + probs[2]) / 3
    eps = 1e-8
    js = sum((p * (p.clamp(min=eps) / mean_prob.clamp(min=eps)).log()).sum(dim=1)
             for p in probs) / 3
    return js


def gate_margin(logits):
    top2 = torch.topk(logits[3].float(), k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def collect_split(model, loader, device, is_id=True, num_known=None):
    all_js, all_margin, all_gate_pred, all_labels = [], [], [], []
    all_branch_pred = [[] for _ in range(3)]

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            od = model(data)
            logits = od['logits']

            js = branch_js(logits)
            margin = gate_margin(logits)
            gate_pred = logits[3].argmax(dim=1)

            all_js.append(js.cpu())
            all_margin.append(margin.cpu())
            all_gate_pred.append(gate_pred.cpu())
            all_labels.append(labels.cpu() if is_id else torch.full_like(labels.cpu(), -1))
            for bi in range(3):
                all_branch_pred[bi].append(logits[bi].argmax(dim=1).cpu())

    result = {
        'js': torch.cat(all_js).numpy(),
        'margin': torch.cat(all_margin).numpy(),
        'gate_pred': torch.cat(all_gate_pred).numpy(),
        'labels': torch.cat(all_labels).numpy(),
        'branch_pred': [torch.cat(p).numpy() for p in all_branch_pred],
    }
    return result


def describe(name, arr):
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    print(f'{name}: mean={arr.mean():.6f} std={arr.std():.6f} '
          f'p10={q[0]:.6f} p25={q[1]:.6f} p50={q[2]:.6f} '
          f'p75={q[3]:.6f} p90={q[4]:.6f}')


def print_js_diagnostics(id_res, ood_res):
    print('\n=== Expert JS: ID vs OOD ===')
    describe('ID_JS ', id_res['js'])
    describe('OOD_JS', ood_res['js'])
    y = np.concatenate([np.zeros_like(id_res['js']), np.ones_like(ood_res['js'])])
    s = np.concatenate([id_res['js'], ood_res['js']])
    try:
        print(f'JS_AUROC(OOD higher) = {roc_auc_score(y, s):.5f}')
    except ValueError:
        print('JS_AUROC unavailable')


def print_margin_bins(id_res, ood_res, bins=5):
    print('\n=== Gate Margin Bins ===')
    for name, res in [('ID', id_res), ('OOD', ood_res)]:
        margin = res['margin']
        js = res['js']
        order = np.argsort(margin)
        chunks = np.array_split(order, bins)
        print(f'[{name}] low margin -> high margin')
        for i, idx in enumerate(chunks):
            if len(idx) == 0:
                continue
            print(f'  bin{i + 1}: n={len(idx):4d} '
                  f'margin_mean={margin[idx].mean():.4f} '
                  f'js_mean={js[idx].mean():.6f}')


def print_per_class_branch_acc(id_res, num_known):
    print('\n=== Per-Class Branch Accuracy on ID Test ===')
    labels = id_res['labels']
    gate_pred = id_res['gate_pred']
    branch_pred = id_res['branch_pred']

    header = 'class | gate   br1    br2    br3   best'
    print(header)
    print('-' * len(header))
    for c in range(num_known):
        mask = labels == c
        if mask.sum() == 0:
            continue
        gate_acc = (gate_pred[mask] == c).mean() * 100
        br_accs = [(branch_pred[bi][mask] == c).mean() * 100 for bi in range(3)]
        best = int(np.argmax(br_accs)) + 1
        print(f'{c:5d} | {gate_acc:5.1f} '
              f'{br_accs[0]:6.1f} {br_accs[1]:6.1f} {br_accs[2]:6.1f}   B{best}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_pos', nargs='?', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--bins', type=int, default=5)
    args = parser.parse_args()

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    options = get_config('osr')
    options['dataset'] = args.dataset_pos or args.dataset or options.get('dataset')
    options['gpu_ids'] = args.gpu_ids
    options['batch_size'] = args.batch_size
    options['seed'] = args.seed if args.seed is not None else options.get('seed', 0)
    options['resume'] = True
    options['ckpt'] = args.ckpt
    options['test_only'] = True
    options['out_dataset'] = None

    splits = SPLITS_AUROC.get(options['dataset'])
    if splits:
        known = splits[0]
        unknown = list(set(range(10)) - set(known))
        if 'tiny' in options['dataset']:
            unknown = list(set(range(200)) - set(known))
        options['item'] = 0
        options['known'] = known
        options['unknown'] = unknown

    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu_ids']
    set_seeding(options['seed'])

    _probe_ckpt_options(options, args.ckpt)
    train_loader, test_loader, out_loader = getLoader(options)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Loading checkpoint: {args.ckpt}')
    model = _load_model(options, args.ckpt, device)

    id_res = collect_split(model, test_loader, device, is_id=True, num_known=options['num_known'])
    ood_res = collect_split(model, out_loader, device, is_id=False, num_known=options['num_known'])

    print_js_diagnostics(id_res, ood_res)
    print_margin_bins(id_res, ood_res, bins=args.bins)
    print_per_class_branch_acc(id_res, options['num_known'])


if __name__ == '__main__':
    main()
