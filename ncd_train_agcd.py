import os
import torch
import numpy as np
import random
import argparse
import csv
import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from datasets.osr_loader import Tiny_ImageNet_Filter
from osr_main import splits_AUROC
from ncd_model import NCDWrapper
from ncd_losses import SupConLoss, InfoNCELoss
from ncd_utils import TwoCropTransform, cluster_acc
from ncd_active_data import ActiveDataset
from ncd_strategies import get_strategy
from core.net import MultiBranchNet
from ncd_model import Adapter, DINOHead
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

# EGDB 损失（仅在 use_egdb=True 时加载）
from egdb_losses import EGDBLoss

# ================================================================
# QA-P²OT：查询感知的渐进偏传输模块（替代 get_hard_transport_targets）
# ================================================================
from ncd_ot import QAP2OTSolver, build_ema_teacher, update_ema_teacher


def set_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _harmonic_mean(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    if np.any(values <= 0):
        return 0.0
    return float(len(values) / np.sum(1.0 / (values + eps)))


def make_ncd_save_dir(args):
    run_time = datetime.datetime.now().strftime("%m%d_%H%M")
    save_root = getattr(args, 'save_root', './ckpt/ncd_al')
    save_dir = os.path.join(save_root, str(args.dataset).lower(), run_time)
    os.makedirs(save_dir, exist_ok=True)
    args.run_time = run_time
    args.save_dir = save_dir
    return save_dir


def save_ncd_checkpoint(save_dir, filename, model, ema_teacher, args, round_idx,
                        metrics, active_dataset=None, dynamic_alignment=None):
    payload = {
        'round': round_idx,
        'model_state_dict': model.state_dict(),
        'ema_teacher_state_dict': ema_teacher.state_dict() if ema_teacher is not None else None,
        'metrics': metrics,
        'args': vars(args),
        'dynamic_alignment': dynamic_alignment,
    }
    if active_dataset is not None:
        payload['labeled_mask'] = active_dataset.labeled_mask.astype(np.bool_)
        payload['labeled_indices'] = np.where(active_dataset.labeled_mask)[0]
    path = os.path.join(save_dir, filename)
    torch.save(payload, path)
    return path


def write_ncd_summary(save_dir, all_results, args, aosdq=None):
    fields = ['round', 'budget', 'acc', 'nmi', 'ari', 'known_acc', 'udr', 'ca', 'osca', 'osdq']
    rows = []
    for r, acc, nmi, ari, known_acc, udr, ca, osca, osdq in all_results:
        rows.append({
            'round': r,
            'budget': r * args.query_size,
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'known_acc': known_acc,
            'udr': udr,
            'ca': ca,
            'osca': osca,
            'osdq': osdq,
        })

    csv_path = os.path.join(save_dir, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(save_dir, 'summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('NCD Active Discovery Summary\n')
        f.write('=' * 96 + '\n')
        f.write(f'dataset={args.dataset}\n')
        f.write(f'known={getattr(args, "known", [])}\n')
        f.write(f'unknown={getattr(args, "unknown", [])}\n')
        f.write(f'query_size={args.query_size}, al_rounds={args.al_rounds}, epochs_per_round={args.epochs_per_round}\n')
        f.write(f'ckpt={args.ckpt}\n')
        f.write(f'save_dir={save_dir}\n\n')
        f.write(f"{'Round':^6} | {'Budget':^8} | {'ACC(%)':^8} | {'NMI(%)':^8} | {'ARI(%)':^8} | {'KAcc(%)':^8} | {'UDR(%)':^8} | {'CA(%)':^7} | {'OSCA(%)':^8} | {'OSDQ(%)':^8}\n")
        f.write('-' * 112 + '\n')
        for row in rows:
            f.write(
                f"{row['round']:^6} | {row['budget']:^8} | {row['acc']:^8.2f} | "
                f"{row['nmi']:^8.2f} | {row['ari']:^8.2f} | {row['known_acc']:^8.2f} | "
                f"{row['udr']:^8.2f} | {row['ca']:^7.2f} | {row['osca']:^8.2f} | {row['osdq']:^8.2f}\n"
            )
        f.write('=' * 96 + '\n')
        if aosdq is not None:
            f.write(f'AOSDQ={aosdq:.2f}%\n')
        if rows:
            best_osdq = max(rows, key=lambda x: x['osdq'])
            best_osca = max(rows, key=lambda x: x['osca'])
            f.write(f"Best OSDQ: round={best_osdq['round']}, budget={best_osdq['budget']}, OSDQ={best_osdq['osdq']:.2f}%\n")
            f.write(f"Best OSCA: round={best_osca['round']}, budget={best_osca['budget']}, OSCA={best_osca['osca']:.2f}%\n")
    return csv_path, txt_path


class RemappedVisionDataset(Dataset):
    """Dataset wrapper whose labels are already remapped to 0..K-1."""

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def _to_pil(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[0] == 3:
                x = np.transpose(x, (1, 2, 0))
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 255).astype(np.uint8)
            return Image.fromarray(x)
        return x

    def __getitem__(self, index):
        img = self._to_pil(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[index])


def infer_dataset_from_ckpt_path(ckpt_path):
    path = str(ckpt_path or '').replace('\\', '/').lower()
    if 'tiny_imagenet' in path or 'tiny-imagenet' in path:
        return 'tiny_imagenet'
    if 'cifar_plus' in path or 'cifar-plus' in path:
        return 'cifar_plus'
    if 'cifar100' in path:
        return 'cifar100'
    if 'cifar10' in path:
        return 'cifar10'
    return None


def load_ckpt_options(ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return {}
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        return dict(ckpt.get('options', {})) if isinstance(ckpt, dict) else {}
    except Exception:
        return {}


def _select_remap(data, targets, classes, offset=0):
    targets = np.asarray(targets)
    cls_to_new = {int(c): offset + i for i, c in enumerate(classes)}
    idx = np.where(np.isin(targets, list(cls_to_new.keys())))[0]
    new_targets = np.asarray([cls_to_new[int(targets[i])] for i in idx], dtype=np.int64)
    return np.asarray(data)[idx], new_targets


def _tiny_data(root, split):
    ds = Tiny_ImageNet_Filter(os.path.join(root, 'tiny-imagenet-200', split), None, fast_tensor=True)
    return ds.memory_data.cpu().numpy(), ds.memory_targets.cpu().numpy()


def _protocol_classes(args, ckpt_options):
    dataset = args.dataset
    item = int(ckpt_options.get('item', getattr(args, 'item', 0)))
    plus_num = int(ckpt_options.get('plus_num', getattr(args, 'plus_num', 10)))

    if ckpt_options.get('known'):
        known = list(ckpt_options['known'])
    elif dataset == 'cifar10':
        known = [0, 1, 2, 4, 5, 9]
    else:
        split_key = 'cifar_plus' if dataset == 'cifar_plus' else dataset
        split = splits_AUROC[split_key]
        known = list(split[min(item, len(split) - 1)])

    if ckpt_options.get('unknown'):
        unknown = list(ckpt_options['unknown'])
    elif dataset == 'cifar_plus':
        key = f'cifar100-{plus_num}'
        if key in splits_AUROC:
            unknown = list(splits_AUROC[key][min(item, len(splits_AUROC[key]) - 1)])
        else:
            rng = np.random.default_rng(item + 42)
            unknown = rng.permutation(100)[:plus_num].tolist()
    else:
        total = 200 if dataset == 'tiny_imagenet' else (100 if dataset == 'cifar100' else 10)
        unknown = sorted(list(set(range(total)) - set(known)))

    return known, unknown, item, plus_num


def build_ncd_protocol(args):
    if str(args.dataset).lower() == 'auto':
        args.dataset = infer_dataset_from_ckpt_path(args.ckpt) or 'cifar10'
    supported = {'cifar10', 'cifar_plus', 'cifar100', 'tiny_imagenet'}
    if args.dataset not in supported:
        raise NotImplementedError(f'Unsupported NCD dataset: {args.dataset}. Supported: {sorted(supported)}')

    ckpt_options = load_ckpt_options(args.ckpt)
    known, unknown, item, plus_num = _protocol_classes(args, ckpt_options)
    args.item = item
    args.plus_num = plus_num
    args.known = known
    args.unknown = unknown
    args.num_known = len(known)
    if not getattr(args, 'estimate_classes', False):
        args.num_unknown_est = len(unknown)

    img_size = 64 if args.dataset == 'tiny_imagenet' else 32
    mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    args.img_size = img_size
    args.ncd_mean = mean
    args.ncd_std = std
    crop_pad = int(img_size * 0.125)

    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=crop_pad),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    ncd_transform = TwoCropTransform(transforms.Compose([
        transforms.RandomCrop(img_size, padding=crop_pad),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]))
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    strong_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=crop_pad),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]) if args.use_egdb else None

    if args.dataset == 'cifar10':
        tr = CIFAR10(root=args.data_root, train=True, download=True)
        te = CIFAR10(root=args.data_root, train=False, download=True)
        train_known_data, train_known_targets = _select_remap(tr.data, tr.targets, known, 0)
        train_unknown_data, train_unknown_targets = _select_remap(tr.data, tr.targets, unknown, len(known))
        test_known_data, test_known_targets = _select_remap(te.data, te.targets, known, 0)
        test_unknown_data, test_unknown_targets = _select_remap(te.data, te.targets, unknown, len(known))
    elif args.dataset == 'cifar100':
        tr = CIFAR100(root=args.data_root, train=True, download=True)
        te = CIFAR100(root=args.data_root, train=False, download=True)
        train_known_data, train_known_targets = _select_remap(tr.data, tr.targets, known, 0)
        train_unknown_data, train_unknown_targets = _select_remap(tr.data, tr.targets, unknown, len(known))
        test_known_data, test_known_targets = _select_remap(te.data, te.targets, known, 0)
        test_unknown_data, test_unknown_targets = _select_remap(te.data, te.targets, unknown, len(known))
    elif args.dataset == 'cifar_plus':
        c10_tr = CIFAR10(root=args.data_root, train=True, download=True)
        c10_te = CIFAR10(root=args.data_root, train=False, download=True)
        c100_tr = CIFAR100(root=args.data_root, train=True, download=True)
        c100_te = CIFAR100(root=args.data_root, train=False, download=True)
        train_known_data, train_known_targets = _select_remap(c10_tr.data, c10_tr.targets, known, 0)
        train_unknown_data, train_unknown_targets = _select_remap(c100_tr.data, c100_tr.targets, unknown, len(known))
        test_known_data, test_known_targets = _select_remap(c10_te.data, c10_te.targets, known, 0)
        test_unknown_data, test_unknown_targets = _select_remap(c100_te.data, c100_te.targets, unknown, len(known))
    else:
        train_all_data, train_all_targets = _tiny_data(args.data_root, 'train')
        val_all_data, val_all_targets = _tiny_data(args.data_root, 'val')
        train_known_data, train_known_targets = _select_remap(train_all_data, train_all_targets, known, 0)
        train_unknown_data, train_unknown_targets = _select_remap(train_all_data, train_all_targets, unknown, len(known))
        test_known_data, test_known_targets = _select_remap(val_all_data, val_all_targets, known, 0)
        test_unknown_data, test_unknown_targets = _select_remap(val_all_data, val_all_targets, unknown, len(known))

    train_data = np.concatenate([train_known_data, train_unknown_data], axis=0)
    train_targets = np.concatenate([train_known_targets, train_unknown_targets], axis=0)
    test_data = np.concatenate([test_known_data, test_unknown_data], axis=0)
    test_targets = np.concatenate([test_known_targets, test_unknown_targets], axis=0)

    trainset_base = RemappedVisionDataset(train_data, train_targets)
    testset_eval = RemappedVisionDataset(test_data, test_targets, transform=test_transform)
    initial_indices = np.where(trainset_base.targets < args.num_known)[0].tolist()

    print(f"   [Protocol] dataset={args.dataset} img_size={img_size}")
    print(f"   [Protocol] known={known}")
    print(f"   [Protocol] unknown={unknown[:20]}{' ...' if len(unknown) > 20 else ''}")
    print(f"   [Protocol] num_known={args.num_known} num_unknown={args.num_unknown_est}")

    return {
        'trainset_base': trainset_base,
        'raw_trainset': trainset_base,
        'testset_eval': testset_eval,
        'initial_indices': initial_indices,
        'train_transform': train_transform,
        'ncd_transform': ncd_transform,
        'test_transform': test_transform,
        'strong_transform': strong_transform,
    }



# NOTE: get_hard_transport_targets 已被 ncd_ot.QAP2OTSolver 替代。
# 保留此注释作为变更标记，原函数使用 Hungarian 均匀分配，
# 新版本使用标注感知 Sinkhorn 偏传输（QA-P²OT）。




def evaluate(model, loader, device, num_known=6, round_idx=0):
    model.eval()
    preds, targets = [], []
    novel_preds = []
    max_known_confs = []

    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            label_remapped = remap_labels(label, device)
            logits, _ = model(img)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(label_remapped.cpu().numpy())
            probs = F.softmax(logits, dim=1)
            known_probs = probs[:, :num_known]
            max_conf, _ = torch.max(known_probs, dim=1)
            max_known_confs.append(max_conf.cpu().numpy())
            novel_logits = logits[:, num_known:]
            n_preds = novel_logits.argmax(dim=1).cpu().numpy() + num_known
            novel_preds.append(n_preds)

    targets_np = np.concatenate(targets)
    preds_np = np.concatenate(preds)
    max_known_confs_np = np.concatenate(max_known_confs)
    novel_preds_np = np.concatenate(novel_preds)

    acc = cluster_acc(targets_np, preds_np) * 100
    nmi = normalized_mutual_info_score(targets_np, preds_np) * 100
    ari = adjusted_rand_score(targets_np, preds_np) * 100

    threshold = 0.85 - (round_idx * 0.02)
    actual_known_mask = targets_np < num_known
    actual_unknown_mask = targets_np >= num_known
    total_actual_unknowns = np.sum(actual_unknown_mask)
    pred_unknown_mask = (preds_np >= num_known) | (max_known_confs_np < threshold)
    known_correct_mask = actual_known_mask & (preds_np == targets_np) & (~pred_unknown_mask)
    known_acc = np.sum(known_correct_mask) / np.sum(actual_known_mask) if np.sum(actual_known_mask) > 0 else 0.0
    identified_unknown_mask = actual_unknown_mask & pred_unknown_mask
    identified_count = np.sum(identified_unknown_mask)

    udr = identified_count / total_actual_unknowns if total_actual_unknowns > 0 else 0.0
    if identified_count > 0:
        ca = cluster_acc(targets_np[identified_unknown_mask], novel_preds_np[identified_unknown_mask])
    else:
        ca = 0.0
    osca = 2 * udr * ca / (udr + ca) * 100 if (udr + ca) > 0 else 0.0
    osdq = _harmonic_mean([known_acc, udr, ca]) * 100

    return acc, nmi, ari, known_acc * 100, udr * 100, ca * 100, osca, osdq


def remap_labels(targets, device):
    return targets.to(device=device, dtype=torch.long)


def get_dynamic_label_mapping(model, loader, device, num_known=6, num_unknown=4):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for img, label, _ in loader:
            img = img.to(device)
            logits, _ = model(img)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(remap_labels(label, device).cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    novel_mask = targets >= num_known
    if not np.any(novel_mask):
        return {i: i for i in range(num_known + num_unknown)}

    novel_preds_nm = preds[novel_mask]
    novel_targets = targets[novel_mask]
    cost_matrix = np.zeros((num_unknown, num_unknown))
    for t, p in zip(novel_targets, novel_preds_nm):
        if p >= num_known:
            cost_matrix[t - num_known, p - num_known] -= 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {i: i for i in range(num_known)}
    for r, c in zip(row_ind, col_ind):
        mapping[r + num_known] = c + num_known

    print(f"   🔄 [Label Mapping] 真实标签 -> 模型神经元映射: {mapping}")
    return mapping


def estimate_novel_classes_only(osr_backbone, active_dataset, args, device, threshold=0.85, max_k=10):
    print("\n🔍 [Novel Discovery] 开启解耦模式：仅对未知池样本进行新类数量估计...")
    osr_backbone.eval()

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    unlabeled_dataset = active_dataset.get_unlabeled_dataset(eval_transform)
    loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False)
    unknown_feats = []

    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = imgs.to(device)
            outputs = osr_backbone(imgs)
            if isinstance(outputs, dict) and 'feat_vecs' in outputs and 'gate_pred' in outputs:
                gate = outputs['gate_pred'].unsqueeze(-1)
                feats = (outputs['feat_vecs'] * gate).sum(dim=1)
            else:
                feats = outputs['fts'] if isinstance(outputs, dict) and 'fts' in outputs else outputs
                if feats.dim() == 3: feats = feats.mean(dim=1)
            if feats.dim() == 4:
                feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
            feats = F.normalize(feats, dim=1, eps=1e-8)

            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            if isinstance(logits, list):
                logits = torch.stack(logits).mean(dim=0)

            probs = F.softmax(logits[:, :args.num_known], dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            unknown_mask = max_conf < threshold
            if unknown_mask.sum() > 0:
                unknown_feats.append(feats[unknown_mask].cpu().numpy())

    if len(unknown_feats) == 0:
        print("   ⚠️ 警告：没有拦截到未知样本，强制设为 1。")
        return 1

    unknown_feats = np.concatenate(unknown_feats)
    best_k, best_score = 2, -1
    for k in range(2, max_k + 1):
        if len(unknown_feats) <= k: break
        preds_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(unknown_feats)
        score = silhouette_score(unknown_feats, preds_k)
        if score > best_score:
            best_score, best_k = score, k

    print(f"✅ [Novel Discovery] 推断新类数量 = {best_k}（轮廓系数: {best_score:.4f}）")
    return best_k


def train_agcd(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting AGCD Training...")

    protocol = build_ncd_protocol(args)
    save_dir = make_ncd_save_dir(args)
    print(f"   💾 [NCD SaveDir] {save_dir}")
    train_transform = protocol['train_transform']
    ncd_transform = protocol['ncd_transform']
    test_transform = protocol['test_transform']
    strong_transform = protocol['strong_transform']
    trainset_base = protocol['trainset_base']
    raw_trainset = protocol['raw_trainset']
    testset_eval = protocol['testset_eval']
    loader_test = DataLoader(testset_eval, batch_size=256, shuffle=False, num_workers=args.num_workers)

    active_dataset = ActiveDataset(trainset_base)
    initial_indices = protocol['initial_indices']
    active_dataset.initialize_labels(initial_indices)

    print(f"   Pool: {len(initial_indices)} Labeled | {len(trainset_base) - len(initial_indices)} Unlabeled")

    ckpt_uses_bacl = False
    ckpt_legacy_split = False
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt_probe = torch.load(args.ckpt, map_location='cpu')
        probe_sd = ckpt_probe.get('net', ckpt_probe.get('state_dict', ckpt_probe))
        probe_sd = {k.replace('module.', ''): v for k, v in probe_sd.items()}
        has_bacl = any(k.startswith('bacl') for k in probe_sd)
        has_se = any(k.startswith('se') for k in probe_sd)
        l4_key = 'branch1_l4.0.0.residual_function.0.weight'
        if l4_key in probe_sd and probe_sd[l4_key].shape[0] == 64:
            ckpt_legacy_split = True
        ckpt_uses_bacl = has_bacl and not has_se
        if ckpt_uses_bacl:
            print("   🔧 [Backbone Compat] 检测到 BACL 版 OSR backbone，NCD 阶段启用 use_bacl=True")
        if ckpt_legacy_split:
            print("   🔧 [Backbone Compat] 检测到旧版 MEDAF 分层，NCD 阶段启用 legacy_split=True")

    backbone_args = {
        'img_size': args.img_size, 'backbone': 'resnet18', 'projection_dim': -1,
        'num_known': args.num_known, 'gate_temp': 0.1,
        'use_bacl': ckpt_uses_bacl,
        'legacy_split': ckpt_legacy_split
    }

    osr_backbone = MultiBranchNet(args=backbone_args).to(device)

    if getattr(args, 'estimate_classes', False):
        estimated_unknown = estimate_novel_classes_only(
            osr_backbone, active_dataset, args, device, threshold=0.85, max_k=10)
        args.num_unknown_est = estimated_unknown

    adapter = Adapter(in_dim=512, out_dim=512)
    dino_head = DINOHead(in_dim=512, out_dim=args.num_known + args.num_unknown_est)
    model = NCDWrapper(osr_backbone, adapter, dino_head).to(device)
    if args.ckpt:
        model._load_osr_weights(args.ckpt)

    criterion_simclr = InfoNCELoss(temperature=0.07).to(device)
    criterion_sup = SupConLoss(temperature=0.07).to(device)

    # ================================================================
    # QA-P²OT 初始化：EMA teacher + 标注感知 Sinkhorn 求解器
    # 在 model 构建完成后立即创建 teacher（参数相同，不参与梯度）
    # ================================================================
    osr_backbone_t = MultiBranchNet(args=backbone_args).to(device)
    adapter_t = Adapter(in_dim=512, out_dim=512)
    dino_head_t = DINOHead(in_dim=512, out_dim=args.num_known + args.num_unknown_est)

    # 组装 Teacher 模型
    ema_teacher = NCDWrapper(osr_backbone_t, adapter_t, dino_head_t).to(device)

    # 完美复制 Student 的当前权重（绕过计算图追踪）
    ema_teacher.load_state_dict(model.state_dict())

    # 彻底切断梯度，作为纯动量教师
    for p in ema_teacher.parameters():
        p.requires_grad_(False)
    ema_teacher.eval()
    ot_solver = QAP2OTSolver(
        num_known=args.num_known,
        num_novel=args.num_unknown_est,
        ema_decay=args.ot_ema_decay,
        eps=args.ot_eps,
        rho_known=1.0,      # 已知类：不过滤（已知类标注多、置信度高）
        rho_novel=args.ot_rho_novel,
        n_iter=30,          # Sinkhorn 迭代次数（30 次足够 10 类 CIFAR-10）
        alpha=1.0,          # Laplace 平滑，防止新类先验为零
        c_virtual=args.ot_c_virtual,
        query_prior_weight=args.ot_query_prior_weight,
        prior_floor=args.ot_prior_floor,
        ot_temp=args.ot_temp,
        lambda_dis=args.ot_lambda_dis,
        lambda_u=args.ot_lambda_u,
        use_adaptive_rho=args.ot_use_adaptive_rho,
        rho_min=args.ot_rho_min,
        rho_max=args.ot_rho_max,
        rho_beta=args.ot_rho_beta,
        rho_gamma=args.ot_rho_gamma,
        novel_only_unlabeled=True
    )
    print(f"   [QA-P²OT] OT solver initialized: "
          f"eps={ot_solver.eps}, rho_novel={ot_solver.rho_novel}, "
          f"q_w={ot_solver.query_prior_weight}, prior_floor={ot_solver.prior_floor}, "
          f"temp={ot_solver.ot_temp}, lambda_dis={ot_solver.lambda_dis}, "
          f"lambda_u={ot_solver.lambda_u}, adaptive_rho={ot_solver.use_adaptive_rho}, "
          f"n_iter={ot_solver.n_iter}")

    # EGDB 损失：feat_loss_w=0 关掉特征对齐，只用分类一致性，梯度不流回 backbone
    criterion_egdb = EGDBLoss(
        temperature=0.1, weight_floor=0.1, expert_temp=1.5, feat_loss_w=0.0
    ).to(device) if args.use_egdb else None

    if args.warmup_epochs > 0 and not args.resume_round0:
        print(f"\n🔥 [Warmup] Pre-training Adapter (SimCLR only)...")
        trainset_warmup = active_dataset.get_unlabeled_dataset(transform=ncd_transform)
        if len(trainset_warmup) == 0:
            trainset_warmup = active_dataset.get_labeled_dataset(transform=ncd_transform)
        loader_warmup = DataLoader(trainset_warmup, batch_size=args.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, momentum=0.9, weight_decay=5e-4)

        for epoch in range(args.warmup_epochs):
            model.train()
            loss_m = []
            pbar = tqdm(loader_warmup, desc=f"Warmup {epoch + 1}", ncols=80)
            for u_images, _, _ in pbar:
                u_inputs = torch.cat([u_images[0], u_images[1]], dim=0).to(device)
                _, u_feats = model(u_inputs)
                loss = criterion_simclr(u_feats)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                loss_m.append(loss.item())
                pbar.set_postfix({'Loss': f"{np.mean(loss_m):.4f}"})

    dynamic_alignment = {i: i for i in range(args.num_known + args.num_unknown_est)}
    all_results = []
    best_osdq = -1.0
    best_osca = -1.0

    for round_idx in range(args.al_rounds + 1):
        print(f"\n======== Round {round_idx} / {args.al_rounds} ========")

        skip_training = False
        if args.resume_round0 and round_idx == 0:
            print(f"   ⏭️ [Resume] 检测到断点参数，直接跳过训练，载入权重: {args.resume_round0}")
            checkpoint = torch.load(args.resume_round0, map_location=device)
            sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(sd)
            skip_training = True

        if not skip_training:
            if args.use_egdb:
                model.adapter_strong.load_state_dict(model.adapter.state_dict())
                model.dino_head_strong.load_state_dict(model.dino_head.state_dict())
                print(f"   🔄 [EGDB Sync] Round {round_idx} 训练开始前，已将主头权重同步至辅头")

            current_lr = args.lr if round_idx == 0 else args.lr * 0.1
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=current_lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs_per_round)

            trainset_labeled = active_dataset.get_labeled_dataset(transform=train_transform)

            # ===================================================================
            # 🚀 绝杀机制：WeightedRandomSampler 解决 30000 vs 300 的阶级不平衡
            # ===================================================================
            labeled_indices = np.where(active_dataset.labeled_mask)[0]
            labeled_targets_raw = [trainset_base.targets[i] for i in labeled_indices]

            labeled_targets_tensor = torch.tensor(labeled_targets_raw, device='cpu')
            labeled_targets_remapped = remap_labels(labeled_targets_tensor, device='cpu').numpy()

            num_known_samples = np.sum(labeled_targets_remapped < args.num_known)
            num_novel_samples = np.sum(labeled_targets_remapped >= args.num_known)

            if num_novel_samples == 0:
                sample_weights = [1.0] * len(labeled_targets_remapped)
            else:
                weight_known = 1.0 / num_known_samples
                weight_novel = 1.0 / num_novel_samples
                sample_weights = [weight_novel if t >= args.num_known else weight_known
                                  for t in labeled_targets_remapped]

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )

            loader_labeled = DataLoader(
                trainset_labeled, batch_size=args.batch_size, sampler=sampler, drop_last=True
            )
            # ===================================================================
            trainset_unlabeled = active_dataset.get_unlabeled_dataset(transform=ncd_transform)
            if len(trainset_unlabeled) == 0: break
            loader_unlabeled = DataLoader(
                trainset_unlabeled, batch_size=args.batch_size, shuffle=True, drop_last=True)

            for epoch in range(args.epochs_per_round):
                model.train()
                loss_meter = []
                rel_meter = []
                ot_stat_meter = {
                    'rho': [], 'edis': [], 'eagr': [], 'real': [],
                    'virt': [], 'wmin': [], 'wmax': []
                }
                egdb_fr_list = []

                # ========================================================
                # 🚀 绝妙的动态 EGDB 调度：区别对待 Round 0 和 主动学习阶段
                # ========================================================
                if round_idx == 0:
                    # Round 0：前期拓荒，让聚类自然生长，只在最后 egdb_tail 个 epoch 强行干预
                    if epoch >= (args.epochs_per_round - args.egdb_tail):
                        egdb_active = args.use_egdb
                        current_egdb_weight = args.egdb_weight
                    else:
                        egdb_active = False
                        current_egdb_weight = 0.0
                else:
                    # Round 1及以后：已有真实标签锚点，为防止 Loss 跳变，全程平滑线性预热
                    current_egdb_weight = args.egdb_weight * (epoch / max(1, args.epochs_per_round - 1))
                    egdb_active = args.use_egdb and current_egdb_weight > 0

                iter_labeled = iter(loader_labeled)
                pbar = tqdm(loader_unlabeled, desc=f"R{round_idx} E{epoch + 1}", ncols=100)



                for u_images, _, u_idxs in pbar:
                    try:
                        l_images, l_targets, _ = next(iter_labeled)
                    except StopIteration:
                        iter_labeled = iter(loader_labeled)
                        l_images, l_targets, _ = next(iter_labeled)

                    l_images, l_targets = l_images.to(device), l_targets.to(device)
                    l_targets_base = remap_labels(l_targets, device)
                    remapped_list = [dynamic_alignment.get(t.item(), t.item()) for t in l_targets_base]
                    l_targets_remapped = torch.tensor(remapped_list, dtype=torch.long, device=device)

                    # ---- 主任务（与基线完全一致）----
                    l_logits, l_feats = model(l_images)
                    loss_sup = criterion_sup(l_feats.unsqueeze(1), l_targets_remapped)
                    loss_ce = F.cross_entropy(l_logits, l_targets_remapped)

                    # u_images 始终是两视图，不受 EGDB 影响
                    u_view1, u_view2 = u_images[0].to(device), u_images[1].to(device)
                    u_inputs = torch.cat([u_view1, u_view2], dim=0)
                    u_logits, u_feats = model(u_inputs)
                    loss_simclr = criterion_simclr(u_feats)

                    # ================================================================
                    # QA-P²OT：用 EMA teacher + 标注感知 Sinkhorn 生成软伪标签
                    # 替代原来的 get_hard_transport_targets(u_logits.detach())
                    #
                    # 原来（Hard OT）：
                    #   with torch.no_grad():
                    #       ot_targets = get_hard_transport_targets(u_logits.detach())
                    #
                    # 现在（QA-P²OT）：
                    #   teacher 输出稳定 logits，Sinkhorn 生成软分配，
                    #   先验 q 由当前标注分布决定（已知类权重 >> 新类权重）
                    # ================================================================
                    with torch.no_grad():
                        ot_expert_logits = None
                        if args.ot_lambda_dis > 0 or args.ot_lambda_u > 0 or args.ot_use_adaptive_rho:
                            try:
                                ot_expert_logits = ema_teacher.forward_experts(
                                    u_inputs, use_strong_head=False
                                )  # [2B, 3, K]
                            except Exception as e:
                                if len(ot_stat_meter['rho']) == 0:
                                    print(f"   ⚠️ [QA-P²OT] Expert logits disabled for OT: {e}")
                        ot_targets, ot_reliability = ot_solver.get_targets(
                            student_logits=u_logits.detach(),
                            teacher_model=ema_teacher,
                            u_inputs=u_inputs,
                            active_dataset=active_dataset,
                            all_targets=trainset_base.targets,  # 🚀 直接把外部最纯粹的标签列表传进去
                            device=device,
                            round_idx=round_idx,
                            epoch=epoch + 1,
                            max_epoch=args.epochs_per_round,
                            expert_logits=ot_expert_logits,
                            return_reliability=True
                        )

                    bs = u_view1.size(0)
                    log_probs = F.log_softmax(u_logits / 0.1, dim=1)
                    weights_swapped = torch.cat([ot_reliability[bs:], ot_reliability[:bs]], dim=0).detach()

                    if args.ot_use_reliability_weight:
                        targets_swapped = torch.cat([ot_targets[bs:], ot_targets[:bs]], dim=0).detach()
                        loss_each = -torch.sum(targets_swapped * log_probs, dim=1)
                        loss_cluster = (
                            weights_swapped * loss_each
                        ).sum() / (weights_swapped.sum().detach() + 1e-8)
                    elif round_idx == 0:
                        # Round 0 still needs broad pseudo-label supervision to form novel clusters.
                        # Use reliability as soft smoothing instead of suppressing low-reliability samples.
                        uniform_targets = torch.ones_like(ot_targets) / ot_targets.size(1)
                        ot_rel_2d = ot_reliability.unsqueeze(1)
                        ot_targets_for_loss = ot_rel_2d * ot_targets + (1 - ot_rel_2d) * uniform_targets
                        targets_swapped = torch.cat([ot_targets_for_loss[bs:], ot_targets_for_loss[:bs]], dim=0)
                        loss_cluster = -torch.mean(torch.sum(targets_swapped * log_probs, dim=1))
                    else:
                        # After the first query round, active labels stabilize the structure; reliability
                        # weighting can down-weight boundary/noisy samples without starving cluster formation.
                        targets_swapped = torch.cat([ot_targets[bs:], ot_targets[:bs]], dim=0).detach()
                        per_sample_cluster = -torch.sum(targets_swapped * log_probs, dim=1)
                        loss_cluster = (
                            weights_swapped * per_sample_cluster
                        ).sum() / weights_swapped.sum().clamp_min(1e-6)

                    loss_ent = -torch.mean(
                        torch.sum(F.softmax(u_logits, dim=1) * F.log_softmax(u_logits, dim=1), dim=1))


                    # ================================================================
                    # 损失权重：恢复 loss_cluster=0.4（之前被错误降到 0.2）
                    # loss_cluster 由 QA-P²OT 的软标签驱动，权重需要足够大才能发现新类
                    # ================================================================
                    loss = (0.5 * loss_sup + 0.5 * loss_ce
                            + 0.1 * loss_simclr + 0.4 * loss_cluster + 0.1 * loss_ent)

                    # ================================================================
                    # 🚀 开放集隔离型 EGDB：只对高置信度已知类做强一致性，保护未知类流形
                    # ================================================================
                    if egdb_active:
                        # 1. 动态评估当前 batch 里哪些样本属于已知类空间分布
                        with torch.no_grad():
                            probs_known = F.softmax(u_logits[:bs, :args.num_known], dim=1)
                            max_known_conf, _ = torch.max(probs_known, dim=1)
                            tau = 0.85 - (round_idx * 0.02)
                            known_mask = max_known_conf > tau  # 判定为已知类相似的样本掩码

                        # 2. 只有当 batch 内存在已知类相似样本时，才激活辅助 EGDB 约束
                        if known_mask.sum() > 0:
                            pil_imgs = [raw_trainset[idx.item()][0] for idx in u_idxs]
                            # 仅对通过 OSR 拦截过滤的已知类相似样本进行强增强前向，省显存且实现语义隔离
                            strong_transform_list = [strong_transform(pil_imgs[i]) for i in range(bs) if known_mask[i]]
                            u_strong = torch.stack(strong_transform_list).to(device)

                            # 强视图走辅头，梯度不回传给主骨干
                            strong_logits, strong_feats = model(u_strong, use_strong_head=True)

                            # 专家看弱视图
                            with torch.no_grad():
                                expert_logits = model.forward_experts(u_view1, use_strong_head=False)  # [B, 3, C]

                            weak_ot_for_strong = ot_targets[:bs].detach()

                            # 🚀 核心过滤输入：只让已知类相似的子集参与强一致性拉扯
                            loss_egdb, egdb_stats = criterion_egdb(
                                strong_logits=strong_logits,
                                weak_ot_targets=weak_ot_for_strong[known_mask],
                                expert_logits=expert_logits[known_mask],
                                warmup=False,
                            )

                            loss = loss + current_egdb_weight * loss_egdb
                            egdb_fr_list.append(egdb_stats['filter_rate'])

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                    # ── EMA teacher 更新（每步 optimizer.step 后调用）──────────
                    update_ema_teacher(
                        model, ema_teacher, ema_decay=args.ot_ema_decay
                    )

                    loss_meter.append(loss.item())
                    rel_meter.append(weights_swapped.mean().item())
                    ot_stats = getattr(ot_solver, 'last_stats', {})
                    if ot_stats:
                        ot_stat_meter['rho'].append(ot_stats.get('rho_current', 0.0))
                        ot_stat_meter['edis'].append(ot_stats.get('expert_dis', 0.0))
                        ot_stat_meter['eagr'].append(ot_stats.get('expert_agreement', 1.0))
                        ot_stat_meter['real'].append(ot_stats.get('real_mass', 0.0))
                        ot_stat_meter['virt'].append(ot_stats.get('virtual_mass', 0.0))
                        ot_stat_meter['wmin'].append(ot_stats.get('weight_min', 0.0))
                        ot_stat_meter['wmax'].append(ot_stats.get('weight_max', 0.0))
                    postfix = {'L': f"{np.mean(loss_meter):.3f}"}
                    postfix['rel'] = f"{np.mean(rel_meter):.2f}"
                    if ot_stat_meter['rho']:
                        postfix['rho'] = f"{np.mean(ot_stat_meter['rho']):.2f}"
                        postfix['ed'] = f"{np.mean(ot_stat_meter['edis']):.4f}"
                        postfix['rm'] = f"{np.mean(ot_stat_meter['real']):.4f}"
                    if egdb_active and egdb_fr_list:
                        postfix['fr'] = f"{np.mean(egdb_fr_list):.2f}"
                    pbar.set_postfix(postfix)

                if ot_stat_meter['rho']:
                    print(
                        f"   [QA-P²OT] rho={np.mean(ot_stat_meter['rho']):.4f} "
                        f"edis={np.mean(ot_stat_meter['edis']):.6f} "
                        f"eagr={np.mean(ot_stat_meter['eagr']):.4f} "
                        f"real_mass={np.mean(ot_stat_meter['real']):.6f} "
                        f"virtual_mass={np.mean(ot_stat_meter['virt']):.6f} "
                        f"w={np.mean(rel_meter):.4f} "
                        f"w_min={np.mean(ot_stat_meter['wmin']):.4f} "
                        f"w_max={np.mean(ot_stat_meter['wmax']):.4f} "
                        f"loss_cluster={loss_cluster.item():.4f}"
                    )

                scheduler.step()

        acc, nmi, ari, known_acc, udr, ca, osca, osdq = evaluate(
            model, loader_test, device, args.num_known, round_idx=round_idx)
        metrics = {
            'acc': acc, 'nmi': nmi, 'ari': ari, 'known_acc': known_acc,
            'udr': udr, 'ca': ca, 'osca': osca, 'osdq': osdq,
        }
        all_results.append((round_idx, acc, nmi, ari, known_acc, udr, ca, osca, osdq))
        print(f"Round {round_idx} | ACC={acc:.2f}% NMI={nmi:.2f}% ARI={ari:.2f}% | KAcc={known_acc:.2f}% UDR={udr:.2f}% CA={ca:.2f}% OSCA={osca:.2f}% OSDQ={osdq:.2f}%")

        round_ckpt = save_ncd_checkpoint(
            save_dir, f'round_{round_idx:02d}.pth', model, ema_teacher, args,
            round_idx, metrics, active_dataset, dynamic_alignment)
        print(f"   💾 [Checkpoint] Saved {round_ckpt}")

        if osdq > best_osdq:
            best_osdq = osdq
            best_path = save_ncd_checkpoint(
                save_dir, 'model_best_osdq.pth', model, ema_teacher, args,
                round_idx, metrics, active_dataset, dynamic_alignment)
            print(f"   ⭐ [Best OSDQ] {best_osdq:.2f}% -> {best_path}")

        if osca > best_osca:
            best_osca = osca
            best_path = save_ncd_checkpoint(
                save_dir, 'model_best_osca.pth', model, ema_teacher, args,
                round_idx, metrics, active_dataset, dynamic_alignment)
            print(f"   ⭐ [Best OSCA] {best_osca:.2f}% -> {best_path}")

        write_ncd_summary(save_dir, all_results, args)

        if round_idx == 0 and not args.resume_round0:
            legacy_round0 = os.path.join(save_dir, 'round0_trained.pth')
            torch.save({'model_state_dict': model.state_dict(), 'metrics': metrics, 'args': vars(args)}, legacy_round0)
            print(f"   💾 [Backup] Round 0 训练特征已自动备份至: {legacy_round0}")

        if round_idx < args.al_rounds:
            print(f"   [Active Learning] 正在初始化策略: {args.strategy}")

            strategy_class = get_strategy(args.strategy)
            strategy_instance = strategy_class(active_dataset, model, args, device)

            # 🚀 统一调用：无论什么策略，都把需要的 Round 信息传进去
            # 如果策略本身不需要这些参数，由于我们定义了 **kwargs，它会自动忽略
            query_idxs = strategy_instance.query(
                args.query_size,
                current_round=round_idx,
                adaptive_round=2
            )

            active_dataset.update_labels(query_idxs)
            print(f"   ✅ [Active Learning] 成功查询并标注 {len(query_idxs)} 个高价值样本！")

            print(f"   🔄 [Label Alignment] 正在使用全量有标签数据更新全局神经元映射...")
            all_labeled_dataset = active_dataset.get_labeled_dataset(transform=test_transform)
            all_labeled_loader = DataLoader(all_labeled_dataset, batch_size=256, shuffle=False,
                                            num_workers=args.num_workers)
            dynamic_alignment = get_dynamic_label_mapping(
                model, all_labeled_loader, device, args.num_known, args.num_unknown_est)

    # ── Final Summary Table ────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"{'Round':^6} | {'ACC(%)':^8} | {'NMI(%)':^8} | {'ARI(%)':^8} | {'KAcc(%)':^8} | {'UDR(%)':^8} | {'CA(%)':^7} | {'OSCA(%)':^8} | {'OSDQ(%)':^8}")
    print("-" * 96)
    for r, acc, nmi, ari, known_acc, udr, ca, osca, osdq in all_results:
        print(f"{r:^6} | {acc:^8.2f} | {nmi:^8.2f} | {ari:^8.2f} | {known_acc:^8.2f} | {udr:^8.2f} | {ca:^7.2f} | {osca:^8.2f} | {osdq:^8.2f}")
    print("=" * 96)
    aosdq = None
    if len(all_results) > 1:
        budgets = np.asarray([r * args.query_size for r, *_ in all_results], dtype=np.float64)
        osdq_values = np.asarray([row[-1] for row in all_results], dtype=np.float64)
        if budgets[-1] > budgets[0]:
            aosdq = float(np.trapz(osdq_values, budgets) / (budgets[-1] - budgets[0]))
            print(f"AOSDQ={aosdq:.2f}%  (area under OSDQ-budget curve)")
    csv_path, txt_path = write_ncd_summary(save_dir, all_results, args, aosdq=aosdq)
    print(f"Saved NCD summary to: {txt_path}")
    print(f"Saved NCD metrics csv to: {csv_path}")
    print(f"Saved NCD checkpoints to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default=r'C:\Users\10943\PycharmProjects\PythonProject\MEDAF\ckpt\osr\tiny_imagenet\0715_2245\medaf_backbone_converted.pth')
    parser.add_argument('--data_root', type=str,
                        default=r'C:\Users\10943\PycharmProjects\PythonProject\MEDAF\data')
    parser.add_argument('--save_root', type=str, default='./ckpt/ncd_al',
                        help='NCD checkpoint root. Saved as save_root/dataset/MMDD_HHMM/.')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dataset', type=str, default='auto',
                        help='auto/cifar10/cifar_plus/cifar100/tiny_imagenet')
    parser.add_argument('--item', type=int, default=0, help='Open-set split index if ckpt has no saved options.')
    parser.add_argument('--plus_num', type=int, default=10, help='For cifar_plus: number of CIFAR100 unknown classes.')
    parser.add_argument('--num_known', type=int, default=6)
    parser.add_argument('--num_unknown_est', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--al_rounds', type=int, default=5)
    parser.add_argument('--epochs_per_round', type=int, default=10)
    parser.add_argument('--query_size', type=int, default=300)
    parser.add_argument('--estimate_classes', action='store_true')
    parser.add_argument('--strategy', type=str, default='BoundaryMarginJSSampling')
    parser.add_argument('--query_js_weight', type=float, default=0.6,
                        help='BoundaryMarginJSSampling 中专家 JS 分歧的权重')
    parser.add_argument('--query_margin_weight', type=float, default=0.4,
                        help='BoundaryMarginJSSampling 中低 margin 不确定性的权重')
    parser.add_argument('--query_stable_ratio', type=float, default=0.7,
                        help='BoundaryMarginJSSampling 中 stable anchor 查询比例')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_round0', type=str, default=None)
    parser.add_argument('--egdb_feat_w', type=float, default=0.2,
                        help='EGDB 内特征一致性损失权重')
    # EGDB 参数
    parser.add_argument('--use_egdb', action='store_true',
                        help='开启 EGDB 正则（每轮最后 egdb_tail 个 epoch 激活）')
    parser.add_argument('--egdb_tail', type=int, default=3,
                        help='每轮最后几个 epoch 激活 EGDB（默认 3，10 epoch 的轮次即 E8/E9/E10）')
    parser.add_argument('--egdb_weight', type=float, default=0.05,
                        help='EGDB 损失权重（默认 0.05，远小于主任务，不争梯度）')

    # QA-P²OT 参数
    parser.add_argument('--ot_eps', type=float, default=0.05,
                        help='QA-P2OT Sinkhorn 温度（越小分配越硬）')
    parser.add_argument('--ot_rho_novel', type=float, default=0.85,
                        help='新类虚拟簇过滤比例（1.0=不过滤，0.7=过滤30%%低置信）')
    parser.add_argument('--ot_ema_decay', type=float, default=0.999,
                        help='EMA teacher 动量')
    parser.add_argument('--ot_temp', type=float, default=1.0,
                        help='QA-P2OT 中 teacher/expert logits 转概率的 temperature')
    parser.add_argument('--ot_c_virtual', type=float, default=1.0,
                        help='虚拟簇基础代价，越大越不容易进入虚拟簇')
    parser.add_argument('--ot_lambda_dis', type=float, default=0.5,
                        help='class-wise expert disagreement 加到真实类代价的权重')
    parser.add_argument('--ot_lambda_u', type=float, default=0.5,
                        help='样本级专家不确定性降低虚拟簇代价的权重')
    parser.add_argument('--ot_use_reliability_weight', action='store_true', default=True,
                        help='使用虚拟簇真实质量作为 cluster loss 的样本可靠性权重')
    parser.add_argument('--ot_no_reliability_weight', dest='ot_use_reliability_weight',
                        action='store_false', help='关闭可靠性加权，回退到旧的 Round0 smoothing/后续加权逻辑')
    parser.add_argument('--ot_query_prior_weight', type=float, default=0.6,
                        help='主动查询分布进入 novel prior 的最大权重；会随 round warmup')
    parser.add_argument('--ot_prior_floor', type=float, default=0.5,
                        help='每个新类至少保留 ot_prior_floor/num_unknown_est 的 prior 容量')
    parser.add_argument('--ot_use_adaptive_rho', action='store_true',
                        help='开启专家一致性感知 adaptive rho')
    parser.add_argument('--ot_rho_min', type=float, default=0.2,
                        help='adaptive rho 最小值')
    parser.add_argument('--ot_rho_max', type=float, default=0.95,
                        help='adaptive rho 最大值')
    parser.add_argument('--ot_rho_beta', type=float, default=0.1,
                        help='adaptive rho EMA 平滑系数')
    parser.add_argument('--ot_rho_gamma', type=float, default=0.7,
                        help='专家一致性修正的保底系数')

    args = parser.parse_args()
    set_seeding(args.seed)
    train_agcd(args)
