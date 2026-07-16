import argparse, csv, math, os, sys
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from core import get_model
from core.net import build_backbone, conv1x1, ODLLoss
from datasets.osr_loader import CIFAR10_Filter, CIFAR100_Filter, Tiny_ImageNet_Filter
from datasets.tools import test_transform
from misc import get_config, set_seeding
from osr_main import getLoader, splits_AUROC

DEFAULT_RUNS = [
    ('tiny_imagenet', r'ckpt/osr/tiny_imagenet/0715_2245/model_best.pth'),
    ('cifar100', r'ckpt/osr/cifar100/0709_2016/model_best.pth'),
    ('cifar10', r'ckpt/osr/cifar10/0630_1557/model_best.pth'),
    ('cifar_plus', r'ckpt/osr/cifar_plus_plus10/0710_1647/split_01/model_best.pth'),
]
EPS = 1e-12
CRITICAL_PREFIXES = ('shared_l3','branch1_l4','branch1_l5','branch1_cls','branch2_l4','branch2_l5','branch2_cls','branch3_l4','branch3_l5','branch3_cls','gate_l3','gate_l4','gate_l5','gate_cls')


def setup_console():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def mkdir(p):
    os.makedirs(p, exist_ok=True)


def clean_float(x):
    try:
        v = float(x)
        return '' if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return x


def write_csv(path, rows, fields=None):
    mkdir(os.path.dirname(path))
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: clean_float(r.get(k, '')) for k in fields})


def rank_avg(x):
    x = np.asarray(x); order = np.argsort(x, kind='mergesort'); sx = x[order]
    r = np.empty(len(x), dtype=np.float64); i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]: j += 1
        r[order[i:j]] = (i + j - 1) / 2.0 + 1.0; i = j
    return r


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b); a = a[m]; b = b[m]
    if len(a) < 2 or np.std(a) < EPS or np.std(b) < EPS: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a); b = np.asarray(b); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2: return np.nan
    return pearson(rank_avg(a[m]), rank_avg(b[m]))


def describe(x):
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    keys = ['count','mean','std','min','max','p10','p25','p50','p75','p90','p95','p99']
    if len(x) == 0: return {k: np.nan for k in keys}
    q = np.percentile(x, [10,25,50,75,90,95,99])
    vals = [int(len(x)), x.mean(), x.std(), x.min(), x.max(), *q]
    return {k: float(v) if k != 'count' else int(v) for k, v in zip(keys, vals)}


def binary_metrics(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64); m = np.isfinite(s)
    y = y[m]; s = s[m]
    out = {'AUROC':np.nan,'TNR@95':np.nan,'FPR@95':np.nan,'DTACC':np.nan,'AUPR_IN':np.nan,'AUPR_OUT':np.nan}
    if len(np.unique(y)) < 2: return out
    out['AUROC'] = float(roc_auc_score(y, s)); fpr, tpr, _ = roc_curve(y, s)
    idx = int(np.argmin(np.abs(tpr - 0.95))); out['FPR@95'] = float(fpr[idx]); out['TNR@95'] = float(1 - fpr[idx])
    out['DTACC'] = float(np.max(0.5 * (tpr + 1 - fpr)))
    p, r, _ = precision_recall_curve(y, s); out['AUPR_OUT'] = float(auc(r, p))
    p, r, _ = precision_recall_curve(1 - y, -s); out['AUPR_IN'] = float(auc(r, p))
    return out


def get_sd(ckpt):
    if isinstance(ckpt, dict):
        for k in ['state_dict','model','model_state_dict']:
            if isinstance(ckpt.get(k), dict): return ckpt[k], k
    return ckpt, '<root>'


def strip_prefix(sd, prefix):
    ks = list(sd.keys())
    if ks and sum(k.startswith(prefix) for k in ks) >= max(1, int(0.8 * len(ks))):
        return {k[len(prefix):]: v for k, v in sd.items()}, True
    return sd, False


def normalize_sd(sd):
    sd = dict(sd); stripped = []
    for p in ['module.', 'model.']:
        sd2, ok = strip_prefix(sd, p)
        if ok: sd = sd2; stripped.append(p)
    return sd, stripped


class LegacyRoutingNet(nn.Module):
    """Inference-only compatibility model for early MEDAF checkpoints.

    These checkpoints have branch backbones/classifiers plus a compact routing gate
    (`routing_scale`, `gate_cls.0/1/3`) but no BACL modules and no separate gate_l3/l4/l5.
    The class is local to this diagnostic script, so training code remains untouched.
    """
    def __init__(self, args):
        super().__init__()
        backbone, feature_dim, self.cam_size = build_backbone(
            img_size=args['img_size'], backbone_name=args.get('backbone', 'resnet18'),
            projection_dim=-1, inchan=3)
        self.gate_temp = args.get('gate_temp', 1.0)
        self.num_known = args['num_known']
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        ch = list(backbone.children())
        self.shared_l3 = nn.Sequential(*ch[:-6])
        self.branch1_l4 = nn.Sequential(*ch[-6:-3])
        self.branch1_l5 = nn.Sequential(*ch[-3])
        self.branch1_cls = conv1x1(feature_dim, self.num_known)
        import copy
        self.branch2_l4 = copy.deepcopy(self.branch1_l4)
        self.branch2_l5 = copy.deepcopy(self.branch1_l5)
        self.branch2_cls = conv1x1(feature_dim, self.num_known)
        self.branch3_l4 = copy.deepcopy(self.branch1_l4)
        self.branch3_l5 = copy.deepcopy(self.branch1_l5)
        self.branch3_cls = conv1x1(feature_dim, self.num_known)
        self.routing_scale = nn.Parameter(torch.ones(()))
        self.gate_cls = nn.Sequential(
            nn.Linear(feature_dim * 3 + self.num_known, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        self.odl_b1 = ODLLoss(num_classes=self.num_known, feat_dim=feature_dim, margin=0.6)
        self.odl_b2 = ODLLoss(num_classes=self.num_known, feat_dim=feature_dim, margin=0.6)
        self.odl_b3 = ODLLoss(num_classes=self.num_known, feat_dim=feature_dim, margin=0.6)

    def _branch(self, x, l4, l5, cls):
        z = l5(l4(x.clone()))
        feat = self.avg_pool(z).view(z.size(0), -1)
        cams = cls(z)
        logits = self.avg_pool(cams).view(z.size(0), -1)
        return feat, cams, logits

    def forward(self, x, y=None, return_ft=False):
        ft = self.shared_l3(x)
        b1_feat, b1_cams, b1_logits = self._branch(ft, self.branch1_l4, self.branch1_l5, self.branch1_cls)
        b2_feat, b2_cams, b2_logits = self._branch(ft, self.branch2_l4, self.branch2_l5, self.branch2_cls)
        b3_feat, b3_cams, b3_logits = self._branch(ft, self.branch3_l4, self.branch3_l5, self.branch3_cls)
        avg_logits = (b1_logits + b2_logits + b3_logits) / 3.0
        gate_in = torch.cat([b1_feat, b2_feat, b3_feat, avg_logits], dim=1)
        gate_pred = F.softmax((self.gate_cls(gate_in) * self.routing_scale) / self.gate_temp, dim=1)
        stacked = torch.stack([b1_logits, b2_logits, b3_logits], dim=-1)
        fused = (stacked * gate_pred.view(gate_pred.size(0), 1, 3)).sum(-1)
        out = {'logits': [b1_logits, b2_logits, b3_logits, fused], 'gate_pred': gate_pred,
               'feat_vecs': torch.stack([b1_feat, b2_feat, b3_feat], dim=1)}
        return out
def detect_arch(options, sd):
    has_bacl = any(k.startswith('bacl') for k in sd)
    has_se = any(k.startswith('se') for k in sd)
    if has_bacl and not has_se:
        options['use_bacl'] = True; options['use_se'] = False
    elif has_se and not has_bacl:
        options['use_bacl'] = False; options['use_se'] = True
    k = 'branch1_l4.0.1.residual_function.4.weight'
    if k in sd and len(sd[k].shape) > 0:
        options['legacy_split'] = int(sd[k].shape[0]) == 64
    for k in ['branch1_cls.weight','branch2_cls.weight','branch3_cls.weight']:
        if k in sd:
            options['num_known'] = int(sd[k].shape[0]); break


def prepare_options(dataset, ckpt_path, args):
    base = dict(get_config('osr'))
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and isinstance(ckpt.get('options'), dict): base.update(ckpt['options'])
    base.update({'dataset': dataset, 'ckpt': ckpt_path, 'resume': True, 'test_only': True,
                 'img_size': 64 if dataset == 'tiny_imagenet' else 32,
                 'out_dataset': None, 'batch_size': args.batch_size, 'num_workers': args.num_workers,
                 'gpu_ids': args.gpu_ids, 'seed': args.seed if args.seed is not None else int(base.get('seed', 0))})
    if args.data_root:
        base['data_root'] = args.data_root
    item = base.get('item', 0)
    item = 0 if item == 'cross' else int(item)
    base['item'] = item
    if not base.get('known'):
        split = splits_AUROC.get(dataset, [[0,1,2,4,5,9]])
        base['known'] = list(split[min(item, len(split)-1)])
    if not base.get('unknown') and dataset != 'cifar_plus':
        total = 200 if dataset == 'tiny_imagenet' else (100 if dataset == 'cifar100' else 10)
        base['unknown'] = sorted(list(set(range(total)) - set(base['known'])))
    return base


def _filtered_loader(dataset, known, transform, batch_size, num_workers, pin_memory, root='./data', train=False):
    dataset.__Filter__(known=known)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)


def get_eval_loaders(options, device):
    """Build only ID/OOD evaluation loaders, avoiding train-set loading and downloads."""
    dataset = options['dataset']
    batch_size = int(options.get('batch_size', 256))
    num_workers = int(options.get('num_workers', 0))
    root = options.get('data_root', './data')
    pin_memory = device.type == 'cuda'
    allow_download = bool(options.get('allow_download', False))

    if dataset == 'cifar10':
        transform = test_transform(32)
        known = list(options['known'])
        unknown = list(options.get('unknown') or sorted(set(range(10)) - set(known)))
        id_set = CIFAR10_Filter(root=root, train=False, download=allow_download, transform=transform)
        ood_set = CIFAR10_Filter(root=root, train=False, download=allow_download, transform=transform)
        return (
            _filtered_loader(id_set, known, transform, batch_size, num_workers, pin_memory, root=root),
            _filtered_loader(ood_set, unknown, transform, batch_size, num_workers, pin_memory, root=root),
        )

    if dataset == 'cifar100':
        transform = test_transform(32)
        known = list(options['known'])
        unknown = list(options.get('unknown') or sorted(set(range(100)) - set(known)))
        id_set = CIFAR100_Filter(root=root, train=False, download=allow_download, transform=transform)
        ood_set = CIFAR100_Filter(root=root, train=False, download=allow_download, transform=transform)
        return (
            _filtered_loader(id_set, known, transform, batch_size, num_workers, pin_memory, root=root),
            _filtered_loader(ood_set, unknown, transform, batch_size, num_workers, pin_memory, root=root),
        )

    if dataset == 'cifar_plus':
        transform = test_transform(32)
        known = list(options['known'])
        plus_num = int(options.get('plus_num', 10))
        item = int(options.get('item', 0))
        c100_key = f'cifar100-{plus_num}'
        if c100_key in splits_AUROC:
            unknown = list(splits_AUROC[c100_key][item])
        else:
            rng = np.random.default_rng(item + 42)
            unknown = rng.permutation(100)[:plus_num].tolist()
        options['unknown'] = unknown
        id_set = CIFAR10_Filter(root=root, train=False, download=allow_download, transform=transform)
        ood_set = CIFAR100_Filter(root=root, train=False, download=allow_download, transform=transform)
        return (
            _filtered_loader(id_set, known, transform, batch_size, num_workers, pin_memory, root=root),
            _filtered_loader(ood_set, unknown, transform, batch_size, num_workers, pin_memory, root=root),
        )

    if dataset == 'tiny_imagenet':
        known = list(options['known'])
        unknown = list(options.get('unknown') or sorted(set(range(200)) - set(known)))
        val_root = os.path.join(root, 'tiny-imagenet-200', 'val')
        fast_eval = bool(options.get('tiny_fast_eval_tensor', True))
        val_memory = Tiny_ImageNet_Filter(val_root, None, fast_tensor=fast_eval)
        id_set = Tiny_ImageNet_Filter(
            val_root, None if fast_eval else test_transform(64),
            memory_data=val_memory.memory_data,
            memory_targets=val_memory.memory_targets,
            fast_tensor=fast_eval)
        id_set.__Filter__(known)
        ood_set = Tiny_ImageNet_Filter(
            val_root, None if fast_eval else test_transform(64),
            memory_data=val_memory.memory_data,
            memory_targets=val_memory.memory_targets,
            fast_tensor=fast_eval)
        ood_set.__Filter__(unknown)
        loader_kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_workers, 'pin_memory': pin_memory}
        return torch.utils.data.DataLoader(id_set, **loader_kwargs), torch.utils.data.DataLoader(ood_set, **loader_kwargs)

    _, id_loader, ood_loader = getLoader(options)
    return id_loader, ood_loader


def load_model(ckpt_path, options, device, log):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    log.append(f'checkpoint path: {ckpt_path}')
    log.append('checkpoint keys: ' + (', '.join(list(ckpt.keys())[:30]) if isinstance(ckpt, dict) else '<state_dict>'))
    sd, source = get_sd(ckpt); sd, stripped = normalize_sd(sd)
    log.append(f'state_dict source: {source}')
    if stripped: log.append('stripped prefixes: ' + ', '.join(stripped))
    if isinstance(ckpt, dict) and isinstance(ckpt.get('options'), dict): options.update(ckpt['options'])
    detect_arch(options, sd)
    use_legacy_routing = ('routing_scale' in sd and not any(k.startswith('gate_l3') for k in sd))
    if use_legacy_routing:
        options['use_bacl'] = False
        options['legacy_split'] = True
        log.append('Detected LegacyRoutingNet checkpoint: routing_scale present, no gate_l3/gate_l4/gate_l5.')
        model = LegacyRoutingNet(options).to(device)
    else:
        model = get_model(options).to(device)
    try:
        model.load_state_dict(sd, strict=True)
        log.append('model parameter loading status: strict=True exact match')
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.append('model parameter loading status: strict=False fallback')
        log.append('strict load error: ' + str(e).split('\n')[0])
        log.append('missing keys: ' + (', '.join(missing) if missing else '<none>'))
        log.append('unexpected keys: ' + (', '.join(unexpected) if unexpected else '<none>'))
        critical_missing = [k for k in missing if k.startswith(CRITICAL_PREFIXES)]
        if options.get('use_bacl', False):
            critical_missing += [k for k in missing if k.startswith(('bacl1','bacl2','bacl3'))]
        critical_unexpected = [k for k in unexpected if k.startswith(CRITICAL_PREFIXES)]
        if critical_missing or critical_unexpected:
            raise RuntimeError(f'Critical keys mismatch. missing={critical_missing}, unexpected={critical_unexpected}')
    model.eval()
    if isinstance(ckpt, dict):
        log.append(f'checkpoint epoch: {ckpt.get("epoch", "<unknown>")}')
        log.append(f'checkpoint best metric: {ckpt.get("auroc", ckpt.get("best_auroc", "<unknown>"))}')
    return model, options


def expert_js(logits, T=1.0):
    ps = []
    for i in range(3):
        p = F.softmax(logits[i].float() / T, dim=1).clamp_min(EPS)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(EPS)
        ps.append(p)
    m = ((ps[0] + ps[1] + ps[2]) / 3).clamp_min(EPS)
    return sum((p * (p.log() - m.log())).sum(1) for p in ps) / 3


def collect(model, loader, device, is_ood, T, start=0):
    rows, logits_all, idx = [], [], start
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device); labels = labels.to(device)
            out = model(images)
            if not isinstance(out, dict) or 'logits' not in out or len(out['logits']) < 4:
                raise RuntimeError('forward must return dict with logits=[b1,b2,b3,fused]')
            lg = [x.float() for x in out['logits']]
            gate = out.get('gate_pred', torch.full((images.size(0), 3), 1/3, device=device))
            be = torch.stack([torch.logsumexp(lg[i], 1) for i in range(3)], 1)
            bm = torch.stack([lg[i].max(1).values for i in range(3)], 1)
            bp = torch.stack([lg[i].argmax(1) for i in range(3)], 1)
            fused = lg[3]; fp = fused.argmax(1); fe = torch.logsumexp(fused, 1); fm = fused.max(1).values
            f_msp = F.softmax(fused, dim=1).max(1).values
            ens = torch.logsumexp((lg[0] + lg[1] + lg[2]) / 3, 1)
            margin = torch.topk(fused, 2, dim=1).values; margin = margin[:,0] - margin[:,1]
            js = expert_js(lg, T)
            correct = (fp == labels) & (not is_ood)
            # Store logits on CPU only for cosine diagnostics; number of classes is small in OSR splits.
            logits_all.append(torch.stack([lg[0], lg[1], lg[2]], 1).detach().cpu())
            for i in range(images.size(0)):
                rows.append({'sample_index': idx, 'is_ood': int(is_ood), 'true_label': int(labels[i].cpu()),
                    'fused_pred': int(fp[i].cpu()), 'branch1_pred': int(bp[i,0].cpu()), 'branch2_pred': int(bp[i,1].cpu()), 'branch3_pred': int(bp[i,2].cpu()),
                    'id_correct': int(correct[i].cpu()), 'fused_msp': float(f_msp[i].cpu()), 'fused_mls': float(fm[i].cpu()), 'fused_energy': float(fe[i].cpu()), 'ensemble_energy': float(ens[i].cpu()),
                    'branch1_energy': float(be[i,0].cpu()), 'branch2_energy': float(be[i,1].cpu()), 'branch3_energy': float(be[i,2].cpu()),
                    'branch_energy_mean': float(be[i].mean().cpu()), 'branch_energy_std': float(be[i].std(unbiased=False).cpu()), 'branch_mls_std': float(bm[i].std(unbiased=False).cpu()),
                    'margin': float(margin[i].cpu()), 'js': float(js[i].cpu()), 'js_residual': 0.0, 'conditional_js_residual': 0.0,
                    'gate_w1': float(gate[i,0].cpu()), 'gate_w2': float(gate[i,1].cpu()), 'gate_w3': float(gate[i,2].cpu())})
                idx += 1
    return rows, torch.cat(logits_all, 0).numpy(), idx


def rows_to_arr(rows):
    return {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in rows[0].keys()}

def fit_residual(arr, bins=10):
    idm = arr['is_ood'] == 0; mg = arr['margin'][idm]; js = arr['js'][idm]
    if len(mg) == 0: return np.zeros_like(arr['js']), np.zeros_like(arr['js']), []
    edges = np.percentile(mg, np.linspace(0,100,bins+1)); edges[0] = -np.inf; edges[-1] = np.inf
    means, stds, fit_rows = [], [], []
    for i in range(bins):
        m = (mg >= edges[i]) & (mg < edges[i+1] if i < bins-1 else mg <= edges[i+1])
        vals = js[m]; mean = float(vals.mean()) if len(vals) else float(js.mean()); std = float(vals.std()) if len(vals) else float(js.std())
        std = max(std, 1e-6); means.append(mean); stds.append(std)
        fit_rows.append({'group':'ID_RESIDUAL_FIT','bin':i+1,'n':int(len(vals)),'margin_low':edges[i],'margin_high':edges[i+1],'js_mean':mean,'js_std':std})
    bi = np.digitize(arr['margin'], edges[1:-1], right=False)
    res = np.clip((arr['js'] - np.asarray(means)[bi]) / (np.asarray(stds)[bi] + 1e-6), -10, 10)
    cond = res * (arr['margin'] <= np.percentile(mg, 50)).astype(np.float64)
    return res, cond, fit_rows


def bins_table(arr, key, mode, bins=5):
    rows = []
    for group, flag in [('ID',0),('OOD',1)]:
        idx = np.where(arr['is_ood'] == flag)[0]
        idx = idx[np.argsort(arr[key][idx])]
        for i, chunk in enumerate(np.array_split(idx, bins)):
            if len(chunk) == 0: continue
            r = {'group':group,'bin':i+1,'n':int(len(chunk))}
            if mode == 'margin':
                r.update({'margin_mean':float(arr['margin'][chunk].mean()), 'js_mean':float(arr['js'][chunk].mean()), 'js_median':float(np.median(arr['js'][chunk])),
                          'energy_mean':float(arr['fused_energy'][chunk].mean()), 'energy_std_mean':float(arr['branch_energy_std'][chunk].mean())})
            else:
                r.update({'energy_mean':float(arr['fused_energy'][chunk].mean()), 'js_mean':float(arr['js'][chunk].mean()), 'margin_mean':float(arr['margin'][chunk].mean())})
            rows.append(r)
    return rows


def per_class_acc(arr, num_known):
    rows, cnt = [], Counter(); lab = arr['true_label'].astype(int); idm = arr['is_ood'] == 0
    for c in range(num_known):
        m = idm & (lab == c)
        if not m.any(): continue
        accs = [float((arr[f'branch{i}_pred'][m].astype(int) == c).mean()) for i in [1,2,3]]
        best = int(np.argmax(accs)) + 1; cnt[best] += 1
        rows.append({'class':c,'n':int(m.sum()),'fused_acc':float((arr['fused_pred'][m].astype(int)==c).mean()),'branch1_acc':accs[0],'branch2_acc':accs[1],'branch3_acc':accs[2],'best_branch':f'B{best}'})
    for i in [1,2,3]: rows.append({'class':f'BEST_COUNT_B{i}','n':cnt[i],'fused_acc':'','branch1_acc':'','branch2_acc':'','branch3_acc':'','best_branch':''})
    return rows, cnt


def pred_consistency(arr, mask):
    p = np.stack([arr['branch1_pred'], arr['branch2_pred'], arr['branch3_pred']], 1).astype(int)[mask]
    if len(p) == 0: return {}
    allsame = (p[:,0] == p[:,1]) & (p[:,0] == p[:,2])
    alldiff = (p[:,0] != p[:,1]) & (p[:,0] != p[:,2]) & (p[:,1] != p[:,2])
    return {'top1_all_same':float(allsame.mean()), 'top1_two_same':float((~allsame & ~alldiff).mean()), 'top1_all_diff':float(alldiff.mean()),
            'disagree_b1_b2':float((p[:,0] != p[:,1]).mean()), 'disagree_b1_b3':float((p[:,0] != p[:,2]).mean()), 'disagree_b2_b3':float((p[:,1] != p[:,2]).mean())}


def logit_cosines(logits3, mask):
    x = logits3[mask]
    if len(x) == 0: return {'logit_cos_b1_b2':np.nan,'logit_cos_b1_b3':np.nan,'logit_cos_b2_b3':np.nan}
    def cos(a,b):
        num = (a*b).sum(1); den = np.linalg.norm(a,axis=1) * np.linalg.norm(b,axis=1) + EPS
        return float(np.mean(num / den))
    return {'logit_cos_b1_b2':cos(x[:,0],x[:,1]), 'logit_cos_b1_b3':cos(x[:,0],x[:,2]), 'logit_cos_b2_b3':cos(x[:,1],x[:,2])}


def summary(arr, logits3, bestcnt):
    y = arr['is_ood'].astype(int); idm = y == 0; oodm = y == 1; rows = []
    def add(sec, met, val): rows.append({'section':sec,'metric':met,'value':val})
    for name, vals in [('ID_JS',arr['js'][idm]),('OOD_JS',arr['js'][oodm])]:
        for k,v in describe(vals).items(): add(name,k,v)
    scores = {'JS_AUROC':arr['js'], 'FusedEnergy_neg':-arr['fused_energy'], 'MSP_neg':-arr['fused_msp'], 'MLS_neg':-arr['fused_mls'], 'EnsembleEnergy_neg':-arr['ensemble_energy'],
              'Br1Energy_neg':-arr['branch1_energy'], 'Br2Energy_neg':-arr['branch2_energy'], 'Br3Energy_neg':-arr['branch3_energy'],
              'BranchEnergyMean_neg':-arr['branch_energy_mean'], 'BranchEnergyStd':arr['branch_energy_std'], 'BranchMLSStd':arr['branch_mls_std']}
    branch_aucs = []
    for name, sc in scores.items():
        m = binary_metrics(y, sc)
        for k,v in m.items(): add(name,k,v)
        if name.startswith('Br') and name.endswith('Energy_neg'): branch_aucs.append(m['AUROC'])
    add('BranchEnergy','mean_of_branch_AUROCs',float(np.nanmean(branch_aucs)) if branch_aucs else np.nan)
    for label, mask in [('correct_ID', idm & (arr['id_correct']==1)), ('wrong_ID', idm & (arr['id_correct']==0))]:
        mm = mask | oodm
        if len(np.unique(y[mm])) == 2: add('JS_conditioned', f'AUROC_{label}_vs_OOD', binary_metrics(y[mm], arr['js'][mm])['AUROC'])
    for group, mask in [('ID',idm),('OOD',oodm),('ALL',np.ones_like(y,dtype=bool))]:
        for k in ['margin','fused_energy','branch_energy_std']:
            add(f'Corr_{group}',f'Pearson_JS_{k}',pearson(arr['js'][mask],arr[k][mask])); add(f'Corr_{group}',f'Spearman_JS_{k}',spearman(arr['js'][mask],arr[k][mask]))
    for group, mask in [('ID',idm),('OOD',oodm)]:
        for k,v in pred_consistency(arr,mask).items(): add(f'PredConsistency_{group}',k,v)
        for k,v in logit_cosines(logits3,mask).items(): add(f'LogitCosine_{group}',k,v)
        for nm,a,b in [('b1_b2','branch1_energy','branch2_energy'),('b1_b3','branch1_energy','branch3_energy'),('b2_b3','branch2_energy','branch3_energy')]:
            add(f'EnergyCorr_{group}',f'Pearson_{nm}',pearson(arr[a][mask],arr[b][mask])); add(f'EnergyCorr_{group}',f'Spearman_{nm}',spearman(arr[a][mask],arr[b][mask]))
    for i in [1,2,3]: add('PerClassBestBranch',f'B{i}_best_class_count',bestcnt[i])
    return rows


def sweep_bds(arr, alphas):
    y = arr['is_ood'].astype(int); out = []
    for a in alphas:
        out.append({'alpha':a, **binary_metrics(y, -(arr['fused_energy'] - a * arr['js']))})
    return out


def sweep_residual(arr, betas):
    y = arr['is_ood'].astype(int); base = -arr['fused_energy']; rows = [{'method':'base_ood_neg_fused_energy','beta':0.0, **binary_metrics(y,base)}]
    for b in betas:
        for name,key in [('raw_js','js'),('js_residual','js_residual'),('conditional_js_residual','conditional_js_residual')]:
            rows.append({'method':name,'beta':b, **binary_metrics(y, base + b * arr[key])})
    return rows

def save_plots(out_dir, dataset, tag, arr, bds, resid):
    y = arr['is_ood'].astype(int); idm = y == 0; oodm = y == 1; title = f'{dataset} | {tag}'
    def sf(name): plt.tight_layout(); plt.savefig(os.path.join(out_dir,name), dpi=160); plt.close()
    plt.figure(); plt.hist(arr['js'][idm],60,alpha=.6,label='ID',density=True); plt.hist(arr['js'][oodm],60,alpha=.6,label='OOD',density=True); plt.title(title+' JS histogram'); plt.xlabel('JS'); plt.ylabel('density'); plt.legend(); sf('js_hist_id_ood.png')
    plt.figure()
    for m,l in [(idm,'ID'),(oodm,'OOD')]:
        v=np.sort(arr['js'][m]);
        if len(v): plt.plot(v,np.linspace(0,1,len(v)),label=l)
    plt.title(title+' JS ECDF'); plt.xlabel('JS'); plt.ylabel('ECDF'); plt.legend(); sf('js_ecdf_id_ood.png')
    rng=np.random.default_rng(0); idx=np.arange(len(y));
    if len(idx)>5000: idx=rng.choice(idx,5000,replace=False)
    c=np.where(y[idx]==1,'tab:red','tab:blue')
    plt.figure(); plt.scatter(arr['margin'][idx],arr['js'][idx],c=c,s=6,alpha=.35); plt.title(title+' JS vs margin'); plt.xlabel('fused margin'); plt.ylabel('JS'); sf('js_vs_margin.png')
    plt.figure(); plt.scatter(arr['fused_energy'][idx],arr['js'][idx],c=c,s=6,alpha=.35); plt.title(title+' JS vs energy'); plt.xlabel('fused Energy (ID score)'); plt.ylabel('JS'); sf('js_vs_energy.png')
    for name,key in [('margin_bin_js.png','margin'),('energy_bin_js.png','fused_energy')]:
        plt.figure()
        for m,l in [(idm,'ID'),(oodm,'OOD')]:
            ii=np.where(m)[0]; ii=ii[np.argsort(arr[key][ii])]; ch=np.array_split(ii,5); plt.plot(range(1,6),[arr['js'][x].mean() if len(x) else np.nan for x in ch],marker='o',label=l)
        plt.title(title+' '+name[:-4]); plt.xlabel('low to high '+key+' bin'); plt.ylabel('JS mean'); plt.legend(); sf(name)
    plt.figure()
    for sc,l in [(-arr['fused_energy'],'OOD=-FusedEnergy'),(arr['js'],'JS')]:
        fpr,tpr,_=roc_curve(y,sc); plt.plot(fpr,tpr,label=f'{l} AUC={roc_auc_score(y,sc):.3f}')
    plt.title(title+' ROC energy vs JS'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); sf('roc_energy_vs_js.png')
    plt.figure(); plt.plot([r['alpha'] for r in bds],[r['AUROC'] for r in bds],marker='o'); plt.title(title+' BDS alpha sweep'); plt.xlabel('alpha'); plt.ylabel('AUROC'); sf('roc_bds_alpha.png')
    plt.figure()
    for method in ['raw_js','js_residual','conditional_js_residual']:
        rr=[r for r in resid if r['method']==method]; plt.plot([r['beta'] for r in rr],[r['AUROC'] for r in rr],marker='o',label=method)
    plt.title(title+' JS residual beta sweep'); plt.xlabel('beta'); plt.ylabel('AUROC'); plt.legend(); sf('roc_js_residual.png')
    plt.figure()
    for k in ['gate_w1','gate_w2','gate_w3']: plt.hist(arr[k],50,alpha=.45,label=k,density=True)
    plt.title(title+' gate weight distribution'); plt.xlabel('gate weight'); plt.ylabel('density'); plt.legend(); sf('gate_weight_distribution.png')


def metric(rows, sec, met):
    for r in rows:
        if r['section']==sec and r['metric']==met: return r['value']
    return np.nan


def make_report(dataset, ckpt, options, arr, sm, mb, eb, bds, resid, pc, loadlog):
    lines = list(loadlog) + [f'dataset: {dataset}', f'known classes: {options.get("known")}', f'unknown classes: {options.get("unknown")}', f'num_known: {options.get("num_known")}',
        'Score direction: project MSP/MLS/Energy are ID scores; this script negates them when computing OOD-score metrics.',
        'Diagnostic only; alpha/beta selected on test OOD is not valid for final reporting.', '']
    idd=describe(arr['js'][arr['is_ood']==0]); odd=describe(arr['js'][arr['is_ood']==1])
    lines.append('=== Expert JS: ID vs OOD ===')
    for name,d in [('ID_JS :',idd),('OOD_JS:',odd)]:
        lines.append(name)
        for k in ['count','mean','std','min','max','p10','p25','p50','p75','p90','p95','p99']: lines.append(f'{k}={d[k] if k=="count" else format(d[k], ".6f")}')
        lines.append('')
    lines += [f'JS_AUROC={metric(sm,"JS_AUROC","AUROC"):.6f}', f'JS_AUPR_OUT={metric(sm,"JS_AUROC","AUPR_OUT"):.6f}', f'JS_AUPR_IN={metric(sm,"JS_AUROC","AUPR_IN"):.6f}',
              f'JS AUROC correct-ID vs OOD={metric(sm,"JS_conditioned","AUROC_correct_ID_vs_OOD"):.6f}', f'JS AUROC wrong-ID vs OOD={metric(sm,"JS_conditioned","AUROC_wrong_ID_vs_OOD"):.6f}', '']
    lines.append('=== Gate Margin Bins ===')
    for g in ['ID','OOD']:
        lines.append(f'[{g}] low margin -> high margin')
        for r in [x for x in mb if x['group']==g]: lines.append('bin{bin}: n={n} margin_mean={margin_mean:.6f} js_mean={js_mean:.6f} js_median={js_median:.6f} energy_mean={energy_mean:.6f} energy_std_mean={energy_std_mean:.6f}'.format(**r))
    lines.append('\n=== Energy Bins ===')
    for g in ['ID','OOD']:
        lines.append(f'[{g}] low energy -> high energy')
        for r in [x for x in eb if x['group']==g]: lines.append('bin{bin}: n={n} energy_mean={energy_mean:.6f} js_mean={js_mean:.6f} margin_mean={margin_mean:.6f}'.format(**r))
    base=metric(sm,'FusedEnergy_neg','AUROC'); msp=metric(sm,'MSP_neg','AUROC'); jsauc=metric(sm,'JS_AUROC','AUROC'); ens=metric(sm,'EnsembleEnergy_neg','AUROC')
    br=[metric(sm,f'Br{i}Energy_neg','AUROC') for i in [1,2,3]]; bestbr=np.nanmax(br)
    best_bds=max(bds,key=lambda r:-np.inf if np.isnan(r['AUROC']) else r['AUROC']); best_res=max(resid,key=lambda r:-np.inf if np.isnan(r['AUROC']) else r['AUROC'])
    best_counts={r['class']:r['n'] for r in pc if isinstance(r['class'],str) and r['class'].startswith('BEST_COUNT')}
    lines += ['', '=== Automatic Interpretation ===',
        f'1. OOD_JS mean ({odd["mean"]:.6f}) is {"higher" if odd["mean"]>idd["mean"] else "not higher"} than ID_JS mean ({idd["mean"]:.6f}).',
        f'2. JS_AUROC={jsauc:.4f}; level={"strong >0.8" if jsauc>=.8 else ("moderate >0.7" if jsauc>=.7 else ("above chance >0.5" if jsauc>=.5 else "not above chance"))}.',
        f'3. Low-margin ID JS={mb[0]["js_mean"]:.6f} if bins exist; low-margin ID often indicates hard ID samples.',
        f'4. High-margin OOD may still have low JS; inspect OOD bin5 in margin_bins.csv.',
        f'5. Base fused Energy AUROC={base:.4f}; MSP AUROC={msp:.4f}; raw JS AUROC={jsauc:.4f}.',
        f'6. Best diagnostic BDS alpha={best_bds["alpha"]}, AUROC={best_bds["AUROC"]:.4f}.',
        f'7. Best diagnostic JS residual method={best_res["method"]}, beta={best_res["beta"]}, AUROC={best_res["AUROC"]:.4f}.',
        f'8. Per-class best branch counts: {best_counts}.', f'9. Ensemble AUROC={ens:.4f}; best single branch AUROC={bestbr:.4f}.']
    benefit='unknown-sensitive disagreement' if jsauc>=.8 else ('variance reduction' if ens>=bestbr else 'class specialization or mixed evidence')
    lines += [f'10. Current multi-expert benefit is closest to: {benefit}.', '', f'Executed command example: python analyze_expert_js.py --dataset {dataset} --ckpt "{ckpt}" --js_temperature {options.get("js_temperature",1.0)}']
    return '\n'.join(lines)


def analyze_one(dataset, ckpt, args):
    ckpt=os.path.normpath(ckpt)
    if not os.path.exists(ckpt): raise FileNotFoundError(ckpt)
    set_seeding(args.seed if args.seed is not None else 0)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    options=prepare_options(dataset,ckpt,args); options['js_temperature']=args.js_temperature
    loadlog=[]; model,options=load_model(ckpt,options,device,loadlog)
    options.update({'dataset':dataset,'ckpt':ckpt,'resume':True,'test_only':True,'out_dataset':None,'batch_size':args.batch_size,'num_workers':args.num_workers,'allow_download':args.allow_download})
    print('\n'.join(loadlog)); print(f'dataset={dataset} known={options.get("known")} unknown={options.get("unknown")} num_known={options.get("num_known")}')
    id_loader, ood_loader = get_eval_loaders(options, device)
    rows, logits_id, nxt = collect(model,id_loader,device,False,args.js_temperature,0)
    rows2, logits_ood, _ = collect(model,ood_loader,device,True,args.js_temperature,nxt); rows += rows2
    logits3=np.concatenate([logits_id,logits_ood],0); arr=rows_to_arr(rows)
    arr['js_residual'],arr['conditional_js_residual'],fit_rows=fit_residual(arr,10)
    for i,r in enumerate(rows): r['js_residual']=float(arr['js_residual'][i]); r['conditional_js_residual']=float(arr['conditional_js_residual'][i])
    tag=os.path.basename(os.path.dirname(ckpt))
    if tag.lower().startswith('split_'):
        tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt))) + '_' + tag
    out=os.path.join('results','js_analysis',f'{dataset}_{tag}'); mkdir(out)
    mb=bins_table(arr,'margin','margin',5); eb=bins_table(arr,'fused_energy','energy',5); pc,bestcnt=per_class_acc(arr,int(options.get('num_known',0)))
    sm=summary(arr,logits3,bestcnt); bds=sweep_bds(arr,args.bds_alpha); resid=sweep_residual(arr,args.beta)
    fields=['sample_index','is_ood','true_label','fused_pred','branch1_pred','branch2_pred','branch3_pred','id_correct','fused_msp','fused_mls','fused_energy','ensemble_energy','branch1_energy','branch2_energy','branch3_energy','branch_energy_mean','branch_energy_std','branch_mls_std','margin','js','js_residual','conditional_js_residual','gate_w1','gate_w2','gate_w3']
    write_csv(os.path.join(out,'sample_scores.csv'),rows,fields); write_csv(os.path.join(out,'summary_metrics.csv'),sm,['section','metric','value']); write_csv(os.path.join(out,'margin_bins.csv'),mb+fit_rows); write_csv(os.path.join(out,'energy_bins.csv'),eb)
    write_csv(os.path.join(out,'per_class_branch_accuracy.csv'),pc); write_csv(os.path.join(out,'bds_alpha_sweep.csv'),bds); write_csv(os.path.join(out,'js_residual_beta_sweep.csv'),resid)
    save_plots(out,dataset,tag,arr,bds,resid)
    report=make_report(dataset,ckpt,options,arr,sm,mb,eb,bds,resid,pc,loadlog)
    with open(os.path.join(out,'analysis_report.txt'),'w',encoding='utf-8') as f: f.write(report)
    print(report); print('\nSaved analysis to: '+out); return out


def main():
    setup_console(); p=argparse.ArgumentParser(description='Analyze MEDAF expert JS divergence without training.')
    p.add_argument('--dataset',choices=['cifar10','cifar100','svhn','tiny_imagenet','cifar_plus']); p.add_argument('--ckpt'); p.add_argument('--run_all',action='store_true')
    p.add_argument('--js_temperature',type=float,default=1.0); p.add_argument('--bds_alpha',nargs='*',type=float,default=[0,0.01,0.05,0.1,0.2,0.5,1.0]); p.add_argument('--beta',nargs='*',type=float,default=[0.01,0.05,0.1,0.2,0.5,1.0])
    p.add_argument('--batch_size',type=int,default=256); p.add_argument('--num_workers',type=int,default=0); p.add_argument('--gpu_ids',default='0'); p.add_argument('--seed',type=int,default=71324); p.add_argument('--cpu',action='store_true')
    p.add_argument('--data_root',default=None,help='Dataset root. Defaults to osr.yml data_root or ./data.')
    p.add_argument('--allow_download',action='store_true',help='Allow torchvision CIFAR/SVHN download if local files are missing.')
    args=p.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_ids
    runs=DEFAULT_RUNS if args.run_all else None
    if runs is None:
        if not args.dataset or not args.ckpt: p.error('Either use --run_all or provide --dataset and --ckpt.')
        runs=[(args.dataset,args.ckpt)]
    print('Diagnostic only; alpha/beta selected on test OOD is not valid for final reporting.')
    outs=[]
    for d,c in runs: outs.append(analyze_one(d,c,args))
    print('\nCompleted analyses:'); [print('  '+x) for x in outs]

if __name__ == '__main__':
    main()








