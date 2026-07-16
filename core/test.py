import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from misc.util import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

MAX_NUM = 999999

def _batch_xy(batch):
    """Return image tensor and mapped label from normal or metadata-rich batches."""
    return batch[0], batch[1]


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def clear_react_thresholds(model):
    target = _unwrap_model(model)
    if hasattr(target, 'clear_react_thresholds'):
        target.clear_react_thresholds()


def set_react_thresholds(model, thresholds):
    target = _unwrap_model(model)
    if hasattr(target, 'set_react_thresholds'):
        target.set_react_thresholds(thresholds)


def compute_react_thresholds(model, train_loader, percentile=90.0, device=None,
                             max_values_per_branch=2000000):
    """Estimate one ReAct clipping threshold for each expert branch.

    The thresholds are computed from branch_l5 activations on the known-class
    training set. To keep memory bounded, each branch stores at most a random
    subset of activations.
    """
    target = _unwrap_model(model)
    if device is None:
        device = next(target.parameters()).device

    clear_react_thresholds(model)
    target.eval()

    names = ['branch1_l5', 'branch2_l5', 'branch3_l5']
    buckets = [[] for _ in range(3)]
    counts = [0, 0, 0]

    def _make_hook(branch_idx):
        def _hook(_module, _inputs, output):
            flat = output.detach().float().reshape(-1).cpu()
            remaining = max_values_per_branch - counts[branch_idx]
            if remaining <= 0:
                return
            if flat.numel() > remaining:
                idx = torch.randperm(flat.numel())[:remaining]
                flat = flat[idx]
            buckets[branch_idx].append(flat)
            counts[branch_idx] += flat.numel()
        return _hook

    hooks = []
    for bi, name in enumerate(names):
        module = getattr(target, name, None)
        if module is None:
            raise AttributeError(f'Model does not expose {name}; cannot compute ReAct thresholds.')
        hooks.append(module.register_forward_hook(_make_hook(bi)))

    with torch.no_grad():
        for batch in train_loader:
            data, _labels = _batch_xy(batch)
            data = data.to(device)
            _ = target(data)
            if all(c >= max_values_per_branch for c in counts):
                break

    for h in hooks:
        h.remove()

    thresholds = []
    for bi, parts in enumerate(buckets):
        if not parts:
            raise RuntimeError(f'No activations collected for branch {bi + 1}.')
        vals = torch.cat(parts)
        q = torch.quantile(vals, float(percentile) / 100.0).item()
        thresholds.append(q)
    return thresholds


def compute_branch_prototypes(model, train_loader, num_classes, device):
    """Per-branch class prototypes from training data for prototype-based OSR scoring."""
    model.eval()
    proto_sums = None
    proto_cnt  = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for batch in train_loader:
            data, labels = _batch_xy(batch)
            data, labels = data.to(device), labels.to(device)
            od = model(data)
            if 'feat_vecs' not in od:
                return None
            fv = od['feat_vecs']  # [B, 3, D]
            if proto_sums is None:
                feat_dim = fv.shape[2]
                proto_sums = [torch.zeros(num_classes, feat_dim).to(device) for _ in range(3)]
            for c in range(num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    for bi in range(3):
                        proto_sums[bi][c] += fv[mask, bi, :].sum(0)
                    proto_cnt[c] += mask.sum().float()

    if proto_sums is None:
        return None
    cnt = proto_cnt.unsqueeze(1).clamp(min=1)
    return [F.normalize(proto_sums[bi] / cnt, p=2, dim=1) for bi in range(3)]


def compute_score(logit_list, softmax_list, score_wgts, branch_opt, fts=None):
    """Returns the composite score used for model selection (per score_wgts config)."""
    msp = softmax_list[branch_opt].max(1)[0]
    mls = logit_list[branch_opt].max(1)[0]

    if score_wgts[2] != 0:
        ftl = fts.mean(dim=[2, 3]).norm(dim=1, p=2)
        return score_wgts[0] * msp + score_wgts[1] * mls + score_wgts[2] * ftl
    else:
        w0, w1 = score_wgts[0], score_wgts[1]
        if w0 == 0 and w1 == 0:
            return torch.logsumexp(logit_list[branch_opt].float(), dim=1)
        return w0 * msp + w1 * mls


def evaluation(model, test_loader, out_loader, **options):
    model.eval()
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    correct = 0
    total = 0
    n = 0

    pred_close = []
    pred_open  = []
    labels_close = []
    labels_open  = []
    score_close  = []
    score_open   = []

    open_labels     = torch.zeros(MAX_NUM)
    score_probs     = torch.zeros(MAX_NUM)  # composite score used for ckpt selection
    mls_probs       = torch.zeros(MAX_NUM)  # max logit score (gate branch)
    energy_probs    = torch.zeros(MAX_NUM)  # logsumexp energy (gate branch) — always independent
    ensemble_probs  = torch.zeros(MAX_NUM)  # energy of avg(b1,b2,b3) ≡ old gate_temp=100
    bds_probs       = torch.zeros(MAX_NUM)  # [B] BDS: energy(gate) - alpha * JS(b1,b2,b3)
    proto_probs     = torch.zeros(MAX_NUM)  # [C] max-branch prototype cosine similarity
    br_energy_probs = [torch.zeros(MAX_NUM) for _ in range(3)]

    need_ft       = (options.get('score_wgts', [0, 1, 0])[2] != 0)
    branch_opt    = options['branch_opt']
    bds_alpha     = options.get('bds_alpha', 1.0)
    use_bds_score = options.get('use_bds_score', False)
    prototypes    = options.get('prototypes', None)  # list of 3 tensors [C, D] or None

    progress_interval = int(options.get('eval_progress_interval', 20))
    eval_start = time.time()
    print(f"Start Evaluation... (Feature Calc: {'ON' if need_ft else 'OFF'})", flush=True)

    with torch.no_grad():
        # --- 1. 已知类测试 ---
        known_start = time.time()
        for batch_idx, batch in enumerate(test_loader):
            if options.get('max_eval_batches') is not None and batch_idx >= int(options.get('max_eval_batches')):
                break
            if progress_interval > 0 and (batch_idx == 0 or (batch_idx + 1) % progress_interval == 0):
                print(f"Testing Known Classes: [{batch_idx + 1}/{len(test_loader)}]", flush=True)

            data, labels = _batch_xy(batch)
            data, labels = data.to(device), labels.to(device)
            batch_size = labels.size(0)

            output_dict = model(data, return_ft=need_ft)
            logits_list  = output_dict['logits']                           # [b1,b2,b3,gate]
            softmax_list = torch.stack(logits_list)                        # [4,B,C]
            softmax_list = torch.softmax(
                softmax_list / options.get('lgs_temp', 1.0), dim=2)

            fts = output_dict.get('fts') if need_ft else None
            score_temp = compute_score(logits_list, softmax_list,
                                       options['score_wgts'], branch_opt, fts=fts)

            # Energy scores — always computed independently from the raw logits
            gate_energy = torch.logsumexp(logits_list[branch_opt].float(), dim=1)
            gate_mls    = logits_list[branch_opt].float().max(1)[0]
            ens_logit   = (logits_list[0].float() +
                           logits_list[1].float() +
                           logits_list[2].float()) / 3
            ens_energy  = torch.logsumexp(ens_logit, dim=1)

            # [B] BDS = energy(gate) − α·JS(p1,p2,p3)
            # 已知类：分支预测一致 → JS小 → BDS接近energy；OOD：分支分歧 → BDS更低
            _p = [F.softmax(logits_list[i].float(), dim=1) for i in range(3)]
            _m = (_p[0] + _p[1] + _p[2]) / 3
            _eps = 1e-8
            _js = sum((p * (p.clamp(min=_eps) / _m.clamp(min=_eps)).log()).sum(dim=1)
                      for p in _p) / 3
            bds_temp = gate_energy - bds_alpha * _js
            if use_bds_score:
                score_temp = bds_temp

            # [C] prototype score: max over branches of max cosine-sim to nearest class prototype
            if prototypes is not None and 'feat_vecs' in output_dict:
                fv = output_dict['feat_vecs']  # [B, 3, D]
                br_sims = []
                for bi in range(3):
                    fn = F.normalize(fv[:, bi, :], p=2, dim=1)
                    br_sims.append((fn @ prototypes[bi].t()).max(dim=1)[0])
                proto_temp = torch.stack(br_sims).max(dim=0)[0]
                proto_probs[n: n + batch_size] = proto_temp.detach().cpu()

            score_probs   [n: n + batch_size] = score_temp.detach().cpu()
            mls_probs     [n: n + batch_size] = gate_mls.detach().cpu()
            energy_probs  [n: n + batch_size] = gate_energy.detach().cpu()
            ensemble_probs[n: n + batch_size] = ens_energy.detach().cpu()
            bds_probs     [n: n + batch_size] = bds_temp.detach().cpu()
            for bi in range(3):
                e_bi = torch.logsumexp(logits_list[bi].float(), dim=1)
                br_energy_probs[bi][n: n + batch_size] = e_bi.detach().cpu()
            open_labels[n: n + batch_size] = 1
            n += batch_size

            score_close.append(score_temp.detach().cpu().numpy())
            pred_label = softmax_list[branch_opt].data.max(1)[1]
            total   += labels.size(0)
            correct += (pred_label == labels.data).sum()
            pred_close  .append(softmax_list[branch_opt].data.cpu().numpy())
            labels_close.append(labels.data.cpu().numpy())
        print(f"Known evaluation done in {time.time() - known_start:.1f}s", flush=True)

        # --- 2. 未知类测试 ---
        unknown_start = time.time()
        for batch_idx, batch in enumerate(out_loader):
            if options.get('max_eval_batches') is not None and batch_idx >= int(options.get('max_eval_batches')):
                break
            if progress_interval > 0 and (batch_idx == 0 or (batch_idx + 1) % progress_interval == 0):
                print(f"Testing Unknown Classes: [{batch_idx + 1}/{len(out_loader)}]", flush=True)

            data, labels = _batch_xy(batch)
            data, labels = data.to(device), labels.to(device)
            batch_size = labels.size(0)
            ood_label  = torch.zeros_like(labels) - 1

            output_dict = model(data, return_ft=need_ft)
            logits_list  = output_dict['logits']
            softmax_list = torch.stack(logits_list)
            softmax_list = torch.softmax(
                softmax_list / options.get('lgs_temp', 1.0), dim=2)

            fts = output_dict.get('fts') if need_ft else None
            score_temp = compute_score(logits_list, softmax_list,
                                       options['score_wgts'], branch_opt, fts=fts)

            gate_energy = torch.logsumexp(logits_list[branch_opt].float(), dim=1)
            gate_mls    = logits_list[branch_opt].float().max(1)[0]
            ens_logit   = (logits_list[0].float() +
                           logits_list[1].float() +
                           logits_list[2].float()) / 3
            ens_energy  = torch.logsumexp(ens_logit, dim=1)

            _p = [F.softmax(logits_list[i].float(), dim=1) for i in range(3)]
            _m = (_p[0] + _p[1] + _p[2]) / 3
            _eps = 1e-8
            _js = sum((p * (p.clamp(min=_eps) / _m.clamp(min=_eps)).log()).sum(dim=1)
                      for p in _p) / 3
            bds_temp = gate_energy - bds_alpha * _js
            if use_bds_score:
                score_temp = bds_temp

            if prototypes is not None and 'feat_vecs' in output_dict:
                fv = output_dict['feat_vecs']
                br_sims = []
                for bi in range(3):
                    fn = F.normalize(fv[:, bi, :], p=2, dim=1)
                    br_sims.append((fn @ prototypes[bi].t()).max(dim=1)[0])
                proto_temp = torch.stack(br_sims).max(dim=0)[0]
                proto_probs[n: n + batch_size] = proto_temp.detach().cpu()

            score_probs   [n: n + batch_size] = score_temp.detach().cpu()
            mls_probs     [n: n + batch_size] = gate_mls.detach().cpu()
            energy_probs  [n: n + batch_size] = gate_energy.detach().cpu()
            ensemble_probs[n: n + batch_size] = ens_energy.detach().cpu()
            bds_probs     [n: n + batch_size] = bds_temp.detach().cpu()
            for bi in range(3):
                e_bi = torch.logsumexp(logits_list[bi].float(), dim=1)
                br_energy_probs[bi][n: n + batch_size] = e_bi.detach().cpu()
            open_labels[n: n + batch_size] = 0
            n += batch_size

            score_open .append(score_temp.detach().cpu().numpy())
            pred_open  .append(softmax_list[branch_opt].data.cpu().numpy())
            labels_open.append(ood_label.data.cpu().numpy())
        print(f"Unknown evaluation done in {time.time() - unknown_start:.1f}s", flush=True)

    # --- 3. 指标计算 ---
    acc = float(correct) * 100. / float(total)

    pred_close   = np.concatenate(pred_close,   0)
    pred_open    = np.concatenate(pred_open,    0)
    labels_close = np.concatenate(labels_close, 0)
    labels_open  = np.concatenate(labels_open,  0)
    score_close  = np.concatenate(score_close,  0)
    score_open   = np.concatenate(score_open,   0)

    open_np = open_labels[:n].numpy()

    def _auroc(scores_tensor):
        arr = scores_tensor[:n].numpy()
        fpr_, tpr_, thr_ = roc_curve(open_np, arr)
        return auc(fpr_, tpr_), fpr_, tpr_, thr_

    auroc_score,    fpr,     tpr,     thresholds = _auroc(score_probs)
    auroc_mls,      *_                           = _auroc(mls_probs)
    auroc_energy,   *_                           = _auroc(energy_probs)
    auroc_ensemble, *_                           = _auroc(ensemble_probs)
    auroc_bds,      *_                           = _auroc(bds_probs)
    auroc_proto     = _auroc(proto_probs)[0] if prototypes is not None else None

    auroc_branches = []
    for bi in range(3):
        a, *_ = _auroc(br_energy_probs[bi])
        auroc_branches.append(a)

    # TNR@95
    thresh_idx_95 = np.abs(np.array(tpr) - 0.95).argmin()
    tnr_at_tpr95  = 1. - fpr[thresh_idx_95]

    # DTACC
    dtacc = 0.5 * (tpr + (1. - fpr)).max()

    # Macro F1
    pred1 = np.argmax(pred_close, axis=1)
    pred2 = np.argmax(pred_open,  axis=1)
    total_pred_label = np.concatenate([pred1, pred2], 0)
    total_label      = np.concatenate([labels_close, labels_open], 0)
    total_pred       = np.concatenate([score_close,  score_open],  0)
    threshold_f1     = thresholds[thresh_idx_95]
    open_pred        = (total_pred > threshold_f1).astype(np.float32)
    macro_f1 = f1_score(total_label,
                        ((total_pred_label + 1) * open_pred) - 1,
                        average='macro')

    # AUPR
    score_arr = score_probs[:n].numpy().reshape(-1)
    precision, recall, _ = precision_recall_curve(open_np, score_arr)
    aupr_in = auc(recall, precision)
    precision, recall, _ = precision_recall_curve(
        np.bitwise_not(open_np.astype(bool)), -score_arr)
    aupr_out = auc(recall, precision)

    print('=' * 50)
    print(f'📊 Evaluation Results:')
    print(f'   Accuracy:          {acc:.3f}%')
    print(f'   AUROC(MLS):        {auroc_mls:.5f}')
    print(f'   AUROC(Energy/Gate):{auroc_energy:.5f}')
    print(f'   AUROC(Ensemble):   {auroc_ensemble:.5f}   <- avg(b1,b2,b3) ≡ old gate_temp=100')
    print(f'   AUROC(BDS):        {auroc_bds:.5f}   <- energy - alpha*JS(b1,b2,b3)')
    if auroc_proto is not None:
        print(f'   AUROC(Proto-Max):  {auroc_proto:.5f}   <- max-branch prototype cosine-sim [Innovation C]')
    print(f'   AUROC(Score/Main): {auroc_score:.5f}   <- ckpt selection metric{"  [=BDS]" if use_bds_score else ""}')
    print(f'   --- Per-Branch Diagnostic ---')
    for bi in range(3):
        print(f'   AUROC(Br{bi+1}-Energy): {auroc_branches[bi]:.5f}')
    br_mean = float(np.mean(auroc_branches))
    print(f'   AUROC(BrMean):     {br_mean:.5f}  (mean of 3 branch AUROCs)')
    print(f'   --- Main Metrics ---')
    print(f'   TNR@95:       {tnr_at_tpr95:.5f}')
    print(f'   DTACC:        {dtacc:.5f}')
    print(f'   AUPR_IN:      {aupr_in:.5f}')
    print(f'   AUPR_OUT:     {aupr_out:.5f}')
    print(f'   Macro F1:     {macro_f1:.5f}')
    print(f'   Eval Time:    {time.time() - eval_start:.1f}s')
    print('=' * 50)

    # --- 分支特征多样性诊断 ---
    try:
        sample_data, _ = next(iter(test_loader))
        sample_data = sample_data[:32].to(device)
        with torch.no_grad():
            od = model(sample_data)
            if isinstance(od, dict) and 'feat_vecs' in od:
                fv = od['feat_vecs']  # [B, 3, 512]
                f1n = torch.nn.functional.normalize(fv[:, 0], dim=1)
                f2n = torch.nn.functional.normalize(fv[:, 1], dim=1)
                f3n = torch.nn.functional.normalize(fv[:, 2], dim=1)
                cos12 = (f1n * f2n).sum(1).mean().item()
                cos13 = (f1n * f3n).sum(1).mean().item()
                cos23 = (f2n * f3n).sum(1).mean().item()
                print(f'   Branch cosine sim: B1-B2={cos12:.3f} B1-B3={cos13:.3f} B2-B3={cos23:.3f}')
                b1_p = torch.softmax(od['logits'][0], dim=1)
                b2_p = torch.softmax(od['logits'][1], dim=1)
                b3_p = torch.softmax(od['logits'][2], dim=1)
                m_p  = (b1_p + b2_p + b3_p) / 3
                def kl(p, q):
                    return (p * (p / (q + 1e-8) + 1e-8).log()).sum(1)
                js = (kl(b1_p, m_p) + kl(b2_p, m_p) + kl(b3_p, m_p)).mean().item() / 3
                print(f'   Branch JS-div (logit): {js:.5f}  (0=identical, ln2={math.log(2):.3f}=max)')
    except Exception:
        pass

    return [acc, auroc_score, aupr_in, aupr_out, macro_f1, tnr_at_tpr95, dtacc]


