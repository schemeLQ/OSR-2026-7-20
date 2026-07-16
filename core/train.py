import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from misc.util import *


def get_dynamic_weight(current_epoch, max_epoch, target_weight=1.0, warmup_epochs=0, mode='sigmoid'):
    """Schedule the attention-diversity weight.

    Tiny-ImageNet is sensitive to early expert dispersion, so the default
    schedule can keep diversity off for a warmup period and then ramp it up.
    """
    if target_weight <= 0:
        return 0.0
    warmup_epochs = max(0, int(warmup_epochs))
    if current_epoch <= warmup_epochs:
        return 0.0
    active_epochs = max(1, int(max_epoch) - warmup_epochs)
    progress = min(max((current_epoch - warmup_epochs) / active_epochs, 0.0), 1.0)
    if mode == 'linear':
        scale = progress
    else:
        # Smoothly starts near 0 after warmup and approaches 1 near the end.
        scale = 1.0 / (1.0 + math.exp(-10.0 * (progress - 0.5)))
    return float(target_weight) * scale


def branch_feat_decorr_loss(feat_vecs):
    # Fix3: 特征空间分支去相关 — feat_vecs: [B, 3, D]
    f = F.normalize(feat_vecs, dim=-1)
    sim12 = (f[:, 0] * f[:, 1]).sum(dim=1).mean()
    sim13 = (f[:, 0] * f[:, 2]).sum(dim=1).mean()
    sim23 = (f[:, 1] * f[:, 2]).sum(dim=1).mean()
    return (sim12 + sim13 + sim23) / 3


def cross_branch_dict_orth_loss(bacl1, bacl2, bacl3):
    """CBDO: 强制三个分支的混杂因子字典互相正交，确保各分支捕获不同的混杂因子。"""
    d1 = F.normalize(bacl1.confounder_dict, p=2, dim=1)  # [K, D]
    d2 = F.normalize(bacl2.confounder_dict, p=2, dim=1)
    d3 = F.normalize(bacl3.confounder_dict, p=2, dim=1)
    l12 = torch.mm(d1, d2.t()).pow(2).mean()
    l13 = torch.mm(d1, d3.t()).pow(2).mean()
    l23 = torch.mm(d2, d3.t()).pow(2).mean()
    return (l12 + l13 + l23) / 3


def inter_branch_center_align_loss(odl1, odl2, odl3):
    """PB-ODL-ISA: 对齐跨分支同类中心方向，使各分支在语义上一致。
    最大化每对分支对应类中心的余弦相似度 → 语义结构跨分支对齐。
    同时 CBDO 保证各分支的 BACL 字典正交 → 处理过程多样。
    结果: diverse process, aligned semantics — 对 GCD downstream 友好。
    """
    c1 = F.normalize(odl1.centers, p=2, dim=1)  # [K, D]
    c2 = F.normalize(odl2.centers, p=2, dim=1)
    c3 = F.normalize(odl3.centers, p=2, dim=1)
    align_12 = (c1 * c2).sum(dim=1).mean()
    align_13 = (c1 * c3).sum(dim=1).mean()
    align_23 = (c2 * c3).sum(dim=1).mean()
    return -(align_12 + align_13 + align_23) / 3  # minimize = maximize alignment


def attnDiv(cams):
    bs = cams.shape[0]
    num_part = cams.shape[1]
    cams = cams.view(bs, num_part, -1)          # [B, 3, H*W]
    cams = F.normalize(cams, p=2, dim=-1)        # unit-norm per spatial map
    mean = cams.mean(dim=-1, keepdim=True).expand_as(cams)
    cams = F.relu(cams - mean)                   # highlight above-average regions

    orthogonal_loss = torch.tensor(0.0, device=cams.device)
    cnt = 0
    for i in range(num_part):
        for j in range(i + 1, num_part):
            # cosine similarity of full spatial vectors: result is [B], then mean → scalar
            orthogonal_loss += F.cosine_similarity(cams[:, i, :], cams[:, j, :], dim=1).mean()
            cnt += 1
    if cnt == 0:
        return orthogonal_loss
    return orthogonal_loss / cnt


def branch_js_divergence(logits):
    """Per-sample Jensen-Shannon divergence among the three expert branches."""
    probs = [F.softmax(logits[i].float(), dim=1) for i in range(3)]
    mean_prob = (probs[0] + probs[1] + probs[2]) / 3
    eps = 1e-8
    js = sum((p * (p.clamp(min=eps) / mean_prob.clamp(min=eps)).log()).sum(dim=1)
             for p in probs) / 3
    return js


def baed_loss(logits, tau=1.0):
    """Boundary-Aware Expert Diversification.

    The gate prediction is used as a semantic anchor for all branches, while
    low-margin samples receive stronger expert-divergence pressure.
    """
    gate_logits = logits[3].float()
    gate_prob = F.softmax(gate_logits.detach(), dim=1)

    anchor = 0.0
    for bi in range(3):
        log_pb = F.log_softmax(logits[bi].float(), dim=1)
        anchor = anchor + F.kl_div(log_pb, gate_prob, reduction='none').sum(dim=1).mean()
    anchor = anchor / 3

    top2 = torch.topk(gate_logits.detach(), k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    boundary_weight = torch.exp(-margin / max(float(tau), 1e-6)).detach()
    js = branch_js_divergence(logits)
    div = -((boundary_weight * js).sum() / boundary_weight.sum().clamp_min(1e-6))
    return anchor, div


def train(train_loader, model, criterion, optimizer, args):
    model.train()
    device = next(model.parameters()).device

    loss_keys = args['loss_keys']
    acc_keys  = args['acc_keys']
    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter  = {p: AverageMeter() for p in acc_keys}
    time_start = time.time()

    current_epoch    = args.get('current_epoch', 1)
    max_epoch        = args.get('epoch_num', 200)
    target_div_w     = args['loss_wgts'][2]
    current_div_w    = get_dynamic_weight(current_epoch, max_epoch, target_weight=target_div_w,
                                          warmup_epochs=args.get('div_warmup_epochs', 0),
                                          mode=args.get('div_schedule', 'sigmoid'))

    use_bacl        = args.get('use_bacl', False)
    odl_start_epoch = int(args.get('odl_start_epoch', 1))
    odl_datasets = args.get('odl_datasets', None)
    dataset_allows_odl = True
    if odl_datasets:
        dataset_allows_odl = str(args.get('dataset', '')).lower() in {
            str(name).lower() for name in odl_datasets
        }
    use_odl         = args.get('use_odl',  False) and dataset_allows_odl and current_epoch >= odl_start_epoch
    use_branch_decorr = args.get('use_branch_decorr', False)
    decorr_weight   = args.get('decorr_weight', 0.1)

    # Fix4: gate_temp 余弦退火 (gate_temp → gate_temp_end)
    gate_temp_start = args.get('gate_temp', 1.0)
    gate_temp_end   = args.get('gate_temp_end', gate_temp_start)
    if gate_temp_start != gate_temp_end:
        progress = current_epoch / max_epoch
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        current_gate_temp = gate_temp_end + (gate_temp_start - gate_temp_end) * cos_factor
        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        m.gate_temp = current_gate_temp
    else:
        current_gate_temp = gate_temp_start

    use_cbdo            = args.get('use_cbdo', False)
    cbdo_weight         = args.get('cbdo_weight', 0.1)
    use_pb_odl          = args.get('use_pb_odl', False)
    # center_align only makes sense when separate ODL instances exist (not with SC-PB-ODL)
    use_center_align    = use_odl and use_pb_odl and args.get('use_center_align', False)
    center_align_weight = args.get('center_align_weight', 0.05)
    use_baed            = args.get('use_baed', False) and current_epoch >= args.get('baed_start_epoch', 1)
    baed_anchor_weight  = args.get('baed_anchor_weight', 0.03)
    baed_div_weight     = args.get('baed_div_weight', 0.03)
    baed_tau            = args.get('baed_tau', 1.0)

    use_se = args.get('use_se', True)

    if args.get('last_printed_epoch', -1) != current_epoch:
        print(f">>> [Epoch {current_epoch}/{max_epoch}] "
              f"div_w={current_div_w:.4f}  gate_T={current_gate_temp:.2f}  "
              f"se={'ON' if use_se else 'OFF'}  "
              f"bacl={'ON' if use_bacl else 'OFF'}  "
              f"odl={'ON' if use_odl else 'OFF'}  "
              f"pb_odl={'ON' if use_pb_odl else 'OFF'}  "
              f"cbdo={'ON' if use_cbdo else 'OFF'}  "
              f"c_align={'ON' if use_center_align else 'OFF'}  "
              f"baed={'ON' if use_baed else 'OFF'}")
        args['last_printed_epoch'] = current_epoch

    zero = torch.tensor(0.0, device=device)

    for batch_idx, data in enumerate(train_loader):
        inputs = data[0].to(device)
        target = data[1].to(device)

        m_for_odl = model.module if isinstance(model, torch.nn.DataParallel) else model
        old_model_use_odl = getattr(m_for_odl, 'use_odl', False)
        if old_model_use_odl != use_odl:
            m_for_odl.use_odl = use_odl
        output_dict  = model(inputs, target)
        if old_model_use_odl != use_odl:
            m_for_odl.use_odl = old_model_use_odl
        logits       = output_dict['logits']   # [b1, b2, b3, gate]
        branch_cams  = output_dict['cams']

        # 1. CE losses (one per branch + gate)
        loss_values = [criterion['entropy'](lg.float(), target.long()) for lg in logits]
        # indices: 0=b1_ce  1=b2_ce  2=b3_ce  3=gate_ce

        # 2. AttnDiv spatial diversity loss
        loss_div = attnDiv(branch_cams)
        loss_values.append(loss_div)
        # index 4=div_loss

        # 3. ODL (optional, Phase1+ replaces with RPL)
        loss_odl = output_dict.get('loss_odl', zero) if use_odl else zero

        # 4. BACL losses (orth + CBDO) — get model ref once
        if use_bacl:
            m = model.module if isinstance(model, torch.nn.DataParallel) else model
            cls_w = m.branch1_cls.weight.view(m.branch1_cls.weight.size(0), -1)
            loss_orth = sum(b.get_orthogonality_loss(cls_w.detach())
                            for b in [m.bacl1, m.bacl2, m.bacl3])
            # [A] 跨分支字典正交损失 (CBDO)
            loss_cbdo = cross_branch_dict_orth_loss(m.bacl1, m.bacl2, m.bacl3) if use_cbdo else zero
        else:
            loss_orth = zero
            loss_cbdo = zero

        # 5. (legacy) 特征空间去相关损失
        if use_branch_decorr and 'feat_vecs' in output_dict:
            loss_decorr = branch_feat_decorr_loss(output_dict['feat_vecs'])
        else:
            loss_decorr = zero

        # 6. [B] 跨分支类中心对齐损失 (Inter-Branch Center Alignment, PB-ODL-ISA)
        if use_center_align:
            _m = model.module if isinstance(model, torch.nn.DataParallel) else model
            if hasattr(_m, 'odl_loss2') and hasattr(_m, 'odl_loss3'):
                loss_center_align = inter_branch_center_align_loss(
                    _m.odl_loss1, _m.odl_loss2, _m.odl_loss3)
            else:
                loss_center_align = zero
        else:
            loss_center_align = zero

        # 7. BAED: keep branch semantics anchored to the fused gate prediction,
        # while encouraging disagreement only around low-margin boundary samples.
        if use_baed:
            loss_baed_anchor, loss_baed_div = baed_loss(logits, tau=baed_tau)
        else:
            loss_baed_anchor, loss_baed_div = zero, zero

        # 8. Total loss
        total_loss = (
            args['loss_wgts'][0] * (loss_values[0] + loss_values[1] + loss_values[2]) +
            args['loss_wgts'][1] * loss_values[3] +
            current_div_w        * loss_values[4] +
            args.get('odl_weight',  0.1) * loss_odl          +
            args.get('orth_weight', 0.1) * loss_orth         +
            decorr_weight               * loss_decorr        +
            cbdo_weight                 * loss_cbdo          +
            center_align_weight         * loss_center_align  +
            baed_anchor_weight          * loss_baed_anchor   +
            baed_div_weight             * loss_baed_div
        )

        loss_values.append(loss_odl)           # index 5  → 'odl'
        loss_values.append(loss_orth)          # index 6  → 'orth'
        loss_values.append(loss_decorr)        # index 7  → 'decorr'
        loss_values.append(loss_cbdo)          # index 8  → 'cbdo'
        loss_values.append(loss_center_align)  # index 9  → 'center_align'
        loss_values.append(loss_baed_anchor)   # index 10 → 'baed_anchor'
        loss_values.append(loss_baed_div)      # index 11 → 'baed_div'
        loss_values.append(total_loss)         # index 12 → 'total'

        # Logging
        multi_loss = {loss_keys[k]: loss_values[k].item() for k in range(len(loss_keys))}
        acc_values = [accuracy(lg, target, topk=(1,))[0] for lg in logits]
        train_accs = {acc_keys[k]: acc_values[k].item() for k in range(len(acc_keys))}
        update_meter(loss_meter, multi_loss, inputs.size(0))
        update_meter(acc_meter,  train_accs, inputs.size(0))

        # Backprop (no gradient projection — removed as no-op)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if args.get('max_train_batches') is not None and (batch_idx + 1) >= int(args.get('max_train_batches')):
            break

    elapsed = time.time() - time_start
    loss_str = "< Loss > "
    for k, v in loss_meter.items():
        if k in ('total', 'odl', 'orth', 'divAttn', 'cbdo', 'center_align', 'baed_anchor', 'baed_div'):
            loss_str += f"{k}:{v.value:.4f} "
    acc_str = " | accGate:"
    if 'accGate' in acc_meter:
        acc_str += f"{acc_meter['accGate'].value:.1f}"
    print(loss_str + acc_str + f"  t:{elapsed:.1f}s")

    return loss_meter[loss_keys[-1]].value


