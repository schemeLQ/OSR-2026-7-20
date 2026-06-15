import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
from misc.util import *


# ==========================================
# 1. 辅助函数
# ==========================================
def get_dynamic_weight(current_epoch, max_epoch, target_weight=1.0):
    """
    [动态权重] Sigmoid 升温策略
    """
    if max_epoch == 0: return target_weight
    progress = current_epoch / max_epoch
    scale = 1 / (1 + math.exp(-10 * (progress - 0.5)))
    return target_weight * scale


def attnDiv(cams):
    """
    [空间差异 Loss] 强迫分支关注不同区域
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    orthogonal_loss = 0
    bs = cams.shape[0]
    num_part = cams.shape[1]

    cams = cams.view(bs, num_part, -1)
    cams = F.normalize(cams, p=2, dim=-1)
    mean = cams.mean(dim=-1).view(bs, num_part, -1).expand(size=[bs, num_part, cams.shape[-1]])
    cams = F.relu(cams - mean)

    cnt = 0
    for i in range(cams.shape[1]):
        for j in range(i + 1, cams.shape[1]):
            orthogonal_loss += cos(cams[:, i, :].view(bs, 1, -1), cams[:, j, :].view(bs, 1, -1)).mean()
            cnt += 1

    if cnt == 0: return torch.tensor(0.0).to(cams.device)
    return orthogonal_loss / cnt


# ==========================================
# 2. 核心训练函数
# ==========================================
def train(train_loader, model, criterion, optimizer, args):
    model.train()

    loss_keys = args['loss_keys']
    acc_keys = args['acc_keys']
    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter = {p: AverageMeter() for p in acc_keys}
    time_start = time.time()

    # 获取动态权重
    current_epoch = args.get('current_epoch', 1)
    max_epoch = args.get('epoch_num', 150)
    target_div_weight = args['loss_wgts'][2]
    current_div_weight = get_dynamic_weight(current_epoch, max_epoch, target_weight=target_div_weight)

    # 打印权重日志 (仅一次)
    if getattr(args, 'last_printed_epoch', -1) != current_epoch:
        # 改个名字，表明现在是 Static Dictionary 模式
        print(
            f">>> [Epoch {current_epoch}] Static Dictionary Mode. Dynamic Div-Weight: {current_div_weight:.6f}")
        args['last_printed_epoch'] = current_epoch

    for i, data in enumerate(train_loader):
        inputs = data[0].cuda()
        target = data[1].cuda()

        output_dict = model(inputs, target)
        loss_odl = output_dict['loss_odl']
        logits = output_dict['logits']
        branch_cams = output_dict['cams']
        # feat_vecs = output_dict['feat_vecs'] # 训练时暂时用不到

        # 1. 基础分类 Loss
        loss_values = [criterion['entropy'](logit.float(), target.long()) for logit in logits]

        # 2. AttnDiv Loss (使用动态权重)
        loss_div = attnDiv(branch_cams)
        loss_values.append(loss_div)

        # 3. BACL Orth Loss (正交 Loss)
        if isinstance(model, torch.nn.DataParallel):
            w = model.module.branch1_cls.weight
            bacl_modules = [model.module.bacl1, model.module.bacl2, model.module.bacl3]
        else:
            w = model.branch1_cls.weight
            bacl_modules = [model.bacl1, model.bacl2, model.bacl3]

        cls_weight = w.view(w.size(0), -1)

        loss_orth = 0
        for bacl in bacl_modules:
            # 🚨🚨🚨 [关键修复] 加上 .detach() 🚨🚨🚨
            # 切断分类器的梯度，只让字典去适应分类器
            loss_orth += bacl.get_orthogonality_loss(cls_weight.detach())

        # 4. 组合 Total Loss
        total_loss = args['loss_wgts'][0] * sum(loss_values[:3]) + \
                     args['loss_wgts'][1] * loss_values[-2] + \
                     current_div_weight * loss_values[-1] + \
                     0.1 * loss_odl + \
                     0.1 * loss_orth

        loss_values.append(loss_odl)
        loss_values.append(loss_orth)
        loss_values.append(total_loss)

        # 记录
        multi_loss = {loss_keys[k]: loss_values[k].item() for k in range(len(loss_keys))}
        acc_values = [accuracy(logit, target, topk=(1,))[0] for logit in logits]
        train_accs = {acc_keys[k]: acc_values[k].item() for k in range(len(acc_keys))}

        update_meter(loss_meter, multi_loss, inputs.size(0))
        update_meter(acc_meter, train_accs, inputs.size(0))

        optimizer.zero_grad()
        loss_orth.backward(retain_graph=True)

        # 记录下所有 extractor 层的混杂梯度向量
        confounder_grads = {}
        for name, param in model.named_parameters():
            if 'extractor' in name and param.grad is not None:
                confounder_grads[name] = param.grad.clone()

        # 2. 计算分类主梯度 (Task Gradient)
        optimizer.zero_grad()
        total_loss.backward()  # 包含分类损失和 odl 等

        # 3. 执行梯度正交投影 (Intervention)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in confounder_grads:
                    g_t = param.grad  # 主任务梯度
                    g_c = confounder_grads[name]  # 环境混杂梯度

                    # 计算投影：从主梯度中减去在混杂方向上的分量
                    dot_product = torch.sum(g_t * g_c)
                    norm_sq = torch.sum(g_c * g_c) + 1e-8

                    # 核心干预公式：强迫 Backbone 学习与环境无关的特征
                    param.grad -= (dot_product / norm_sq) * g_c

        # 4. 更新参数
        optimizer.step()

    time_eclapse = time.time() - time_start
    tmp_str = "< Training Loss >\n"
    # 只打印关键 Loss
    for k, v in loss_meter.items():
        if k in ['total', 'odl', 'orth']:
            tmp_str += f"{k}:{v.value:.4f} "

    tmp_str += "\n< Training Accuracy >\n"
    for k, v in acc_meter.items():
        if k == 'accGate':
            tmp_str += f"{k}:{v.value:.1f} "

    print(tmp_str + f"t:{time_eclapse:.1f}s")

    return loss_meter[loss_keys[-1]].value