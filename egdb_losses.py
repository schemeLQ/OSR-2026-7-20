"""
egdb_losses.py (修复版)

核心修复：
  - weak_ot_targets 必须由调用方从弱视图生成后传入，不能在内部用 strong_logits 自己生成
  - 新增可选的特征一致性损失 (feat_loss_w)：强迫强视图特征向弱视图特征靠拢
  - 保留 hard filter + soft weighting 逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EGDBLoss(nn.Module):
    """
    Expert-Guided Dual-Branch Augmentation Loss (修复版)

    调用方职责：
        weak_ot_targets 必须在训练循环里由弱视图的 logits 生成后传进来。
        示例：
            with torch.no_grad():
                weak_ot_targets = get_hard_transport_targets(u_logits[:bs].detach())
            loss_egdb, stats = criterion_egdb(
                strong_logits=strong_logits,
                weak_ot_targets=weak_ot_targets,
                expert_logits=expert_logits,
                strong_feats=strong_feats,
                weak_feats=u_feats[:bs].detach(),
            )

    Args:
        temperature  : 聚类损失温度，默认 0.1
        weight_floor : 一致样本的权重下限，默认 0.1
        expert_temp  : 专家 softmax 温度（防止方差崩塌为 0），默认 1.5
        feat_loss_w  : 特征一致性损失权重，默认 0.5；设为 0 退化为纯聚类损失
    """

    def __init__(self, temperature=0.1, weight_floor=0.1, expert_temp=1.5, feat_loss_w=0.5):
        super().__init__()
        self.temperature = temperature
        self.weight_floor = weight_floor
        self.expert_temp = expert_temp
        self.feat_loss_w = feat_loss_w

    def forward(self, strong_logits, weak_ot_targets, expert_logits,
                strong_feats=None, weak_feats=None, warmup=False):
        """
        Args:
            strong_logits    : [B, C]    强增强视图的分类 logits（有梯度）
            weak_ot_targets  : [B, C]    弱增强视图的 OT 软标签（已 detach，外部传入）
            expert_logits    : [B, 3, C] 三专家对强增强视图的预测（无梯度）
            strong_feats     : [B, D]    强增强视图特征（可选）
            weak_feats       : [B, D]    弱增强视图特征（可选，已 detach）
            warmup           : bool，True 时返回零损失
        """
        if warmup:
            return strong_logits.sum() * 0.0, {
                'filter_rate': 0.0, 'mean_weight': 0.0,
                'loss_cls': 0.0, 'loss_feat': 0.0,
            }

        device = strong_logits.device

        # 🚀 修复一：获取真实的 Batch Size，用于后面正确的归一化
        B = strong_logits.size(0)

        # ============================================================
        # Step 1: 🚀 修复惨案二 —— 改为多数派投票 (Majority Vote)
        # 只要三个专家中有任意两个达成一致，我们就信任这个伪标签！
        # ============================================================
        expert_probs = F.softmax(expert_logits / self.expert_temp, dim=-1)  # [B, 3, C]
        expert_preds = expert_probs.argmax(dim=-1)  # [B, 3]

        # 分别判断两两之间是否一致
        agree_01 = (expert_preds[:, 0] == expert_preds[:, 1])
        agree_12 = (expert_preds[:, 1] == expert_preds[:, 2])
        agree_02 = (expert_preds[:, 0] == expert_preds[:, 2])

        # 用 "或 (|)" 代替 "与 (&)"，只要有一对达成共识即可
        is_consistent = agree_01 | agree_12 | agree_02  # [B], bool

        aug_weights = is_consistent.float()

        # ============================================================
        # Step 2: 🚀 修复惨案一 —— Batch 均值归一化，消灭梯度爆炸
        # ============================================================
        if aug_weights.sum() < 1e-6:
            return strong_logits.sum() * 0.0, {
                'filter_rate': 0.0, 'mean_weight': 0.0,
                'loss_cls': 0.0, 'loss_feat': 0.0,
            }

        # 绝对不能除以 weight_sum，必须除以全局 Batch Size B！
        # 这样才能保证梯度的量级与标准的 CrossEntropy(reduction='mean') 完全一致
        aug_weights_norm = aug_weights / float(B)  # [B]

        # ============================================================
        # Step 3: 加权聚类损失（Weak → Strong 一致性）
        # ============================================================
        log_probs_strong = F.log_softmax(strong_logits / self.temperature, dim=1)
        per_sample_cls = -torch.sum(weak_ot_targets * log_probs_strong, dim=1)  # [B]
        loss_cls = torch.sum(aug_weights_norm * per_sample_cls)

        # ============================================================
        # Step 4: 特征一致性损失（可选）
        # ============================================================
        loss_feat = torch.tensor(0.0, device=device)
        if self.feat_loss_w > 0 and strong_feats is not None and weak_feats is not None:
            s_norm = F.normalize(strong_feats, dim=1)
            w_norm = F.normalize(weak_feats.detach(), dim=1)
            cosine_sim = (s_norm * w_norm).sum(dim=1)  # [B]
            margin = 0.85
            # 只有当相似度小于 0.85 时，才产生梯度惩罚
            per_sample_feat = F.relu(margin - cosine_sim)
            loss_feat = torch.sum(aug_weights_norm * per_sample_feat)

        loss = loss_cls + self.feat_loss_w * loss_feat

        # ============================================================
        # 统计信息（用于日志）
        # ============================================================
        filter_rate = is_consistent.float().mean().item()
        mean_weight = aug_weights[is_consistent].mean().item() if is_consistent.any() else 0.0

        return loss, {
            'filter_rate': filter_rate,
            'mean_weight': mean_weight,
            'loss_cls': loss_cls.item(),
            'loss_feat': loss_feat.item() if isinstance(loss_feat, torch.Tensor) else 0.0,
        }