"""
ncd_ot.py
=========
查询感知的渐进偏传输模块（QA-P²OT）
Query-Aware Progressive Partial Optimal Transport

核心思想：
  原始 get_hard_transport_targets 的三个问题：
    1. Hungarian 分配每 batch 独立求解，无跨 batch 一致性，每步梯度后剧烈抖动
    2. 先验假设每类样本数量均等（但你的已知类有 30000、新类只有 0-300）
    3. 所有样本强制被分配，低置信度样本的噪声标签直接污染训练

  本模块的改动：
    1. EMA Teacher：用平滑后的 teacher 输出代替瞬时 student logits，稳定 OT 输入
    2. 标注感知先验：q_k ∝ labeled_count_k + α，已知类自然获得更高权重
    3. 虚拟簇偏传输：低置信度样本被吸收到虚拟簇，不产生硬伪标签
    4. Sinkhorn（log-domain）：数值稳定，GPU 上运行，替代 CPU 上的 Hungarian

使用方法（在 ncd_train_agcd.py 里）：
  from ncd_ot import QAP2OTSolver, build_ema_teacher, update_ema_teacher

  # 初始化（在 train_agcd 函数开头，model 构建完之后）
  teacher = build_ema_teacher(model)
  ot_solver = QAP2OTSolver(
      num_known=args.num_known,
      num_novel=args.num_unknown_est,
      ema_decay=0.999,
      eps=0.05,
      rho_known=1.0,   # 已知类：保证全部被分配
      rho_novel=0.85,  # 新类：过滤 15% 低置信度样本
      n_iter=30,
      alpha=1.0        # Laplace 平滑
  )

  # 在每步训练后（optimizer.step() 之后）更新 teacher
  update_ema_teacher(model, teacher, ema_decay=0.999)

  # 在 loss 计算前，用 teacher 和当前标注分布生成伪标签
  ot_targets = ot_solver.get_targets(
      student_logits=u_logits.detach(),
      teacher_model=teacher,
      u_inputs=u_inputs,
      active_dataset=active_dataset,
      device=device,
      round_idx=round_idx
  )
"""

import copy
import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 标签映射（和 ncd_train_agcd.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────
_REMAP = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 9: 5, 3: 6, 6: 7, 7: 8, 8: 9}


def _compute_annotation_prior(active_dataset, all_targets, num_known, num_novel, alpha=1.0, device='cpu'):
    K = num_known + num_novel
    label_counts = torch.zeros(K, device=device)

    labeled_indices = np.where(active_dataset.labeled_mask)[0]

    for idx in labeled_indices:
        raw_t = int(all_targets[idx])
        mapped = _REMAP.get(raw_t, raw_t)
        if 0 <= mapped < K:
            label_counts[mapped] += 1.0

    novel_labeled = label_counts[num_known:].sum().item()
    if novel_labeled == 0:
        # Round 0：新类没有任何标注，退回均匀先验避免新类神经元被饿死
        q = torch.ones(K, device=device) / K
    else:
        q = label_counts + alpha
        q = q / q.sum()
    return q


# ─────────────────────────────────────────────────────────────────────────────
# Sinkhorn（log-domain，数值稳定，GPU 友好）
# ─────────────────────────────────────────────────────────────────────────────
def _sinkhorn_log(C, p, q, eps, n_iter):
    """
    Log-domain Sinkhorn，求解最优传输计划 T。

    最小化：<T, C> - eps * H(T)
    约束：  T @ 1 = p（行边际），T^T @ 1 = q（列边际）

    Args:
        C:      [B, K+1] 代价矩阵（最后一列是虚拟簇代价）
        p:      [B] 样本权重（均匀 = 1/B）
        q:      [K+1] 类先验（最后一个是虚拟簇权重）
        eps:    float，熵正则化强度（越小分配越硬，越大越软）
        n_iter: int，迭代次数

    Returns:
        T: [B, K+1] 传输矩阵
    """
    log_K = -C / eps                             # [B, K+1]
    log_p = torch.log(p + 1e-9)                 # [B]
    log_q = torch.log(q + 1e-9)                 # [K+1]

    log_u = torch.zeros_like(p)                 # [B]
    log_v = torch.zeros_like(q)                 # [K+1]

    for _ in range(n_iter):
        # 更新 v：使列边际满足 q
        log_v = log_q - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        # 更新 u：使行边际满足 p
        log_u = log_p - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

    log_T = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return torch.exp(log_T)


# ─────────────────────────────────────────────────────────────────────────────
# 核心 OT 求解器
# ─────────────────────────────────────────────────────────────────────────────
class QAP2OTSolver:
    """
    查询感知渐进偏传输求解器（QA-P²OT）

    与原始 P²OT 的区别：
      - 先验 q 由当前标注分布动态计算，而非固定均匀分布
      - 已知类和新类使用不同的虚拟簇过滤比例（rho_known / rho_novel）
      - 已知类 rho_known=1.0 → 不过滤（已知类置信度本来就高）
      - 新类 rho_novel=0.85 → 过滤 15% 最低置信度样本（新类伪标签噪声大）
      - 渐进性由主动学习轮次（round_idx）提供，不需要额外的 m 调度
    """

    def __init__(self,
                 num_known,
                 num_novel,
                 ema_decay=0.999,
                 eps=0.05,
                 rho_known=1.0,
                 rho_novel=0.85,
                 n_iter=30,
                 alpha=1.0,
                 c_virtual=2.0,
                 novel_only_unlabeled=True):
        """
        Args:
            num_known (int):   已知类数量
            num_novel (int):   新类估计数量
            ema_decay (float): EMA teacher 更新动量（0.999 适合 NCD，比 SimGCD 的 0.996 更保守）
            eps (float):       Sinkhorn 熵正则化（0.05 在 CIFAR-10 上有良好平衡）
            rho_known (float): 已知类的传输质量比例（1.0 = 不过滤）
            rho_novel (float): 新类的传输质量比例（0.85 = 过滤 15% 低置信）
            n_iter (int):      Sinkhorn 迭代次数（30 次足够 CIFAR-10 收敛）
            alpha (float):     先验 Laplace 平滑（防止新类先验为 0）
            c_virtual (float): 虚拟簇代价（越大 = 越少样本被过滤）
        """
        self.num_known = num_known
        self.num_novel = num_novel
        self.K = num_known + num_novel
        self.ema_decay = ema_decay
        self.eps = eps
        self.rho_known = rho_known
        self.rho_novel = rho_novel
        self.n_iter = n_iter
        self.alpha = alpha
        self.c_virtual = c_virtual
        self.novel_only_unlabeled = novel_only_unlabeled

    @torch.no_grad()
    def get_targets(self, student_logits, teacher_model, u_inputs,
                    active_dataset, all_targets, device, round_idx=0):
        """
        生成 OT 软伪标签目标。

        对应原来代码里的：
            with torch.no_grad():
                ot_targets = get_hard_transport_targets(u_logits.detach())

        Args:
            student_logits: [2B, K] student 模型的 logits（两视图 cat）
            teacher_model:  EMA teacher 模型（已设为 eval，不参与梯度）
            u_inputs:       [2B, C, H, W] 无标签样本的两视图（cat 后）
            active_dataset: ActiveDataset 实例（用于计算先验）
            device:         torch.device
            round_idx (int): 当前主动学习轮次（Round 0 不用过滤新类）

        Returns:
            ot_targets: [2B, K] 软伪标签，dtype=float32，每行和为 1
        """
        teacher_model.eval()

        # ── 1. 用 teacher 生成稳定 logits ────────────────────────────
        teacher_logits, _ = teacher_model(u_inputs)
        teacher_logits = torch.nan_to_num(teacher_logits,
                                          nan=0.0, posinf=1e5, neginf=-1e5)
        B_2, K = teacher_logits.shape  # B_2 = 2 * batch_size

        # ── 2. 标注感知先验 ─────────────────────────────────────────
        q_real = _compute_annotation_prior(
            active_dataset, all_targets, self.num_known, self.num_novel,
            alpha=self.alpha, device=device)  # [K]

        # All known-class training samples are labeled at initialization in
        # this AGCD protocol, so the unlabeled loader contains novel classes.
        # Assigning OT mass to known heads directly suppresses novel clusters.
        if self.novel_only_unlabeled:
            q_real = torch.cat([
                torch.zeros(self.num_known, device=device),
                torch.full(
                    (self.num_novel,),
                    1.0 / max(1, self.num_novel),
                    device=device
                )
            ])

        # ── 3. 构造分类别的虚拟簇先验 ────────────────────────────────
        # Round 0：新类还没有标注，强制 rho_novel=1.0（不过滤）
        # Round 1+：新类开始有标注，允许过滤低置信度新类样本
        rho_novel_eff = 1.0 if round_idx == 0 else self.rho_novel

        # 对已知类和新类分别设置 rho，拼成一个 [K] 的 rho 向量
        rho_per_class = torch.cat([
            torch.full((self.num_known,), self.rho_known, device=device),
            torch.full((self.num_novel,), rho_novel_eff, device=device)
        ])  # [K]

        # 加权先验（rho 越小 → 该类越多样本进虚拟簇）
        # 总真实质量 = sum(rho_k * q_k)，虚拟簇质量 = 1 - 总真实质量
        q_real_weighted = q_real * rho_per_class   # [K]
        total_real_mass = q_real_weighted.sum()
        virtual_mass = 1.0 - total_real_mass

        # 扩展先验：[K+1]（最后一个是虚拟簇）
        q_hat = torch.cat([q_real_weighted,
                            virtual_mass.unsqueeze(0).clamp(min=1e-6)])
        q_hat = q_hat / q_hat.sum()

        # ── 4. 构造代价矩阵（加虚拟簇列）───────────────────────────
        C_real = -teacher_logits                   # [2B, K]
        C_virtual = torch.full((B_2, 1), self.c_virtual, device=device)
        C_hat = torch.cat([C_real, C_virtual], dim=1)  # [2B, K+1]

        # ── 5. Sinkhorn ──────────────────────────────────────────────
        p = torch.ones(B_2, device=device) / B_2  # 均匀样本权重
        T_hat = _sinkhorn_log(C_hat, p, q_hat, self.eps, self.n_iter)
        # T_hat: [2B, K+1]

        # ── 6. 从传输矩阵提取软标签 ─────────────────────────────────
        T_real = T_hat[:, :K]                      # [2B, K]，忽略虚拟簇列

        # 行归一化 → 软概率分布
        row_sum = T_real.sum(dim=1, keepdim=True).clamp(min=1e-9)
        soft_targets = T_real / row_sum            # [2B, K]

        # ── 7. 虚拟簇进入量作为置信度权重：低置信样本向均匀分布后退 ─
        # virtual_weight[i] 越大 = 第 i 个样本越不确定
        # Each row sums to p_i=1/(2B). Convert transported virtual mass into
        # a per-sample rejection probability before mixing the target.
        virtual_weight = (T_hat[:, K] / p).clamp(0.0, 1.0).unsqueeze(1)
        uniform = torch.ones_like(soft_targets) / K
        # 混合：confident 样本用 OT 分配，uncertain 样本用均匀分布
        soft_targets = (1 - virtual_weight) * soft_targets + virtual_weight * uniform

        return soft_targets.float()


# ─────────────────────────────────────────────────────────────────────────────
# EMA Teacher 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def build_ema_teacher(model):
    """
    从 student model 创建 EMA teacher，参数相同但不参与梯度。

    Args:
        model: NCDWrapper 实例

    Returns:
        teacher: 深拷贝的 NCDWrapper，requires_grad=False
    """
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(student, teacher, ema_decay=0.999):
    """
    用 EMA 更新 teacher 参数。应在每步 optimizer.step() 之后调用。

    θ_teacher = decay * θ_teacher + (1 - decay) * θ_student

    Args:
        student:   当前训练中的 student model
        teacher:   EMA teacher model
        ema_decay: 动量参数（0.999 对 10-epoch 轮次合适）
    """
    for param_t, param_s in zip(teacher.parameters(), student.parameters()):
        param_t.data.mul_(ema_decay).add_(param_s.data, alpha=1.0 - ema_decay)


# ─────────────────────────────────────────────────────────────────────────────
# 兼容层（可选）：如果你想保留原来的函数名，用这个 wrapper
# ─────────────────────────────────────────────────────────────────────────────
_global_solver = None
_global_teacher = None


def init_global_solver(model, num_known, num_novel, **kwargs):
    """
    一次性初始化全局 solver 和 teacher（给不想改调用接口的场合用）。
    """
    global _global_solver, _global_teacher
    _global_solver = QAP2OTSolver(num_known=num_known, num_novel=num_novel, **kwargs)
    _global_teacher = build_ema_teacher(model)
    return _global_solver, _global_teacher
