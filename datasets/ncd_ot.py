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


def _compute_annotation_prior(active_dataset, all_targets, num_known, num_novel, alpha=1.0, device='cpu'):
    K = num_known + num_novel
    label_counts = torch.zeros(K, device=device)

    labeled_indices = np.where(active_dataset.labeled_mask)[0]

    for idx in labeled_indices:
        mapped = int(all_targets[idx])
        if 0 <= mapped < K:
            label_counts[mapped] += 1.0

    novel_labeled = label_counts[num_known:].sum().item()
    if novel_labeled == 0:
        # Round 0：新类没有任何标注，退回均匀先验避免新类神经元被饿死
        q = torch.ones(K, device=device) / K
    else:
        q = label_counts + alpha
        q = q / q.sum()
    return q, novel_labeled


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


def _norm01(x, eps=1e-8):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def _as_mbk_expert_logits(expert_logits, batch_size, num_classes):
    """Accept [B, M, K] or [M, B, K], return [M, B, K]."""
    if expert_logits is None or expert_logits.dim() != 3:
        return None
    if expert_logits.shape[-1] != num_classes:
        return None
    if expert_logits.shape[0] == batch_size:
        return expert_logits.permute(1, 0, 2).contiguous()
    if expert_logits.shape[1] == batch_size:
        return expert_logits.contiguous()
    return None


def compute_classwise_disagreement(expert_logits, batch_size, num_classes, temp=1.0):
    """
    Args:
        expert_logits: [B, M, K] or [M, B, K]
    Returns:
        D_class: [B, K], class-wise expert disagreement
        U_norm:  [B], normalized sample uncertainty
        stats:   dict for logging
    """
    expert_logits = _as_mbk_expert_logits(expert_logits, batch_size, num_classes)
    if expert_logits is None:
        device = torch.device('cpu')
        D_class = torch.zeros(batch_size, num_classes, device=device)
        U_norm = torch.zeros(batch_size, device=device)
        return D_class, U_norm, {
            'expert_dis': 0.0,
            'expert_agreement': 1.0,
            'expert_valid': 0.0,
        }

    expert_logits = torch.nan_to_num(expert_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    device = expert_logits.device
    expert_probs = F.softmax(expert_logits / max(temp, 1e-6), dim=-1)  # [M, B, K]
    expert_mean = expert_probs.mean(dim=0)                             # [B, K]
    D_class = ((expert_probs - expert_mean.unsqueeze(0)) ** 2).mean(dim=0)
    U = D_class.mean(dim=1)
    U_norm = _norm01(U)
    sample_agreement = 1.0 - U_norm
    return D_class, U_norm, {
        'expert_dis': U.mean().detach().item(),
        'expert_agreement': sample_agreement.mean().detach().item(),
        'expert_valid': 1.0,
    }


def update_adaptive_rho(
    epoch,
    max_epoch,
    expert_logits,
    prev_rho,
    rho_min=0.2,
    rho_max=0.95,
    beta=0.1,
    gamma=0.7,
    temp=1.0,
):
    """
    Expert-consistency-aware rho update.
    Returns:
        rho_new: scalar tensor
        stats: dict with rho_time / agreement / disagreement
    """
    if expert_logits is not None:
        device = expert_logits.device
    elif torch.is_tensor(prev_rho):
        device = prev_rho.device
    else:
        device = torch.device('cpu')

    t = max(0.0, min(1.0, float(epoch) / max(1.0, float(max_epoch))))
    ramp = torch.exp(torch.tensor(-5.0 * (1.0 - t) ** 2, device=device))
    rho_time = rho_min + (rho_max - rho_min) * ramp

    if expert_logits is None:
        mean_agreement = torch.tensor(1.0, device=device)
        mean_disagreement = torch.tensor(0.0, device=device)
    else:
        expert_logits = torch.nan_to_num(expert_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        expert_probs = F.softmax(expert_logits / max(temp, 1e-6), dim=-1)
        expert_mean = expert_probs.mean(dim=0)
        sample_dis = ((expert_probs - expert_mean.unsqueeze(0)) ** 2).mean(dim=(0, 2))
        sample_dis_norm = _norm01(sample_dis)
        sample_agreement = 1.0 - sample_dis_norm
        mean_agreement = sample_agreement.mean().detach()
        mean_disagreement = sample_dis.mean().detach()

    reliability_factor = gamma + (1.0 - gamma) * mean_agreement
    rho_target = rho_time * reliability_factor
    prev_rho_t = torch.as_tensor(prev_rho, device=device, dtype=rho_target.dtype)
    rho_new = (1.0 - beta) * prev_rho_t + beta * rho_target
    rho_new = torch.clamp(rho_new, min=rho_min, max=rho_max)
    return rho_new.detach(), {
        'rho_time': rho_time.detach().item(),
        'rho_target': rho_target.detach().item(),
        'expert_dis': mean_disagreement.detach().item(),
        'expert_agreement': mean_agreement.detach().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 核心 OT 求解器
# ─────────────────────────────────────────────────────────────────────────────
class QAP2OTSolver:
    """
    查询感知渐进偏传输求解器（QA-P²OT）

    与原始 P²OT 的区别：
      - 先验 q 由当前标注分布动态计算，而非固定均匀分布
      - 查询先验会与均匀先验混合，避免主动采样偏差饿死少标注新类
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
                 query_prior_weight=0.6,
                 prior_floor=0.5,
                 ot_temp=1.0,
                 lambda_dis=0.0,
                 lambda_u=0.0,
                 use_adaptive_rho=False,
                 rho_min=0.2,
                 rho_max=0.95,
                 rho_beta=0.1,
                 rho_gamma=0.7,
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
            query_prior_weight (float): 查询分布最大混合权重；会随 round 逐步升高
            prior_floor (float): 每个新类至少保留 prior_floor/num_novel 的容量
            ot_temp (float):     teacher/expert logits 转概率时的 temperature
            lambda_dis (float):  class-wise expert disagreement 的真实类代价权重
            lambda_u (float):    样本不确定性降低虚拟簇代价的权重
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
        self.query_prior_weight = query_prior_weight
        self.prior_floor = prior_floor
        self.ot_temp = ot_temp
        self.lambda_dis = lambda_dis
        self.lambda_u = lambda_u
        self.use_adaptive_rho = use_adaptive_rho
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_beta = rho_beta
        self.rho_gamma = rho_gamma
        self.rho_current = torch.tensor(rho_novel)
        self.last_stats = {}
        self.novel_only_unlabeled = novel_only_unlabeled

    @torch.no_grad()
    def get_targets(self, student_logits, teacher_model, u_inputs,
                    active_dataset, all_targets, device, round_idx=0,
                    epoch=0, max_epoch=1, expert_logits=None,
                    return_reliability=False, return_stats=False):
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
        if expert_logits is not None:
            expert_logits = expert_logits.to(device)

        # ── 2. 标注感知先验 ─────────────────────────────────────────
        q_real, novel_labeled = _compute_annotation_prior(
            active_dataset, all_targets, self.num_known, self.num_novel,
            alpha=self.alpha, device=device)  # [K]

        # All known-class training samples are labeled at initialization in
        # this AGCD protocol, so the unlabeled loader contains novel classes.
        # Assigning OT mass to known heads directly suppresses novel clusters.
        # If queried novel labels exist, keep their empirical/smoothed prior
        # over novel heads; otherwise fall back to a uniform novel prior.
        if self.novel_only_unlabeled:
            uniform_prior = torch.full(
                (self.num_novel,),
                1.0 / max(1, self.num_novel),
                device=device
            )
            if novel_labeled > 0:
                empirical_prior = q_real[self.num_known:].clamp_min(0)
                empirical_prior = empirical_prior / empirical_prior.sum().clamp_min(1e-9)

                # Active queries are intentionally biased toward informative
                # samples, so their class histogram is not a reliable dataset
                # prior. Warm up the query prior and keep a capacity floor for
                # every novel head to preserve unknown-class coverage.
                warmup = min(1.0, max(0, round_idx) / 3.0)
                query_w = self.query_prior_weight * warmup
                novel_prior = (1.0 - query_w) * uniform_prior + query_w * empirical_prior
                min_mass = self.prior_floor / max(1, self.num_novel)
                novel_prior = novel_prior.clamp_min(min_mass)
                novel_prior = novel_prior / novel_prior.sum().clamp_min(1e-9)
            else:
                novel_prior = uniform_prior
            q_real = torch.cat([
                torch.zeros(self.num_known, device=device),
                novel_prior
            ])

        # ── 3. 构造分类别的虚拟簇先验 ────────────────────────────────
        # Round 0：新类还没有标注，强制 rho_novel=1.0（不过滤）
        # Round 1+：新类开始有标注，允许过滤低置信度新类样本
        expert_logits_mbk = _as_mbk_expert_logits(expert_logits, B_2, K)
        rho_stats = {}
        if self.use_adaptive_rho and round_idx > 0:
            self.rho_current = self.rho_current.to(device)
            rho_new, rho_stats = update_adaptive_rho(
                epoch=epoch,
                max_epoch=max_epoch,
                expert_logits=expert_logits_mbk,
                prev_rho=self.rho_current,
                rho_min=self.rho_min,
                rho_max=self.rho_max,
                beta=self.rho_beta,
                gamma=self.rho_gamma,
                temp=self.ot_temp,
            )
            self.rho_current = rho_new.to(device)
            rho_novel_eff = self.rho_current
        else:
            rho_novel_eff = torch.tensor(
                1.0 if round_idx == 0 else self.rho_novel,
                device=device
            )

        # 对已知类和新类分别设置 rho，拼成一个 [K] 的 rho 向量
        rho_per_class = torch.cat([
            torch.full((self.num_known,), self.rho_known, device=device),
            torch.full((self.num_novel,), float(rho_novel_eff.detach().item()), device=device)
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
        log_p_teacher = F.log_softmax(teacher_logits / max(self.ot_temp, 1e-6), dim=1)
        C_real = -log_p_teacher                    # [2B, K]
        D_class, U, dis_stats = compute_classwise_disagreement(
            expert_logits_mbk, B_2, K, temp=self.ot_temp
        )
        D_class = D_class.to(device)
        U = U.to(device)
        if self.lambda_dis > 0:
            C_real = C_real + self.lambda_dis * D_class
        C_virtual = self.c_virtual - self.lambda_u * U.unsqueeze(1)
        C_virtual = C_virtual.clamp_min(1e-6)
        C_hat = torch.cat([C_real, C_virtual], dim=1)  # [2B, K+1]
        C_hat = torch.nan_to_num(C_hat, nan=0.0, posinf=1e4, neginf=-1e4)

        # ── 5. Sinkhorn ──────────────────────────────────────────────
        p = torch.ones(B_2, device=device) / B_2  # 均匀样本权重
        T_hat = _sinkhorn_log(C_hat, p, q_hat, self.eps, self.n_iter)
        # T_hat: [2B, K+1]

        # ── 6. 从传输矩阵提取软标签 ─────────────────────────────────
        T_real = T_hat[:, :K]                      # [2B, K]，忽略虚拟簇列
        T_virtual = T_hat[:, K]                    # [2B]
        real_mass = T_real.sum(dim=1)              # [2B]

        # 行归一化 → 只在真实类别内归一化为 soft target
        soft_targets = T_real / real_mass.unsqueeze(1).clamp_min(1e-8)

        # ── 7. 真实类质量作为可靠性权重 ───────────────────────────
        reliability = (real_mass / p).clamp(0.0, 1.0)  # [2B]

        self.last_stats = {
            'rho_current': float(rho_novel_eff.detach().item()),
            'rho_time': rho_stats.get('rho_time', float(rho_novel_eff.detach().item())),
            'rho_target': rho_stats.get('rho_target', float(rho_novel_eff.detach().item())),
            'expert_dis': dis_stats['expert_dis'],
            'expert_agreement': dis_stats['expert_agreement'],
            'expert_valid': dis_stats['expert_valid'],
            'real_mass': real_mass.mean().detach().item(),
            'virtual_mass': T_virtual.mean().detach().item(),
            'weight_mean': reliability.mean().detach().item(),
            'weight_min': reliability.min().detach().item(),
            'weight_max': reliability.max().detach().item(),
        }

        if return_stats:
            return soft_targets.float().detach(), reliability.float().detach(), dict(self.last_stats)

        if return_reliability:
            return soft_targets.float().detach(), reliability.float().detach()

        # Backward-compatible fallback: callers that do not consume reliability
        # still receive softened labels instead of hard noisy targets.
        uniform = torch.ones_like(soft_targets) / K
        soft_targets = reliability.unsqueeze(1) * soft_targets + (1 - reliability.unsqueeze(1)) * uniform
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
