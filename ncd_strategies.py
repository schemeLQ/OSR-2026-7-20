import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import stats
from sklearn.metrics import pairwise_distances
# ==========================================
# 1. 策略基类 (保持不变)
# ==========================================
class Strategy:
    def __init__(self, dataset, net, args, device):
        self.dataset = dataset
        self.net = net
        self.args = args
        self.device = device

    def _get_clean_loader(self, idxs):
        mean = tuple(getattr(self.args, 'ncd_mean', (0.4914, 0.4822, 0.4465)))
        std = tuple(getattr(self.args, 'ncd_std', (0.2023, 0.1994, 0.2010)))
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        temp_dataset = self.dataset.get_subset_by_idxs(idxs)
        if hasattr(temp_dataset, 'transform'):
            temp_dataset.transform = eval_transform
        elif hasattr(temp_dataset, 'dataset'):
            temp_dataset.dataset.transform = eval_transform
        return DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers)

    def get_logits(self, idxs):
        self.net.eval()
        loader = self._get_clean_loader(idxs) # 🚀 调用新方法
        logits_list = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs = imgs.to(self.device)
                out = self.net(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                logits_list.append(logits.cpu())
        return torch.cat(logits_list) if logits_list else torch.tensor([])

    def get_expert_logits(self, idxs):
        self.net.eval()
        loader = self._get_clean_loader(idxs) # 🚀 调用新方法
        expert_logits_list = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs = imgs.to(self.device)
                expert_logits = self.net.forward_experts(imgs)
                expert_logits_list.append(expert_logits.cpu())
        return torch.cat(expert_logits_list) if expert_logits_list else torch.tensor([])


# ==========================================
# 2. 基础策略 (Entropy & Random)
# ==========================================
class EntropySampling(Strategy):
    def query(self, n, current_round=None, adaptive_round=None):
        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        logits = self.get_logits(unlabeled_idxs)
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1)
        sorted_idxs = entropy.sort(descending=True)[1]
        return unlabeled_idxs[sorted_idxs[:n].numpy()]


class RandomSampling(Strategy):
    def query(self, n, current_round=None, adaptive_round=None):
        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        np.random.shuffle(unlabeled_idxs)
        return unlabeled_idxs[:n]


# ==========================================
# 3. 整合你上传的 Novel 系列策略
# ==========================================

class NovelSamplingRandom(Strategy):
    """从预测为新类的样本中随机采样"""

    def query(self, n, current_round=None, adaptive_round=None, **kwargs):
        num_known = self.args.num_known
        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        logits = self.get_logits(unlabeled_idxs)
        preds = logits.argmax(dim=1)

        # 筛选预测为新类的索引
        novel_idxs = unlabeled_idxs[preds >= num_known]

        if len(novel_idxs) < n:
            return np.random.choice(unlabeled_idxs, n, replace=False)
        return np.random.choice(novel_idxs, n, replace=False)


class NovelEntropySampling(Strategy):
    """新类簇内熵采样"""

    def query(self, n, current_round=None, adaptive_round=None, **kwargs):
        num_known = self.args.num_known
        num_unknown = self.args.num_unknown_est
        num_per_class = int(n / num_unknown) if num_unknown > 0 else n

        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        logits = self.get_logits(unlabeled_idxs)
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-10)
        uncertainties = -(probs * log_probs).sum(dim=1)
        preds = logits.argmax(dim=1)

        # 按熵从大到小排序
        sorted_idxs = uncertainties.sort(descending=True)[1]
        unlabeled_idxs = unlabeled_idxs[sorted_idxs.numpy()]
        preds = preds[sorted_idxs]

        final_idxs_list = []
        for i in range(num_unknown):
            novel_idx_i = unlabeled_idxs[preds == (i + num_known)]
            final_idxs_list.append(novel_idx_i[:min(len(novel_idx_i), num_per_class)])

        final_idxs = np.concatenate(final_idxs_list) if len(final_idxs_list) > 0 else np.array([], dtype=int)

        # 补齐
        if len(final_idxs) < n:
            diff_idxs = np.setdiff1d(unlabeled_idxs, final_idxs)
            final_idxs = np.concatenate([final_idxs, diff_idxs[:(n - len(final_idxs))]])
        return final_idxs[:n]

class MarginSampling(Strategy):
    """
    边际采样：挑选预测概率最大的前两名之间差距 $p_1 - p_2$ 最小的样本。
    """
    def query(self, n, current_round=None, adaptive_round=None, **kwargs):
        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        logits = self.get_logits(unlabeled_idxs)
        probs = F.softmax(logits, dim=1)
        probs_sorted, _ = probs.sort(descending=True, dim=1)
        # 计算 Margin: 差距越小，模型越分不清前两个类
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        return unlabeled_idxs[uncertainties.sort()[1][:n].numpy()]


class NovelMarginSampling(Strategy):
    """新类簇内 Margin 采样"""

    def query(self, n, current_round=None, adaptive_round=None, **kwargs):
        num_known = self.args.num_known
        num_unknown = self.args.num_unknown_est
        num_per_class = int(n / num_unknown) if num_unknown > 0 else n

        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        logits = self.get_logits(unlabeled_idxs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        # 计算 Margin (最小的表示最模糊)
        probs_sorted, _ = probs.sort(descending=True, dim=1)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]

        # 按 Margin 升序排列
        sorted_indices = uncertainties.sort()[1]
        unlabeled_idxs = unlabeled_idxs[sorted_indices.numpy()]
        preds = preds[sorted_indices]

        final_idxs_list = []
        for i in range(num_unknown):
            novel_idx_i = unlabeled_idxs[preds == (i + num_known)]
            final_idxs_list.append(novel_idx_i[:min(len(novel_idx_i), num_per_class)])

        final_idxs = np.concatenate(final_idxs_list) if len(final_idxs_list) > 0 else np.array([], dtype=int)

        if len(final_idxs) < n:
            diff_idxs = np.setdiff1d(unlabeled_idxs, final_idxs)
            final_idxs = np.concatenate([final_idxs, diff_idxs[:(n - len(final_idxs))]])
        return final_idxs[:n]


class NovelMarginSamplingAdaptive(NovelMarginSampling):
    """Early rounds prefer stable samples; later rounds query boundaries."""

    def query(self, n, current_round=0, adaptive_round=2, **kwargs):
        num_known = self.args.num_known
        num_unknown = self.args.num_unknown_est
        num_per_class = max(1, int(n / num_unknown)) if num_unknown > 0 else n

        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        n = min(n, len(unlabeled_idxs))
        if n == 0:
            return np.array([], dtype=int)

        logits = self.get_logits(unlabeled_idxs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        probs_sorted, _ = probs.sort(descending=True, dim=1)
        margins = probs_sorted[:, 0] - probs_sorted[:, 1]

        # Stable pseudo-novel samples bootstrap early rounds. Once queried
        # labels exist, small-margin samples refine cluster boundaries.
        descending = current_round < adaptive_round
        order = margins.sort(descending=descending)[1]
        sorted_pool = unlabeled_idxs[order.numpy()]
        sorted_preds = preds[order]

        selected = []
        for class_id in range(num_known, num_known + num_unknown):
            class_pool = sorted_pool[sorted_preds == class_id]
            selected.append(class_pool[:min(len(class_pool), num_per_class)])

        final_idxs = (
            np.concatenate(selected) if selected else np.array([], dtype=int)
        )
        if len(final_idxs) < n:
            remaining = np.setdiff1d(sorted_pool, final_idxs, assume_unique=False)
            final_idxs = np.concatenate([final_idxs, remaining[:n - len(final_idxs)]])
        return final_idxs[:n]


class BadgeSampling(Strategy):
    """
    BADGE: 通过梯度空间中的 KMeans++ 同时确保样本的“不确定性”和“多样性”。
    """

    def query(self, n, current_round=None, adaptive_round=None, **kwargs):
        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]

        # 1. 计算梯度嵌入 (Gradient Embeddings)
        # 这里模拟损失函数对最后一层参数的梯度
        self.net.eval()
        temp_dataset = self.dataset.get_subset_by_idxs(unlabeled_idxs)
        loader = DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=False)

        grad_embeddings = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs = imgs.to(self.device)
                logits, feats = self.net(imgs)  # 获取投影后的特征和分类 Logits
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                # 核心逻辑：g_i = (probs - 1_y) * feature
                for j in range(len(imgs)):
                    p = probs[j].cpu().numpy()
                    f = feats[j].cpu().numpy()
                    max_p_idx = preds[j].item()

                    # 简化版梯度嵌入：只考虑预测最准的一行梯度以节省内存
                    p[max_p_idx] -= 1.0
                    grad_embeddings.append(p[max_p_idx] * f)

        grad_embeddings = np.array(grad_embeddings)

        # 2. 执行 KMeans++ 初始化挑选中心 (init_centers)
        chosen = self.init_centers(grad_embeddings, n)
        return unlabeled_idxs[chosen]

    def init_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        D2[i] = newD[i]

            if sum(D2) == 0.0: break
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
        return indsAll

# ==========================================
# 4. 专家分歧策略 (DDEUS)
# ==========================================
class ExpertDisagreementAdaptiveSampling(Strategy):
    def query(self, n, current_round=0, adaptive_round=2, **kwargs):
        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        n = min(n, len(unlabeled_idxs))
        if n == 0: return np.array([], dtype=int)

        expert_logits = self.get_expert_logits(unlabeled_idxs)
        expert_probs = F.softmax(expert_logits, dim=-1)
        mean_probs = expert_probs.mean(dim=1)
        preds = mean_probs.max(1)[1]
        variance = torch.var(expert_probs, dim=1)
        disagreement_score = variance.mean(dim=1)

        num_known = self.args.num_known
        novel_preds = preds[preds >= num_known]
        active_novel_clusters = np.unique(novel_preds.numpy())
        num_per_class = int(n / len(active_novel_clusters)) if len(active_novel_clusters) > 0 else n

        # 🚀 显式初始化 final_idxs_list 和 final_idxs
        final_idxs_list = []
        final_idxs = np.array([], dtype=int)

        if len(active_novel_clusters) > 0:
            for cluster_id in active_novel_clusters:
                mask = (preds == cluster_id)
                idx_in_cluster = unlabeled_idxs[mask]
                disag_in_cluster = disagreement_score[mask]

                # 1. 按分歧从小到大排序 (Stable 在前)
                sorted_indices = disag_in_cluster.sort(descending=False)[1]
                sorted_idx_in_cluster = idx_in_cluster[sorted_indices.numpy()]

                n_select = min(len(sorted_idx_in_cluster), num_per_class)

                # 2. 比例计算 (Stable 70%, Informative 30%)
                n_stable = int(n_select * 0.7)
                n_info = n_select - n_stable

                # 3. 选 Stable 部分 (前70%)
                if n_stable > 0:
                    final_idxs_list.append(sorted_idx_in_cluster[:n_stable])
                # 4. 选 Informative 部分 (后30%)
                if n_info > 0:
                    final_idxs_list.append(sorted_idx_in_cluster[-n_info:])

            if len(final_idxs_list) > 0:
                final_idxs = np.concatenate(final_idxs_list)

        # 补齐逻辑：如果选出的样本数不足 n，从剩余样本补齐
        if len(final_idxs) < n:
            diff_idxs = np.setdiff1d(unlabeled_idxs, final_idxs)
            if len(diff_idxs) > 0:
                needed = n - len(final_idxs)
                final_idxs = np.concatenate([final_idxs, diff_idxs[:needed]])

        return final_idxs[:n]


class BoundaryMarginJSSampling(Strategy):
    """Stage-wise stable-boundary expert querying.

    Early AGCD rounds need reliable pseudo-novel anchors to form class centers;
    later rounds need boundary samples to repair fine-grained class separation.
    The stable/boundary ratio is therefore selected from current round metrics
    instead of staying fixed at 70/30.
    """

    @staticmethod
    def _norm01(x):
        x_min, x_max = x.min(), x.max()
        return (x - x_min) / (x_max - x_min + 1e-8)

    def _stage_stable_ratio(self, current_round=0, udr=None, ca=None):
        mode = str(getattr(self.args, 'query_stage_mode', 'metric')).lower()
        fixed_ratio = float(getattr(self.args, 'query_stable_ratio', 0.7))
        early_ratio = float(getattr(self.args, 'query_early_stable_ratio', 0.85))
        mid_ratio = float(getattr(self.args, 'query_mid_stable_ratio', 0.55))
        late_ratio = float(getattr(self.args, 'query_late_stable_ratio', 0.25))
        udr_thr = float(getattr(self.args, 'query_udr_threshold', 97.0))
        ca_thr = float(getattr(self.args, 'query_ca_threshold', 85.0))

        if mode == 'fixed':
            ratio, stage = fixed_ratio, 'fixed'
        elif mode == 'round':
            if current_round <= 0:
                ratio, stage = early_ratio, 'early-anchor'
            elif current_round < int(getattr(self.args, 'query_boundary_start_round', 2)):
                ratio, stage = mid_ratio, 'mixed'
            else:
                ratio, stage = late_ratio, 'boundary-refine'
        else:
            if current_round <= 0:
                ratio, stage = early_ratio, 'early-anchor'
            elif udr is not None and float(udr) < udr_thr:
                ratio, stage = early_ratio, 'discovery-anchor'
            elif ca is not None and float(ca) < ca_thr:
                ratio, stage = mid_ratio, 'structure-mixed'
            else:
                ratio, stage = late_ratio, 'boundary-refine'

        ratio = min(max(float(ratio), 0.0), 1.0)
        return ratio, stage

    @staticmethod
    def _split_counts(n_select, stable_ratio):
        if n_select <= 0:
            return 0, 0
        n_stable = int(round(n_select * stable_ratio))
        if stable_ratio >= 0.5:
            n_stable = max(1, n_stable)
        elif stable_ratio <= 0.0:
            n_stable = 0
        n_stable = min(max(n_stable, 0), n_select)
        return n_stable, n_select - n_stable

    def query(self, n, current_round=0, adaptive_round=2, **kwargs):
        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        n = min(n, len(unlabeled_idxs))
        if n == 0:
            return np.array([], dtype=int)

        expert_logits = self.get_expert_logits(unlabeled_idxs)  # [N, 3, C]
        expert_probs = F.softmax(expert_logits, dim=-1)
        mean_probs = expert_probs.mean(dim=1)
        preds = mean_probs.argmax(dim=1)

        eps = 1e-8
        js = (expert_probs * (
            expert_probs.clamp(min=eps) / mean_probs.unsqueeze(1).clamp(min=eps)
        ).log()).sum(dim=2).mean(dim=1)

        probs_sorted, _ = mean_probs.sort(descending=True, dim=1)
        margin = probs_sorted[:, 0] - probs_sorted[:, 1]
        margin_uncertainty = 1.0 - self._norm01(margin)
        js_score = self._norm01(js)

        js_w = float(getattr(self.args, 'query_js_weight', 0.6))
        margin_w = float(getattr(self.args, 'query_margin_weight', 0.4))
        boundary_score = js_w * js_score + margin_w * margin_uncertainty
        stable_score = js_w * (1.0 - js_score) + margin_w * self._norm01(margin)
        stable_ratio, stage = self._stage_stable_ratio(
            current_round=current_round,
            udr=kwargs.get('udr', None),
            ca=kwargs.get('ca', None),
        )
        print(
            f"   [Stage-wise Query] stage={stage} stable_ratio={stable_ratio:.2f} "
            f"boundary_ratio={1.0 - stable_ratio:.2f} "
            f"UDR={kwargs.get('udr', None)} CA={kwargs.get('ca', None)}"
        )

        num_known = self.args.num_known
        novel_clusters = np.unique(preds[preds >= num_known].numpy())
        if len(novel_clusters) == 0:
            n_stable, _ = self._split_counts(n, stable_ratio)
            stable_order = stable_score.sort(descending=True)[1].numpy()
            boundary_order = boundary_score.sort(descending=True)[1].numpy()
            chosen = list(unlabeled_idxs[stable_order[:n_stable]])
            for idx in unlabeled_idxs[boundary_order]:
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= n:
                    break
            return np.array(chosen[:n], dtype=int)

        num_per_class = max(1, int(np.ceil(n / len(novel_clusters))))
        selected = []
        for cluster_id in novel_clusters:
            mask = preds == cluster_id
            idx_in_cluster = unlabeled_idxs[mask.numpy()]
            stable_in_cluster = stable_score[mask]
            boundary_in_cluster = boundary_score[mask]

            n_select = min(len(idx_in_cluster), num_per_class)
            n_stable, _ = self._split_counts(n_select, stable_ratio)

            stable_order = stable_in_cluster.sort(descending=True)[1].numpy()
            boundary_order = boundary_in_cluster.sort(descending=True)[1].numpy()

            cluster_selected = list(idx_in_cluster[stable_order[:n_stable]])
            for idx in idx_in_cluster[boundary_order]:
                if idx not in cluster_selected:
                    cluster_selected.append(idx)
                if len(cluster_selected) >= n_select:
                    break
            selected.append(np.array(cluster_selected[:n_select], dtype=int))

        final_idxs = np.concatenate(selected) if selected else np.array([], dtype=int)
        if len(final_idxs) < n:
            remaining = np.setdiff1d(unlabeled_idxs, final_idxs, assume_unique=False)
            remaining_positions = np.nonzero(np.isin(unlabeled_idxs, remaining))[0]
            remaining_idxs = unlabeled_idxs[remaining_positions]
            fill_score = stable_score if stable_ratio >= 0.5 else boundary_score
            remaining_score = fill_score[remaining_positions]
            order = remaining_score.sort(descending=True)[1].numpy()
            final_idxs = np.concatenate([final_idxs, remaining_idxs[order[:n - len(final_idxs)]]])
        return final_idxs[:n]


# ==========================================
# 5. 工厂函数
# ==========================================
def get_strategy(name):
    if name in ['BoundaryMarginJSSampling', 'BAEDSampling', 'MarginJS']:
        return BoundaryMarginJSSampling
    elif name in ['ExpertDisagreementAdaptiveSampling', 'Adaptive']:
        return ExpertDisagreementAdaptiveSampling
    elif name == 'EntropySampling':
        return EntropySampling
    elif name == 'MarginSampling':
        return MarginSampling
    elif name == 'BadgeSampling':
        return BadgeSampling
    elif name == 'RandomSampling':
        return RandomSampling
    elif name == 'NovelRandom':
        return NovelSamplingRandom
    elif name == 'NovelEntropy':
        return NovelEntropySampling
    elif name == 'NovelMargin':
        return NovelMarginSampling
    elif name in ['NovelAdaptive', 'NovelMarginSamplingAdaptive']:
        return NovelMarginSamplingAdaptive
    else:
        raise NotImplementedError(f"策略 {name} 未在 ncd_strategies.py 中注册！")
