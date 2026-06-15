import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
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
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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

    def query(self, n, current_round=None):
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

    def query(self, n, current_round=None):
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
    def query(self, n, current_round=None):
        unlabeled_idxs = np.where(~self.dataset.labeled_mask)[0]
        logits = self.get_logits(unlabeled_idxs)
        probs = F.softmax(logits, dim=1)
        probs_sorted, _ = probs.sort(descending=True, dim=1)
        # 计算 Margin: 差距越小，模型越分不清前两个类
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        return unlabeled_idxs[uncertainties.sort()[1][:n].numpy()]


class NovelMarginSampling(Strategy):
    """新类簇内 Margin 采样"""

    def query(self, n, current_round=None):
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


class BadgeSampling(Strategy):
    """
    BADGE: 通过梯度空间中的 KMeans++ 同时确保样本的“不确定性”和“多样性”。
    """

    def query(self, n, current_round=None):
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


# ==========================================
# 5. 工厂函数
# ==========================================
def get_strategy(name):
    if name in ['ExpertDisagreementAdaptiveSampling', 'Adaptive']:
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
    elif name == 'NovelAdaptive':
        return NovelMarginSamplingAdaptive
    else:
        raise NotImplementedError(f"策略 {name} 未在 ncd_strategies.py 中注册！")