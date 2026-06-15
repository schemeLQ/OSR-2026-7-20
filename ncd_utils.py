import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

class TwoCropTransform:
    """生成双视图 (View1, View2)"""
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def cluster_acc(y_true, y_pred):
    """计算聚类准确率 (Hungarian Matching)"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

@torch.no_grad()
def sinkhorn_knopp(logits, epsilon=0.05, iterations=3):
    """
    [SimGCD 核心] Sinkhorn-Knopp 归一化
    作用：将 Logits 转换为 Doubly Stochastic Matrix (行和=1, 列和=1)
    效果：强制 Batch 内的预测类别分布均匀，彻底防止模型坍塌。
    """
    # 转换为概率分布 Q [K, B]
    Q = torch.exp(logits / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]

    # 归一化总和
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(iterations):
        # 1. 行归一化：每个类别的总概率 -> 1/K (防止某类独大)
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # 2. 列归一化：每个样本的总概率 -> 1/B (标准概率定义)
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= sum_of_cols
        Q /= B

    Q *= B # 放大回概率尺度
    return Q.t() # [B, K]