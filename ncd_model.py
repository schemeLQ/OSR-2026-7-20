import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import os
import sys
import copy

try:
    from core.net import MultiBranchNet
except ImportError:
    try:
        sys.path.append(os.path.join(os.getcwd(), 'core'))
        from core.net import MultiBranchNet
    except ImportError:
        try:
            from net import MultiBranchNet
        except ImportError:
            raise ImportError("Error: Cannot find core/net.py")


class Adapter(nn.Module):
    # 简单的非线性映射，负责“修复”特征
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers=3, hidden_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.last_layer = weight_norm(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.last_layer(x)
        return x


class NCDWrapper(nn.Module):
    def __init__(self, backbone, adapter, dino_head):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.dino_head = dino_head
        self.adapter_strong = copy.deepcopy(adapter)
        # 🚀 物理隔离塔
        self.adapter_strong = copy.deepcopy(adapter)
        self.dino_head_strong = DINOHead(in_dim=512, out_dim=dino_head.last_layer.weight.shape[0])
        self.dino_head_strong.load_state_dict(dino_head.state_dict())
        # 冻结所有共享层和低层分支
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 仅解冻每个分支的最后一层 (Layer 5) 和相关的 BACL 门控层
        # 这样模型既能保留 OSR 阶段的“去偏”骨架，又能微调“语义”
            # 🟢 [半冻结逻辑] 安全解冻模式
            # 我们希望解冻每个分支的最后一层以及相关的门控/分类层，但不同网络结构的命名可能不同
            # 这里使用 hasattr 进行安全检查，有的层就解冻，没有的就跳过
            # 🟢 [半冻结逻辑] 安全解冻模式 (增加 Layer 4 缓解灾难性遗忘)
            modules_to_unfreeze = [
                'branch1_l5', 'branch2_l5', 'branch3_l5',  # 分支最后一层
                'branch1_l4', 'branch2_l4', 'branch3_l4',  # 🚨 新增：解冻倒数第二层，提供更大容量
                'gate_l5', 'gate',
                'bacl1', 'bacl2', 'bacl3',
                'classifier1', 'classifier2', 'classifier3'
            ]

            for mod_name in modules_to_unfreeze:
                if hasattr(self.backbone, mod_name):
                    module = getattr(self.backbone, mod_name)
                    for param in module.parameters():
                        param.requires_grad = True

    def _load_osr_weights(self, path):
        print(f"🔄 Loading weights: {path}")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint.get('net', checkpoint.get('state_dict', checkpoint))
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(new_state_dict, strict=False)
            print("✅ Weights loaded.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")

    def forward(self, x, use_strong_head=False):
        outputs = self.backbone(x)

        # 1. 统一提取特征，适配 MultiBranchNet 输出
        if isinstance(outputs, dict) and 'feat_vecs' in outputs and 'gate_pred' in outputs:
            gate = outputs['gate_pred'].unsqueeze(-1)
            feats = (outputs['feat_vecs'] * gate).sum(dim=1)
        else:
            feats = outputs['fts'] if isinstance(outputs, dict) and 'fts' in outputs else outputs
            if isinstance(feats, torch.Tensor) and feats.dim() == 3:
                feats = feats.mean(dim=1)

        # 2. 强制转换成 [B, 512] 的标准维度
        if isinstance(feats, torch.Tensor) and feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)

        if feats is None: raise RuntimeError("Cannot extract features.")

        # 3. 标准化 -> 修复层 (Adapter) -> 标准化 -> 分类
        feats_norm = F.normalize(feats, dim=1, eps=1e-8)

        if use_strong_head:
            # 1. 物理切断：.detach() 保证梯度绝对不会回传到 Backbone
            feats_detached = feats.detach()
            # 2. 独立适配：走独立的 adapter_strong
            feats_norm = F.normalize(feats_detached, dim=1, eps=1e-8)
            adapted_feats = self.adapter_strong(feats_norm)
            adapted_feats = F.normalize(adapted_feats, dim=1, eps=1e-8)
            logits = self.dino_head_strong(adapted_feats)
        else:
            # 主分支走原有的 adapter 和 dino_head，保护专家方差
            feats_norm = F.normalize(feats, dim=1, eps=1e-8)
            adapted_feats = self.adapter(feats_norm)
            adapted_feats = F.normalize(adapted_feats, dim=1, eps=1e-8)
            logits = self.dino_head(adapted_feats)

        return logits, adapted_feats

    def forward_experts(self, x, use_strong_head=False): # 🚀 必须加上这个参数！
        outputs = self.backbone(x)
        """
        专门为主动学习 DDEUS 策略设计：
        不融合特征，让3个分支独立穿过 Adapter 和 DINOHead，输出3个独立的预测。
        """
        outputs = self.backbone(x)

        # 提取 3 个分支的独立特征 [B, 3, 512]
        if isinstance(outputs, dict) and 'feat_vecs' in outputs:
            feat_vecs = outputs['feat_vecs']
        else:
            raise RuntimeError("Backbone did not return 'feat_vecs'. DDEUS requires multi-branch features.")

        B, K, D = feat_vecs.shape  # B: Batch, K: 3(分支数), D: 512

        # 将其展平为 [B*3, 512]，以便一次性并行通过后续网络
        feats_flat = feat_vecs.view(B * K, D)
        feats_flat = F.normalize(feats_flat, dim=1, eps=1e-8)

        if use_strong_head:
            adapted_flat = self.adapter_strong(feats_flat.detach())
        else:
            adapted_flat = self.adapter(feats_flat)  # 这一步保证了专家方差来源于纯净 Adapter

        adapted_flat = F.normalize(adapted_flat, dim=1, eps=1e-8)
        logits_flat = self.dino_head_strong(adapted_flat) if use_strong_head else self.dino_head(adapted_flat)
        return logits_flat.view(B, K, -1)