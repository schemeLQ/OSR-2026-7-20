import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


# ==========================================
# [组件] BranchSE: 轻量通道注意力 (替代 BACL)
# 每个 branch 独立学习 channel 重要性权重 → 自然形成 channel-level 多样性
# ==========================================
class BranchSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = x.mean(dim=[2, 3])                         # [B, C] global avg pool
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
        return x * w


# ==========================================
# [组件] VisualBACL: 视觉后门调整模块
# ==========================================
class VisualBACL(nn.Module):
    def __init__(self, in_dim, num_confounders=32):
        super(VisualBACL, self).__init__()
        # 1. 混杂因子字典
        self.confounder_dict = nn.Parameter(torch.Tensor(num_confounders, in_dim))

        # 2. 注意力层
        self.w_q = nn.Linear(in_dim, in_dim)
        self.w_k = nn.Linear(in_dim, in_dim)
        self.w_v = nn.Linear(in_dim, in_dim)

        # 3. 门控层
        self.gate_layer = nn.Linear(in_dim, in_dim)
        self.ln = nn.LayerNorm(in_dim)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.confounder_dict)
        init.xavier_uniform_(self.w_q.weight)
        init.xavier_uniform_(self.w_k.weight)
        init.xavier_uniform_(self.w_v.weight)
        init.zeros_(self.gate_layer.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]

        q = self.w_q(x_flat)
        k = self.w_k(self.confounder_dict).unsqueeze(0).expand(b, -1, -1)
        v = self.w_v(self.confounder_dict).unsqueeze(0).expand(b, -1, -1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (c ** 0.5)
        attn_probs = F.softmax(attn, dim=-1)
        z_b = torch.matmul(attn_probs, v)

        bg_gate = torch.sigmoid(self.gate_layer(z_b))
        x_adjusted = x_flat * (1.0 - bg_gate)
        x_adjusted = self.ln(x_adjusted)
        return x_adjusted.transpose(1, 2).reshape(b, c, h, w)

    def get_orthogonality_loss(self, classifier_weights):
        dict_norm = F.normalize(self.confounder_dict, p=2, dim=1)
        cls_norm = F.normalize(classifier_weights, p=2, dim=1)
        similarity = torch.matmul(dict_norm, cls_norm.t())
        return torch.mean(similarity ** 2)


# ==========================================
# ODLLoss
# ==========================================
class ODLLoss(nn.Module):
    def __init__(self, num_classes=6, feat_dim=128, margin=20):
        super(ODLLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.register_buffer("classes", torch.arange(self.num_classes).long())
        self.margin = margin

    def forward(self, x, labels):
        device = x.device
        batch_size = x.size(0)

        # 输入 x 已经过 F.normalize（单位球），Centers 投影到同一球面保证距离有界 [0,4]
        # 原始代码用 randn 初始化的 centers 范数 ≈ sqrt(512)≈22，导致距离≈513，
        # 被 clamp(max=100) 全部截断，梯度归零，ODL 无法学习
        c = F.normalize(self.centers, p=2, dim=1)  # [K, D]，单位范数，梯度仍流回 self.centers

        distmat = (torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +
                   torch.pow(c, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
                   - 2.0 * x.mm(c.t()))

        self.distcenter = torch.cdist(c, c, p=2) ** 2
        dist_2center, index_2center = torch.sort(self.distcenter)
        nearest_center_dis = dist_2center[:, 1]
        nearest_center_index = index_2center[:, 1]

        dir_c2c = torch.sub(c.unsqueeze(1).expand(self.num_classes, self.num_classes, self.feat_dim), c)

        inter_cos = torch.tensor(0.).to(device)
        for i in range(self.num_classes):
            nearest = nearest_center_index[i]
            dir_2nearest = dir_c2c[i, nearest].unsqueeze(0)
            dir_near2near = dir_c2c[i]
            dis_near2near = self.distcenter[nearest]

            into_compute = (self.classes.ne(i) & self.classes.ne(nearest))
            dis_near2near = torch.masked_select(dis_near2near, into_compute)
            cos_near = torch.cosine_similarity(dir_2nearest, dir_near2near, dim=1)
            cos_near = torch.masked_select(cos_near, into_compute)

            neardis_weight = torch.softmax(-dis_near2near.unsqueeze(0), dim=1)
            inter_cos += ((1 + cos_near.unsqueeze(0)) * neardis_weight).sum()
        inter_cos = inter_cos / self.num_classes

        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(self.classes.expand(batch_size, self.num_classes))
        no_mask = labels_expand.ne(self.classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        neighbor_dist = torch.masked_select(distmat, no_mask).view(batch_size, -1)
        dis_weight = torch.softmax(-neighbor_dist, dim=1)

        centers_exp = c.unsqueeze(0).expand(batch_size, self.num_classes, self.feat_dim)
        samples_exp = x.unsqueeze(1).expand(batch_size, self.num_classes, self.feat_dim)
        dir_s2cs = torch.sub(centers_exp, samples_exp)
        dir_s2other = dir_s2cs[no_mask].view(batch_size, -1, self.feat_dim)
        dir_s2c = dir_s2cs[mask].unsqueeze(1).expand(batch_size, -1, self.feat_dim)
        cosine_dir = torch.cosine_similarity(dir_s2c, dir_s2other, dim=2)

        # 归一化后距离在 [0, 4]，clamp max=4.5 即可，梯度完全流通
        loss_repel = torch.clamp(self.margin - nearest_center_dis, 1e-12, 10.0).sum() / self.num_classes
        loss_direction = ((1 - cosine_dir) * dis_weight).sum() / batch_size
        loss_center = dist.clamp(min=1e-12, max=4.5).sum() / batch_size

        loss = loss_center + loss_direction + 0.1 * (loss_repel + inter_cos)
        return loss


# ==========================================
# ResNet Backbone
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, mastermodel, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.mastermodel = mastermodel

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_block=[2, 2, 2, 2], avg_output=False, output_dim=-1, resprestride=1,
                 res1ststride=1, res2ndstride=1, inchan=3, stem_type='cifar'):
        super().__init__()
        img_chan = inchan
        if stem_type == 'imagenet':
            self.conv1 = nn.Sequential(
                nn.Conv2d(img_chan, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(img_chan, 64, kernel_size=3, padding=1, bias=False, stride=resprestride),
                nn.BatchNorm2d(64),
                nn.LeakyReLU())
        self.in_channels = 64
        self.conv2_x = self._make_layer(block, 64, num_block[0], res1ststride)
        self.conv3_x = self._make_layer(block, 128, num_block[1], res2ndstride)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.conv6_x = nn.Identity() if output_dim <= 0 else self.conv_layer(512, output_dim, 1, 0)
        self.conv6_is_identity = output_dim <= 0
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if output_dim > -1:
            self.output_dim = output_dim
        else:
            self.output_dim = 512 * block.expansion
        self.avg_output = avg_output

    def conv_layer(self, input_channel, output_channel, kernel_size=3, padding=1):
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2))
        return res

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(None, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.conv6_x(output)
        if self.avg_output:
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
        return output


def build_backbone(img_size, backbone_name, projection_dim, inchan=3, stem_type='cifar'):
    if backbone_name == 'resnet18':
        backbone = ResNet(output_dim=projection_dim, inchan=inchan, resprestride=1, res1ststride=1, res2ndstride=2, stem_type=stem_type)
        cam_size = int(img_size / (32 if stem_type == 'imagenet' else 8))
    elif backbone_name == 'resnet34':
        backbone = ResNet(output_dim=projection_dim, inchan=inchan, num_block=[3, 4, 6, 3], resprestride=1,
                          res1ststride=2, res2ndstride=2, stem_type=stem_type)
        cam_size = int(img_size / 32)
    else:
        raise Exception(f'Backbone \"{backbone_name}\" is not defined.')
    return backbone, backbone.output_dim, cam_size


class BaselineNet(nn.Module):
    def __init__(self, args):
        super(BaselineNet, self).__init__()
        backbone, feature_dim, _ = build_backbone(img_size=args['img_size'],
                                                  backbone_name=args['backbone'],
                                                  projection_dim=-1,
                                                              inchan=3, stem_type=args.get('stem_type', 'cifar'))
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = conv1x1(feature_dim, args['num_known'])

    def forward(self, x, y=None):
        x = self.backbone(x)
        ft = self.classifier(x)
        logits = self.pool(ft)
        logits = logits.view(logits.size(0), -1)
        outputs = {'logits': [logits]}
        return outputs

    def get_params(self, prefix='extractor'):
        extractor_params = list(self.backbone.parameters())
        extractor_params_ids = list(map(id, self.backbone.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim() == 1: logit = logit.unsqueeze(0)
        return logit


# ==========================================
# [核心组件] MultiBranchNet
# ==========================================
class MultiBranchNet(nn.Module):
    def __init__(self, args=None):
        super(MultiBranchNet, self).__init__()
        backbone, feature_dim, self.cam_size = build_backbone(img_size=args['img_size'],
                                                              backbone_name=args['backbone'],
                                                              projection_dim=-1,
                                                              inchan=3, stem_type=args.get('stem_type', 'cifar'))
        self.img_size  = args['img_size']
        self.gate_temp = args['gate_temp']
        self.num_known = args['num_known']
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.use_bacl  = args.get('use_bacl', False)
        self.use_odl   = args.get('use_odl',  False)

        _ch = list(backbone.children())  # conv1, conv2_x, conv3_x, conv4_x, conv5_x, conv6_x, avg_pool
        self.legacy_split = args.get('legacy_split', False)
        if self.legacy_split:
            # Compatibility with old MEDAF checkpoints such as 0513_2201:
            # shared_l3=conv1, branch_l4=conv2_x+conv3_x+conv4_x.
            self.shared_l3 = nn.Sequential(*_ch[:1])
            self.branch1_l4 = nn.Sequential(*_ch[1:4])
        else:
            self.shared_l3 = nn.Sequential(*_ch[:3])    # shared: conv1, conv2_x, conv3_x
            self.branch1_l4 = nn.Sequential(*_ch[3:4])  # branch-private: conv4_x
        self.branch1_l5 = nn.Sequential(*_ch[4])
        self.branch1_cls = conv1x1(feature_dim, self.num_known)

        self.branch2_l4 = copy.deepcopy(self.branch1_l4)
        self.branch2_l5 = copy.deepcopy(self.branch1_l5)
        self.branch2_cls = conv1x1(feature_dim, self.num_known)

        self.branch3_l4 = copy.deepcopy(self.branch1_l4)
        self.branch3_l5 = copy.deepcopy(self.branch1_l5)
        self.branch3_cls = conv1x1(feature_dim, self.num_known)

        # SE channel-attention per branch (lightweight diversity; kept for ablation)
        self.se1 = BranchSE(feature_dim)
        self.se2 = BranchSE(feature_dim)
        self.se3 = BranchSE(feature_dim)

        # VisualBACL kept for ablation experiments (disabled by default via use_bacl=False)
        self.bacl1 = VisualBACL(in_dim=feature_dim, num_confounders=32)
        self.bacl2 = VisualBACL(in_dim=feature_dim, num_confounders=32)
        self.bacl3 = VisualBACL(in_dim=feature_dim, num_confounders=32)

        self.gate_l3 = copy.deepcopy(self.shared_l3)
        self.gate_l4 = copy.deepcopy(self.branch1_l4)
        self.gate_l5 = copy.deepcopy(self.branch1_l5)
        self.gate_cls = nn.Sequential(Classifier(feature_dim, int(feature_dim / 4), bias=True),
                                      Classifier(int(feature_dim / 4), 3, bias=True))

        self.use_pb_odl = args.get('use_pb_odl', False)
        # Single ODL instance with shared class centers.
        # SC-PB-ODL: called once per branch → each branch independently organized
        # around the SAME class geometry; no conflicting gradients in shared L3.
        self.odl_loss1 = ODLLoss(num_classes=self.num_known, feat_dim=feature_dim, margin=0.6)

    def set_react_thresholds(self, thresholds):
        """Enable ReAct clipping with one scalar threshold per expert branch."""
        self.react_thresholds = [float(t) for t in thresholds]

    def clear_react_thresholds(self):
        """Disable ReAct clipping. Called before training resumes."""
        if hasattr(self, 'react_thresholds'):
            delattr(self, 'react_thresholds')

    def _react_clip(self, x, branch_idx):
        thresholds = getattr(self, 'react_thresholds', None)
        if thresholds is None:
            return x
        return torch.clamp(x, max=thresholds[branch_idx])

    def forward(self, x, y=None, return_ft=False):
        b = x.size(0)
        ft_till_l3 = self.shared_l3(x)

        # --- Branch 1 ---
        branch1_l4 = self.branch1_l4(ft_till_l3.clone())
        branch1_l5 = self.branch1_l5(branch1_l4)
        branch1_l5 = self._react_clip(branch1_l5, 0)
        branch1_out = self.bacl1(branch1_l5) if self.use_bacl else self.se1(branch1_l5)
        b1_feat    = self.avg_pool(branch1_out).view(b, -1)
        b1_ft_cams = self.branch1_cls(branch1_out)
        b1_logits  = self.avg_pool(b1_ft_cams).view(b, -1)

        # --- Branch 2 ---
        branch2_l4 = self.branch2_l4(ft_till_l3.clone())
        branch2_l5 = self.branch2_l5(branch2_l4)
        branch2_l5 = self._react_clip(branch2_l5, 1)
        branch2_out = self.bacl2(branch2_l5) if self.use_bacl else self.se2(branch2_l5)
        b2_feat    = self.avg_pool(branch2_out).view(b, -1)
        b2_ft_cams = self.branch2_cls(branch2_out)
        b2_logits  = self.avg_pool(b2_ft_cams).view(b, -1)

        # --- Branch 3 ---
        branch3_l4 = self.branch3_l4(ft_till_l3.clone())
        branch3_l5 = self.branch3_l5(branch3_l4)
        branch3_l5 = self._react_clip(branch3_l5, 2)
        branch3_out = self.bacl3(branch3_l5) if self.use_bacl else self.se3(branch3_l5)
        b3_feat    = self.avg_pool(branch3_out).view(b, -1)
        b3_ft_cams = self.branch3_cls(branch3_out)
        b3_logits  = self.avg_pool(b3_ft_cams).view(b, -1)

        # --- Gate Calculation (提前到这里) ---
        gate_l5 = self.gate_l5(self.gate_l4(self.gate_l3(x)))
        gate_pool = self.avg_pool(gate_l5).view(b, -1)
        gate_pred = F.softmax(self.gate_cls(gate_pool) / self.gate_temp, dim=1)

        gate_logits = torch.stack([b1_logits, b2_logits, b3_logits], dim=-1)
        gate_logits = gate_logits * gate_pred.view(gate_pred.size(0), 1, gate_pred.size(1))
        gate_logits = gate_logits.sum(-1)

        logits_list = [b1_logits, b2_logits, b3_logits, gate_logits]

        # 🚨🚨🚨 [修复] 先初始化 outputs 字典 🚨🚨🚨
        outputs = {'logits': logits_list, 'gate_pred': gate_pred}

        # 🚨🚨🚨 [修复] 再赋值 feat_vecs 🚨🚨🚨
        outputs['feat_vecs'] = torch.stack([b1_feat, b2_feat, b3_feat], dim=1)

        # --- CAM Calculation ---
        if y is not None:
            target_idx = y
        else:
            target_idx = gate_logits.max(1)[1]

        cams = torch.cat([
            b1_ft_cams.gather(dim=1, index=target_idx[:, None, None, None].repeat(1, 1, b1_ft_cams.shape[-2],
                                                                                  b1_ft_cams.shape[-1])),
            b2_ft_cams.gather(dim=1, index=target_idx[:, None, None, None].repeat(1, 1, b2_ft_cams.shape[-2],
                                                                                  b2_ft_cams.shape[-1])),
            b3_ft_cams.gather(dim=1, index=target_idx[:, None, None, None].repeat(1, 1, b3_ft_cams.shape[-2],
                                                                                  b3_ft_cams.shape[-1])),
        ], dim=1)

        outputs['cams'] = cams

        # --- Feature Return ---
        if return_ft:
            fts = b1_ft_cams.detach().clone() + b2_ft_cams.detach().clone() + b3_ft_cams.detach().clone()
            if y is None:
                outputs['fts'] = fts

        # --- ODL Loss ---
        if y is not None and self.use_odl:
            if self.use_pb_odl:
                # SC-PB-ODL: shared centers called once per branch.
                # Each branch independently pulled toward the SAME class geometry;
                # L3 gradients from all three branches are consistent → stable training.
                b1n = F.normalize(b1_feat, p=2, dim=1)
                b2n = F.normalize(b2_feat, p=2, dim=1)
                b3n = F.normalize(b3_feat, p=2, dim=1)
                outputs['loss_odl'] = (self.odl_loss1(b1n, y) +
                                       self.odl_loss1(b2n, y) +
                                       self.odl_loss1(b3n, y)) / 3
            else:
                fused_feat = F.normalize(b1_feat + b2_feat + b3_feat, p=2, dim=1)
                outputs['loss_odl'] = self.odl_loss1(fused_feat, y)

        return outputs

    def get_params(self, prefix='extractor'):
        extractor_params = (list(self.shared_l3.parameters()) +
                            list(self.branch1_l4.parameters()) + list(self.branch1_l5.parameters()) +
                            list(self.branch2_l4.parameters()) + list(self.branch2_l5.parameters()) +
                            list(self.branch3_l4.parameters()) + list(self.branch3_l5.parameters()) +
                            list(self.gate_l3.parameters()) + list(self.gate_l4.parameters()) +
                            list(self.gate_l5.parameters()) +
                            list(self.se1.parameters()) + list(self.se2.parameters()) +
                            list(self.se3.parameters()) +
                            list(self.bacl1.parameters()) + list(self.bacl2.parameters()) +
                            list(self.bacl3.parameters()))

        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())

        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
