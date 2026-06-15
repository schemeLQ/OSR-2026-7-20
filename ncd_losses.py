import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """SimCLR Loss: 拉近正例对，推开负例对"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        # features shape: [2*B, Dim]
        device = features.device
        batch_size = features.shape[0] // 2

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature

        targets = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        return F.cross_entropy(logits, targets)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features.dim() == 3: features = features.squeeze(1)
        device = features.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(features.shape[0]).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = - (1.0 / 1.0) * mean_log_prob_pos
        return loss.mean()