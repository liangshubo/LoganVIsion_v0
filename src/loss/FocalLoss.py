import torch.nn.functional as F

import  torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)  # 模型预测正确的概率
        focal_loss = self.alpha * (1 - pt )**self.gamma * ce_loss
        return focal_loss.mean()