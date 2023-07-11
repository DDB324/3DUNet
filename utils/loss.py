import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i].sum(dim=1).sum(dim=1).sum(dim=1) / (
                    pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target[:, i].pow(2).sum(dim=1).sum(dim=1)
                    .sum(dim=1) + smooth))
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)


class ELDiceLoss(nn.Module):
    def __init__(self):
        super(ELDiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i].sum(dim=1).sum(dim=1).sum(dim=1) / (
                    pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target[:, i].pow(2).sum(dim=1).sum(dim=1)
                    .sum(dim=1) + smooth))
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):
        smooth = 1
        dice = 0.
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                    target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        jaccard = 0.
        for i in range(pred.size(1)):
            jaccard += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                    target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:, i] * target[:, i])
                    .sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)


class SSLoss(nn.Module):
    def __init__(self):
        super(SSLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        loss = 0.
        for i in range(pred.size(1)):
            s1 = ((pred[:, i] - target[:, i]).pow(2) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    smooth + target[:, i].sum(dim=1).sum(dim=1).sum(dim=1))
            s2 = ((pred[:, i] - target[:, i]).pow(2) * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    smooth + (1 - target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1))
            loss += (0.05 * s1 + 0.95 * s2)
        return loss / pred.size(1)


class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.
        for i in range(pred.size(1)):
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                    (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                    0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                            (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)
