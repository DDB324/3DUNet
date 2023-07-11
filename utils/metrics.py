import numpy as np
import torch


class LossAverage(object):
    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)


class DiceAverage(object):
    def __init__(self, class_num):
        self.count = None
        self.sum = None
        self.avg = None
        self.value = None
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    @staticmethod
    def get_dices(logit, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logit[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logit[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)

    def update(self, logit, targets):
        self.value = DiceAverage.get_dices(logit, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)