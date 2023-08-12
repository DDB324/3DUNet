import numpy as np
import torch


# 跟踪和更新损失值，并计算损失的平均值
class LossAverage:
    def __init__(self):
        self.total_loss = 0.0
        self.count = 0

    def update(self, loss: float, batch_size: int = 1):
        self.total_loss += loss * batch_size
        self.count += batch_size

    def get_average(self):
        if self.count == 0:
            return 0.0
        average_loss = np.around(self.total_loss / self.count, 4)
        return average_loss


class DiceAverage(object):
    def __init__(self, n_labels: int):
        self.sum = np.asarray([0] * n_labels, dtype='float64')
        self.count = 0

    def update(self, prediction: torch.Tensor, targets: torch.Tensor):
        value = self._get_dices(prediction, targets)
        self.sum += value
        self.count += 1
        # print(self.value)

    @staticmethod
    def _get_dices(prediction: torch.Tensor, targets: torch.Tensor):
        dices = []
        num_classes = targets.size()[1]
        for class_index in range(num_classes):
            inter = torch.sum(prediction[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(prediction[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)

    def get_average(self):
        return np.around(self.sum / self.count, 4)
