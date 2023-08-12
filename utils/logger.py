import csv
import os
from collections import OrderedDict, defaultdict

import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, save_path, save_name):
        self.log = defaultdict(list)
        self.summary = SummaryWriter(save_path)
        self.save_path = save_path
        self.save_name = save_name

    def update_train_val(self, epoch, train_log, val_log):
        item = {'epoch': epoch}
        item.update(train_log)
        item.update(val_log)
        print("\033[0;33mTrain:\033[0m", train_log)
        print("\033[0;33mValid:\033[0m", val_log)
        self._update_log(item)
        self._update_tensorboard(item)

    def update_test(self, img_name, test_log):
        item = {'img_name': img_name}
        item.update(test_log)
        print("\033[0;33mTrain:\033[0m", test_log)
        self._update_log(item)

    def _update_log(self, item):
        for key, value in item.items():
            self.log[key].append(value)

        # Save to CSV file
        self._save_to_csv(item)

    def _update_tensorboard(self, item):
        epoch = item['epoch']
        for key, value in item.items():
            if key != 'epoch':
                self.summary.add_scalar(key, value, epoch)

    def _save_to_csv(self, item):
        file_path = os.path.join(self.save_path, f"{self.save_name}.csv")
        header = not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=item.keys())
            if header:
                writer.writeheader()
            writer.writerow(item)

    def close(self):
        self.summary.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def dict_round(dic, num):
    for key, value in dic.items():
        dic[key] = round(value, num)
    return dic
