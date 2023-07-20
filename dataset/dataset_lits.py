import sys
from transforms import RandomCrop, RandomFlipLR, RandomFlipUD, Compose, CenterCrop
from torch.utils.data import Dataset, DataLoader
import os
from utils.common import load_file_name_list, read_image
import SimpleITK as sitk
import numpy as np
import torch


def process_image(path, img_type, norm_factor):
    np_array = read_image(path, img_type)

    if img_type == 'ct':
        np_array = np_array / norm_factor
        np_array = np_array.astype(np.float32)

    np_array = torch.FloatTensor(np_array).unsqueeze(0)
    return np_array


class LiTSDataset(Dataset):
    def __init__(self, config_parameters, dataset_type):
        # 判断数据集类型，只能接收train和val
        if dataset_type not in ['train', 'val']:
            raise ValueError("Invalid parameter value. Expected 'train' or 'val'.")

        # 根据train和val，采取不同的处理方法
        self.args = config_parameters
        if dataset_type == 'train':
            self.dataset_file_name = 'train_path_list.txt'
            self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                RandomFlipLR(prob=0.5),
                RandomFlipUD(prob=0.5),
            ])
        else:
            self.dataset_file_name = 'val_path_list.txt'
            self.transforms = Compose([CenterCrop(base=16, max_size=config_parameters.val_crop_max_size)])

        self.filename_list = load_file_name_list(os.path.join(config_parameters.dataset_path, self.dataset_file_name))

    def __getitem__(self, index):
        ct_array = process_image(self.filename_list[index][0], 'ct', self.args.norm_factor)
        seg_array = process_image(self.filename_list[index][1], 'seg', self.args.norm_factor)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)


if __name__ == "__main__":
    sys.path.append(r'C:\Users\DDB\PycharmProjects\unet')
    from config import args

    # 训练集
    # train_dataset = LiTSDataset(args, 'train')
    # train_dl = DataLoader(train_dataset, 2, False, num_workers=1)
    # for i, (ct, seg) in enumerate(train_dl):
    #     print(i, ct.size(), seg.size())

    # 验证集
    val_dataset = LiTSDataset(args, 'val')
    val_dl = DataLoader(val_dataset, 2, False, num_workers=1)
    for i, (ct, seg) in enumerate(val_dl):
        print(i, ct.size(), seg.size())
