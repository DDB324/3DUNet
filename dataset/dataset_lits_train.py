from transforms import RandomCrop, RandomFlipLR, RandomFlipUD, Compose
import SimpleITK as sitk
import torch
import numpy as np
from torch.utils.data import Dataset as dataset
import os

# 读取 dataset/train_path_list.txt文件内容并将它放到数组中作为返回值
def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list


class TrainDataset(dataset):
    def __init__(self, args):
        self.args = args
        self.filename_list = load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))
        self.transforms = Compose([
            RandomCrop(self.args.crop_size),
            RandomFlipLR(prob=0.5),
            RandomFlipUD(prob=0.5),
        ])

    def __getitem__(self, index):
        ct_image = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg_image = sitk.ReadImage(self.filename_list[index][1], sitk.sitkInt8)

        ct_array = sitk.GetArrayFromImage(ct_image)
        seg_array = sitk.GetArrayFromImage(seg_image)

        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)
