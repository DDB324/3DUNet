import sys
from torch.utils.data import Dataset, DataLoader
import os
from dataset.transforms import Compose, RandomCrop, RandomFlip, CenterCrop
from utils.common import load_file_name_list, read_image
import numpy as np
import torch


class LiTSDataset(Dataset):
    def __init__(self, args, dataset_type: str):
        # 判断数据集类型，只能接收train和val
        if dataset_type not in ['train', 'val']:
            raise ValueError("Invalid parameter value. Expected 'train' or 'val'.")

        # 根据train和val，采取不同的处理方法
        self.args = args
        if dataset_type == 'train':
            self.dataset_file_name: str = 'train_path_list.txt'
            self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                RandomFlip(prob=0.5, flip_lr=True, flip_ud=True),
            ])
        else:
            self.dataset_file_name = 'val_path_list.txt'
            self.transforms = Compose([CenterCrop(crop_max_size=args.val_crop_max_size, base=16)])

        self.filename_list = load_file_name_list(os.path.join(args.dataset_path, self.dataset_file_name))

    @staticmethod
    def _process_image(path: str, img_type: str, norm_factor: int = 200) -> torch.Tensor:
        np_array = read_image(path, img_type)[1]

        if img_type == 'ct':
            np_array = np_array / norm_factor  # 归一化
            np_array = np_array.astype(np.float32)  # 保证数据类型与 PyTorch 中的 FloatTensor 匹配

        tensor_array: torch.Tensor = torch.FloatTensor(np_array).unsqueeze(0)
        return tensor_array

    def __getitem__(self, index):
        ct_tensor_array: torch.Tensor = self._process_image(self.filename_list[index][0], 'ct', self.args.norm_factor)
        seg_tensor_array: torch.Tensor = self._process_image(self.filename_list[index][1], 'seg', self.args.norm_factor)

        if self.transforms:
            ct_tensor_array, seg_tensor_array = self.transforms(ct_tensor_array, seg_tensor_array)

        return ct_tensor_array, seg_tensor_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)


def main():
    sys.path.append(r'C:\Users\DDB\PycharmProjects\unet')
    from config import args

    # 训练集
    train_dataset = LiTSDataset(args, 'train')
    train_dl = DataLoader(train_dataset, 2, False, num_workers=1)
    for batch_idx, (ct, seg) in enumerate(train_dl):
        print(f"Batch {batch_idx}:")
        print("CT array shape:", ct.shape)  # [2,1,48,256,256] val[1,1,96,256,256]
        # print("CT array content:")
        # print(ct)
        print("Segmentation array shape:", seg.shape)  # [2,48,256,256] val[1,96,256,256]
        # print("Segmentation array content:")
        # print(seg)
        print("\n")


if __name__ == "__main__":
    main()
