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


def process_image(img, kind, norm_factor):
    image = sitk.ReadImage(img, sitk.sitkInt16 if kind == 'ct' else sitk.sitkInt8)
    array = sitk.GetArrayFromImage(image)

    if kind == 'ct':
        array = array / norm_factor
        array = array.astype(np.float32)

    array = torch.FloatTensor(array).unsqueeze(0)
    return array


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
        ct_array = process_image(self.filename_list[index][0], 'ct', self.args.norm_factor)
        seg_array = process_image(self.filename_list[index][1], 'seg', self.args.norm_factor)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)


if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args

    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i, ct.size(), seg.size())
