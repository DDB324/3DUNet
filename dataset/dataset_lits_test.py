import sys
from glob import glob
import SimpleITK as sitk
import numpy as np
import os
import torch
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from utils.common import read_image, clip_array, padding_img, extract_ordered_overlap


class LiTSTestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.test_data_path = args.test_data_path
        self.ct_list = sorted(glob(os.path.join(self.test_data_path, 'ct/*')))
        self.seg_list = sorted(glob(os.path.join(self.test_data_path, 'label/*')))

    def __getitem__(self, item):
        ct_array = read_image(self.ct_list[item], 'ct')[1]
        seg_array = read_image(self.seg_list[item], 'seg')[1]
        return ct_array, seg_array

    def __len__(self):
        return len(self.ct_list)


def main():
    sys.path.append(r'C:\Users\DDB\PycharmProjects\unet')
    from config import args

    test_dataset = LiTSTestDataset(args)
    test_loader = DataLoader(test_dataset, 1, False, num_workers=1)
    for batch_idx, (ct, seg) in enumerate(test_loader):
        print(f"Batch {batch_idx}:")
        print("CT array shape:", ct.shape)
        print("CT array content:")
        # print(ct)
        print("Segmentation array shape:", seg.shape)
        print("Segmentation array content:")
        # print(seg)
        print("\n")


class TestDataset(Dataset):
    def __init__(self, data_path, label_path, args):
        self.ct_path = data_path
        self.seg_path = label_path
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride
        self.slice_down_scale: float = args.slice_down_scale
        self.xy_down_scale: float = args.xy_down_scale
        self.upper = args.upper
        self.lower = args.lower
        self.norm_factor = args.norm_factor

        self.ori_shape = None
        self.resized_shape = None
        self.padding_shape = None

        self.processed_ct = self._process_ct()
        self.processed_seg = self._process_label()

        # 预测结果保存
        self.result = None

    def update_result(self, tensor):
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def merge_segmentation_results(self):
        """
        将分割后的结果合并为原始图像的预测结果。
        通常在深度学习图像分割任务中，为了避免边缘处信息丢失的问题，
        我们会对图像进行重叠分割（overlapping patches），然后对分割结果进行合并，得到最终的预测结果。
        """
        patch_size = self.result.shape[2]  # 获取分割结果self.result中每个块(patch)的s轴大小（通常代表通道数）

        num_patches = (self.padding_shape[0] - patch_size) // self.cut_stride + 1  # 计算原始图像在s轴上能够得到多少个重叠分割块

        # 初始化两个全零张量：full_prob和full_sum，用于存储像素预测概率的累加和以及像素点出现的次数。
        total_probabilities = np.zeros((self.n_labels,
                                        self.padding_shape[0],
                                        self.ori_shape[1],
                                        self.ori_shape[2]), dtype=np.float32)
        total_counts = np.zeros((self.n_labels,
                                 self.padding_shape[0],
                                 self.ori_shape[1],
                                 self.ori_shape[2]), dtype=np.float32)

        for i in range(num_patches):
            # 将第i个分割块的预测结果self.result[i]叠加到对应位置的full_prob中
            total_probabilities[:, i * self.cut_stride:i * self.cut_stride + patch_size] += self.result[i]
            # 在对应位置的full_sum中加1，记录每个像素点出现的次数
            total_counts[:, i * self.cut_stride:i * self.cut_stride + patch_size] += 1

        # 计算每个像素点的平均预测概率，得到最终的平均预测结果
        final_avg = total_probabilities / total_counts

        # 从最终平均预测结果中截取出与原始图像尺寸相同的部分
        merged_result = final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]

        # 返回最终的平均预测结果，并在第0维添加一个维度，使其与原始图像形状相匹配
        return merged_result[np.newaxis, :]

    def _process_ct(self):
        ct: sitk.Image
        ct_array: np.ndarray
        ct, ct_array = read_image(self.ct_path, 'ct')
        self.ori_shape = ct_array.shape

        ct_array_down_scale = ndimage.zoom(ct_array,
                                           (self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                           order=3)  # 双三次重采样

        ct_array_clip = clip_array(ct_array_down_scale, self.lower, self.upper)
        ct_array_norm = ct_array_clip / self.norm_factor
        self.resized_shape = ct_array_norm.shape

        # 扩展一定数量的slices，以保证卷积下采样合理运算
        ct_array_add_slices = padding_img(ct_array_norm, self.cut_size, self.cut_stride)
        self.padding_shape = ct_array_add_slices.shape

        ct_array_overlapping_patches = extract_ordered_overlap(ct_array_add_slices, self.cut_size, self.cut_stride)

        return ct_array_overlapping_patches

    def _process_label(self):
        seg: sitk.Image
        seg_aray: np.ndarray
        seg, seg_aray = read_image(self.seg_path, 'seg')

        if self.n_labels == 2:
            seg_aray[seg_aray > 0] = 1

        label: torch = torch.from_numpy(np.expand_dims(seg_aray, axis=0)).long
        return label

    def __getitem__(self, index):
        data = torch.from_numpy(self.processed_ct[index])
        data = torch.FloatTensor(data).unsqueeze(0)
        return data

    def __len__(self):
        return len(self.processed_ct)


def Test_Dataset(dataset_path, args):
    ct_list = sorted(glob(os.path.join(dataset_path, 'ct/*')))
    seg_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print("The number of test samples is: ", len(ct_list))
    for ct_path, seg_path in zip(ct_list, seg_list):
        print("\nStart Evaluate: ", ct_path)
        yield TestDataset(ct_path, seg_path, args=args), ct_path.split('-')[-1]


if __name__ == "__main__":
    main()
