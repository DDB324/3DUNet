import argparse
from typing import List, Union, Tuple
import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config
from utils.common import read_image, clip_array


class LiTSPreprocess:
    def __init__(self, raw_path: str, fixed_path: str, args: argparse.Namespace):
        self.raw_root_path: str = raw_path
        self.fixed_path: str = fixed_path
        self.classes: int = args.n_labels
        self.upper: int = args.upper
        self.lower: int = args.lower
        self.expand_slice: int = args.expand_slice
        self.size: int = args.min_slices
        self.xy_down_scale: float = args.xy_down_scale
        self.slice_down_scale: float = args.slice_down_scale
        self.valid_rate: float = args.valid_rate

    def fix_data(self):
        self._create_save_directory()

        file_list: List[str] = os.listdir(join(self.raw_root_path, 'ct'))
        numbers: int = len(file_list)
        print('Total numbers of samples is :', numbers)

        self._write_dataset(file_list, numbers)

    def write_train_val_name_list(self):
        data_name_list = os.listdir(join(self.fixed_path, 'ct'))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is:', data_num)

        train_name_list, val_name_list = self._generate_train_val_list(data_name_list, data_num)

        self._write_name_list(train_name_list, 'train_path_list.txt')
        self._write_name_list(val_name_list, 'val_path_list.txt')

    # 从seg图像获取有标注的图像的范围，并前后各增加20张。
    def _get_liver_start_end_slice(self, seg_array: np.ndarray):
        # 找到肝脏区域开始和结束的slice，并各向外扩张来增加切片的数量
        # z = [False False False False False False False False False False False False
        #  False False False False False False False False False False False False
        #  False False False False False False False False False False False False
        #  False False False False False False False False False  True  True  True
        #   True  True  True  True  True  True  True  True  True  True  True  True
        #   True  True  True  True  True  True  True  True  True  True  True  True
        #   True  True False]
        z: np.ndarray[bool] = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        # 两个方向上各增加self.expand个slice，例如start和end为[45,73]，增加后为[25,93]
        if start_slice - self.expand_slice < 0:
            start_slice = 0
        else:
            start_slice -= self.expand_slice

        if end_slice + self.expand_slice >= seg_array.shape[0]:
            end_slice = seg_array.shape[0] - 1
        else:
            end_slice += self.expand_slice
        print('Cut out range:', str(start_slice) + '--' + str(end_slice))
        return start_slice, end_slice

    def _save_specified_area(self, img: sitk.Image, np_array: np.ndarray) -> sitk.Image:
        # 保存为对应的格式
        new_array = sitk.GetImageFromArray(np_array)
        new_array.SetDirection(img.GetDirection())  # 获取原始图像的方向
        new_array.SetOrigin(img.GetOrigin())  # 获取原始图像的原点
        new_array.SetSpacing((img.GetSpacing()[0] * int(1 / self.xy_down_scale),
                              img.GetSpacing()[1] * int(1 / self.xy_down_scale),
                              self.slice_down_scale))
        return new_array

    def _down_sampling(self, ct: sitk.Image, ct_array_clip: np.ndarray, seg_array: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        # ct.GetSpacing() = (0.703125, 0.703125, 5.0)
        # order=3为三次样条插值
        # 三次样条插值（Cubic Spline Interpolation）： 三次样条插值是一种更精细的插值方法，它使用三次多项式来逼近像素之间的曲线。
        # 这种方法通常能够产生更平滑的插值结果，但相对计算量较大。
        spacing = ct.GetSpacing()[-1]
        ct_array_down_scale = ndimage.zoom(ct_array_clip,
                                           (spacing / self.slice_down_scale, self.xy_down_scale,
                                            self.xy_down_scale),
                                           order=3)
        # order=0为最近邻插值
        # 最近邻插值可以保持 mask 图像中的不同区域的分割边界，不会引入额外的像素值。
        # 这对于分割任务来说是重要的，因为保持准确的区域边界可以有助于正确分割不同的结构。
        seg_array_down_scale = ndimage.zoom(seg_array,
                                            (spacing / self.slice_down_scale, self.xy_down_scale,
                                             self.xy_down_scale),
                                            order=0)
        return ct_array_down_scale, seg_array_down_scale

    def _get_reserved_slices(self, ct_array_down_scale: np.ndarray, seg_array_down_scale: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        start_slice, end_slice = self._get_liver_start_end_slice(seg_array_down_scale)
        ct_array_reserved = ct_array_down_scale[start_slice:end_slice + 1, :, :]
        seg_array_reserved = seg_array_down_scale[start_slice:end_slice + 1, :, :]
        return ct_array_reserved, seg_array_reserved

    def _process(self, ct_path: str, ct_file: str, seg_path: str, classes: int):
        ct, ct_array = read_image(ct_path, 'ct')
        seg, seg_array = read_image(seg_path, 'seg')
        print('Original shape:', ' ct:', ct_array.shape, ' seg:', seg_array.shape)

        # combine ground truth for liver and liver tumor into one
        if classes == 2:
            seg_array[seg_array > 0] = 1

        # truncate the value of the gray value outside the threshold
        ct_array_clip = clip_array(ct_array, self.lower, self.upper)

        # 降采样，对x和y轴进行降采样. 插值，slice轴的spacing进行插值
        ct_array_down_scale, seg_array_down_scale = self._down_sampling(ct, ct_array_clip, seg_array)

        # 截取有标注的图像的范围
        ct_array_reserved, seg_array_reserved = self._get_reserved_slices(ct_array_down_scale, seg_array_down_scale)
        print('Preprocessed shape:', 'ct:', ct_array_reserved.shape, ' seg:', seg_array_reserved.shape)

        # 保存为对应的格式
        new_ct = self._save_specified_area(ct, ct_array_reserved)
        new_seg = self._save_specified_area(ct, seg_array_reserved)
        return new_ct, new_seg

    def _create_save_directory(self):
        if not os.path.exists(self.fixed_path):  # create save directory
            os.makedirs(join(self.fixed_path, 'ct'))
            os.makedirs(join(self.fixed_path, 'label'))

    def _write_dataset(self, file_list: List[str], numbers: int):
        for ct_file, i in zip(file_list, range(numbers)):
            print('==== {} | {}/{} ==='.format(ct_file, i + 1, numbers))
            ct_path: str = os.path.join(self.raw_root_path, 'ct', ct_file)
            seg_path: str = os.path.join(self.raw_root_path, 'label', ct_file.replace('volume', 'segmentation'))

            new_ct: Union[None, sitk.Image]
            new_seg: Union[None, sitk.Image]
            new_ct, new_seg = self._process(ct_path, ct_file, seg_path, classes=self.classes)
            sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))
            sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'label',
                                                  ct_file.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))

    def _generate_train_val_list(self, data_name_list, data_num):
        random.shuffle(data_name_list)
        assert self.valid_rate < 1.0
        train_index = int(data_num * (1 - self.valid_rate))
        train_name_list = data_name_list[:train_index]
        val_name_list = data_name_list[train_index:]
        return train_name_list, val_name_list

    def _write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'ct', name)
            seg_path = os.path.join(self.fixed_path, 'label', name.replace('volume', 'segmentation'))
            f.write(ct_path + ' ' + seg_path + '\n')
        f.close()


def main():
    raw_dataset_path: str = r'D:\dataset\LiTS\train'
    fixed_dataset_path: str = r'D:\dataset\LiTS\fixed_lits_512'

    args: argparse.Namespace = config.args
    tool = LiTSPreprocess(raw_dataset_path, fixed_dataset_path, args)
    tool.fix_data()
    tool.write_train_val_name_list()


if __name__ == '__main__':
    main()
