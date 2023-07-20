import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config
from utils.common import read_image, clip_array


class LiTSPreprocess:
    def __init__(self, raw_path, fixed_path, config_parameters):
        self.raw_root_path = raw_path
        self.fixed_path = fixed_path
        self.classes = config_parameters.n_labels
        self.upper = config_parameters.upper
        self.lower = config_parameters.lower
        self.expand_slice = config_parameters.expand_slice
        self.size = config_parameters.min_slices
        self.xy_down_scale = config_parameters.xy_down_scale
        self.slice_down_scale = config_parameters.slice_down_scale
        self.valid_rate = config_parameters.valid_rate

    def add_slice(self, seg_array):
        # 找到肝脏区域开始和结束的slice，并各向外扩张来增加切片的数量
        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        # 两个方向上各增加slice
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

    def save_specified_area(self, img, np_array):
        # 保存为对应的格式
        new_array = sitk.GetImageFromArray(np_array)
        new_array.SetDirection(img.GetDirection())
        new_array.SetOrigin(img.GetOrigin())
        new_array.SetSpacing((img.GetSpacing()[0] * int(1 / self.xy_down_scale),
                              img.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        return new_array

    def process(self, ct_path, ct_file, seg_path, classes=None):
        ct, ct_array = read_image(ct_path, 'ct', '2')
        seg, seg_array = read_image(seg_path, 'seg', '2')
        print('Original shape:', ' ct:', ct_array.shape, ' seg:', seg_array.shape)

        # combine ground truth for liver and liver tumor into one
        if classes == 2:
            seg_array[seg_array > 0] = 1

        # truncate the value of the gray value outside the threshold
        ct_array = clip_array(ct_array, self.lower, self.upper)

        # 降采样，对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale
        ct_array = ndimage.zoom(ct_array,
                                (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                order=3)
        seg_array = ndimage.zoom(seg_array,
                                 (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                 order=0)

        start_slice, end_slice = self.add_slice(seg_array)

        # 如果这时候剩下的slice数量小于最小切片数量，直接放弃
        if end_slice - start_slice + 1 < self.size:
            print('Too little slice, give up the sample:', ct_file)
            return None, None
        # 截取保留区域
        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]
        print('Preprocessed shape:', ct_array.shape, seg_array.shape)
        # 保存为对应的格式
        new_ct = self.save_specified_area(ct, ct_array)
        new_seg = self.save_specified_area(ct, seg_array)
        return new_ct, new_seg

    def create_save_directory(self):
        if not os.path.exists(self.fixed_path):  # create save directory
            os.makedirs(join(self.fixed_path, 'ct'))
            os.makedirs(join(self.fixed_path, 'label'))

    def write_dataset(self, file_list, numbers):
        for ct_file, i in zip(file_list, range(numbers)):
            print('==== {} | {}/{} ==='.format(ct_file, i + 1, numbers))
            ct_path = os.path.join(self.raw_root_path, 'ct', ct_file)
            seg_path = os.path.join(self.raw_root_path, 'label', ct_file.replace('volume', 'segmentation'))
            new_ct, new_seg = self.process(ct_path, ct_file, seg_path, classes=self.classes)
            if new_ct is not None and new_seg is not None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'label',
                                                      ct_file.replace('volume', 'segmentation').replace('.nii',
                                                                                                        '.nii.gz')))

    def fix_data(self):
        self.create_save_directory()

        file_list = os.listdir(join(self.raw_root_path, 'ct'))
        numbers = len(file_list)
        print('Total numbers of samples is :', numbers)

        self.write_dataset(file_list, numbers)

    def generate_train_val_list(self, data_name_list, data_num):
        random.shuffle(data_name_list)
        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num * (1 - self.valid_rate))]
        val_name_list = data_name_list[
                        int(data_num * (1 - self.valid_rate)):int(data_num * ((1 - self.valid_rate) + self.valid_rate))]
        return train_name_list, val_name_list

    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'ct', name)
            seg_path = os.path.join(self.fixed_path, 'label', name.replace('volume', 'segmentation'))
            f.write(ct_path + ' ' + seg_path + '\n')
        f.close()

    def write_train_val_name_list(self):
        data_name_list = os.listdir(join(self.fixed_path, 'ct'))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is:', data_num)

        train_name_list, val_name_list = self.generate_train_val_list(data_name_list, data_num)

        self.write_name_list(train_name_list, 'train_path_list.txt')
        self.write_name_list(val_name_list, 'val_path_list.txt')


if __name__ == '__main__':
    raw_dataset_path = r'C:\Users\DDB\Desktop\LiTS\train'
    fixed_dataset_path = r'C:\Users\DDB\PycharmProjects\unet\dataset\fixed_lits'

    args = config.args
    tool = LiTSPreprocess(raw_dataset_path, fixed_dataset_path, args)
    tool.fix_data()
    tool.write_train_val_name_list()
