from typing import Tuple
import numpy as np
import random
import torch
import SimpleITK as sitk


def to_one_hot_3d(tensor: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
    """输入四维张量，返回one-hot编码后的五维张量"""
    b, s, h, w = tensor.size()
    one_hot = torch.zeros(b, n_classes, s, h, w).scatter_(1, tensor.view(b, 1, s, h, w), 1)
    return one_hot


def padding_img(img: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    扩展一定数量的slices，以保证卷积下采样合理运算
    将输入的3D图像数组进行填充(padding)操作，使得图像在s轴上的维度（depth/通道数）能够被整除以给定的stride。
    目的是：数据对齐：在深度学习中，经常需要对输入数据进行批处理（batch processing）以提高计算效率。
    对于图像数据，要求它们在批处理中具有相同的尺寸，这样才能一次性进行并行计算。
    """
    assert (len(img.shape) == 3)  # 3D array
    img_s, img_h, img_w = img.shape
    leftover_s = (img_s - size) % stride

    if leftover_s != 0:
        target_s = img_s + (stride - leftover_s)
    else:
        target_s = img_s

    new_img = np.zeros((target_s, img_h, img_w), dtype=np.float32)
    new_img[:img_s] = img
    print("Padded images shape: " + str(new_img.shape))
    return new_img


def extract_ordered_overlap(img: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    对数据按步长进行分patch操作，以防止显存溢出
    将输入的3D图像数组（通常是一个带有多个通道的图像）分割成大小为size的重叠（overlapping）块（patches）并返回这些块的数组
    这样的分割在一些深度学习任务中有助于增加数据量和利用图像的空间上下文信息
    """
    img_s, img_h, img_w = img.shape
    assert (img_s - size) % stride == 0
    patches_img_num = (img_s - size) // stride + 1
    print("Patches number of the image:{}".format(patches_img_num))

    patches = np.empty((patches_img_num, size, img_h, img_w), dtype=np.float32)

    for i in range(patches_img_num):
        patch = img[i * stride: i * stride + size]
        patches[i] = patch

    return patches


def random_crop_3d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random + crop_size[1], z_random:z_random + crop_size[2]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                 z_random:z_random + crop_size[2]]

    return crop_img, crop_label


def center_crop_3d(img, label, slice_num=16):
    if img.shape[0] < slice_num:
        return None
    left_x = img.shape[0] // 2 - slice_num // 2
    right_x = img.shape[0] // 2 + slice_num // 2

    crop_img = img[left_x:right_x]
    crop_label = label[left_x:right_x]
    return crop_img, crop_label


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list


def print_network(net):
    num_params = sum(param.numel() for param in net.parameters())
    print(net)
    print('Total number of parameters: %d' % num_params)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_v2(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def read_image(path: str, img_type: str) -> Tuple[sitk.Image, np.ndarray]:
    if img_type not in ['ct', 'seg']:
        raise ValueError("Invalid parameter value. Expected 'ct' or 'seg'.")
    img: sitk.Image = sitk.ReadImage(path, sitk.sitkInt16 if img_type == 'ct' else sitk.sitkInt8)
    np_array: np.ndarray = sitk.GetArrayFromImage(img)
    return img, np_array


# 灰度截断（Grayscale Clipping）或灰度阈值截断（Grayscale Threshold Clipping）
# 预处理方法的目标是去除图像中灰度值过高或过低的像素，以便增强图像的对比度和细节
def clip_array(np_array: np.ndarray, lower: int, upper: int) -> np.ndarray:
    np_array[np_array > upper] = upper
    np_array[np_array < lower] = lower
    return np_array
