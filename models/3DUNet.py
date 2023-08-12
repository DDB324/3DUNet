"""
Class Name: 3DUNet
Description: The implementation of 3DUNet
Author: DDB
Created: 2023-08-12
Version: 3DUNet v1
"""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up_conv_block = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.up_conv_block(x)
        x = torch.cat([x, skip], dim=1)
        return x


def test_batchnorm():
    # 创建一个随机输入数据
    input_data = torch.randn(2, 3, 32, 128, 128)

    # 创建一个BatchNorm3d层
    batch_norm_layer = nn.BatchNorm3d(num_features=3)

    # 经过BatchNorm3d处理的数据
    output_data = batch_norm_layer(input_data)

    # 打印处理前的第一个样本的均值和标准差
    print("处理前的均值:", torch.mean(input_data[0]))
    print("处理前的标准差:", torch.std(input_data[0]))

    # 打印处理后的第一个样本的均值和标准差
    print("处理后的均值:", torch.mean(output_data[0]))
    print("处理后的标准差:", torch.std(output_data[0]))


def test_conv3d_dilation():
    # 创建一个随机输入数据
    input_data = torch.randn(2, 3, 32, 32, 32)  # 输入尺寸为 (batch_size, channels, depth, height, width)

    # 创建一个Conv3d层
    conv_layer = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

    # 创建一个Conv3d层，应用扩张卷积
    conv_layer_dilation = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, dilation=2, padding=1)

    # 经过Conv3d层处理的数据
    output_data = conv_layer(input_data)
    output_data_dilation = conv_layer_dilation(input_data)

    # 打印输出数据的形状
    print("输入数据尺寸:", input_data.shape)
    print("输出数据尺寸:", output_data.shape)
    print("输出数据尺寸dilation:", output_data.shape)


if __name__ == '__main__':
    test_conv3d_dilation()
