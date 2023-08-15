"""
Class Name: ResUNetV2
Description: 
Author: DDB
Created: 2023-08-13
Version: 3DUNet v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __int__(self, in_channels, out_channels):
        super(ResidualBlock, self).__int__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x


class ResUNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResUNet3D, self).__init__()
        self.encoder = nn.ModuleList([
            ResidualBlock(in_channels, 64),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock(64, 128),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock(128, 256),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock(256, 512),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock(512, 1024),
            nn.MaxPool3d(kernel_size=2, stride=2),
        ])

        self.decoder = nn.ModuleList([
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        ])

        self.final_conv = nn.Conv3d(64,num_classes,kernel_size=1)

    def forward(self,x):
        encoder_outputs = []

        for block in self.encoder:
            x = block(x)
            encoder_outputs.append(x)

        for i, block in enumerate(self.decoder):
            x = block(x)
            x = torch.cat([x,])

class ResUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super().__init__()

        self.training = training
        self.drop_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, padding=5, dilation=5),
            nn.ReLU(inplace=True),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, out_channel, kernel_size=1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, kernel_size=1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, kernel_size=1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),

            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, kernel_size=1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.drop_rate, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.drop_rate, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.drop_rate, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4
