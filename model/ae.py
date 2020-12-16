"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
The Y-shaped autoencoder model file.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, depth=3):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(depth, 8, kernel_size=3, stride=1, padding=1),  # in_channel, out_channel, -> 256, 256, 8
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),  # -> 128, 128, 8
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # -> 128, 128, 16
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # -> 64, 64, 16
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> 64, 64, 32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # -> 32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> 32, 32, 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # -> 16, 16, 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> 16, 16, 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # apply函数用法：https://blog.csdn.net/dss_dssssd/article/details/83990511
        self.encoder.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.5, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # 共享单元
        self.shared = nn.Sequential(
            # 转置卷积(反卷积)，padding指的是在输入shape每一边零填充的层数，output_padding输出的每一条边补充0的层数
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),  # 8, 8, 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16, 16, 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0),  # 16, 16, 32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 分割器
        self.segmenter = nn.Sequential(

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=0),  # 32, 32, 16
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64, 64, 16
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0),  # 64, 64, 8
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128, 128, 8
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1, output_padding=0),  # 128, 128, 2
            nn.Softmax(dim=1)
        )

        # 解码器
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=0),  # 32, 32, 16
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64, 64, 16
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0),  # 64, 64, 8
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128, 128, 8
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, depth, kernel_size=3, stride=1, padding=1, output_padding=0),  # 128, 128, 3
            nn.Tanh()
        )

        self.segmenter.apply(self.weights_init)
        self.decoder.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.5, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        latent = self.shared(x)
        seg = self.segmenter(latent)
        rect = self.decoder(latent)

        return seg, rect


class ActivationLoss(nn.Module):
    def __init__(self):
        super(ActivationLoss, self).__init__()

    # zero和one为隐向量的一半激活
    def forward(self, zero, one, labels):

        loss_act = torch.abs(one - labels.data) + torch.abs(zero - (1.0 - labels.data))
        return 1 / labels.shape[0] * loss_act.sum()


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, reconstruction, groundtruth):

        return self.loss(reconstruction, groundtruth.data)


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, segment, groundtruth):
        # view与resize功能类似
        return self.loss(segment.view(segment.shape[0], segment.shape[1], segment.shape[2] * segment.shape[3]), 
            groundtruth.data.view(groundtruth.shape[0], groundtruth.shape[1] * groundtruth.shape[2]))
