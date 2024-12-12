#!/usr/bin/env python3
#
# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed and modified from MP-SENet,
# public available at https://github.com/yxlu-0102/MP-SENet

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubPixelConvTranspose2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 3),
                 stride=(1, 2),
                 padding=(0, 1)):
        super(SubPixelConvTranspose2d, self).__init__()
        self.upscale_width_factor = stride[1]
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels * self.upscale_width_factor,
            kernel_size=kernel_size,
            padding=padding)  # only change the width

    def forward(self, x):
        b, c, t, f = x.size()
        x = self.conv1(x)
        x = x.view(b, c, self.upscale_width_factor, t, f).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(b, c, t, f * self.upscale_width_factor)
        return x


class DenseBlockV2(nn.Module):
    """
    A denseblock for ZipEnhancer
    """

    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super(DenseBlockV2, self).__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2**i
            pad_length = kernel_size[0] + (dil - 1) * (kernel_size[0] - 1) - 1
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(
                    h.dense_channel * (i + 1),
                    h.dense_channel,
                    kernel_size,
                    dilation=(dil, 1)),
                # nn.Conv2d(h.dense_channel * (i + 1), h.dense_channel, kernel_size, dilation=(dil, 1),
                #           padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel))
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for block in self.dense_block[:self.depth]:
            x = block(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):

    def __init__(self, h, in_channel):
        """
        Initialize the DenseEncoder module.

        Args:
        h (object): Configuration object containing various hyperparameters and settings.
        in_channel (int): Number of input channels. Example: mag + phase: 2 channels
        """
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

        self.dense_block = DenseBlockV2(h, depth=4)

        encoder_pad_kersize = (0, 1)
        # Here pad was originally (0, 0)，now change to (0, 1)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(
                h.dense_channel,
                h.dense_channel, (1, 3), (1, 2),
                padding=encoder_pad_kersize),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        return self.dense_conv_2(self.dense_block(self.dense_conv_1(x)))


class BaseDecoder(nn.Module):

    def __init__(self, h):
        """
        Initialize the BaseDecoder module.

        Args:
        h (object): Configuration object containing various hyperparameters and settings.
        including upsample_type, dense_block_type.
        """
        super(BaseDecoder, self).__init__()

        self.upsample_module_class = SubPixelConvTranspose2d

        # for both mag and phase decoder
        self.dense_block = DenseBlockV2(h, depth=4)


class MappingDecoder(BaseDecoder):

    def __init__(self, h, out_channel=1):
        """
        Initialize the MappingDecoderV3 module.

        Args:
        h (object): Configuration object containing various hyperparameters and settings.
        out_channel (int): Number of output channels. Default is 1. The number of output spearkers.
        """
        super(MappingDecoder, self).__init__(h)
        decoder_final_kersize = (1, 2)

        self.mask_conv = nn.Sequential(
            self.upsample_module_class(h.dense_channel, h.dense_channel,
                                       (1, 3), (1, 2)),
            # nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, decoder_final_kersize))
        # Upsample at F dimension

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.mask_conv(self.dense_block(x)))


class PhaseDecoder(BaseDecoder):

    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__(h)

        # now change to (1, 2), previous (1, 1)
        decoder_final_kersize = (1, 2)

        self.phase_conv = nn.Sequential(
            self.upsample_module_class(h.dense_channel, h.dense_channel,
                                       (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel,
                                      decoder_final_kersize)
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel,
                                      decoder_final_kersize)

    def forward(self, x):
        x = self.phase_conv(self.dense_block(x))
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))
