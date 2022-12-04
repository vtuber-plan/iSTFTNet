

from functools import reduce
import operator
from typing import List, Union
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from ..commons import init_weights, get_padding

import numpy as np

LRELU_SLOPE = 0.1

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5, 7)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList()
        for dilation_size in dilation:
            conv_kernel = Conv1d(channels, channels, kernel_size, 1, dilation=dilation_size, padding=get_padding(kernel_size, dilation_size))
            self.convs1.append(conv_kernel)
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList()
        for dilation_size in dilation:
            conv_kernel = Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
            self.convs2.append(conv_kernel)
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

class ResizeConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResizeConv1d, self).__init__()
        self.stride = stride
        self.conv = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same')

    def forward(self, x):
        interpolated_x = torch.nn.functional.interpolate(x, size=(self.stride * x.shape[2],), mode='linear', align_corners=True)
        return self.conv(interpolated_x)

class iSTFTNetGenerator(torch.nn.Module):
    def __init__(self, istft_n_fft: int,
                    initial_channel: int,
                    resblock_kernel_sizes: List[int],
                    resblock_dilation_sizes: List[int],
                    upsample_rates: List[int],
                    upsample_initial_channel: int,
                    upsample_kernel_sizes: List[int]):
        super(iSTFTNetGenerator, self).__init__()
        self.istft_n_fft = istft_n_fft
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(in_channels=initial_channel, out_channels=upsample_initial_channel, kernel_size=7, stride=1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(in_channels=upsample_initial_channel//(2**i),
                            out_channels=upsample_initial_channel//(2**(i+1)),
                            kernel_size=k,
                            stride=u,
                            padding=(k-u)//2)
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(channels=ch, kernel_size=k, dilation=d))

        self.conv_post = Conv1d(ch, self.istft_n_fft + 2, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = []
            for j in range(self.num_kernels):
                xs.append(self.resblocks[i * self.num_kernels + j](x))
            x = reduce(operator.add, xs) / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.istft_n_fft // 2 + 1, :])
        phase = torch.tanh(x[:, self.istft_n_fft // 2 + 1:, :]) * 2 * np.pi # (-pi, pi)

        return spec, phase

