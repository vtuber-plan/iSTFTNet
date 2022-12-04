

from functools import reduce
import operator
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from ..commons import init_weights, get_padding

LRELU_SLOPE = 0.1

class TransformerGenerator(torch.nn.Module):
    def __init__(self, istft_n_fft: int,
                    initial_channel: int):
        super(TransformerGenerator, self).__init__()
        self.istft_n_fft = istft_n_fft
        self.initial_channel = initial_channel
        self.fft_channel = istft_n_fft // 2 + 1
        self.n_head = 8
        self.inter_channel = self.fft_channel // 2 // self.n_head * self.n_head

        self.spec_conv_pre = Conv1d(in_channels=initial_channel, out_channels=self.inter_channel, kernel_size=7, stride=1, padding=3)
        self.spec_encoder_layer = nn.TransformerEncoderLayer(d_model=self.inter_channel, nhead=self.n_head, batch_first=True, activation="gelu")
        self.spec_encoder = nn.TransformerEncoder(self.spec_encoder_layer, num_layers=6)
        self.spec_conv_post = Conv1d(in_channels=self.inter_channel, out_channels=self.fft_channel, kernel_size=7, stride=1, padding=3)
        
        self.phase_conv_pre = Conv1d(in_channels=initial_channel, out_channels=self.inter_channel, kernel_size=7, stride=1, padding=3)
        self.phase_encoder_layer = nn.TransformerEncoderLayer(d_model=self.inter_channel, nhead=self.n_head, batch_first=True, activation="gelu")
        self.phase_encoder = nn.TransformerEncoder(self.phase_encoder_layer, num_layers=6)
        self.phase_conv_post = Conv1d(in_channels=self.inter_channel, out_channels=self.fft_channel, kernel_size=7, stride=1, padding=3)
        
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

    def forward(self, x):
        x = self.reflection_pad(x)

        spec_out = self.spec_conv_pre(x)
        spec_out = self.spec_encoder(spec_out.transpose(1,2)).transpose(1,2)
        spec_out = self.spec_conv_post(spec_out)

        phase_out = self.phase_conv_pre(x)
        phase_out = self.phase_encoder(phase_out.transpose(1,2)).transpose(1,2)
        phase_out = self.phase_conv_post(phase_out)
        
        # spec = torch.exp(spec_out)
        # phase = torch.sin(phase_out)

        spec = torch.exp(spec_out)
        phase = torch.tanh(phase_out) * 2 * np.pi

        return spec, phase

