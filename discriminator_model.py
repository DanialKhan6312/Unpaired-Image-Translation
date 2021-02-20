# -*- coding: utf-8 -*-
"""Discriminator model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fSmXSfGOijUOIYIMlRFRteYrlo0g356V
"""

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

class Discriminator(nn.Module):
  def __init__(self, channels_img, features_d):
    super(Discriminator, self).__init__()
    self.disc = nn.Sequential(
        # input is 3x256x256
        nn.Conv2d(
            channels_img, features_d, kernel_size=4, stride=2, padding=1
        ), # 128x128
        nn.LeakyReLU(0.2),
        self._block(feautres_d, features_d*2, 4, 2, 1), # 64x64
        self._block(feautres_d*2, features_d*4, 4, 2, 1), # 32x32
        self._block(feautres_d*4, features_d*8, 4, 2, 1), # 16x16
        self._block(feautres_d*8, features_d*16, 4, 2, 1), # 8x8
        self._block(feautres_d*16, features_d*32, 4, 2, 1), # 4x4
        nn.Conv2d(features_d*32, 1, kernel_size=4, stride=2, padding=0),
        nn.Sigmoid()
    )
  def block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )

  def forward(self, x):
    return self.disc(x)