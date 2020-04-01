import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as nnu

import data
import selfattention as sa

class Discriminator(nn.Module):
    ngpu = 1
    nf = 64
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (data.CHANNELS) x 64 x 64
            nnu.spectral_norm(nn.Conv2d(
                in_channels = data.CHANNELS,
                out_channels = self.nf,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False
            )),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.nf) x 32 x 32
            nnu.spectral_norm(nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # sa.SelfAttention(self.nf * 2),
            # state size. (self.nf*2) x 16 x 16
            nnu.spectral_norm(nn.Conv2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.nf*4) x 8 x 8
            nnu.spectral_norm(nn.Conv2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.nf*8) x 4 x 4
            nnu.spectral_norm(nn.Conv2d(self.nf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, inpt):
        return self.main(inpt)

if __name__ == '__main__':
    import utils

    disc = Discriminator()
    print('Trainable parameters:', utils.count_parameters(disc))