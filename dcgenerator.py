import numpy as np

import torch
import torch.nn as nn

import data
import utils

class DCNNGenerator(nn.Module):
    '''
    Based on
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    '''
    nz = 128
    nf = 64

    def __init__(self):
        super(DCNNGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.nf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.nf * 8),
            nn.ReLU(True),
            # state size. (self.nf*8) x 4 x 4
            nn.ConvTranspose2d(self.nf * 8, self.nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU(True),
            # state size. (self.nf*4) x 8 x 8
            nn.ConvTranspose2d(self.nf * 4, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.ReLU(True),
            # state size. (self.nf*2) x 16 x 16
            nn.ConvTranspose2d(self.nf * 2, self.nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(True),
            # state size. (self.nf) x 32 x 32
            nn.ConvTranspose2d(self.nf, data.CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def parameters_for_optimizer(self):
        return self.parameters()
    def forward(self, x):
        x = x[:, :, None, None]
        return self.main(x)

if __name__ == '__main__':
    gen = DCNNGenerator()
    print('Trainable parameters:', utils.count_parameters(gen))