import torch
import torch.nn as nn
import torch.nn.utils as nnu

import data
import selfattention as sa

class SelfAttentionGenerator(nn.Module):
    nz = 128
    nf = 64
    def __init__(self, save_attention = False):
        super(SelfAttentionGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nnu.spectral_norm(nn.ConvTranspose2d(self.nz, self.nf * 8, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(self.nf * 8),
            nn.ReLU(True),
            # state size. (self.nf*8) x 4 x 4
            nnu.spectral_norm(nn.ConvTranspose2d(self.nf * 8, self.nf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU(True),
            # state size. (self.nf*4) x 8 x 8
            nnu.spectral_norm(nn.ConvTranspose2d(self.nf * 4, self.nf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.nf * 2),
            nn.ReLU(True),
            # state size. (self.nf*2) x 16 x 16
            nnu.spectral_norm(nn.ConvTranspose2d(self.nf * 2, self.nf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(True),
            sa.SelfAttention(self.nf, save_attention=save_attention),
            # state size. (self.nf) x 32 x 32
            nnu.spectral_norm(nn.ConvTranspose2d(self.nf, data.CHANNELS, 4, 2, 1, bias=False)),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def parameters_for_optimizer(self):
        return self.parameters()
    def get_sa_module(self):
        return self.main[-3]
    def forward(self, x):
        x = x[:, :, None, None]
        return self.main(x)

if __name__ == '__main__':
    import utils
    gen = SelfAttentionGenerator()
    print('Trainable parameters:', utils.count_parameters(gen))