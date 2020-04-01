import torch
import torch.nn as nn
import torch.nn.utils as nnu

class SelfAttention(nn.Module):
    '''
    Based on
    https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    '''
    def __init__(self, in_dim, k = 8, save_attention = False):
        super(SelfAttention, self).__init__()
        self.indim = in_dim
        self.k = k

        self.f = nnu.spectral_norm(nn.Conv2d(in_dim, in_dim // k, 1, bias=False))
        self.g = nnu.spectral_norm(nn.Conv2d(in_dim, in_dim // k, 1, bias=False))

        self.softmax = nn.Softmax(-1)

        self.h = nnu.spectral_norm(nn.Conv2d(in_dim, in_dim // k, 1, bias=False))
        self.v = nnu.spectral_norm(nn.Conv2d(in_dim // k, in_dim, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.save_attention = save_attention
    def forward(self, x):
        batchsize, channels, w, h = x.size()
        fx = self.f(x).view(batchsize, channels // self.k, w * h).permute(0, 2, 1) # Transposed
        gx = self.g(x).view(batchsize, channels // self.k, w * h)
        fxgx = torch.bmm(fx, gx)
        attention = self.softmax(fxgx)
        if self.save_attention:
            self.saved_attention = attention

        hx = self.h(x).view(batchsize, channels // self.k, w * h)
        attention_hx = torch.bmm(hx, attention).view(batchsize, channels // self.k, w, h)

        unscaled_result = self.v(attention_hx)
        result = self.gamma * unscaled_result + x
        return result