import torch
import torch.nn as nn
import torch.nn.functional as tnf

import utils


'''
https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py
and https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py
were very helpful when building these
'''

class ExponentialMovingAverage: # TODO: Maybe probably start using
    def __init__(self, beta):
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + new * (1 - self.beta)

class MappingNetwork(nn.Module):
    def __init__(self, nz, depth = 6):
        super(MappingNetwork, self).__init__()
        # TODO: Normalize?
        self.nz = nz
        self.depth = depth
        layers = []
        for _ in range(self.depth):
            layers.append(nn.Linear(self.nz, self.nz))
            layers.append(nn.LeakyReLU(.2))
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)

class ModulatedConv2D(nn.Module):
    def __init__(self, in_chan, out_chan, style_dim, kernel, demodulate = True):
        super(ModulatedConv2D, self).__init__()
        self.out_chan = out_chan
        self.demodulate = demodulate
        self.kernel = kernel

        self.style_converter = nn.Linear(style_dim, in_chan)
        self.style_converter.bias = nn.Parameter(torch.ones(in_chan)) # Bias should be initially 1

        self.weight = nn.Parameter(torch.empty((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x, w):
        batch_size, channels, width, height = x.shape

        style = self.style_converter(w)

        s = style[:, None, :, None, None]
        weights = self.weight[None, :, :, :, :] # Add batch dimension
        weights = s * weights

        if self.demodulate:
            discriminator = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdims=True) + utils.EPSILON)
            weights = weights * discriminator

        x = x.reshape(1, -1, width, height)
        _, _, *ws = weights.shape
        weights = weights.reshape(batch_size * self.out_chan, *ws)

        x = tnf.conv2d(x, weights, groups=batch_size, padding=1 if self.kernel > 1 else 0)
        x = x.reshape(-1, self.out_chan, width, height)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, output_channels, upsample=False, noise=True):
        super(GeneratorBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else None
        self.noise = noise

        self.conv = ModulatedConv2D(input_channels, output_channels, latent_dim, 3)

        self.noise_strength = nn.Parameter(torch.zeros(1))
        #TODO: self.bias = nn.Parameter(torch.zeros(input_channels))
        self.activation = nn.LeakyReLU(.2)
    def forward(self, x, w):
        if self.upsample is not None: x = self.upsample(x)

        batch_size, channels, width, height = x.shape

        x = self.conv(x, w)

        x = x + self.noise_strength * torch.randn((batch_size, 1, width, height), device=utils.get_device())
        return self.activation(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, in_chan):
        super(RGBBlock, self).__init__()
        self.main = ModulatedConv2D(in_chan, 3, latent_dim, 1, demodulate=False)
        # TODO: We might want bias here too! Maybe as a parameter for ModulatedConv2D?
    def forward(self, x, w):
        return self.main(x, w)


class SynthesisNetwork(nn.Module):
    nf = 64
    def __init__(self, nz):
        super(SynthesisNetwork, self).__init__()
        self.nz = nz

        self.mapper = MappingNetwork(self.nz)
        self.const = nn.Parameter(torch.randn((self.nf, 4, 4))) # TODO: Revisit this

        self.blocks = nn.ModuleList([
            GeneratorBlock(self.nz, self.nf, 512), # 4x4
            GeneratorBlock(self.nz, 512, 256, upsample=True), # 8x8
            GeneratorBlock(self.nz, 256, 256),
            GeneratorBlock(self.nz, 256, 128, upsample=True), # 16x16
            GeneratorBlock(self.nz, 128, 128),
            GeneratorBlock(self.nz, 128, 64, upsample=True), # 32x32
            GeneratorBlock(self.nz, 64, 64),
            GeneratorBlock(self.nz, 64, 32, upsample=True), # 64x64
            GeneratorBlock(self.nz, 32, 32),
        ])

        self.torgb = RGBBlock(self.nz, 32)
    
    def forward(self, styles):
        batch_size = styles.shape[0]

        x = self.const.expand(batch_size, -1, -1, -1)
        styles = styles.transpose(0, 1) # TODO: Needed?

        for block, style in zip(self.blocks, styles):
            x = block(x, style)
        
        rgb = self.torgb(x, styles[-1])
        return rgb

class StyleGenerator(nn.Module):
    nz = 256
    mix_prob = .9
    def __init__(self):
        super(StyleGenerator, self).__init__()
        self.mapping = MappingNetwork(self.nz)
        self.synthesis = SynthesisNetwork(self.nz)
        self.mixing = True
    def parameters_for_optimizer(self):
        return self.parameters()
        '''
        TODO: Slow down mapping network learning rate?
        return [
            {'params': self.synthesis.parameters()},
            {'params': self.mapping.parameters(), 'lr': 0.00001},
        ]
        '''
    def forward(self, x, mixed_x = None, mixing_cutoff = None):
        # TODO: Truncation trick, EMA stuff
        batch_size = x.shape[0]
        styles = self.mapping(x)
        if self.mixing and torch.rand(()) < self.mix_prob:
            mixed_latent = torch.randn(batch_size, self.nz, device=utils.get_device()) if mixed_x is None else mixed_x
            cutoff = int(torch.rand(()).numpy() * 10) if mixing_cutoff is None else mixing_cutoff
            mixed_styles = self.mapping(mixed_latent)[:, None, :].expand(-1, 10 - cutoff, -1)
            styles = torch.cat((styles[:, None, :].expand(-1, cutoff, -1), mixed_styles), dim=1)
        else:
            styles = styles[:, None, :].expand(-1, 10, -1)
        imgs = self.synthesis(styles)
        return imgs

if __name__ == '__main__':
    gen = StyleGenerator()
    print('Trainable parameters:', utils.count_parameters(gen))