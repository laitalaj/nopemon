import torch
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from itertools import product

import stylegenerator
import utils

def load_stylegen(name):
    device = utils.get_device()
    gen = utils.load_model(name, stylegenerator.StyleGenerator)
    return gen, device

def visualize_stylegenerator(name, out=None, latents=8):
    gen, device = load_stylegen(name)

    noises = [torch.randn(1, gen.nz, device=device) for _ in range(latents)]
    results = []
    for a, b in product(noises, repeat=2):
        results.append(gen(a, b, 4).detach())

    plt.figure(figsize=(8, 8))
    plt.imshow(utils.make_torch_grid(torch.cat(results)))
    plt.axis('off')
    if out:
        plt.savefig(out, bbox_inches='tight')
    plt.show()

def interactive_stylegen(name):
    initial_cutoff = 5

    class StyleGenContainer:
        def __init__(self, gen, device, cutoff):
            self.gen = gen
            self.device = device
            self.gen_lower_styles(False)
            self.gen_upper_styles(False)
            self.set_cutoff(cutoff)
        def gen_lower_styles(self, generate=True):
            self.a = torch.randn(1, self.gen.nz, device=self.device)
            if generate: self.generate()
        def gen_upper_styles(self, generate=True):
            self.b = torch.randn(1, self.gen.nz, device=self.device)
            if generate: self.generate()
        def set_cutoff(self, cutoff, generate=True):
            self.cutoff = cutoff
            if generate: self.generate()
        def generate(self):
            result = self.gen(self.a, self.b, self.cutoff).detach()[0]
            offset = result.min()
            scale = result.max() - offset
            self.result = utils.im_to_numpy(result, -offset, scale)

    gen, device = load_stylegen(name)
    container = StyleGenContainer(gen, device, initial_cutoff)

    plt.figure(figsize=(9, 9))
    im = plt.imshow(container.result)

    axgenl = plt.axes([0.20, 0.01, 0.20, 0.06])
    bgenl = widgets.Button(axgenl, 'New lower styles')
    axgenu = plt.axes([0.55, 0.01, 0.20, 0.06])
    bgenu = widgets.Button(axgenu, 'New upper styles')
    axslider = plt.axes([0.20, 0.90, 0.65, 0.03])
    cutoffslider = widgets.Slider(axslider, 'Style cutoff', 1, 9, valinit=initial_cutoff, valstep=1)

    def gen_lower(event):
        container.gen_lower_styles()
        im.set_data(container.result)
        plt.draw()

    def gen_upper(event):
        container.gen_upper_styles()
        im.set_data(container.result)
        plt.draw()

    def set_cutoff(event):
        container.set_cutoff(int(cutoffslider.val))
        im.set_data(container.result)
        plt.draw()

    bgenl.on_clicked(gen_lower)
    bgenu.on_clicked(gen_upper)
    cutoffslider.on_changed(set_cutoff)

    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    name = '20epoch/style_gen.pt'
    static = False

    if static:
        torch.manual_seed(69)
        visualize_stylegenerator(name, 'out/style_visualized.png')
    else:
        interactive_stylegen(name)