import torch
import matplotlib.pyplot as plt
from itertools import product

import stylegenerator
import utils

def visualize_stylegenerator(name, out=None, latents=8):
    device = utils.get_device()
    gen = utils.load_model(name, stylegenerator.StyleGenerator)

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

if __name__ == '__main__':
    torch.manual_seed(69)
    visualize_stylegenerator('20epoch/style_gen.pt', 'out/style_visualized.png')