import torch
import torch.nn.functional as tnf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

import sagenerator
import utils

def load_sagen(name):
    device = utils.get_device()
    gen = utils.load_model(name, sagenerator.SelfAttentionGenerator, save_attention=True)
    return gen, device

def gen_sagen(gen, device):
    noise = torch.randn(1, gen.nz, device=device)
    result = gen(noise)
    attention_maps = gen.get_sa_module().saved_attention
    return result.detach(), attention_maps.detach()

def get_attention_map(attention_maps, x, y):
    x //= 2
    y //= 2

    j = y * 32 + x
    attention_map = attention_maps[0, :, j].view(1, 1, 32, 32)

    value_min = torch.min(attention_map)
    value_max = torch.max(attention_map)
    value_range = value_max - value_min
    attention_map = ((attention_map - value_min) / value_range) * 2 - 1

    attention_map = tnf.interpolate(attention_map, scale_factor=2, mode='bilinear', align_corners=True)
    return attention_map.repeat(1, 3, 1, 1)

def visualize_sagenerator(name, out=None, locations=[]):
    gen, device = load_sagen(name)

    result, attention_maps = gen_sagen(gen, device)
    maps_for_locations = []

    for l in locations:
        x, y = l
        maps_for_locations.append(get_attention_map(attention_maps, x, y))
    
    plt.figure(figsize=((len(locations) + 1) * 3, 3))
    plt.imshow(utils.make_torch_grid(torch.cat([result, *maps_for_locations])))

    colors = plt.cycler('color', plt.cm.Set1.colors)
    for i, data in enumerate(zip(colors, locations)):
        color, loc = data
        color = color['color']
        plt.plot(loc[0], loc[1], c=color, marker='o', markersize=10, markeredgecolor='w')
        plt.plot(loc[0] + 64 * (i + 1), loc[1], c=color, marker='o', markersize=10, markeredgecolor='w')
    plt.axis('off')
    if out:
        plt.savefig(out, bbox_inches='tight')
    plt.show()

def interactive_sagenerator(name):
    class SAGenContainer:
        def __init__(self, gen, device):
            self.gen = gen
            self.device = device
            self.generate()
        def generate(self):
            result, self.attention_maps = gen_sagen(self.gen, self.device)
            self.result = utils.im_to_numpy(result[0])
            self.attention = np.zeros((64, 64, 3))
        def select_attention(self, x, y):
            attention = get_attention_map(self.attention_maps, x, y)[0]
            self.attention = utils.im_to_numpy(attention)[:, :, 0]


    gen, device = load_sagen(name)
    container = SAGenContainer(gen, device)
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    ax1, ax2 = ax
    im = ax1.imshow(container.result)
    im_dot, = ax1.plot([], [], c='red', marker='o', markersize=10, markeredgecolor='w')
    att = ax2.imshow(container.attention, vmin=0, vmax=1)
    att_dot, = ax2.plot([], [], c='red', marker='o', markersize=10, markeredgecolor='w')

    def gen_new(event):
        container.generate()
        im.set_data(container.result)
        im_dot.set_data([], [])
        att.set_data(container.attention)
        att_dot.set_data([], [])
        plt.draw()

    def select_attention(event):
        if event.inaxes != ax1:
            return
        x, y = int(event.xdata), int(event.ydata)
        container.select_attention(x, y)
        att.set_data(container.attention)
        im_dot.set_data(x, y)
        att_dot.set_data(x, y)
        plt.draw()

    axgen = plt.axes([0.75, 0.10, 0.15, 0.06])
    bgen = widgets.Button(axgen, 'Generate new')
    bgen.on_clicked(gen_new)

    fig.canvas.mpl_connect('button_press_event', select_attention)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()

if __name__ == '__main__':
    name = '30epoch/sa_gen.pt'
    static = False

    if static:
        seeds = (42, 10)
        locations = [
            [(29, 12), (57, 14), (46, 42), (10, 10)],
            [(23, 52), (22, 30), (43, 14), (55, 55)],
        ]
        for seed, locs in zip(seeds, locations):
            torch.manual_seed(seed)
            visualize_sagenerator(name, f'out/sa_{seed}_visualized.png', locations=locs)
    else:
        torch.manual_seed(3)
        interactive_sagenerator(name)
