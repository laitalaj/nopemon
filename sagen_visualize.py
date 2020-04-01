import torch
import torch.nn.functional as tnf
import matplotlib.pyplot as plt

import sagenerator
import utils

def visualize_sagenerator(name, out=None, locations=[]):
    device = utils.get_device()
    gen = utils.load_model(name, sagenerator.SelfAttentionGenerator, save_attention=True)

    noise = torch.randn(1, gen.nz, device=device)
    result = gen(noise)

    attention_maps = gen.get_sa_module().saved_attention
    maps_for_locations = []

    for l in locations:
        x, y = l
        x //= 2
        y //= 2

        j = y * 32 + x
        attention_map = attention_maps[0, :, j].view(1, 1, 32, 32)

        value_min = torch.min(attention_map)
        value_max = torch.max(attention_map)
        value_range = value_max - value_min
        attention_map = ((attention_map - value_min) / value_range) * 2 - 1

        attention_map = tnf.interpolate(attention_map, scale_factor=2, mode='bilinear', align_corners=True)
        maps_for_locations.append(attention_map.repeat(1, 3, 1, 1))
    
    plt.figure(figsize=((len(locations) + 1) * 3, 3))
    plt.imshow(utils.make_torch_grid(torch.cat([result, *maps_for_locations]).detach()))

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

if __name__ == '__main__':
    seeds = (42, 10)
    locations = [
        [(29, 12), (57, 14), (46, 42), (10, 10)],
        [(23, 52), (22, 30), (43, 14), (55, 55)],
    ]
    for seed, locs in zip(seeds, locations):
        torch.manual_seed(seed)
        visualize_sagenerator('20epoch/sa_gen.pt', f'out/sa_{seed}_visualized.png', locations=locs)
