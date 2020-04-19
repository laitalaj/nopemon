import os
from os import path

import imageio
import PIL
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils as tu
import torchvision.datasets as tvd
import torchvision.transforms as tvt
import torchvision.utils as tvu

import utils

SPRITE_LOCATIONS = (
    '3d/3D Battlers/Front',
    '3d/3D Battlers/Front/Female',
    '3d/3D Battlers/FrontShiny',
    '3d/3D Battlers/FrontShiny/Female',
)

IMAGE_SIZE = 64
CHANNELS = 3

DATA_ROOT = 'data'
DATA_DIR = path.join(DATA_ROOT, 'pokemon')

def iter_sprite_paths():
    for d in SPRITE_LOCATIONS:
        for p in map(lambda f: path.join(d, f), os.listdir(d)):
            if p.endswith('.png'): yield p

def load_sprite(p, keep_frames=10):
    img = imageio.imread(p)
    if img.shape[1] % img.shape[0] != 0:
        raise f'Non-square sprites for {p} - please get rid of it!'
    frame_count = img.shape[1] // img.shape[0]
    frames = np.array(np.split(img, frame_count, 1))
    keep_count = min(keep_frames, frame_count)
    keep = np.random.choice(np.arange(len(frames)), keep_count, False)
    return frames[keep]

def ensure_black_at_alpha0(sprite):
    alpha0 = sprite[:, :, 3] == 0
    sprite[alpha0] = [0, 0, 0, 0]

def save_sprite(i, sprite):
    fname = path.join(DATA_DIR, f'{i:05d}.png')
    imageio.imwrite(fname, sprite[:, :, :CHANNELS])

def generate_dataset(report_interval = 100):
    i = 0
    next_report = report_interval
    for j, p in enumerate(iter_sprite_paths()):
        if j == next_report:
            print(j)
            next_report += report_interval

        frames = load_sprite(p)
        for f in frames:
            ensure_black_at_alpha0(f)
            save_sprite(i, f)
            i += 1

def get_dataloader(batch_size = 64, workers=2):
    dataset = tvd.ImageFolder(
        root = DATA_ROOT,
        transform = tvt.Compose([
            tvt.RandomAffine(10, shear=10, resample=PIL.Image.BILINEAR, fillcolor=(0, 0, 0)),
            tvt.Resize(IMAGE_SIZE),
            tvt.CenterCrop(IMAGE_SIZE),
            tvt.ColorJitter(brightness=.1, contrast=.2, saturation=.3, hue=.4),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor(),
            tvt.Normalize((.5, .5, .5), (.5, .5, .5)),
        ])
    )
    return tu.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

if __name__ == '__main__':
    if not path.isdir(DATA_DIR): os.makedirs(DATA_DIR)
    if not os.listdir(DATA_DIR): generate_dataset()
    loader = get_dataloader()
    batch = next(iter(loader))
    device = utils.get_device()
    plt.figure(figsize = (8, 8))
    plt.axis('off')
    plt.imshow(
        utils.make_torch_grid(batch[0].to(device))
    )
    plt.savefig('out/data.png', bbox_inches='tight', pad_inches=0)
    plt.show()
