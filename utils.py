import os

import numpy as np
import imageio
import skimage as ski
import matplotlib.pyplot as plt

import torch
import torch.autograd as ta
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision.utils as tvu

OUTPUT_DIR = 'out'
MODEL_DIR = 'models'

EPSILON = 1e-12

DEVICE = None

def get_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return DEVICE

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def interpolate(x1, x2):
    batch_size = x1.shape[0]
    u = torch.rand(batch_size, 1, 1, 1, device=get_device())
    return x1 * u + x2 * (1 - u)

def gradient_penalty(disc, real, fake, weight = 10):
    interpolated = interpolate(real, fake)
    interpolated.requires_grad_()
    results = disc(interpolated)

    grad = ta.grad(results, interpolated,
        grad_outputs=torch.ones(results.size(), device=get_device()),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad = grad.view(grad.shape[0], -1) # flatten
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + EPSILON)
    return weight * ((grad_norm - 1) ** 2).mean()

def disc_logistic_loss(real_out, fake_out):
    return (tnf.softplus(fake_out) + tnf.softplus(-real_out)).mean()

def gen_logistic_ns_loss(fake_out):
    return tnf.softplus(-fake_out).mean()

def disc_r1_regularizer(real_in, real_out, gamma=10):
    grads = ta.grad(torch.sum(real_out), real_in, only_inputs=True, retain_graph=True)[0]
    penalty = (grads ** 2).sum(dim=(1, 2, 3))
    return .5 * gamma * penalty.mean()

def im_to_numpy(im, offset = 1, scale = 2):
    return (np.transpose(im.cpu(), (1, 2, 0)) + offset) / scale

def make_torch_grid(imgs):
    return np.transpose(
        tvu.make_grid(imgs, padding=1, normalize=True).cpu(),
        (1, 2, 0)
    )

def save_image(img, name):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    imageio.imwrite(
        os.path.join(OUTPUT_DIR, name), ski.img_as_ubyte(img)
    )

def save_weights(model, name):
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, name))

def load_model(name, model_class, *model_args, **model_kwargs):
    weights = torch.load(os.path.join(MODEL_DIR, name), map_location=get_device())
    model = model_class(*model_args, **model_kwargs).to(get_device())
    model.load_state_dict(weights)
    return model

def plot_losses(iter_axis, gen_axis, disc_axis):
    plt.figure(figsize=(10, 8))
    plt.plot(iter_axis, gen_axis, label='Generator', alpha=.75)
    plt.plot(iter_axis, disc_axis, label='Discriminator', alpha=.75)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
