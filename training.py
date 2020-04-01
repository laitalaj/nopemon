from itertools import count

import torch
import torch.nn as nn
import torch.optim as to

import data
import utils

REAL_LABEL = .9
FAKE_LABEL = 0

def train_discriminator(gen, disc, batch, loss_func, d_optimizer):
    device = utils.get_device()

    # Train discriminator w/ all-real batch
    disc.zero_grad()
    device_batch = batch[0].to(device)
    batch_size = device_batch.size(0)
    labels = torch.full((batch_size,), REAL_LABEL, device=device)

    output = disc(device_batch).view(-1)

    loss_real = loss_func(output, labels)
    loss_real.backward()
    acc_real = output.mean().item()

    # Train discriminator w/ all-fake batch
    noise = torch.randn(batch_size, gen.nz, device=device)
    fake_batch = gen(noise)
    labels.fill_(FAKE_LABEL)

    output = disc(fake_batch.detach()).view(-1)

    loss_fake = loss_func(output, labels)
    loss_fake.backward()
    acc_fake = output.mean().item()

    loss_disc = loss_real + loss_fake

    # Update discriminator
    d_optimizer.step()

    return loss_disc, acc_real, acc_fake

def train_generator(gen, disc, loss_func, g_optimizer, batch_size = 64):
    device = utils.get_device()

    # Train generator
    noise = torch.randn(batch_size, gen.nz, device=device)
    fake_batch = gen(noise)

    gen.zero_grad()

    labels = torch.full((batch_size,), REAL_LABEL, device=device)

    output = disc(fake_batch).view(-1)

    loss_gen = loss_func(output, labels)
    loss_gen.backward()
    acc_gen = output.mean().item()

    # Update generator
    g_optimizer.step()

    return loss_gen, acc_gen

def train(name, gen, disc, gen_train_ratio = 5, epochs=-1):
    device = utils.get_device()
    dataloader = data.get_dataloader()

    loss_func = nn.BCELoss()
    g_optimizer = to.Adam(gen.parameters(), lr=.0003, betas=(.5, .9))
    d_optimizer = to.Adam(disc.parameters(), lr=.0003, betas=(.5, .9))

    iter_axis = []
    g_loss_axis = []
    d_loss_axis = []

    fixed_noise = torch.randn(64, gen.nz, device=device)
    epoch_iterator = range(epochs) if epochs >= 0 else count(0)
    iters = 0
    for epoch in epoch_iterator:
        for i, batch in enumerate(dataloader):
            loss_disc, acc_real, acc_fake = train_discriminator(gen, disc, batch, loss_func, d_optimizer)

            for _ in range(gen_train_ratio):
                loss_gen, acc_gen = train_generator(gen, disc, loss_func, g_optimizer)

            # Training stats
            if i % 50 == 0:
                print(
                    f'[{epoch:2d}/{epochs}][{i:3d}/{len(dataloader)}]\t' +\
                     f'Loss_D: {loss_disc:3.4f}\tLoss_G: {loss_gen:3.4f}\t' +\
                     f'D(x): {acc_real:3.4f}\tD(G(z)): {acc_fake:3.4f} / {acc_gen:3.4f}'
                )
            
            if (iters % 10 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                iter_axis.append(iters)
                g_loss_axis.append(loss_gen)
                d_loss_axis.append(loss_disc)

            # Save output
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                print('Saving images...')
                with torch.no_grad():
                    gen.mixing = False
                    fake = gen(fixed_noise).detach().cpu()
                    gen.mixing = True
                    utils.save_image(utils.make_torch_grid(fake), f'{name}_{iters}.png')
            
            # Save weights
            if (iters % 1000 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                print('Saving weights...')
                utils.save_weights(gen, f'{name}_gen.pt')
                utils.save_weights(disc, f'{name}_disc.pt')
            
            iters += 1
    
    utils.plot_losses(iter_axis, g_loss_axis, d_loss_axis)

if __name__ == '__main__':
    import discriminator
    import dcgenerator
    import sagenerator
    import stylegenerator

    device = utils.get_device()

    disc = discriminator.Discriminator().to(device)

    name, gen = 'dcnn', dcgenerator.DCNNGenerator().to(device)
    #name, gen = 'sa', sagenerator.SelfAttentionGenerator().to(device)
    #name, gen = 'style', stylegenerator.StyleGenerator().to(device)

    train(name, gen, disc, gen_train_ratio = 1, epochs=30)