# nopemon

Generating Pokémon with various GAN generator architectures,
using pytorch.
Part of the Seminar on Computer Vision,
University of Helsinki,
2020.

## What's this?

I got Generative Adversarial Networks as the subject for my seminar report,
and I decided to pit two advanced GAN generator architectures agains each other,
using a tried-and-true DCNN as the baseline.
The seminar report,
with a bunch of images, explanation and results,
is available [here](http://julius.laita.la/pdfs/nopemon).

## Disclaimer

The code is still quite messy.
I'll try to get around to cleaning it a bit soon!

## Getting started

1. Make sure you have Python 3 installed
2. Install requirements: `pip3 install -r requirements.txt`
3. Check that everything works: `python3 ./discriminator.py`
    * This should result in `Trainable parameters: 2765568`
4. You're good to go!

## Data

The raw data is expected to consist of
* .png -format images with
* one row of rectangular frames each

Personally, I used [this dataset](https://www.pokecommunity.com/showthread.php?t=416344),
and the folders in `data.py` are set up so that this dataset will work if extracted under
`3d/3D Battlers` in the project directory.
Alternatively, you can change `SPRITE_LOCATIONS` to load data from elsewhere.

To generate a dataset for training, just make sure that you have valid sprites in `SPRITE_LOCATIONS`
and run `data.py`.
This will split the input sprites,
pick a given number of frames from them and save the frames under `data/pokemon`
(where PyTorch expects it to be).

## Training

To train the models,
just comment out the relevant rows under `if __name__ == '__main__'` in `training.py`
and run it.
(I know, it's a bit of a mess, sorry!)

## Visualization

This repo includes some nice interactive visualizations for SAGAN and StyleGAN.
To see those, run `sagen_visualize.py` or `stylegen_visualize.py`, respectively.
