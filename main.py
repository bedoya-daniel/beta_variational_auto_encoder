#-*-encoding:UTF-8-*-
"""
Beta Variational Auto-Encoder
Derived from Pytorch tutorial at
https://github.com/yunjey/pytorch-tutorial
"""
#%% Librairies
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms

from framework import modVAE
from framework.utils import to_var

from toyDataset import dataset as dts
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np

import librosa
#%% PARAMETERS
# Parameters, dataset
N_FFT = 1024
N_EXAMPLES = 100
#LEN_EXAMPLES = 38400
LEN_EXAMPLES = 128000
# Net parameters
Z_DIM, H_DIM = 5, 100

#%% Importing DATASET
# Creating dataset
DATASET = dts.toyDataset(batchSize=N_EXAMPLES,length_sample=LEN_EXAMPLES, n_fft=N_FFT)
_ = DATASET.get_minibatch()

# dataset = DATASET.get_minibatch()
DATA_LOADER = torch.utils.data.DataLoader(dataset=DATASET,
                                            batch_size = N_EXAMPLES,
                                            shuffle=False)

#%% Saving original image
FIXED_INDEX = randint(N_EXAMPLES)

# Saving an item from the dataset to debug
FIXED_X, FIXED_X_PARAMS = DATASET.__getitem__(FIXED_INDEX)
FIXED_X = to_var(torch.Tensor(FIXED_X.contiguous())).view(FIXED_X.size(0), -1)
HEIGHT,WIDTH = FIXED_X.size()

# SAVING fixed x as an image
torchvision.utils.save_image(DATASET.__getitem__(FIXED_INDEX)[0].contiguous(), 
                            './data/SOUND/real_images.png',
                            normalize=False)

#%% CREATING THE Beta-VAE

betaVAE = modVAE.VAE(image_size= WIDTH,z_dim=Z_DIM, h_dim=H_DIM)

# BETA: Regularisation factor
# 0: Maximum Likelihood
# 1: Bayes solution
BETA = 1

# GPU computing if available
if torch.cuda.is_available():
    betaVAE.cuda()
    print('GPU acceleration enabled')

# OPTIMIZER
OPTIMIZER = torch.optim.Adam(betaVAE.parameters(), lr=0.001)

ITER_PER_EPOCH = N_EXAMPLES
NB_EPOCH = 15;


#%%
""" TRAINING """
for epoch in range(NB_EPOCH):    
    # Epoch
    CURRENT_ITERATION = 0;
    for images,params in DATASET:
        CURRENT_ITERATION = CURRENT_ITERATION + 1
        
        # Formatting
        images = to_var(torch.Tensor(images.contiguous())).view(images.size(0), -1)
        out, mu, log_var = betaVAE(images)

        # Compute reconstruction loss and KL-divergence
        reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
        kl_divergence = torch.sum(0.5 * (mu**2
                                         + torch.exp(log_var)
                                         - log_var -1))

        # Backprop + Optimize
        total_loss = reconst_loss + BETA*kl_divergence
        OPTIMIZER.zero_grad()
        total_loss.backward()
        OPTIMIZER.step()

        # PRINT
        # Prints stats at each epoch
        if CURRENT_ITERATION % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f"
                   %(epoch+1,
                     NB_EPOCH,
                     CURRENT_ITERATION,
                     ITER_PER_EPOCH,
                     total_loss.data[0],
                     reconst_loss.data[0],
                     kl_divergence.data[0])
                  )

    # Save the reconstructed images
    reconst_images, _, _ = betaVAE(FIXED_X)
    #reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
                                 './data/SOUND/reconst_images_%d.png' %(epoch+1))
#%% SAMPLING FROM LATENT SPACE
# Random input
FIXED_Z = torch.randn(HEIGHT, Z_DIM)
FIXED_Z = to_var(torch.Tensor(FIXED_Z.contiguous()))

# Sampling from model, reconstructing from spectrogram
sampled_image = betaVAE.sample(FIXED_Z)
sampled_image_numpy = sampled_image.data.numpy()

#%%
plt.figure
plt.subplot(1,2,1)
plt.imshow(sampled_image_numpy)
plt.title('Sampled image from latent space')
plt.subplot(1,2,2)
plt.imshow(FIXED_X.data.numpy())
plt.title('Original image')
#%%
reconst_sound = DATASET.audio_engine.griffinlim(sampled_image_numpy, N_iter=250)
output_name = 'sampled_sound.wav'
librosa.output.write_wav('data/SOUND/sampled_sound.wav',reconst_sound,DATASET.Fs)

# Saving spectrogram image
torchvision.utils.save_image(sampled_image.data.cpu(),
                             './data/SOUND/sampled_image.png')
