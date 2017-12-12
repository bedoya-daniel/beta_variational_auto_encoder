#-*-encoding:UTF-8-*-
"""
Beta Variational Auto-Encoder
"""

#%% Librairies
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms

from framework import modVAE
from framework.utils import to_var,zdim_analysis

from toyDataset import dataset as dts
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np

import librosa

#%% PARAMETERS
# Parameters, dataset
N_FFT = 1024
MNBATCH_SIZE = 20
#LEN_EXAMPLES = 38400
LEN_EXAMPLES = 16000
# Net parameters
Z_DIM, H_DIM, CONV1D_OUT = 5, 100, 400
FS = 8000

#%% Importing DATASET
# Creating dataset
DATASET = dts.toyDataset(length_sample=LEN_EXAMPLES, 
                         n_fft=N_FFT, 
                         Fe_Hz=FS,
                         data='sound')

SOUND_LENGTH = np.shape(DATASET.__getitem__(9)[0])[0]

for i in xrange(25,50):
    if (SOUND_LENGTH % i) == 0:
        MNBATCH_SIZE = i
        print('Mini_batch size is %d'%(MNBATCH_SIZE))
        break

DATA_LOADER = torch.utils.data.DataLoader(dataset=DATASET,
                                            batch_size = MNBATCH_SIZE,
                                            shuffle=True)

#%% Saving original image
FIXED_INDEX = randint(MNBATCH_SIZE)

# Saving an item from the dataset to debug
DATA_ITER = iter(DATA_LOADER)
FIXED_X,_ = next(DATA_ITER)
FIXED_X = torch.Tensor(FIXED_X.type(torch.FloatTensor))
HEIGHT,WIDTH = FIXED_X.size()

#%% SAVING fixed x as an image
FIXED_X = to_var(FIXED_X)

#%% CREATING THE Beta-VAE
betaVAE = modVAE.conv1dVAE(sound_length=SOUND_LENGTH, z_dim=Z_DIM, h_dim=H_DIM, out_conv_dim = CONV1D_OUT)

# BETA: Regularisation factor
# 0: Maximum Likelihood
# 1: Bayes solution
BETA = 0.5


# GPU computing if available
if torch.cuda.is_available():
    betaVAE.cuda()
    print('GPU acceleration enabled')

# OPTIMIZER
OPTIMIZER = torch.optim.Adam(betaVAE.parameters(), lr=0.001)

ITER_PER_EPOCH = len(DATASET)/MNBATCH_SIZE
NB_EPOCH = 100;


#%%
""" TRAINING """
for epoch in range(NB_EPOCH):    
    # Epoch
    print(' ')
    print('\t \t  /=======================================\\')
    print('\t \t  | - | - | - | EPOCH [%d/%d] | - | - | - | '%(epoch+1, NB_EPOCH))
    print('\t \t  \\=======================================/')
    print(' ')
    for i,(images,params) in enumerate(DATA_LOADER):

        # Formatting
        images = images.type(torch.FloatTensor)
        images = images.unsqueeze(1)#.unsqueeze(3)
        images = to_var(torch.Tensor(images))
        
        # Input in the net
        out, mu, log_var = betaVAE(images)

        # Compute reconstruction loss and KL-divergence
        reconst_loss = F.binary_cross_entropy(out, images, size_average=True)
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
        if i%10 == 0:
            print ("Step [%d/%d] \t Total Loss: %.2f \t Reconst Loss: %.2f \t KL Div: %.3f"
                   %(i,
                     ITER_PER_EPOCH,
                     total_loss.data[0],
                     reconst_loss.data[0],
                     kl_divergence.data[0])
                  )

    # Save the reconstructed images
    reconst_images, _, _ = betaVAE(FIXED_X)
    reconst_images = reconst_images.view(MNBATCH_SIZE,1,N_FFT/2+1,-1)
    torchvision.utils.save_image(reconst_images.data.cpu(),'./data/SOUND/reconst_images_%d.png' %(epoch+1))
    
#%% SAMPLING FROM LATENT SPACE
FIXED_Z = zdim_analysis(MNBATCH_SIZE, Z_DIM, Z_DIM, -1, 1)
FIXED_Z = to_var(torch.Tensor(FIXED_Z.contiguous()))

# Sampling from model, reconstructing from spectrogram
sampled_image = betaVAE.sample(FIXED_Z)
reconst_images = sampled_image.view(MNBATCH_SIZE,1,N_FFT/2+1,-1)
torchvision.utils.save_image(reconst_images.data.cpu(),'./data/SOUND/sampled_images.png')

#%%
sampled_image_numpy = sampled_image.data.numpy()
sampled_image_numpy =sampled_image_numpy[1,:].reshape(N_FFT/2+1,-1)

reconst_sound = DATASET.audio_engine.griffinlim(sampled_image_numpy, N_iter=500)
output_name = 'sampled_sound.wav'
librosa.output.write_wav('data/SOUND/sampled_sound.wav',reconst_sound,DATASET.Fs)

