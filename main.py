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


#%% Importing DATASET

sel_dataset = 'MNIST'
#sel_dataset = 'toydataset'

if sel_dataset == 'MNIST':
    # MNIST dataset
    DATASET = datasets.MNIST(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)    
    # Data loader
    DATA_LOADER = torch.utils.data.DataLoader(dataset=DATASET,
                                               batch_size=100,
                                               shuffle=True)
elif sel_dataset == 'toydataset':
    # Toy Dataset loading
    # Parameters
    n_fft = 1024
    
    # Creating dataset
    DATASET = dts.toyDataset()
    # dataset = DATASET.get_minibatch()
    DATA_LOADER = torch.utils.data.DataLoader(dataset=DATASET,
                                             batch_size = 100,
                                             shuffle=False)


#%%
""" TRAINING THE VAE MODEL """

if sel_dataset == 'MNIST':    
    ITER_PER_EPOCH = len(DATA_LOADER)
    DATA_ITER = iter(DATA_LOADER)
    
    # fixed inputs for debugging
    FIXED_X, _ = next(DATA_ITER)
    torchvision.utils.save_image(FIXED_X.cpu(), './data/real_images.png')
    FIXED_X = to_var(FIXED_X.view(FIXED_X.size(0), -1))
    
    DATA_ITER = enumerate(DATA_LOADER)    
elif sel_dataset == 'toydataset':    
    # fixed inputs for debugging
    FIXED_Z = to_var(torch.randn(100, 20))
    # Saving an item from the dataset to debug
    FIXED_X, _ = DATASET.__getitem__(9)
    FIXED_X = torch.Tensor(FIXED_X).contiguous() # As a contiguous memory block
    
    # Retrieving Height and width
    #HEIGHT,WIDTH = FIXED_X.size()
    
    ## SAVING fixed x as an image
    torchvision.utils.save_image(FIXED_X, 
                             './data/real_images.png', 
                             normalize=False)


#%% CREATING THE Beta-VAE
betaVAE = modVAE.VAE()

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

ITER_PER_EPOCH = len(DATA_LOADER)
NB_EPOCH = 5;

#%%
""" TRAINING """
for epoch in range(NB_EPOCH):
    
    # Setting-up the relevant data_iterator
    if sel_dataset == 'MNIST':
        DATA_ITER = enumerate(DATA_LOADER)
    elif sel_dataset == 'toydataset':
        DATA_ITER = DATA_LOADER
    
    # Epoch
    for i,(images,param) in DATA_ITER:
        # Formatting
        images = to_var(torch.Tensor(images)).view(images.size(0), -1)
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
        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f"
                   %(epoch+1,
                     NB_EPOCH,
                     i+1,
                     ITER_PER_EPOCH,
                     total_loss.data[0],
                     reconst_loss.data[0],
                     kl_divergence.data[0])
                  )

    # Save the reconstructed images
    reconst_images, _, _ = betaVAE(FIXED_X)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
                                 './data/reconst_images_%d.png' %(epoch+1))
#%% SAMPLING 
# Creating a z vector to decode from
FIXED_Z = to_var(torch.randn(100, 20))

# Sampling from model
sampled_images = betaVAE.sample(FIXED_Z)

# Saving
sampled_images = sampled_images.view(sampled_images.size(0), 1, 28, 28)
torchvision.utils.save_image(sampled_images.data.cpu(),
                             './data/sampled_image.png')
