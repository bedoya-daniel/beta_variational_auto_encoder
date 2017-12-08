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

# MNIST dataset
DATASET = datasets.MNIST(root='./data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)    
# Data loader
DATA_LOADER = torch.utils.data.DataLoader(dataset=DATASET,
                                           batch_size=100,
                                           shuffle=True)

#%%
""" TRAINING THE VAE MODEL """

ITER_PER_EPOCH = len(DATA_LOADER)
DATA_ITER = iter(DATA_LOADER)

# fixed inputs for debugging
FIXED_X, _ = next(DATA_ITER)
torchvision.utils.save_image(FIXED_X.cpu(), './data/MNIST/real_images.png')
FIXED_X = to_var(FIXED_X.view(FIXED_X.size(0), -1))

DATA_ITER = enumerate(DATA_LOADER)    

#%% CREATING THE Beta-VAE
betaVAE = modVAE.VAE(z_dim=2)

# BETA: Regularisation factor
# 0: Maximum Likelihood
# 1: Bayes solution
BETA = 4

# GPU computing if available
if torch.cuda.is_available():
    betaVAE.cuda()
    print('GPU acceleration enabled')

# OPTIMIZER
OPTIMIZER = torch.optim.Adam(betaVAE.parameters(), lr=0.001)

ITER_PER_EPOCH = len(DATA_LOADER)
NB_EPOCH = 10;

#%%
""" TRAINING """
for epoch in range(NB_EPOCH):
    # Epoch
    for i,(images,params) in enumerate(DATA_LOADER):
        
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
                                 './data/MNIST/reconst_images_%d.png' %(epoch+1))
#%% SAMPLING 
# Creating a z vector to decode from
FIXED_Z = to_var(torch.randn(100, 2))

# Sampling from model
sampled_images = betaVAE.sample(FIXED_Z)

# Saving
sampled_images = sampled_images.view(sampled_images.size(0), 1, 28, 28)
torchvision.utils.save_image(sampled_images.data.cpu(),
                             './data/MNIST/sampled_image.png')

#%%                           Test 3: analyse z=space (Guassian mesh)
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
# Create Guassian distribution mesh

x, y = np.mgrid[0:1:1.0/np.sqrt(100), 0:1:1.0/np.sqrt(100)]
x = norm.ppf(x)
y = norm.ppf(y) 
x[0,:]= - 2
y[:,0] = -2
x = x.reshape(100,)
y = y.reshape(100,)
z = np.array([x,y]).T
z = Variable(torch.from_numpy(z).float())

# Decode mesh to images
samples = betaVAE.decoder(z).data.numpy()

# Saving
sampled_images = sampled_images.view(sampled_images.size(0), 1, 28, 28)
torchvision.utils.save_image(sampled_images.data.cpu(),
                             './data/MNIST/sampled_image_2.png')

# Plot
#fig = plt.figure()
#for i in range(1,100):
#    ax = fig.add_subplot(10,10,i)
#    ax.plot(samples[i,:])
#fig.show()
