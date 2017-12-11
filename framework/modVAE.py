""" modVAE.py
Implementation of a VAE model """

# Librairies
import torch
import torch.nn as nn
from torch.autograd import Variable
from framework.utils import to_var

# VAE model
class VAE(nn.Module):
    """ Variational Audo-Encoder (VAE) class """

    def __init__(self, image_size=784, h_dim=400, z_dim=5):
        """ Creates a VAE net object. 
        
        Arguments:
            - image_size (int, def: 784): size of the flattened image (1D)
            - h_dim (int, def:400): size of the hidden layer
            - z_dim (int, def:5): number of latent dimensions

        Returns:
            - VAE object
        """
        super(VAE, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2))  # 2 for mean and variance.

        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid())

    def reparametrize(self, mu, log_var):
        """" Does the reparametrize trick on the beta-VAE:
            $z = mean + eps * sigma$ where 'eps' is sampled from N(0, 1).

        Arguments: 
            - mu: mean of the distribution
            - log_var: log value of the variance

        Returns:
            - output: Reparametrized data (mu + eps * exp(log_var/2))
        
        """
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        output = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return output

    def forward(self, data):
        """ Do the forward pass in the VAE model: encodes, reparameterize and
        decodes the input data.
        
        Arguments:
            - data (array[SAMPLE_INDEX][DATA_VECTOR]):

        Returns:
            - output (array like data): data encoded, reparametrized then decoded
            - mu: mean 
            - log_var: log value of the variance
        
        """
        encoded_vec = self.encoder(data)
        mu, log_var = torch.chunk(encoded_vec, 2, dim=1)  # mean and log variance.
        reparam = self.reparametrize(mu, log_var)
        output = self.decoder(reparam)
        
        return output, mu, log_var

    def sample(self, latent_vec):
        """ Decodes the given latent vector.
        
        Arguments:
            - latent_vec: matrix containing a number of encoded vectors

        Returns:
            - output: decoded vector
        """
        return self.decoder(latent_vec)


