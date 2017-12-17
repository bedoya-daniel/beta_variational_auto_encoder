""" ---------------------------------------------
VAE with a convnet, for pure sound input
"""

# Librairies
import torch
import torch.nn as nn
from torch.autograd import Variable
from framework.utils import to_var


class conv1dVAE(nn.Module):
    """ Variational Audo-Encoder (VAE) class, with a 1D convolutional net """

    def __init__(self, sound_length=784, out_conv_ch=400, kernel_size=5, h_dim=400, z_dim=5, stride=1):
        """ Creates a VAE net object. 
        
        Arguments:
            - image_size (int, def: 784): size of the flattened image (1D)
            - h_dim (int, def:400): size of the hidden layer
            - z_dim (int, def:5): number of latent dimensions

        Returns:
            - VAE object
        """
        super(conv1dVAE, self).__init__()
        
        # Calculating output size after conv1D
        LENGTH_OUT_CONV1D = (sound_length)/stride 

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv1d(1, out_conv_ch, kernel_size, stride=stride),
            nn.AdaptiveMaxPool1d(LENGTH_OUT_CONV1D),
            nn.Linear(LENGTH_OUT_CONV1D,1200),
            nn.ReLU(),
            nn.Linear(1200, h_dim),
            nn.Sigmoid(),
            nn.Linear(h_dim, z_dim*2),
            nn.Tanh())  # 2 for mean and variance.

        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1200),
            nn.ReLU6(),
            nn.Linear(1200, LENGTH_OUT_CONV1D),
            nn.ConvTranspose1d(out_conv_ch, 1, kernel_size ,stride=stride),
            nn.AdaptiveMaxPool1d(sound_length))

        # Initializing weights
        self.encoder.apply(self.init_weight)
        self.decoder.apply(self.init_weight)

    def init_weight(self, module):
        """ This function initialize the weight of a linear layer

        Arguments:
            - module: the sequential to initialize weights

        """
        if type(module) == nn.Linear:
            nn.init.xavier_uniform(module.weight)

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
        
        encoded_vec = self.encoder(data).squeeze(1)
        mu, log_var = torch.chunk(encoded_vec, 2, dim=1)  # mean and log variance.
        reparam = self.reparametrize(mu, log_var)
        reparam = reparam.unsqueeze(1)
        output = self.decoder(reparam)
        output = output.squeeze(1)
        
        return output, mu, log_var

    def sample(self, latent_vec):
        """ Decodes the given latent vector.
        
        Arguments:
            - latent_vec: matrix containing a number of encoded vectors

        Returns:
            - output: decoded vector
        """
        return self.decoder(latent_vec)
