#-*-encoding:UTF-8-*-
""" Toy Dataset class:
    Create a dataset depending on arguments """

# Librairies
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import Librosa as lib
import generateParameterSpace as gps

# Classe du toyDataset
class toyDataset:
    def __init__(self, args):
        """ args: dictionnaire contenant les paramètres du toyDataset 
                ex: {"pente spectrale",[0.01 0.02 0.03 ... 0.9]} """

                # Contient toutes les permutations de paramètres
                # Nombre de dimensions: nombre de clés
        
        self.toyDataSet = [];
        self.defSampleParams = {'f0': 300, # fréquence fondamentale
                                'inh':0.3, # facteur d'inharmonicité
                                'PH': 0,   # présence des harmoniques
                                'PS': 1, # pente spectrale
                                'AB': 0.3} # amplitude du bruit

    def get_minibatch(self, batchSize=100, render=True):
        """ Outputs a dataset for the bVAE. If render = True, recalculate a new 
        minibatch. If False, just return the old one (self.toyDataset)

        DEV:
            - calls get_rand_params(N): N size of the dataset
            - calls render_batch: render the whole list into audio
            - calls to_pytorch_dataset so it converts it into a pytorch dataset

        INPUT:
            - batchSize: number of list of parameter to be taken
        """


    def render_sample(self, sampleParams,
                            length=self.lengthSample):
        """ Renders a unique sample from a list of sample 
        INPUT:
            - parameters: dictionnary of parameters
            - length (opt): length of the recrding (in samples)

        DEV:
            Will probably be replaced by a module audioEngine with a 
            render_sample method
        """

    def render_batch(self, paramsBatch=self.paramsToyDataset):
        """ Render the given batch of samples from the input arguments. 
        Default is the whole array of parameter toyDataset

        DEV:
            - Allocate all the memory we will need
            - Iterate on each set of parameter
            - calls render_sample
            
            create a framework.audioengine module?

        INPUT:
            - paramsPatch (array 1*N): N list of parameters 
                                       for sample rendering
        """

    def get_rand_params(self, batchSize):
        """ Generate a set of parameters of size 'batchSize' 

        INPUT:
            - batchSize: Number of set of parameter to generate from the dict

        OUTPUT:
            - paramBatch: array of size [batchSize x numberOfParam] containing the 
            sets of parameters

        DEV:
            - for loop over batchSize
            - calls get_rand_parameters in the genetaeParameterSpace module,
              which is a method of the object paramterSpace

        unit test: it sends back an array of the good size with no 'None' value
        """
        return 0

    def to_pytorch_dataset(self, batch=self.toyDataset):
        """ Convert the given batch of dataset into a pytorch dataset to input
        in the Net"""

