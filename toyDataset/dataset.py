#-*-encoding:UTF-8-*-
""" Toy Dataset class:
    Create a dataset depending on arguments """

# Librairies
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import librosa as lib
import generateParameterSpace as gps
import audioEngine as aud

# Classe du toyDataset
class toyDataset(Dataset):
    def __init__(self, Fe_Hz=16000, length_sample=64000, batchSize=100):
        """ ToyDataset object.
        INPUT:
            -(opt) Fs: samplerate (def: 16kHz)
        """
        # initializing variables
        self.Fs = Fe_Hz
        self.batch_size = batchSize
        self.length_sample = length_sample

        # init data structures
        self.batch_parameters = {}
        self.sound_data = [];
        self.spectrograms = [];

        self.defSampleParams = {'f0': 300, # fréquence fondamentale
                                'inh':0.3, # facteur d'inharmonicité
                                'PH': 0,   # présence des harmoniques
                                'SnR': 1, # pente spectrale
                                'AB': 0.3} # amplitude du bruit

        
        # Create parameterSpace
        self.paramSpace = gps.parameterSpace()
        self.paramSpace.generate_parameter_space()

        # Create the audio engine for audio rendering
        self.audio_engine = aud.audioEngine(Fs_Hz=self.Fs)

    def get_minibatch(self, batchSize=100, render=True):
        """ Outputs a dataset for the bVAE. If render = True, recalculate a new 
        minibatch. If False, just return the old one (self.toyDataset)

        INPUT:
            - batchSize: number of list of parameter to be taken
        """
        if render:
            self.batch_parameters = [{} for i in xrange(batchSize)]
            self.batch_size = batchSize
            
            self.batch_parameters = self.get_rand_params(self.batch_size)

            # Create sounds
            self.sound_data = self.render_batch()

            # Converts them into spectrograms
            self.spectrograms = self.audio_engine.spectrogram(self.sound_data)

        # returns the spectrograms
        return self.spectrograms

    def __getitem__(self, index):
        """ Returns a sample which is a dict with key "images" and "parameters"

        INPUT:
            - index: index corresponing to a sample in the dataset

        OUTPUT:
            - sample: dict {'image':spectrogram, 'parameters':[parameters]}

        UNIT TEST:
            - Vérifier que la fonction retourne bien un dico avec un numpy array 
            pour la première clé et un ndarray de taille [1xparam]
        """

        sample = {}
        sample['spectro'] = self.spectrograms[index]
        sample['parameters'] = self.batch_parameters[index]

        return sample

    def __len__(self):
        """ Returns the length of the label vector
        OUTPUT:
            - len: length of the parameter vector
        """
        return len(self.defSampleParams)

    def render_batch(self):
        """ Render the given batch of samples from the input arguments. 
        Default is the whole array of parameter toyDataset

        INPUT:
            - paramsPatch (array 1*N): N list of parameters 
                                       for sample rendering
        """
        # INIT: allocate memory
        sound_data = [[None in xrange(self.length_sample)] \
                      for j in xrange(self.batch_size)]

        # FOR LOOP: renders each sound
        for i in xrange(self.batch_size):
            sound_data[i] = self.audio_engine.\
                    render_sound(self.batch_parameters[i],\
                                self.length_sample)
        
        # Returns the rendered audio
        return sound_data


    def get_rand_params(self, batchSize):
        """ Generate a set of parameters of size 'batchSize' 

        INPUT:
            - batchSize: Number of set of parameter to generate from the dict

        OUTPUT:
            - paramBatch: array of size [batchSize x numberOfParam] containing the 
            sets of parameters

        unit test: it sends back an array of the good size with no 'None' value
        """
        # INIT
        batch_parameters = [{} for i in xrange(self.batch_size)]

        # FOR LOOP: generates the batch of random parameters
        for i in xrange(self.batch_size):
            batch_parameters[i] = self.paramSpace.get_rand_parameters()

        # Returns the data
        return batch_parameters


    def to_pytorch_dataset(self, audioBatch):
        """ Convert the given batch of dataset into a pytorch dataset to input
        in the Net"""

