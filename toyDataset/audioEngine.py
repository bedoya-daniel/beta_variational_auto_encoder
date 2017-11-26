""" AudioEngine module, part of package toyDataset 
ATIAM 2017 """

import librosa as lib
import numpy as np

class audioEngine:
    def __init__(Fs=16000):
        self.Fs = Fs
        
    def render_sample(self, params):
        """ Render the sample from a dictionnary of parameters

        INPUT: 
            - Dictionnary of parameters

        OUTPUT:
            - Numpy array of size (N x 1) containing the sample
        """
