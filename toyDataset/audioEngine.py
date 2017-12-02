""" AudioEngine module, part of package toyDataset 
ATIAM 2017 """

import librosa as lib
import numpy as np

class audioEngine:
    def __init__(self,Fs_Hz=16000, n_fft=1024):
        self.Fs = Fs_Hz
        self.n_fft = n_fft
        
    def render_sound(self, params, sound_length):
        """ Render the sample from a dictionnary of parameters

        INPUT: 
            - Dictionnary of parameters

        OUTPUT:
            - Numpy array of size (N x 1) containing the sample
        """

        return np.random.rand((sound_length))


    def spectrogram(self, data):
        """ Returns the spectrograms of the array of sounds 'data' """
        # M: number of sounds
        # N: number of samples
        (M,N) = np.shape(data)
        
        # Allocating the memory needed
        spectrograms = [lib.stft(data[1], n_fft=self.n_fft) * 0 for i in xrange(M)]

        # FOR LOOP: computing spectrogram
        for i in xrange(M):
            spectrograms[i] = np.abs( lib.stft(data[i], n_fft=self.n_fft) )

        return spectrograms
