""" AudioEngine module, part of package toyDataset 
ATIAM 2017 """

import librosa as lib
import numpy as np

class audioEngine:
    def __init__(self,Fs_Hz=16000):
        self.Fs = Fs_Hz
        
    def render_sound(self, params, sound_length):
        """ Render the sample from a dictionnary of parameters

        INPUT: 
            - Dictionnary of parameters

        OUTPUT:
            - Numpy array of size (N x 1) containing the sample
        """

        return np.random.rand((sound_length))


    def spectrogram(self, data, n_fft=1024):
        """ an n-array of sounds into spectrograms """
        # M: number of sounds
        # N: number of samples
        (M,N) = np.shape(data)

        spectrograms = [lib.stft(data[1], n_fft=n_fft)*0 for i in xrange(M)]

        # FOR LOOP: computing spectrogram
        for i in xrange(M):
            spectrograms[i] = np.abs(lib.stft(data[i], n_fft=n_fft))

        return spectrograms
