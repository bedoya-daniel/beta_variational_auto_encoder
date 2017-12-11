#-*-encoding:UTF-8-*-
""" AudioEngine module, part of package toyDataset 
ATIAM 2017 """

import librosa as lib
import numpy as np
import torch as to

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

         # Retrieve parameters
        f0 = params['f0']
        slope = params['PS']
        harmonicPresence = params['PH']
        inharmonicity = params['inh']
        SNR = params['SnR']
        
        # Sampling parameters
        Fs = self.Fs # Sampling rate
        Ts = 1.0/Fs
        length = sound_length/Fs # length of the signal (in secondes)
        N = sound_length # number of samples
        t = np.arange(0,N,1)*Ts # time vector
        
        # Additive synthesis
        n_modes = 10 # nombre de modes
        if harmonicPresence == 0: # tous les modes
            modes = np.arange(1, n_modes+1)
        elif harmonicPresence == 1: # que les harmoniques impaires
            modes = np.arange(1, n_modes+1, 2)
            n_modes = n_modes/2
        elif harmonicPresence == 2: # que des harmoniques paires
            modes = np.arange(2, n_modes+1, 2)
            modes = np.concatenate((np.array([1]), modes))
            n_modes = n_modes/2 + 1        
        
        freq = modes * f0 * np.sqrt(1 + inharmonicity * pow(modes, 2))
        amp = (modes-1)*slope + n_modes
        amp = (amp/float(n_modes)).clip(min=0)
        x = np.zeros(N) # signal
        
        for k in range(n_modes-1):
            x = x + amp[k]*np.sin(2*np.pi*freq[k]*t)
                
        # Noise
        noise = (np.random.rand(N) - 0.5) * 2
        energy_x = np.sum(pow(np.abs(x), 2))
        energy_noise = np.sum(pow(np.abs(noise), 2))
        a = np.sqrt(energy_x/(energy_noise*SNR))
        noise = a*noise
        
        # Final signal
        #y = x + noise
        y = x
        y = y/max(y)
        
        return y


    def spectrogram(self, data):
        """ Returns the absolute value of the stft of the array of sounds 'data'
        flattened on a line vector.
        Size : (1, N_fft * N_frame)
        
        INPUT:
            data: array of n sounds (n*SOUND_LENGTH)
            
        OUTPUT:
            output: absolute value of the stft of every sound in the sound array
        
        """
        # M: number of sounds
        # N: number of samples
        (M,N) = np.shape(data)
        
        # Allocating the memory needed
        temp_spec_flattened = lib.stft(data[0], n_fft=self.n_fft)#.reshape(1,-1)*0
        spectrograms = [np.zeros_like(temp_spec_flattened) for i in xrange(M)]

        # FOR LOOP: computing spectrogram
        for i in xrange(M):
            spectrograms[i] = np.abs( lib.stft(data[i], n_fft=self.n_fft) )#.reshape(1,-1)

        return spectrograms
    
        # La fonction prend en paramètres un spectrogramme S    
        # et le nombre de pts de la NFFT désirée.
        # Elle retourne un vecteur correspondant à l'audio reconstruit
    
    def griffinlim(self, S, N_iter=100):
        """ Returns a sound, reconstructed from a spectrogram with NFFT points.
        Griffin and lim algorithm
        
        INPUT:
            - S: spectrogram (array) (absolute value of the stft)
            - N_iter (def: 100): number of iteration for the reconstruction
        
        OUTPUT:
            - x: signal """
        # ---- INIT ----
        # Create empty STFT & Back from log amplitude
        n_fft = S.shape[0]
        S = np.log1p(np.abs(S))

        a = np.exp(S) - 1
        
        # Phase reconstruction
        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
        
        # LOOP: iterative reconstruction
        for i in range(N_iter):
            S = a * np.exp(1j*p)
            x = lib.istft(S)
            p = np.angle(lib.stft(x, self.n_fft))
    
        return np.real(x)
