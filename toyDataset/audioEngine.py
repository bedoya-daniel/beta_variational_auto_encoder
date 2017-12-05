#-*-encoding:UTF-8-*-
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

         # Retrieve parameters
        f0 = params['f0']
        slope = params['PS']
        harmonicPresence = params['PH']
        inharmonicity = params['inh']
        SNR = params['SnR']
        
        # Sampling parameters
        Fs = self.Fs # Sampling rate
        Ts = 1.0/Fs
        length = 1.0 # length of the signal (in seconds)
        N = int(Fs*length) # number of samples
        t = np.arange(0, length, Ts) # time vector
        
        # Additive synthesis
        n_modes = 10 # nombre de modes
        if harmonic == 0: # tous les modes
            modes = np.arange(1, n_modes+1)
        elif harmonic == 1: # que les harmoniques impaires
            modes = np.arange(1, n_modes+1, 2)
            n_modes = n_modes/2
        
        
        freq = modes * f0 * np.sqrt(1 + beta * pow(modes, 2))
        amp = (modes-1)*slope + n_modes
        amp = (amp/float(n_modes)).clip(min=0)
        x = np.zeros(N) # signal
        
        for k in range(n_modes-1):
            print freq[k], amp[k] 
            x = x + amp[k]*np.sin(2*np.pi*freq[k]*t)
        
        
        # Noise
        noise = (np.random.rand(N) - 0.5) * 2
        energy_x = np.sum(pow(np.abs(x), 2))
        energy_noise = np.sum(pow(np.abs(noise), 2))
        a = np.sqrt(energy_x/(energy_noise*SNR))
        noise = a*noise
        
        # Final signal
        y = x + noise
        y = y/max(y)
        
        return y


    def spectrogram(self, data):
        """ Returns the absolute value of the stft of the array of sounds 'data'
        
        INPUT:
            data: array of n sounds (n*SOUND_LENGTH)
            
        OUTPUT:
            output: absolute value of the stft of every sound in the sound array
        
        """
        # M: number of sounds
        # N: number of samples
        (M,N) = np.shape(data)
        
        # Allocating the memory needed
        spectrograms = [lib.stft(data[1], n_fft=self.n_fft) * 0 for i in xrange(M)]

        # FOR LOOP: computing spectrogram
        for i in xrange(M):
            spectrograms[i] = np.abs( lib.stft(data[i], n_fft=self.n_fft) )

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
