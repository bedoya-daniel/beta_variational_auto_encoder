# -*- coding: utf-8 -*-

# --- IMPORT STATEMENTS. ---
import numpy as np
import matplotlib.pyplot as plt





# ------------------------------------------------
# --------------- HILBERT CURVE ------------------
# ------------------------------------------------
"""
HILBERT CURVE
Function to create a Hilbert curve and the corresponding indices to asign to a nxn
spectrogram matrix
"""

#%% HILBERT CURVE FUNCTION by Alexander Mordvintsev (https://github.com/znah/notebooks)
def hilbert_curve(n):
    """ Generate Hilbert curve indexing for (n, n) array. 'n' must be a power of two.
        
        Arguments:
            - n : size of the matrix
        Returns:
            - Square matrix with the order of the points to fill the Hilbert curve from 0 to n
        """
    # recursion base
    if n == 1:  
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbert_curve(n//2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size*2
    d = np.flipud(np.rot90(t, -1)) + t.size*3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))

#%% CREATE VECTOR FROM SPECTROGRAM FUNCTION
def hilbert_vector(STFT):
    """ Reshapes the STFT matrix into a vector but keeping the spectro temporal continuity
    by using the Hilbert curve, which makes the index points converge regardless of the
    STFT resolution
        
        Arguments:
            - STFT : Short Term Fourier Transform square matrix (it can have complex values)
        Returns:
            - X_vector : Reshaped STFT coefficients into a vector
        """
    matrix_size = len(STFT)
    idx = hilbert_curve(matrix_size)
    y, x = np.indices(idx.shape).reshape(2, -1)
    x[idx.ravel()], y[idx.ravel()] = x.copy(), y.copy()

    # Plotting section
    plot = 0 # Change to 1 if you want to plot the resulting Hilbert curve (just indices)
    if plot==1:
        plt.plot(x, y)
        plt.axis('equal')
        plt.axis('off')
#    _=plt.ylim(ymin=-1)
    
    X_vector = []       # Initialize the variable
    
    # Asign the order indices to the new STFT vector
    for ii in range (0,len(x)):
        temp = STFT[x[ii]][y[ii]]
        X_vector.append(temp)
    
    return X_vector

#%% CREATE STFT MATRIX FROM VECTOR
def inv_hilbert_vector(X_vector):
    """ Performs the inverse transformation of hilbert_vector
    (turns a vector into a STFT matrix)
        
        Arguments:
            - X_vector : STFT vector (it can have complex values)
        Returns:
            - STFT : Reshaped Short Term Fourier Transform square matrix
        """
    matrix_size = int(np.sqrt(len(X_vector)))
    idx = hilbert_curve(matrix_size)
    y, x = np.indices(idx.shape).reshape(2, -1)
    x[idx.ravel()], y[idx.ravel()] = x.copy(), y.copy()
 
    STFT = np.zeros((matrix_size, matrix_size))      # Initialize the variable
    STFT = STFT + 0j                                 # Turn into a complex array
    
    # Asign the order indices to the new STFT matrix
    for ii in range (0,len(x)-1):
        STFT[x[ii]][y[ii]] = X_vector[ii]
    
    return STFT
