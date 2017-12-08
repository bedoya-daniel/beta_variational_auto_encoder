# Librairies
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# ------------------------------------------------------------------------
# to_var(data)
# If CUDA is available on the system, the data gets transfered on the GPU
# for acceleration, and is converts into a Pytorch (torch) tensor
# -----
def to_var(tensor):
    """ to_var(tensor):
            Converts data into a torch tensor, 
            and puts it on the GPU if available"""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


# ------------------------------------------------------------------------
    
def zdim_analysis(BATCH_SIZE, TOTAL_Z_DIM, Z_DIM, start=0, stop=100):
    """ Returns a matrix where only one dimension changes 
    
    INPUT:
        - BATCH_SIZE: size of the batch to input
        - TOTAL_Z_DIM: number of latent dim
        - Z_DIM: which z_dim to variate
        - start: start value
        - stop: stop value
        -inc: increment
        
    OUTPUT:
        - sampling_data: desired matrix [BATCH_SIZE * TOTAL_Z_DIM]
    """
    a = torch.arange(start, stop, step=(np.abs(stop-start)/BATCH_SIZE))
    b = torch.ones(BATCH_SIZE, Z_DIM-1)
    c = torch.ones(BATCH_SIZE, TOTAL_Z_DIM-Z_DIM)
    
    return torch.cat((b,a,c),dim=1)