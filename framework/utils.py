# Librairies
import torch
import torch.nn as nn
from torch.autograd import Variable


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
