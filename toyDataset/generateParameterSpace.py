""" generateParameterSpace Module
Generate the parameters' space according to the rules configured
by the user
"""

# Librairies
import numpy as np

class parameterSpace:
    def __init__(args):
        paramF0 = {'f0' : np.arange(100, 1000, 50)}

    def generate_parameter_space(self):
        """ Generate all the values of the parameters specfied in the __init__
        function: [start, increment, end]. It allows non linear increment

        INPUT:

        OUTPUT:
            - self.parameterSpace: dictionnaire, where each paramter's name is a 
            key and its value is a list of values
        """
        return 0

    def get_rand_parameters(self):
        """ outputs a random selection of parameters from the parameter space

        INPUT:
        OUTPUT:
            - list of paramters (1 x Nparams)

        UNIT test: Sends an array with good size and no 'None' value (or empty)
        """

        return np.zeros(1,5)
