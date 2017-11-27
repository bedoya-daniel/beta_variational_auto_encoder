#-*-encoding:UTF-8-*-
""" generateParameterSpace Module
Generate the parameters' space according to the rules configured
by the user
"""

# Librairies
import numpy as np
from numpy.random import randint as rand

class parameterSpace:
    def __init__(self):
        # Initialisation
        self.params = {'f0' : [100, 50, 1000],
                       'PS' : [-0.01, -0.01, -0.1],
                       'PH' : [0, 1 ,2],
                       'inh': [0, 0.1, 5],
                       'AB' : [0, 0.05, 1]}

        self.parameter_space = dict.fromkeys(self.params)


        # Generating parameters
        self.generate_parameter_space()

    def generate_parameter_space(self):
        """ Generate all the values of the parameters specfied in the __init__
        function: [start, increment, end]. It allows non linear increment

        INPUT:

        OUTPUT:
            - self.parameterSpace: dictionnaire, where each paramter's
            name is a key and its value is a list of values
        """

        for key, value in self.params.iteritems():
            # Extracing params
            start = value[0]
            stop = value[2]
            inc = value[1]

            # Vector
            self.parameter_space[key] = np.arange(start, stop+inc, inc)

        return 0

    def get_rand_parameters(self):
        """ outputs a random selection of parameters from the parameter space

        INPUT:
        OUTPUT:
            - list of paramters (1 x Nparams)

        UNIT test: Sends an array with good size and no 'None' value (or empty)
        """

        # Init
        output_params = dict.fromkeys(self.parameter_space)

        # Generating parameters
        for key, value in self.parameter_space.iteritems():
            number_value = len(value)
            rand_index = rand(0, number_value)

            output_params[key] = value[rand_index]

        return output_params
