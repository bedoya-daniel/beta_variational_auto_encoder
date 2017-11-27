#-*-encoding:UTF-8-*-

# IMPORT STATEMENTS
import unittest
import toyDataset.generateParameterSpace as gps


class parameterSpaceTest(unittest.TestCase):
    """ Tests the toyDataset.generateParameterSpace module """

    def setUp(self):
        """ Creating a parameterSpace object """
        self.gps_example = gps.parameterSpace()
        self.Nparams = 5

    def test_generateParamSpace(self):
        """ Test that the object paramSpace has Nparams keys """
        sampleLength = len(self.gps_example.parameter_space.keys())

        self.assertEqual(sampleLength, self.Nparams)

    def test_getRandParam(self):
        """ Test de la fonction get_random_parameter()"""
        sample = self.gps_example.get_rand_parameters()

        self.assertEqual(len(sample), self.Nparams)
