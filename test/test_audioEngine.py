#-*-encoding:UTF-8-*-

# IMPORT STATEMENTS
import unittest
import numpy as np
from  numpy.random import randint
import toyDataset.audioEngine as aud
import toyDataset.generateParameterSpace as gps
import toyDataset.dataset as dts

class audioEngineTest(unittest.TestCase):
    """ Test the audio engine module """

    def setUp(self):
        """ Creating a dict loaded with parameter
        """
        self.fixed_length = 60000
        self.batch_size = 40
        self.paramSpace = gps.parameterSpace()
        self.audio_engine = aud.audioEngine()
        self.paramSpace.generate_parameter_space()
        self.test_param = self.paramSpace.get_rand_parameters()
        self.dataset = dts.toyDataset(self.fixed_length)
        self.data = self.dataset.get_minibatch(batchSize = self.batch_size)
        self.sounds = self.dataset.sound_data


    def test_render_sound(self):
        sound = self.audio_engine.render_sound(self.test_param, self.fixed_length)
        length = len(sound)
        self.assertEqual(length, self.fixed_length)

    def test_spectrograms(self):
        test_spectro = self.audio_engine.spectrogram(self.sounds)
        test_shape = np.shape(test_spectro)

        self.assertEqual(test_shape[0],self.dataset.batch_size)
