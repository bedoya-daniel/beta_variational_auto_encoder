# toyDataset

This package is related to our dataset. It contains three differents modules:
* **Parameter space** `generateParameterSpace.py`: performs every permutations
  of sampled dimensions of our dataset. This list is created when a
  `toyDataset` object is instanciated. It is stored, and then passed to the audio
  engine for rendering.
* **Audio engine** `audioEngine.py`: Allows audio, spectrogram and CQT
  rendering. Called the same way as `generateParameterSpace.py`, but later.
* **Dataset** `dataset.py`: this is the main script.


`utils.py` contains attempts to implement a hilbert transform to flatten image
while conserving spatial coherence.
