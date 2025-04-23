import nni
from nni.nas.nn.pytorch import ModelSpace

from nni.mutable import MutableExpression

import torch.nn as nn
import torchaudio.transforms as T

import numpy as np


class Experiment2SearchSpace(ModelSpace):
    def __init__(self, **kwargs):
        orig_sr = kwargs.pop('orig_sr', 16000)
        super().__init__(**kwargs)

        candidate_sampling_rates = [4000, 6000, 8000, 11000, 16000, 22050, 44100]
        # remove sampling rates that are higher than the original sampling rate
        candidate_sampling_rates = [sr for sr in candidate_sampling_rates if sr <= orig_sr]

        # Experiment Nr. 2 optimizes the provided preprocessing parameters
        self.add_mutable(nni.choice('method', ["mel","mfcc", "wavelet"]))
        self.add_mutable(nni.choice('sample_rate', candidate_sampling_rates))
        self.add_mutable(nni.choice('use_db', [False, True]))

        self.add_mutable(nni.choice('n_fft', [128, 256, 512, 1024, 2048]))
        self.add_mutable(nni.choice('hop_length', [32, 64, 128, 256, 512]))
        self.add_mutable(nni.choice('n_mels', [20, 30, 40, 60, 80, 100, 128]))
        self.add_mutable(nni.choice('stft_power', [1, 2]))
        self.add_mutable(nni.choice('n_mfcc', [12, 13, 20, 30, 40]))

        self.add_mutable(nni.choice('wavelet_scaling', ["lin", "log"]))
        self.add_mutable(nni.choice('wavelet_resize', [64, 128]))

    def freeze(self, sample):
        with nni.nas.space.model_context(sample):
            return self.__class__()  # Return a new instance with the frozen context
