"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import numpy as np


class GaussianNoise:
    """
    Simple Gaussian noise for TD3 exploration.

    TD3 paper uses N(0, 0.1) - simple uncorrelated Gaussian noise.
    This is different from DDPG which uses temporally correlated OU noise.

    Reference: TD3 paper Table 1 - "Exploration Policy: N(0, 0.1)"
    """
    def __init__(self, shape, sigma: float = 0.1):
        self._shape = shape
        self._sigma = sigma

    def __call__(self):
        return np.random.normal(0, self._sigma, size=self._shape)

    def reset(self):
        # No state to reset for Gaussian noise (stateless)
        pass
