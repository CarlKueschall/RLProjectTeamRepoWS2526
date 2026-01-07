"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import numpy as np
class OUNoise:
    # Ornstein-Uhlenbeck noise for exploration in continuous action spaces
    # Provides temporally correlated noise that is more suitable for continuous
    # control than independent Gaussian noise.
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        #########################################################
        # We initialize the shape of the noise array, the speed of mean reversion
        # and the time step for noise generation
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self):
        #NOTE: using the class via __call__ is just efficient way of doing it, it's not necessary but it's a good practice so I went back to this
        noise = (
            self.noise_prev
            + self._theta * (-self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self):
        self.noise_prev = np.zeros(self._shape)
