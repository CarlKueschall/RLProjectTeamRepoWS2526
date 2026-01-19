"""
Distribution utilities for DreamerV3.

Contains:
- TanhNormal: Squashed Gaussian for bounded continuous actions
- ContDist: Wrapper for continuous distributions with mode/sample

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import torch
import torch.nn as nn
from torch import distributions as D
import numpy as np


class TanhNormal:
    """
    Gaussian distribution followed by tanh squashing.

    This produces actions bounded in [-1, 1] while maintaining
    differentiable sampling via the reparameterization trick.

    The log probability accounts for the tanh transformation
    using the change of variables formula.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Args:
            mean: Mean of the underlying Gaussian
            std: Standard deviation of the underlying Gaussian
        """
        self.mean = mean
        self.std = std
        self._normal = D.Normal(mean, std)

    def sample(self) -> torch.Tensor:
        """Sample with reparameterization (allows gradients)."""
        x = self._normal.rsample()
        return torch.tanh(x)

    def rsample(self) -> torch.Tensor:
        """Alias for sample() - both use reparameterization."""
        return self.sample()

    @property
    def mode(self) -> torch.Tensor:
        """Most likely action (mean squashed through tanh)."""
        return torch.tanh(self.mean)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        Log probability of action under this distribution.

        Uses change of variables: if y = tanh(x), then
        log p(y) = log p(x) - log |dy/dx|
                 = log p(x) - log(1 - tanh(x)^2)
                 = log p(x) - log(1 - y^2)
        """
        # Inverse tanh (atanh) to get the underlying Gaussian sample
        # Clip to avoid numerical issues at boundaries
        action_clipped = torch.clamp(action, -0.9999, 0.9999)
        x = torch.atanh(action_clipped)

        # Log prob of underlying Gaussian
        log_prob = self._normal.log_prob(x)

        # Jacobian correction for tanh transformation
        # d(tanh(x))/dx = 1 - tanh(x)^2
        log_prob = log_prob - torch.log(1 - action_clipped.pow(2) + 1e-6)

        # Sum over action dimensions
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """
        Approximate entropy of the TanhNormal distribution.

        Exact entropy is intractable, so we use the Gaussian entropy
        as an approximation (slightly overestimates).
        """
        return self._normal.entropy().sum(dim=-1)


class ContDist:
    """
    Wrapper for continuous distributions providing consistent interface.

    Provides .mode, .sample(), .log_prob() for any underlying distribution.
    Optionally clips samples to a maximum absolute value.
    """

    def __init__(self, dist: D.Distribution, absmax: float = None):
        """
        Args:
            dist: Underlying PyTorch distribution
            absmax: Optional clipping for samples
        """
        self._dist = dist
        self._absmax = absmax

    def sample(self) -> torch.Tensor:
        sample = self._dist.rsample()
        if self._absmax is not None:
            sample = torch.clamp(sample, -self._absmax, self._absmax)
        return sample

    def rsample(self) -> torch.Tensor:
        return self.sample()

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self._dist, 'mean'):
            mode = self._dist.mean
        elif hasattr(self._dist, 'base_dist'):
            mode = self._dist.base_dist.mean
        else:
            # Fallback: sample and hope for the best
            mode = self._dist.sample()

        if self._absmax is not None:
            mode = torch.clamp(mode, -self._absmax, self._absmax)
        return mode

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()


class GaussianDist:
    """
    Simple Gaussian distribution for latent states.

    Used for the stochastic component of the RSSM world model.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        self._dist = D.Normal(mean, std)

    def sample(self) -> torch.Tensor:
        return self._dist.rsample()

    def rsample(self) -> torch.Tensor:
        return self.sample()

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(dim=-1)

    def kl_divergence(self, other: 'GaussianDist') -> torch.Tensor:
        """KL divergence from self to other: KL(self || other)."""
        kl = D.kl_divergence(self._dist, other._dist)
        return kl.sum(dim=-1)


def build_normal_dist(mean: torch.Tensor, std: torch.Tensor) -> GaussianDist:
    """Factory function for Gaussian distributions."""
    return GaussianDist(mean, std)
