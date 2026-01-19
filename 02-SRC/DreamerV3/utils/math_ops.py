"""
Mathematical operations for DreamerV3.

Contains:
- symlog/symexp: Symmetric logarithm for handling varied reward scales
- lambda_returns: TD(λ) return computation for value targets

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithm transformation.

    Compresses large values while preserving sign:
    symlog(x) = sign(x) * ln(|x| + 1)

    This allows neural networks to handle rewards/values
    across many orders of magnitude with fixed hyperparameters.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of symlog.

    symexp(x) = sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """
    Compute TD(λ) returns for value function targets.

    This implements Generalized Advantage Estimation (GAE) style returns:
    G_t = r_t + γ * ((1-λ) * V_{t+1} + λ * G_{t+1})

    Args:
        rewards: Rewards at each timestep (T, B) or (T, B, 1)
        values: Value estimates at each timestep (T, B) or (T, B, 1)
        continues: Continue probabilities (1 - done) (T, B) or (T, B, 1)
        bootstrap: Value estimate for final state (B,) or (B, 1)
        gamma: Discount factor
        lambda_: TD(λ) mixing parameter

    Returns:
        Lambda returns for each timestep (T, B) or (T, B, 1)
    """
    # Handle dimension variations
    if rewards.dim() == 3:
        rewards = rewards.squeeze(-1)
    if values.dim() == 3:
        values = values.squeeze(-1)
    if continues.dim() == 3:
        continues = continues.squeeze(-1)
    if bootstrap.dim() == 2:
        bootstrap = bootstrap.squeeze(-1)

    T, B = rewards.shape
    returns = torch.zeros(T, B, device=rewards.device, dtype=rewards.dtype)

    # Start from the end and work backwards
    next_return = bootstrap

    for t in reversed(range(T)):
        # TD(λ) formula:
        # G_t = r_t + γ * c_t * ((1-λ) * V_{t+1} + λ * G_{t+1})
        # where c_t is the continue probability

        next_value = values[t + 1] if t + 1 < T else bootstrap
        td_target = rewards[t] + gamma * continues[t] * next_value

        returns[t] = (1 - lambda_) * td_target + lambda_ * (
            rewards[t] + gamma * continues[t] * next_return
        )
        next_return = returns[t]

    return returns


class ReturnNormalizer:
    """
    Normalize returns using running percentile estimates.

    DreamerV3 uses 5th and 95th percentile to normalize returns,
    which allows fixed entropy scale across different reward magnitudes.
    """

    def __init__(self, decay: float = 0.99, scale_min: float = 1.0, device='cpu'):
        """
        Args:
            decay: EMA decay rate for percentile estimates
            scale_min: Minimum scale to avoid amplifying noise
            device: Torch device
        """
        self.decay = decay
        self.scale_min = scale_min
        self.device = device

        # Running estimates of 5th and 95th percentiles
        self.low = torch.tensor(0.0, device=device)
        self.high = torch.tensor(0.0, device=device)
        self.initialized = False

    def update(self, returns: torch.Tensor):
        """Update percentile estimates with new returns."""
        flat = returns.detach().flatten()

        if len(flat) < 2:
            return

        # Compute percentiles
        low = torch.quantile(flat, 0.05)
        high = torch.quantile(flat, 0.95)

        if not self.initialized:
            self.low = low
            self.high = high
            self.initialized = True
        else:
            # Exponential moving average
            self.low = self.decay * self.low + (1 - self.decay) * low
            self.high = self.decay * self.high + (1 - self.decay) * high

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns using current percentile estimates."""
        if not self.initialized:
            return returns

        scale = torch.clamp(self.high - self.low, min=self.scale_min)
        return (returns - self.low) / scale

    def state_dict(self):
        return {
            'low': self.low,
            'high': self.high,
            'initialized': self.initialized,
        }

    def load_state_dict(self, state):
        self.low = state['low']
        self.high = state['high']
        self.initialized = state['initialized']
