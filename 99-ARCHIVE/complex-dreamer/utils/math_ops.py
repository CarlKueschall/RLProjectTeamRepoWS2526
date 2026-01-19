"""
Mathematical operations for DreamerV3.

Contains:
- symlog/symexp: Symmetric logarithm for handling varied reward scales
- TwoHotDist: Two-hot encoding for discrete regression (DreamerV3 paper)
- lambda_returns: TD(λ) return computation for value targets
- adaptive_gradient_clip: Per-layer adaptive gradient clipping (AGC)

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def adaptive_gradient_clip(parameters, clip_factor: float = 0.3, eps: float = 1e-3):
    """
    Adaptive Gradient Clipping (AGC) from DreamerV3 / NFNet papers.

    Clips gradients based on the ratio of gradient norm to parameter norm,
    applied per-layer. This is more stable than global norm clipping.

    AGC clips gradients when: ||g|| > clip_factor * ||w||

    Args:
        parameters: Iterable of model parameters
        clip_factor: Maximum allowed ratio of grad norm to param norm (default: 0.3)
        eps: Small constant to avoid division by zero

    Returns:
        total_norm: Total gradient norm before clipping (for logging)
    """
    total_norm = 0.0

    for p in parameters:
        if p.grad is None:
            continue

        # Compute norms
        param_norm = p.data.norm(2).item()
        grad_norm = p.grad.data.norm(2).item()

        total_norm += grad_norm ** 2

        # Skip if parameter is too small (avoid division issues)
        if param_norm < eps:
            continue

        # Compute maximum allowed gradient norm
        max_grad_norm = clip_factor * param_norm

        # Clip if necessary
        if grad_norm > max_grad_norm:
            clip_coef = max_grad_norm / (grad_norm + eps)
            p.grad.data.mul_(clip_coef)

    return total_norm ** 0.5


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


class TwoHotDist:
    """
    Two-Hot Encoding Distribution for Discrete Regression (DreamerV3).

    Instead of predicting a scalar directly (which fails with MSE on sparse data),
    DreamerV3 predicts a categorical distribution over fixed bins. The target is
    encoded as a "two-hot" vector where probability mass is distributed between
    the two bins adjacent to the target value.

    Key benefits:
    1. Zero logits = uniform distribution (not zero output)
    2. Cross-entropy loss provides gradients even for rare events
    3. Symlog transform keeps gradients stable across reward magnitudes

    Usage:
        dist = TwoHotDist(logits, bins)  # logits from network
        loss = dist.log_prob(target)     # cross-entropy with two-hot target
        mode = dist.mode                  # most likely value (for inference)
    """

    def __init__(
        self,
        logits: torch.Tensor,
        bins: torch.Tensor,
        low: float = -20.0,
        high: float = 20.0,
    ):
        """
        Args:
            logits: Raw logits from network (..., num_bins)
            bins: Bin centers in symlog space (num_bins,)
            low: Lower bound of bin range (in symlog space)
            high: Upper bound of bin range (in symlog space)
        """
        self.logits = logits
        self.bins = bins
        self.low = low
        self.high = high
        self.num_bins = bins.shape[0]

        # Softmax to get probabilities
        self.probs = F.softmax(logits, dim=-1)

    @property
    def mode(self) -> torch.Tensor:
        """
        Most likely value (expected value under the distribution).

        Returns the weighted sum of bin centers, then applies symexp
        to convert from symlog space back to original space.
        """
        # Expected value in symlog space
        symlog_value = (self.probs * self.bins).sum(dim=-1)
        # Convert back to original space
        return symexp(symlog_value)

    @property
    def mean(self) -> torch.Tensor:
        """Alias for mode (expected value)."""
        return self.mode

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of target under the distribution.

        Uses two-hot encoding: the target is converted to symlog space,
        then probability is distributed between the two adjacent bins.

        Args:
            target: Target values in original space (...,)

        Returns:
            Log probability (...,)
        """
        # Convert target to symlog space
        target_symlog = symlog(target)

        # Compute two-hot encoding
        two_hot = self._two_hot_encode(target_symlog)

        # Cross-entropy: -sum(two_hot * log_softmax(logits))
        log_probs = F.log_softmax(self.logits, dim=-1)
        return (two_hot * log_probs).sum(dim=-1)

    def _two_hot_encode(self, target_symlog: torch.Tensor) -> torch.Tensor:
        """
        Encode target as two-hot vector.

        Distributes probability between the two bins adjacent to the target.
        If target falls exactly on a bin, all mass goes to that bin.

        Args:
            target_symlog: Target values in symlog space (...,)

        Returns:
            Two-hot encoding (..., num_bins)
        """
        # Clamp to valid range
        target_clamped = torch.clamp(target_symlog, self.low, self.high)

        # Find bin indices
        # bins is (num_bins,), target_clamped is (...)
        # We need to find which bin each target falls into

        # Compute distance from each target to each bin
        # Expand dimensions for broadcasting
        target_expanded = target_clamped.unsqueeze(-1)  # (..., 1)
        bins_expanded = self.bins  # (num_bins,)

        # Find the two closest bins
        diff = target_expanded - bins_expanded  # (..., num_bins)

        # Get indices of bins below and above target
        below_mask = diff >= 0
        above_mask = diff <= 0

        # Find the closest bin below (largest index where diff >= 0)
        below_idx = below_mask.long().sum(dim=-1) - 1
        below_idx = torch.clamp(below_idx, 0, self.num_bins - 2)

        # The bin above is just the next one
        above_idx = below_idx + 1

        # Get the bin values
        below_val = self.bins[below_idx]
        above_val = self.bins[above_idx]

        # Compute interpolation weights
        # Weight for above bin = how far target is from below bin
        # Weight for below bin = how far target is from above bin
        total_dist = above_val - below_val
        # Avoid division by zero (when target is exactly on a bin)
        total_dist = torch.clamp(total_dist, min=1e-8)

        above_weight = (target_clamped - below_val) / total_dist
        below_weight = 1.0 - above_weight

        # Clamp weights to [0, 1]
        above_weight = torch.clamp(above_weight, 0.0, 1.0)
        below_weight = torch.clamp(below_weight, 0.0, 1.0)

        # Create two-hot vector
        batch_shape = target_symlog.shape
        two_hot = torch.zeros(*batch_shape, self.num_bins, device=target_symlog.device)

        # Scatter the weights
        two_hot.scatter_(-1, below_idx.unsqueeze(-1), below_weight.unsqueeze(-1))
        two_hot.scatter_(-1, above_idx.unsqueeze(-1), above_weight.unsqueeze(-1))

        return two_hot


def create_bins(num_bins: int = 255, low: float = -20.0, high: float = 20.0,
                device: str = 'cpu') -> torch.Tensor:
    """
    Create uniformly-spaced bin centers for two-hot encoding.

    The bins are in symlog space, so they cover a wide range of values
    in the original space (from symexp(-20) ≈ -485M to symexp(20) ≈ 485M).

    For hockey with rewards in [-1, +1]:
    - symlog(-1) ≈ -0.69
    - symlog(+1) ≈ +0.69
    So most of the bins are unused, but the resolution is fine near 0.

    Args:
        num_bins: Number of bins (default: 255, from DreamerV3 paper)
        low: Lower bound in symlog space
        high: Upper bound in symlog space
        device: Torch device

    Returns:
        Tensor of bin centers (num_bins,)
    """
    return torch.linspace(low, high, num_bins, device=device)


class TwoHotDistLayer(nn.Module):
    """
    Neural network layer that outputs a TwoHotDist.

    Use this in place of a scalar output head for reward/value prediction.
    Zero initialization of logits = uniform distribution (safe start).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_bins: Number of output bins
            low: Lower bound in symlog space
            high: Upper bound in symlog space
        """
        super().__init__()

        self.num_bins = num_bins
        self.low = low
        self.high = high

        # MLP to predict logits
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

        # Initialize final layer to zero (uniform distribution)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Register bins as buffer (not a parameter)
        self.register_buffer('bins', create_bins(num_bins, low, high))

    def forward(self, features: torch.Tensor) -> TwoHotDist:
        """
        Predict distribution over values.

        Args:
            features: Input features (..., input_dim)

        Returns:
            TwoHotDist distribution
        """
        logits = self.net(features)
        return TwoHotDist(logits, self.bins, self.low, self.high)


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
