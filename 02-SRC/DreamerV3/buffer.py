"""
Replay Buffer for DreamerV3.

Simple sequence-based buffer that stores transitions and samples
contiguous sequences for training.

Includes DreamSmooth: temporal reward smoothing for sparse reward environments.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import numpy as np
import torch


def dreamsmooth_ema(rewards, dones, alpha=0.5):
    """
    Apply bidirectional EMA smoothing to rewards within episodes.

    DreamSmooth (arXiv:2311.01450) smooths rewards before world model training,
    making reward prediction easier and providing denser learning signal.

    Instead of sparse spikes (0, 0, 0, +10, 0, 0), produces (0, 0.5, 2.5, 6.25, 3.1, 1.5)

    Args:
        rewards: Reward tensor of shape (batch, seq_len, 1) or (batch, seq_len)
        dones: Done flags of shape (batch, seq_len, 1) or (batch, seq_len)
        alpha: Smoothing factor (0-1). Higher = more smoothing.
                0.5 is recommended for sparse rewards.

    Returns:
        Smoothed rewards with same shape as input
    """
    # Handle shape
    squeeze_last = False
    if rewards.dim() == 3 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        squeeze_last = True

    batch_size, seq_len = rewards.shape
    smoothed = torch.zeros_like(rewards)

    # Forward pass: smooth from past to present
    for b in range(batch_size):
        running = 0.0
        for t in range(seq_len):
            # Reset at episode boundary
            if t > 0 and dones[b, t-1] > 0.5:
                running = 0.0
            running = alpha * rewards[b, t] + (1 - alpha) * running
            smoothed[b, t] = running

    # Backward pass: smooth from future to present (adds anticipation signal)
    for b in range(batch_size):
        running = 0.0
        for t in range(seq_len - 1, -1, -1):
            # Reset at episode boundary (looking backward)
            if t < seq_len - 1 and dones[b, t] > 0.5:
                running = 0.0
            running = alpha * smoothed[b, t] + (1 - alpha) * running
            smoothed[b, t] = running

    if squeeze_last:
        smoothed = smoothed.unsqueeze(-1)

    return smoothed


class ReplayBuffer:
    """
    Replay buffer storing transitions for sequence sampling.

    Stores (obs, action, reward, next_obs, done) tuples and samples
    contiguous sequences for world model training.
    """

    def __init__(self, observationSize, actionSize, config, device):
        """
        Args:
            observationSize: Dimension of observation (scalar for 1D, tuple for images)
            actionSize: Dimension of action
            config: Config with 'capacity' field and optional DreamSmooth settings
            device: torch device for sampling
        """
        self.config = config
        self.device = device
        self.capacity = int(config.capacity)

        # DreamSmooth settings (for sparse reward environments)
        self.useDreamSmooth = getattr(config, 'useDreamSmooth', False)
        self.dreamsmoothAlpha = getattr(config, 'dreamsmoothAlpha', 0.5)

        # Handle both 1D (int) and multi-D (tuple) observations
        if isinstance(observationSize, int):
            obsShape = (observationSize,)
        else:
            obsShape = observationSize

        self.observations = np.empty((self.capacity, *obsShape), dtype=np.float32)
        self.nextObservations = np.empty((self.capacity, *obsShape), dtype=np.float32)
        self.actions = np.empty((self.capacity, actionSize), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

        self.bufferIndex = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, action, reward, nextObservation, done):
        """Add a single transition to the buffer."""
        self.observations[self.bufferIndex] = observation
        self.actions[self.bufferIndex] = action
        self.rewards[self.bufferIndex] = reward
        self.nextObservations[self.bufferIndex] = nextObservation
        self.dones[self.bufferIndex] = done

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        """
        Sample a batch of contiguous sequences.

        Args:
            batchSize: Number of sequences to sample
            sequenceSize: Length of each sequence

        Returns:
            Batch object with observations, actions, rewards, nextObservations, dones
        """
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), \
            f"Not enough data: need {batchSize} sequences of length {sequenceSize}, have {len(self)} transitions"

        # Sample starting indices
        maxIdx = self.capacity if self.full else lastFilledIndex
        sampleIndex = np.random.randint(0, maxIdx, batchSize).reshape(-1, 1)
        sequenceOffset = np.arange(sequenceSize).reshape(1, -1)
        sampleIndex = (sampleIndex + sequenceOffset) % self.capacity

        # Fetch data
        observations = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        nextObservations = torch.as_tensor(self.nextObservations[sampleIndex], device=self.device).float()
        actions = torch.as_tensor(self.actions[sampleIndex], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[sampleIndex], device=self.device).float()
        dones = torch.as_tensor(self.dones[sampleIndex], device=self.device).float()

        # Apply DreamSmooth if enabled (temporal reward smoothing for sparse rewards)
        if self.useDreamSmooth:
            rewards = dreamsmooth_ema(rewards, dones, alpha=self.dreamsmoothAlpha)

        # Return as simple namespace
        class Batch:
            pass

        batch = Batch()
        batch.observations = observations
        batch.actions = actions
        batch.rewards = rewards
        batch.nextObservations = nextObservations
        batch.dones = dones

        return batch
