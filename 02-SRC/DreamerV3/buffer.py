"""
Replay Buffer for DreamerV3.

Simple sequence-based buffer that stores transitions and samples
contiguous sequences for training.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import numpy as np
import torch


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
            config: Config with 'capacity' field
            device: torch device for sampling
        """
        self.config = config
        self.device = device
        self.capacity = int(config.capacity)

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
