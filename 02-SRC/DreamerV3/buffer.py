"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

# Replay buffer for DreamerV3. Stores transitions, samples contiguous sequences.
# DreamSmooth: temporal reward smoothing for sparse rewards (arXiv:2311.01450).

import numpy as np
import torch


def dreamsmooth_ema(rewards, dones, alpha=0.5):
    """
    Bidirectional EMA smoothing on rewards within episodes.
    Turns sparse spikes (0,0,0,+10,0) into something denser like (0,0.5,2.5,6.25,3.1,1.5).
    alpha 0.5 works well for sparse rewards.
    """
    squeeze_last = False
    if rewards.dim() == 3 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        squeeze_last = True

    batch_size, seq_len = rewards.shape
    smoothed = torch.zeros_like(rewards)

    for b in range(batch_size):
        running = 0.0
        for t in range(seq_len):
            if t > 0 and dones[b, t-1] > 0.5:
                running = 0.0
            running = alpha * rewards[b, t] + (1 - alpha) * running
            smoothed[b, t] = running

    for b in range(batch_size):
        running = 0.0
        for t in range(seq_len - 1, -1, -1):
            if t < seq_len - 1 and dones[b, t] > 0.5:
                running = 0.0
            running = alpha * smoothed[b, t] + (1 - alpha) * running
            smoothed[b, t] = running

    if squeeze_last:
        smoothed = smoothed.unsqueeze(-1)

    return smoothed


class ReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) and samples contiguous sequences
    for world model training.
    """

    def __init__(self, observationSize, actionSize, config, device):
        self.config = config
        self.device = device
        self.capacity = int(config.capacity)

        self.useDreamSmooth = getattr(config, 'useDreamSmooth', False)
        self.dreamsmoothAlpha = getattr(config, 'dreamsmoothAlpha', 0.5)

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
        self.observations[self.bufferIndex] = observation
        self.actions[self.bufferIndex] = action
        self.rewards[self.bufferIndex] = reward
        self.nextObservations[self.bufferIndex] = nextObservation
        self.dones[self.bufferIndex] = done

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), \
            f"Not enough data: need {batchSize} sequences of length {sequenceSize}, have {len(self)} transitions"

        maxIdx = self.capacity if self.full else lastFilledIndex
        sampleIndex = np.random.randint(0, maxIdx, batchSize).reshape(-1, 1)
        sequenceOffset = np.arange(sequenceSize).reshape(1, -1)
        sampleIndex = (sampleIndex + sequenceOffset) % self.capacity

        observations = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        nextObservations = torch.as_tensor(self.nextObservations[sampleIndex], device=self.device).float()
        actions = torch.as_tensor(self.actions[sampleIndex], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[sampleIndex], device=self.device).float()
        dones = torch.as_tensor(self.dones[sampleIndex], device=self.device).float()

        if self.useDreamSmooth:
            rewards = dreamsmooth_ema(rewards, dones, alpha=self.dreamsmoothAlpha)

        class Batch:
            pass

        batch = Batch()
        batch.observations = observations
        batch.actions = actions
        batch.rewards = rewards
        batch.nextObservations = nextObservations
        batch.dones = dones

        return batch
