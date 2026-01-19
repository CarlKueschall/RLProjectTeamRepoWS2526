"""
Episode-based replay buffer for DreamerV3.

DreamerV3 trains on sequences of transitions, so we store
complete episodes and sample fixed-length subsequences.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional


class EpisodeBuffer:
    """
    Episode-based replay buffer for sequence model training.

    Stores complete episodes and samples fixed-length sequences
    for world model training. Uses FIFO eviction when capacity is reached.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
    ):
        """
        Args:
            capacity: Maximum number of transitions to store
            obs_shape: Shape of observations
            action_shape: Shape of actions
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Store complete episodes
        self.episodes = deque()
        self.total_transitions = 0

        # Current episode being built
        self._current_episode = {
            'obs': [],
            'action': [],
            'reward': [],
            'is_first': [],
            'is_terminal': [],
        }

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        is_first: bool = False,
    ):
        """
        Add a single transition to the current episode.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            done: Whether episode terminated
            is_first: Whether this is the first step of episode
        """
        self._current_episode['obs'].append(obs.astype(np.float32))
        self._current_episode['action'].append(action.astype(np.float32))
        self._current_episode['reward'].append(float(reward))
        self._current_episode['is_first'].append(is_first)
        self._current_episode['is_terminal'].append(done)

        if done:
            self._finalize_episode()

    def _finalize_episode(self):
        """Store completed episode and reset current episode."""
        if len(self._current_episode['obs']) == 0:
            return

        # Convert lists to arrays
        episode = {
            'obs': np.stack(self._current_episode['obs']),
            'action': np.stack(self._current_episode['action']),
            'reward': np.array(self._current_episode['reward'], dtype=np.float32),
            'is_first': np.array(self._current_episode['is_first'], dtype=np.float32),
            'is_terminal': np.array(self._current_episode['is_terminal'], dtype=np.float32),
        }

        episode_length = len(episode['obs'])
        self.episodes.append(episode)
        self.total_transitions += episode_length

        # FIFO eviction to stay under capacity
        while self.total_transitions > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.total_transitions -= len(removed['obs'])

        # Reset current episode
        self._current_episode = {
            'obs': [],
            'action': [],
            'reward': [],
            'is_first': [],
            'is_terminal': [],
        }

    def sample(self, batch_size: int, seq_length: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Sample a batch of sequences for training.

        Args:
            batch_size: Number of sequences to sample
            seq_length: Length of each sequence

        Returns:
            Dictionary with batched sequences:
            - obs: (batch, seq_length, *obs_shape)
            - action: (batch, seq_length, *action_shape)
            - reward: (batch, seq_length)
            - is_first: (batch, seq_length)
            - is_terminal: (batch, seq_length)
        """
        if len(self.episodes) == 0:
            return None

        # Filter episodes that are long enough
        valid_episodes = [ep for ep in self.episodes if len(ep['obs']) >= seq_length]

        if len(valid_episodes) == 0:
            # Fall back to padding shorter episodes
            valid_episodes = list(self.episodes)

        batch = {
            'obs': [],
            'action': [],
            'reward': [],
            'is_first': [],
            'is_terminal': [],
        }

        for _ in range(batch_size):
            # Sample random episode
            ep_idx = np.random.randint(len(valid_episodes))
            episode = valid_episodes[ep_idx]
            ep_length = len(episode['obs'])

            # Sample random starting position
            if ep_length >= seq_length:
                start = np.random.randint(0, ep_length - seq_length + 1)
                end = start + seq_length
            else:
                # Episode too short - start from 0 and pad
                start = 0
                end = ep_length

            # Extract sequence
            seq_obs = episode['obs'][start:end]
            seq_action = episode['action'][start:end]
            seq_reward = episode['reward'][start:end]
            seq_is_first = episode['is_first'][start:end]
            seq_is_terminal = episode['is_terminal'][start:end]

            # Pad if necessary
            actual_len = end - start
            if actual_len < seq_length:
                pad_len = seq_length - actual_len

                seq_obs = np.concatenate([
                    seq_obs,
                    np.zeros((pad_len,) + self.obs_shape, dtype=np.float32)
                ])
                seq_action = np.concatenate([
                    seq_action,
                    np.zeros((pad_len,) + self.action_shape, dtype=np.float32)
                ])
                seq_reward = np.concatenate([
                    seq_reward,
                    np.zeros(pad_len, dtype=np.float32)
                ])
                seq_is_first = np.concatenate([
                    seq_is_first,
                    np.zeros(pad_len, dtype=np.float32)
                ])
                seq_is_terminal = np.concatenate([
                    seq_is_terminal,
                    np.ones(pad_len, dtype=np.float32)  # Pad with terminal=True
                ])

            batch['obs'].append(seq_obs)
            batch['action'].append(seq_action)
            batch['reward'].append(seq_reward)
            batch['is_first'].append(seq_is_first)
            batch['is_terminal'].append(seq_is_terminal)

        # Stack into arrays
        return {
            'obs': np.stack(batch['obs']),
            'action': np.stack(batch['action']),
            'reward': np.stack(batch['reward']),
            'is_first': np.stack(batch['is_first']),
            'is_terminal': np.stack(batch['is_terminal']),
        }

    def __len__(self) -> int:
        """Return total number of transitions stored."""
        return self.total_transitions

    @property
    def num_episodes(self) -> int:
        """Return number of complete episodes stored."""
        return len(self.episodes)

    def stats(self) -> Dict:
        """Return buffer statistics."""
        if len(self.episodes) == 0:
            return {
                'num_episodes': 0,
                'total_transitions': 0,
                'avg_episode_length': 0,
            }

        lengths = [len(ep['obs']) for ep in self.episodes]
        return {
            'num_episodes': len(self.episodes),
            'total_transitions': self.total_transitions,
            'avg_episode_length': np.mean(lengths),
            'min_episode_length': np.min(lengths),
            'max_episode_length': np.max(lengths),
        }
