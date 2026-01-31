"""
Self-Play Manager for DreamerV3 Hockey Training.

Manages opponent pool and selection for curriculum learning through self-play.
Works with DreamerV3 agents using state()/restore_state() for serialization.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import os
import pathlib
from collections import deque
import numpy as np
import torch

from .pfsp import pfsp_weight


class SelfPlayOpponent:
    """
    Wrapper for a DreamerV3 agent loaded from checkpoint to act as opponent.

    This opponent maintains recurrent state across steps within an episode.
    """

    def __init__(self, agent_state, agent_class, obs_size, action_size,
                 action_low, action_high, device, config):
        """
        Initialize self-play opponent from saved state.

        Args:
            agent_state: State dict from agent.state()
            agent_class: Dreamer class for creating opponent
            obs_size: Observation dimension
            action_size: Action dimension
            action_low: Action lower bounds
            action_high: Action upper bounds
            device: Torch device
            config: Agent config
        """
        self.device = device

        # Create agent instance and restore state
        self.agent = agent_class(obs_size, action_size, action_low, action_high, device, config)
        self.agent.restore_state(agent_state)

        # Recurrent states for episode
        self.h = None
        self.z = None
        self.prev_action = None

    def act(self, obs):
        """
        Get action from opponent.

        Args:
            obs: Observation (18 dims)

        Returns:
            action: Action (4 dims)
        """
        action, self.h, self.z = self.agent.act(obs, self.h, self.z, self.prev_action)
        self.prev_action = action
        return action

    def reset(self):
        """Reset recurrent states for new episode."""
        self.h = None
        self.z = None
        self.prev_action = None


class SelfPlayManager:
    """
    Manage self-play opponent pool and selection.

    Features:
    - Pool management: FIFO circular buffer of past checkpoints
    - PFSP opponent selection: Prioritize opponents with most learning signal
    - Anchor balancing: Maintain mix of fixed (weak/strong) and self-play opponents
    - Metrics tracking: Win rates, opponent selection stats for W&B
    """

    def __init__(self, pool_size=10, save_interval=500, weak_ratio=0.3,
                 device=None, use_pfsp=True, pfsp_mode="variance",
                 agent_class=None, obs_size=18, action_size=4,
                 action_low=None, action_high=None, config=None):
        """
        Initialize self-play manager.

        Args:
            pool_size: Maximum number of past checkpoints to keep
            save_interval: Episodes between adding new opponents to pool
            weak_ratio: Probability of training against anchor (weak/strong) vs pool
            device: Torch device
            use_pfsp: Enable PFSP opponent selection
            pfsp_mode: PFSP mode ('variance' or 'hard')
            agent_class: Dreamer class for creating opponents
            obs_size: Observation dimension
            action_size: Action dimension
            action_low: Action lower bounds
            action_high: Action upper bounds
            config: Agent config for creating opponents
        """
        self.pool_size = pool_size
        self.save_interval = save_interval
        self.weak_ratio = weak_ratio
        self.device = device
        self.use_pfsp = use_pfsp
        self.pfsp_mode = pfsp_mode

        # Agent creation settings
        self.agent_class = agent_class
        self.obs_size = obs_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.config = config

        # Self-play state
        self.active = False
        self.start_episode = 0
        self.pool = []  # List of checkpoint paths
        self.current_opponent = None  # Currently loaded SelfPlayOpponent
        self.current_opponent_path = None
        self.current_opponent_idx = -1
        self.current_opponent_episode = 0
        self.last_opponent_type = None  # 'weak', 'strong', or 'self-play'

        # PFSP tracking (win rates per opponent)
        self.opponent_results = {}  # {path: deque of results}
        self.opponent_games = {}  # {path: total games}
        self.opponent_episodes = {}  # {path: episode when added}

        # Age-stratified self-play win rate tracking
        # Pool is FIFO: index 0 = oldest, index -1 = newest
        # We split into thirds: oldest, middle, newest
        self.age_stratified_results = {
            'oldest': deque(maxlen=200),
            'middle': deque(maxlen=200),
            'newest': deque(maxlen=200),
        }
        self.overall_sp_results = deque(maxlen=500)

        # Anchor balance tracking
        self.anchor_weak_count = 0
        self.anchor_strong_count = 0
        self.selfplay_count = 0

    def activate(self, episode, checkpoints_dir, agent):
        """
        Activate self-play and seed the pool with current agent.

        Args:
            episode: Current episode number
            checkpoints_dir: Directory to save self-play checkpoints
            agent: Current agent to seed pool with
        """
        if self.active:
            return

        self.active = True
        self.start_episode = episode

        # Reset counters
        self.anchor_weak_count = 0
        self.anchor_strong_count = 0
        self.selfplay_count = 0

        # Create checkpoints directory
        selfplay_dir = pathlib.Path(checkpoints_dir) / 'selfplay'
        selfplay_dir.mkdir(parents=True, exist_ok=True)

        # Seed pool with current agent
        seed_path = selfplay_dir / f'selfplay_seed_ep{episode}.pth'
        agent_state = agent.state()
        torch.save({'agent_state': agent_state, 'episode': episode}, seed_path)
        self._add_to_pool(str(seed_path), episode)

        print("\n" + "=" * 70)
        print(f"SELF-PLAY ACTIVATED AT EPISODE {episode}!")
        print("=" * 70)
        print(f"Pool seeded with 1 opponent")
        print(f"PFSP enabled: {self.use_pfsp} (mode: {self.pfsp_mode})")
        print(f"Weak ratio: {self.weak_ratio} (prob of anchor vs self-play)")
        print(f"Pool size: {self.pool_size}, Save interval: {self.save_interval}")
        print("=" * 70 + "\n")

    def _add_to_pool(self, path, episode):
        """Add checkpoint path to pool with tracking."""
        self.pool.append(path)
        self.opponent_results[path] = deque(maxlen=100)
        self.opponent_games[path] = 0
        self.opponent_episodes[path] = episode

    def should_activate(self, episode, start_episode):
        """Check if self-play should activate."""
        return not self.active and episode >= start_episode

    def update_pool(self, episode, agent, checkpoints_dir):
        """
        Add current agent to opponent pool if it's time.

        Args:
            episode: Current episode number
            agent: Current agent to potentially add
            checkpoints_dir: Directory for checkpoints

        Returns:
            Path of removed opponent if pool was full, None otherwise
        """
        if not self.active:
            return None

        if episode % self.save_interval != 0:
            return None

        # Save current agent to pool
        selfplay_dir = pathlib.Path(checkpoints_dir) / 'selfplay'
        selfplay_dir.mkdir(parents=True, exist_ok=True)
        new_path = selfplay_dir / f'selfplay_pool_ep{episode}.pth'

        agent_state = agent.state()
        torch.save({'agent_state': agent_state, 'episode': episode}, new_path)
        self._add_to_pool(str(new_path), episode)

        # Remove oldest if pool is full
        removed_path = None
        if len(self.pool) > self.pool_size:
            removed_path = self.pool.pop(0)
            # Clean up tracking
            if removed_path in self.opponent_results:
                del self.opponent_results[removed_path]
                del self.opponent_games[removed_path]
                del self.opponent_episodes[removed_path]

        return removed_path

    def select_opponent(self):
        """
        Select opponent type for this episode.

        Returns:
            str: 'weak', 'strong', or 'self-play'
        """
        if not self.active or not self.pool:
            self.last_opponent_type = 'weak'
            return 'weak'

        # Decide: anchor (weak/strong) or self-play
        use_anchor = np.random.random() < self.weak_ratio

        if use_anchor:
            # Balance weak vs strong (target 50/50)
            total_anchor = self.anchor_weak_count + self.anchor_strong_count

            if total_anchor == 0:
                opponent_type = 'weak'
            else:
                weak_ratio_actual = self.anchor_weak_count / total_anchor
                if weak_ratio_actual < 0.5:
                    opponent_type = 'weak'
                elif weak_ratio_actual > 0.5:
                    opponent_type = 'strong'
                else:
                    opponent_type = 'weak' if np.random.random() < 0.5 else 'strong'

            # Update counts
            if opponent_type == 'weak':
                self.anchor_weak_count += 1
            else:
                self.anchor_strong_count += 1

            self.current_opponent = None
            self.current_opponent_path = None
            self.last_opponent_type = opponent_type
            return opponent_type

        else:
            # Select from self-play pool
            self.selfplay_count += 1

            if self.use_pfsp:
                selected_path = self._pfsp_select()
            else:
                selected_path = np.random.choice(self.pool)

            self._load_opponent(selected_path)
            self.last_opponent_type = 'self-play'
            return 'self-play'

    def _pfsp_select(self):
        """Select opponent using PFSP weighting."""
        weights = []
        valid_paths = []

        for path in self.pool:
            if path in self.opponent_results and len(self.opponent_results[path]) >= 5:
                # Compute win rate from recent results
                results = list(self.opponent_results[path])
                wins = sum(1 for r in results if r == 1)
                winrate = wins / len(results)
                weight = pfsp_weight(winrate, mode=self.pfsp_mode)
            else:
                # Not enough data, use uniform
                weight = 1.0

            weights.append(max(weight, 0.01))  # Minimum weight to ensure sampling
            valid_paths.append(path)

        # Normalize and sample
        weights = np.array(weights)
        weights = weights / weights.sum()
        return np.random.choice(valid_paths, p=weights)

    def _load_opponent(self, path):
        """Load opponent from checkpoint path."""
        if path == self.current_opponent_path and self.current_opponent is not None:
            return  # Already loaded

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'agent_state' in checkpoint:
                agent_state = checkpoint['agent_state']
                episode = checkpoint.get('episode', 0)
            else:
                agent_state = checkpoint
                episode = 0

            # Create opponent agent
            self.current_opponent = SelfPlayOpponent(
                agent_state=agent_state,
                agent_class=self.agent_class,
                obs_size=self.obs_size,
                action_size=self.action_size,
                action_low=self.action_low,
                action_high=self.action_high,
                device=self.device,
                config=self.config
            )

            self.current_opponent_path = path
            self.current_opponent_idx = self.pool.index(path)
            self.current_opponent_episode = episode

        except Exception as e:
            print(f"Failed to load self-play opponent from {path}: {e}")
            import traceback
            traceback.print_exc()
            self.current_opponent = None
            self.current_opponent_path = None

    def get_opponent(self):
        """
        Get current self-play opponent.

        Returns:
            SelfPlayOpponent or None if not in self-play mode
        """
        return self.current_opponent

    def record_result(self, winner, current_episode=None):
        """
        Record game result against current opponent for PFSP tracking.

        Args:
            winner: 1 for win, -1 for loss, 0 for draw
            current_episode: Current training episode (for age-stratified metrics)
        """
        if self.last_opponent_type != 'self-play':
            return

        if self.current_opponent_path in self.opponent_results:
            self.opponent_results[self.current_opponent_path].append(winner)
            self.opponent_games[self.current_opponent_path] += 1

        # Age-stratified tracking: classify opponent into tercile by pool position
        # Pool is FIFO: index 0 = oldest, index -1 = newest
        pool_size = len(self.pool)
        idx = self.current_opponent_idx
        if pool_size > 0 and 0 <= idx < pool_size:
            tercile_size = pool_size / 3.0
            if idx < tercile_size:
                self.age_stratified_results['oldest'].append(winner)
            elif idx < 2 * tercile_size:
                self.age_stratified_results['middle'].append(winner)
            else:
                self.age_stratified_results['newest'].append(winner)
            self.overall_sp_results.append(winner)

    def reset_opponent(self):
        """Reset opponent's recurrent state for new episode."""
        if self.current_opponent is not None:
            self.current_opponent.reset()

    def get_stats(self):
        """
        Get self-play metrics for W&B logging.

        Returns:
            dict: Metrics for logging
        """
        stats = {
            'selfplay/active': 1.0 if self.active else 0.0,
            'selfplay/pool_size': len(self.pool),
            'selfplay/weak_ratio_target': self.weak_ratio,
            'selfplay/episodes_since_activation': 0,
        }

        if self.active:
            total_episodes = self.anchor_weak_count + self.anchor_strong_count + self.selfplay_count
            stats['selfplay/episodes_since_activation'] = total_episodes

            # Anchor stats
            stats['selfplay/anchor_weak_count'] = self.anchor_weak_count
            stats['selfplay/anchor_strong_count'] = self.anchor_strong_count
            stats['selfplay/selfplay_count'] = self.selfplay_count

            if total_episodes > 0:
                stats['selfplay/anchor_ratio_actual'] = (
                    (self.anchor_weak_count + self.anchor_strong_count) / total_episodes
                )
                stats['selfplay/selfplay_ratio_actual'] = self.selfplay_count / total_episodes

            total_anchor = self.anchor_weak_count + self.anchor_strong_count
            if total_anchor > 0:
                stats['selfplay/anchor_weak_ratio'] = self.anchor_weak_count / total_anchor

            # PFSP stats
            if self.use_pfsp and self.opponent_results:
                winrates = []
                for path, results in self.opponent_results.items():
                    if len(results) >= 5:
                        wins = sum(1 for r in results if r == 1)
                        winrates.append(wins / len(results))

                if winrates:
                    stats['selfplay/pfsp_num_tracked'] = len(winrates)
                    stats['selfplay/pfsp_avg_winrate'] = np.mean(winrates)
                    stats['selfplay/pfsp_std_winrate'] = np.std(winrates)
                    stats['selfplay/pfsp_min_winrate'] = np.min(winrates)
                    stats['selfplay/pfsp_max_winrate'] = np.max(winrates)

            # Current opponent info
            if self.current_opponent_idx >= 0:
                stats['selfplay/current_opponent_idx'] = self.current_opponent_idx
                stats['selfplay/current_opponent_episode'] = self.current_opponent_episode

            # Age-stratified win rates
            for tercile in ('oldest', 'middle', 'newest'):
                results = self.age_stratified_results[tercile]
                if len(results) >= 5:
                    wins = sum(1 for r in results if r == 1)
                    stats[f'selfplay/winrate_{tercile}_third'] = wins / len(results)

            if len(self.overall_sp_results) >= 5:
                wins = sum(1 for r in self.overall_sp_results if r == 1)
                stats['selfplay/winrate_vs_pool_overall'] = wins / len(self.overall_sp_results)

            # Pool age info (in episodes)
            if self.pool and self.opponent_episodes:
                ages = []
                for path in self.pool:
                    if path in self.opponent_episodes:
                        ages.append(self.opponent_episodes[path])
                if ages:
                    stats['selfplay/oldest_opponent_episode'] = min(ages)
                    stats['selfplay/newest_opponent_episode'] = max(ages)

        return stats
