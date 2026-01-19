"""
Self-Play Manager for Hockey Training.

Agent-agnostic implementation - works with any agent that provides:
- state() method returning serializable state
- restore_state(state) method

AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
Updated with Claude Code assistance for DreamerV3 compatibility.
"""

import os
import pathlib
from collections import deque
import numpy as np
import torch


class SelfPlayManager:
    """
    Manage self-play opponent pool and selection.
    Features: Pool management + PFSP opponent selection
    """
    def __init__(self, pool_size=10, save_interval=500,
                 weak_ratio=0.5, device=None,
                 use_pfsp=False, pfsp_mode="variance",
                 observation_space=None, action_space=None, enabled=True):
        """
        Initialize self-play manager.

        Arguments:
            pool_size: Number of past checkpoints to keep
            save_interval: Save interval for adding new opponents
            weak_ratio: Ratio of episodes to train against weak opponent
            device: Torch device for loading opponents
            use_pfsp: Use PFSP opponent selection
            pfsp_mode: PFSP mode ('variance' or 'hard')
            observation_space: Gym observation space for creating opponents
            action_space: Gym action space for creating opponents
            enabled: Enable/disable self-play
        """
        self.pool_size = pool_size
        self.save_interval = save_interval
        self.weak_ratio = weak_ratio
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space

        # PFSP settings
        self.use_pfsp = use_pfsp
        self.pfsp_mode = pfsp_mode

        # Enable/disable flag for self-play
        self.enabled = enabled

        # Self-play state
        self.active = False
        self.start_episode = 0
        self.pool = []  # List of checkpoint paths
        self.opponent = None  # Currently loaded opponent network
        self.opponent_path = None
        self.current_opponent_idx = -1
        self.current_opponent_episode = 0

        # PFSP tracking
        self.opponent_winrates = {}  # {path: deque(results)}
        self.opponent_games_played = {}  # {path: int}
        self.opponent_episodes = {}  # {path: episode_num}

        # Anchor buffer balance tracking (for weak/strong balance)
        self.anchor_weak_count = 0
        self.anchor_strong_count = 0

    def activate(self, episode, checkpoints_dir, agent):
        """Activate self-play and seed the pool with current agent."""
        if self.active:
            return

        self.active = True
        self.start_episode = episode

        # Reset anchor balance tracking when self-play activates
        self.anchor_weak_count = 0
        self.anchor_strong_count = 0

        # Seed the pool with current agent
        seed_path = pathlib.Path(checkpoints_dir) / f'selfplay_seed_ep{episode}.pth'
        agent_state = agent.state()
        torch.save({'agent_state': agent_state}, seed_path)
        self.pool.append(str(seed_path))

        if self.use_pfsp:
            self.opponent_winrates[str(seed_path)] = deque(maxlen=100)
            self.opponent_games_played[str(seed_path)] = 0
            self.opponent_episodes[str(seed_path)] = episode

        # Seed pool with historical checkpoints (for diversity)
        parent_dir = pathlib.Path(checkpoints_dir).parent / 'checkpoints'
        if parent_dir.exists():
            seed_interval = max(500, (episode - self.start_episode) // (self.pool_size - 1)) if episode > 1000 else 500
            for ep in range(max(1000, episode - seed_interval * (self.pool_size - 1)), episode, seed_interval):
                checkpoint_pattern = f'*_ep{ep}_*.pth'
                matching_files = list(parent_dir.glob(checkpoint_pattern))
                if matching_files and len(self.pool) < self.pool_size:
                    checkpoint_path = str(matching_files[0])
                    self.pool.append(checkpoint_path)
                    if self.use_pfsp:
                        self.opponent_winrates[checkpoint_path] = deque(maxlen=100)
                        self.opponent_games_played[checkpoint_path] = 0
                        self.opponent_episodes[checkpoint_path] = ep

        print("\n" + "="*70)
        print(f"SELF-PLAY ACTIVATED AT EPISODE {episode}!")
        print("="*70)
        print(f"Pool seeded with {len(self.pool)} opponent(s)")
        print(f"PFSP enabled: {self.use_pfsp}")
        print("="*70 + "\n")

    def should_activate(self, episode, eval_vs_weak=None):
        """Check if self-play should activate."""
        if not self.enabled:
            return False
        return episode >= self.start_episode

    def update_pool(self, episode, agent, checkpoints_dir):
        """Add current agent to opponent pool."""
        if not self.active or episode % self.save_interval != 0:
            return None

        new_path = pathlib.Path(checkpoints_dir) / f'selfplay_pool_ep{episode}.pth'
        torch.save({'agent_state': agent.state()}, new_path)
        self.pool.append(str(new_path))

        if self.use_pfsp:
            self.opponent_winrates[str(new_path)] = deque(maxlen=100)
            self.opponent_games_played[str(new_path)] = 0
            self.opponent_episodes[str(new_path)] = episode

        # Keep pool bounded
        removed_episode = None
        if len(self.pool) > self.pool_size:
            old_path = self.pool.pop(0)
            removed_episode = self.opponent_episodes.get(old_path, "unknown")

            if self.use_pfsp and old_path in self.opponent_winrates:
                del self.opponent_winrates[old_path]
                del self.opponent_games_played[old_path]
                del self.opponent_episodes[old_path]

        return removed_episode

    def select_opponent(self, episode):
        """
        Select opponent from pool for this episode.
        Returns: 'weak', 'strong', or 'self-play'
        """
        if not self.active or not self.pool:
            return 'weak'

        # Decide whether to use anchor opponent (weak/strong) or self-play
        use_anchor = np.random.random() < self.weak_ratio

        if use_anchor:
            # Balance weak vs strong (target: 50/50)
            total_anchor = self.anchor_weak_count + self.anchor_strong_count

            if total_anchor == 0:
                self.anchor_weak_count += 1
                self.opponent = None
                return 'weak'

            weak_ratio_in_anchor = self.anchor_weak_count / total_anchor

            if weak_ratio_in_anchor < 0.5:
                self.anchor_weak_count += 1
                self.opponent = None
                return 'weak'
            elif weak_ratio_in_anchor > 0.5:
                self.anchor_strong_count += 1
                self.opponent = None
                return 'strong'
            else:
                if np.random.random() < 0.5:
                    self.anchor_weak_count += 1
                    self.opponent = None
                    return 'weak'
                else:
                    self.anchor_strong_count += 1
                    self.opponent = None
                    return 'strong'
        else:
            # Select from self-play pool
            if self.use_pfsp:
                selected_path = self._pfsp_select()
            else:
                selected_path = np.random.choice(self.pool)

            self._load_opponent(selected_path)
            return 'self-play'

    def _pfsp_select(self):
        """Select opponent using PFSP weighting."""
        weights = []
        valid_opponents = []

        for opp_path in self.pool:
            if opp_path in self.opponent_winrates and len(self.opponent_winrates[opp_path]) >= 10:
                results = list(self.opponent_winrates[opp_path])
                wins = sum(1 for r in results if r == 1)
                winrate = wins / len(results)

                from .pfsp import pfsp_weight
                weight = pfsp_weight(winrate, mode=self.pfsp_mode)
                weights.append(weight)
                valid_opponents.append(opp_path)
            else:
                weights.append(1.0)
                valid_opponents.append(opp_path)

        if sum(weights) > 0:
            weights_array = np.array(weights) / sum(weights)
            return np.random.choice(valid_opponents, p=weights_array)
        else:
            return np.random.choice(self.pool)

    def _load_opponent(self, path):
        """
        Load opponent checkpoint from path.

        Note: This is now agent-agnostic. It loads the checkpoint but doesn't
        create a specific agent type. The calling code should use
        get_opponent_state() to get the loaded state and create the appropriate
        agent type.
        """
        if path == self.opponent_path:
            return  # Already loaded

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Extract agent state (format depends on how it was saved)
            if isinstance(checkpoint, dict) and 'agent_state' in checkpoint:
                self.opponent_state = checkpoint['agent_state']
            else:
                self.opponent_state = checkpoint

            self.opponent_path = path
            self.current_opponent_idx = self.pool.index(path)
            self.current_opponent_episode = self.opponent_episodes.get(path, 0)

            # For backwards compatibility, set opponent to a marker
            # The calling code should use get_opponent_state() instead
            self.opponent = "loaded"

        except Exception as e:
            print(f"Failed to load self-play opponent: {e}")
            import traceback
            traceback.print_exc()
            self.opponent = None
            self.opponent_state = None

    def get_opponent_state(self):
        """
        Get the loaded opponent's state dict.

        Returns:
            The opponent's state (format depends on agent type) or None if not loaded.
        """
        return getattr(self, 'opponent_state', None)

    def get_action(self, obs):
        """
        DEPRECATED: Use get_opponent_state() and create your own agent instead.

        This method is kept for backwards compatibility but may not work
        with all agent types.
        """
        # This method is now deprecated - the calling code should
        # handle opponent action generation
        raise NotImplementedError(
            "get_action() is deprecated. Use get_opponent_state() to get the "
            "opponent's state and create an appropriate agent to generate actions."
        )

    def record_result(self, winner, use_weak):
        """Record game result against opponent for PFSP tracking."""
        if use_weak or not self.use_pfsp:
            return

        if self.opponent_path in self.opponent_winrates:
            self.opponent_winrates[self.opponent_path].append(winner)
            self.opponent_games_played[self.opponent_path] += 1

    def get_stats(self):
        """Get self-play metrics for W&B tracking."""
        stats = {
            'selfplay/active': 1.0 if self.active else 0.0,
            'selfplay/pool_size': len(self.pool),
            'selfplay/weak_ratio_target': self.weak_ratio,
        }

        # Anchor buffer balance
        total_anchor = self.anchor_weak_count + self.anchor_strong_count
        if total_anchor > 0:
            anchor_weak_ratio = self.anchor_weak_count / total_anchor
            stats['selfplay/anchor_weak_episodes'] = self.anchor_weak_count
            stats['selfplay/anchor_strong_episodes'] = self.anchor_strong_count
            stats['selfplay/anchor_weak_ratio'] = anchor_weak_ratio

        # PFSP metrics
        if self.use_pfsp and len(self.opponent_winrates) > 0:
            opponent_winrates_list = []

            for opp_path in self.opponent_winrates:
                results = list(self.opponent_winrates[opp_path])
                if len(results) >= 10:
                    wins = sum(1 for r in results if r == 1)
                    winrate = wins / len(results)
                    opponent_winrates_list.append(winrate)

            if opponent_winrates_list:
                stats['selfplay/pfsp_num_opponents_tracked'] = len(opponent_winrates_list)
                stats['selfplay/pfsp_avg_winrate'] = np.mean(opponent_winrates_list)
                stats['selfplay/pfsp_std_winrate'] = np.std(opponent_winrates_list)
                stats['selfplay/pfsp_min_winrate'] = np.min(opponent_winrates_list)
                stats['selfplay/pfsp_max_winrate'] = np.max(opponent_winrates_list)

            if self.current_opponent_idx >= 0 and self.active:
                stats['selfplay/opponent_pool_index'] = self.current_opponent_idx
                stats['selfplay/opponent_checkpoint_episode'] = self.current_opponent_episode

        return stats
