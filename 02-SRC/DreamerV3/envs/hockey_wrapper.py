"""
Hockey environment wrapper for DreamerV3.

This wrapper adapts the laser-hockey-gym environment for use with DreamerV3.
Key features:
- Sparse rewards only (no PBRS) - DreamerV3 handles credit assignment via imagination
- Compatible with gymnasium API
- Supports different opponents (weak, strong, self-play)

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import hockey.hockey_env as h_env


class HockeyEnvDreamer(gym.Env):
    """
    Hockey environment wrapper for DreamerV3.

    Uses SPARSE REWARDS ONLY - this is critical for DreamerV3.
    The world model + imagination handles credit assignment,
    so we don't need (and shouldn't use) reward shaping.

    Reward structure (from Robot Air Hockey Challenge 2023):
    - Win (score goal): +1.0
    - Loss (concede goal): -1.0
    - Draw/timeout: 0.0
    - Fault: -0.33 (optional, can be disabled)
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        mode: str = "NORMAL",
        opponent: str = "weak",
        include_fault_penalty: bool = True,
        fault_penalty: float = -0.33,
        seed: int = None,
    ):
        """
        Initialize hockey environment for DreamerV3.

        Args:
            mode: Game mode - "NORMAL", "TRAIN_SHOOTING", or "TRAIN_DEFENSE"
            opponent: Opponent type - "weak", "strong", or None (for self-play)
            include_fault_penalty: Whether to penalize faults
            fault_penalty: Penalty for faults (default -0.33 from Robot Air Hockey paper)
            seed: Random seed
        """
        super().__init__()

        # Environment mode
        mode_map = {
            'NORMAL': h_env.Mode.NORMAL,
            'TRAIN_SHOOTING': h_env.Mode.TRAIN_SHOOTING,
            'TRAIN_DEFENSE': h_env.Mode.TRAIN_DEFENSE,
        }
        self.mode = mode_map.get(mode, h_env.Mode.NORMAL)

        # Create hockey environment
        self.env = h_env.HockeyEnv(mode=self.mode)

        # Opponent setup
        self.opponent_type = opponent
        if opponent == "weak":
            self.opponent = h_env.BasicOpponent(weak=True)
        elif opponent == "strong":
            self.opponent = h_env.BasicOpponent(weak=False)
        else:
            self.opponent = None  # For self-play, opponent set externally

        # Reward settings
        self.include_fault_penalty = include_fault_penalty
        self.fault_penalty = fault_penalty

        # Observation space: 18-dimensional (keep_mode=True)
        # [0:2]   player position (x, y)
        # [2:3]   player angle
        # [3:5]   player velocity (vx, vy)
        # [5:6]   player angular velocity
        # [6:8]   opponent position (x, y)
        # [8:9]   opponent angle
        # [9:11]  opponent velocity (vx, vy)
        # [11:12] opponent angular velocity
        # [12:14] puck position (x, y)
        # [14:16] puck velocity (vx, vy)
        # [16:18] keep mode flags (has_puck_player1, has_puck_player2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )

        # Action space: 4-dimensional continuous
        # [0] x force
        # [1] y force
        # [2] rotation
        # [3] shoot (keep mode)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Episode tracking
        self._episode_step = 0
        self._max_episode_steps = 250 if mode == "NORMAL" else 80

        # For DreamerV3 compatibility
        self._last_obs = None

        if seed is not None:
            self.reset(seed=seed)

    def set_opponent(self, opponent):
        """
        Set opponent for self-play.

        Args:
            opponent: Object with act(obs) method
        """
        self.opponent = opponent

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Returns:
            obs: Initial observation (18-dim)
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)

        obs, info = self.env.reset()
        self._episode_step = 0
        self._last_obs = obs.astype(np.float32)

        return self._last_obs, info

    def step(self, action):
        """
        Take a step in the environment.

        IMPORTANT: Uses SPARSE REWARDS ONLY.
        DreamerV3's imagination-based planning handles credit assignment.

        Args:
            action: 4-dimensional action for our agent

        Returns:
            obs: Next observation
            reward: Sparse reward (win/loss/fault only)
            terminated: Whether episode ended (goal scored)
            truncated: Whether episode was truncated (timeout)
            info: Additional information including winner
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Get opponent action
        if self.opponent is not None:
            obs_opponent = self.env.obs_agent_two()
            action_opponent = self.opponent.act(obs_opponent)
        else:
            # No opponent set - use zero actions (shouldn't happen in normal use)
            action_opponent = np.zeros(4, dtype=np.float32)

        # Combine actions: [our_action, opponent_action]
        action_combined = np.hstack([action, action_opponent])

        # Step environment
        obs, env_reward, terminated, truncated, info = self.env.step(action_combined)

        # === SPARSE REWARD ONLY ===
        # This is the key difference from TD3 - no PBRS!
        reward = 0.0

        # Win/loss based on winner info
        winner = info.get('winner', 0)
        if winner == 1:  # We won
            reward = 1.0
        elif winner == -1:  # We lost
            reward = -1.0

        # Optional fault penalty
        if self.include_fault_penalty:
            if info.get('reward_touch_puck', 0) < 0:  # Our fault
                reward += self.fault_penalty

        # Update tracking
        self._episode_step += 1
        self._last_obs = obs.astype(np.float32)

        # Check for truncation (max steps reached)
        if self._episode_step >= self._max_episode_steps and not terminated:
            truncated = True

        # Add useful info for logging
        info['episode_step'] = self._episode_step
        info['sparse_reward'] = reward

        return self._last_obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        """Render the environment."""
        return self.env.render(mode=mode)

    def close(self):
        """Clean up environment."""
        self.env.close()

    def obs_agent_two(self):
        """Get observation from opponent's perspective (for self-play)."""
        return self.env.obs_agent_two()

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.env


class HockeyVecEnv:
    """
    Vectorized hockey environment for parallel data collection.

    DreamerV3 benefits from parallel environments for faster data collection.
    This is a simple synchronous vectorized environment.
    """

    def __init__(self, num_envs: int, **env_kwargs):
        """
        Create multiple hockey environments.

        Args:
            num_envs: Number of parallel environments
            **env_kwargs: Arguments passed to HockeyEnvDreamer
        """
        self.num_envs = num_envs
        self.envs = [HockeyEnvDreamer(**env_kwargs) for _ in range(num_envs)]

        # Copy spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed=None):
        """Reset all environments."""
        obs_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)

        return np.stack(obs_list), info_list

    def step(self, actions):
        """Step all environments."""
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.array(reward_list),
            np.array(terminated_list),
            np.array(truncated_list),
            info_list
        )

    def set_opponent(self, opponent, env_idx=None):
        """Set opponent for one or all environments."""
        if env_idx is not None:
            self.envs[env_idx].set_opponent(opponent)
        else:
            for env in self.envs:
                env.set_opponent(opponent)

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
