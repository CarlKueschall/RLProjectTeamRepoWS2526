"""
Potential-Based Reward Shaping (PBRS) for DreamerV3 Air Hockey.

AI Usage Declaration:
This file was developed with assistance from Claude Code.

Based on research from:
- Ng et al. (1999): Policy invariance under reward transformations
- Wiewiora (2003): Potential-based shaping and Q-value initialization

Design Philosophy: "Dense Path to Goal"
- TWO components only: phi_chase + phi_attack
- Creates continuous gradient: Agent -> Puck -> Opponent Goal
- Abstract enough to avoid reward hacking
- Agent discovers HOW to score, not specific tactics

Mathematical Guarantee:
F(s,a,s') = gamma * phi(s') - phi(s) preserves optimal policy.
This is PROVEN - shaped rewards don't change what's optimal, only speed learning.
"""

import numpy as np


# Environment constants for air hockey
TABLE_LENGTH = 9.0   # -4.5 to +4.5 in x
TABLE_WIDTH = 5.0    # -2.5 to +2.5 in y
MAX_DISTANCE = np.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)  # ~10.3
OPPONENT_GOAL_X = 4.5
OPPONENT_GOAL = np.array([OPPONENT_GOAL_X, 0.0])

# Default weights (W_ATTACK > W_CHASE ensures forward shooting is net positive)
DEFAULT_W_CHASE = 0.5
DEFAULT_W_ATTACK = 1.0
POTENTIAL_SCALE = 1.0


def compute_potential(obs, w_chase=DEFAULT_W_CHASE, w_attack=DEFAULT_W_ATTACK):
    """
    Two-component potential function for air hockey.

    Arguments:
        obs: 18-dimensional observation
            [0:2]   - player position (x, y)
            [12:14] - puck position (x, y)
        w_chase: Weight for chase component (default: 0.5)
        w_attack: Weight for attack component (default: 1.0)

    Returns:
        phi: Potential value for this state
    """
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])

    # phi_chase: Agent proximity to puck (negative distance, normalized)
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase = -dist_to_puck / MAX_DISTANCE  # Range: [-1, 0]

    # phi_attack: Puck proximity to opponent goal (negative distance, normalized)
    dist_puck_to_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
    phi_attack = -dist_puck_to_goal / MAX_DISTANCE  # Range: [-1, 0]

    phi = w_chase * phi_chase + w_attack * phi_attack
    return POTENTIAL_SCALE * phi


def compute_pbrs(obs, obs_next, done, gamma, w_chase=DEFAULT_W_CHASE, w_attack=DEFAULT_W_ATTACK):
    """
    Compute Potential-Based Reward Shaping.

    Formula: F(s,a,s') = gamma * phi(s') - phi(s)

    CRITICAL: Terminal state potential MUST be 0 for policy invariance.

    Arguments:
        obs: Current observation (18 dims)
        obs_next: Next observation (18 dims)
        done: Boolean, True if episode terminated
        gamma: Discount factor (MUST match training gamma)
        w_chase: Weight for chase component
        w_attack: Weight for attack component

    Returns:
        shaping_reward: F(s,a,s') to add to sparse reward
    """
    phi_current = compute_potential(obs, w_chase, w_attack)

    # Terminal state has zero potential (required for policy invariance)
    if done:
        phi_next = 0.0
    else:
        phi_next = compute_potential(obs_next, w_chase, w_attack)

    shaping_reward = gamma * phi_next - phi_current
    return shaping_reward


class PBRSRewardShaper:
    """
    PBRS wrapper for DreamerV3 training.

    Usage:
        shaper = PBRSRewardShaper(gamma=0.997, scale=0.03)

        # In episode loop:
        shaped_reward = sparse_reward + shaper.shape(obs, obs_next, done)

        # After episode, record outcome for reward hacking detection:
        shaper.record_episode_outcome(sparse_reward, outcome)
    """

    def __init__(
        self,
        gamma: float = 0.997,
        scale: float = 0.03,
        w_chase: float = DEFAULT_W_CHASE,
        w_attack: float = DEFAULT_W_ATTACK,
        clip: float = None,
    ):
        """
        Initialize PBRS reward shaper.

        Arguments:
            gamma: Discount factor (MUST match DreamerV3 training gamma)
            scale: Global scaling for shaping reward (default: 0.03)
            w_chase: Weight for chase component (default: 0.5)
            w_attack: Weight for attack component (default: 1.0)
            clip: Optional clipping for per-step shaping (None = no clip)
        """
        self.gamma = gamma
        self.scale = scale
        self.w_chase = w_chase
        self.w_attack = w_attack
        self.clip = clip

        # Per-episode tracking
        self._episode_shaping = 0.0
        self._episode_chase = 0.0
        self._episode_attack = 0.0
        self._step_count = 0
        self._episode_sparse = 0.0
        self._episode_outcome = None

        # Rolling window for reward hacking detection (last 100 episodes)
        self._history_sparse = []
        self._history_pbrs = []
        self._history_outcomes = []  # 1=win, 0=draw, -1=loss
        self._history_chase = []
        self._history_attack = []
        self._max_history = 100

    def shape(self, obs, obs_next, done):
        """
        Compute shaping reward for a single transition.

        Arguments:
            obs: Current observation
            obs_next: Next observation
            done: Whether episode terminated

        Returns:
            Shaping reward to ADD to sparse reward
        """
        # Compute component potentials for detailed tracking
        player_pos = np.array([obs[0], obs[1]])
        puck_pos = np.array([obs[12], obs[13]])

        phi_chase_curr = -np.linalg.norm(player_pos - puck_pos) / MAX_DISTANCE
        phi_attack_curr = -np.linalg.norm(puck_pos - OPPONENT_GOAL) / MAX_DISTANCE

        if done:
            phi_chase_next = 0.0
            phi_attack_next = 0.0
        else:
            player_pos_next = np.array([obs_next[0], obs_next[1]])
            puck_pos_next = np.array([obs_next[12], obs_next[13]])
            phi_chase_next = -np.linalg.norm(player_pos_next - puck_pos_next) / MAX_DISTANCE
            phi_attack_next = -np.linalg.norm(puck_pos_next - OPPONENT_GOAL) / MAX_DISTANCE

        # Compute component shaping rewards
        chase_shaping = self.w_chase * (self.gamma * phi_chase_next - phi_chase_curr) * self.scale
        attack_shaping = self.w_attack * (self.gamma * phi_attack_next - phi_attack_curr) * self.scale

        shaped = chase_shaping + attack_shaping

        if self.clip is not None:
            shaped = np.clip(shaped, -self.clip, self.clip)

        # Track components
        self._episode_shaping += shaped
        self._episode_chase += chase_shaping
        self._episode_attack += attack_shaping
        self._step_count += 1

        return shaped

    def record_episode_outcome(self, sparse_reward, outcome):
        """
        Record episode outcome for reward hacking detection.

        Arguments:
            sparse_reward: Total sparse (game) reward for episode
            outcome: 'win', 'loss', or 'draw'
        """
        self._episode_sparse = sparse_reward
        self._episode_outcome = outcome

        # Add to history
        self._history_sparse.append(sparse_reward)
        self._history_pbrs.append(self._episode_shaping)
        self._history_chase.append(self._episode_chase)
        self._history_attack.append(self._episode_attack)
        outcome_val = 1 if outcome == 'win' else (-1 if outcome == 'loss' else 0)
        self._history_outcomes.append(outcome_val)

        # Trim history
        if len(self._history_sparse) > self._max_history:
            self._history_sparse.pop(0)
            self._history_pbrs.pop(0)
            self._history_chase.pop(0)
            self._history_attack.pop(0)
            self._history_outcomes.pop(0)

    def get_episode_stats(self):
        """Get comprehensive episode statistics for logging and reward hacking detection."""
        stats = {
            # Basic PBRS stats
            'pbrs/episode_total': self._episode_shaping,
            'pbrs/episode_mean': self._episode_shaping / max(1, self._step_count),

            # Component breakdown
            'pbrs/chase_total': self._episode_chase,
            'pbrs/attack_total': self._episode_attack,
            'pbrs/chase_ratio': self._episode_chase / (abs(self._episode_shaping) + 1e-8),
            'pbrs/attack_ratio': self._episode_attack / (abs(self._episode_shaping) + 1e-8),
        }

        # Reward hacking detection metrics (need history)
        if len(self._history_sparse) > 0:
            sparse_arr = np.array(self._history_sparse)
            pbrs_arr = np.array(self._history_pbrs)
            outcomes_arr = np.array(self._history_outcomes)

            # Total reward composition
            total_reward = sparse_arr + pbrs_arr
            abs_sparse = np.abs(sparse_arr)
            abs_pbrs = np.abs(pbrs_arr)
            abs_total = abs_sparse + abs_pbrs + 1e-8

            stats.update({
                # Reward composition (CRITICAL for hacking detection)
                'reward_composition/sparse_fraction': np.mean(abs_sparse / abs_total),
                'reward_composition/pbrs_fraction': np.mean(abs_pbrs / abs_total),
                'reward_composition/sparse_mean': np.mean(sparse_arr),
                'reward_composition/pbrs_mean': np.mean(pbrs_arr),
                'reward_composition/total_mean': np.mean(total_reward),

                # Ratio metrics
                'reward_composition/pbrs_to_sparse_ratio': np.mean(abs_pbrs) / (np.mean(abs_sparse) + 1e-8),

                # Correlation between PBRS and winning (hacking = high PBRS but low wins)
                'reward_hacking/pbrs_std': np.std(pbrs_arr),
                'reward_hacking/sparse_std': np.std(sparse_arr),
            })

            # Compute correlation if we have variance
            if np.std(pbrs_arr) > 1e-6 and np.std(outcomes_arr) > 1e-6:
                correlation = np.corrcoef(pbrs_arr, outcomes_arr)[0, 1]
                stats['reward_hacking/pbrs_winrate_correlation'] = correlation
                # High PBRS should correlate with wins; if not, might be hacking
            else:
                stats['reward_hacking/pbrs_winrate_correlation'] = 0.0

            # Win-conditioned PBRS (is PBRS higher in wins vs losses?)
            wins_mask = outcomes_arr == 1
            losses_mask = outcomes_arr == -1
            if np.sum(wins_mask) > 0:
                stats['reward_hacking/pbrs_when_win'] = np.mean(pbrs_arr[wins_mask])
            else:
                stats['reward_hacking/pbrs_when_win'] = 0.0

            if np.sum(losses_mask) > 0:
                stats['reward_hacking/pbrs_when_loss'] = np.mean(pbrs_arr[losses_mask])
            else:
                stats['reward_hacking/pbrs_when_loss'] = 0.0

            # Hacking indicator: PBRS high in losses (agent optimizes PBRS but loses)
            if np.sum(losses_mask) > 0 and np.sum(wins_mask) > 0:
                pbrs_win = np.mean(pbrs_arr[wins_mask])
                pbrs_loss = np.mean(pbrs_arr[losses_mask])
                # If PBRS is higher when losing, that's suspicious
                stats['reward_hacking/loss_pbrs_minus_win_pbrs'] = pbrs_loss - pbrs_win
            else:
                stats['reward_hacking/loss_pbrs_minus_win_pbrs'] = 0.0

        return stats

    def reset(self):
        """Reset episode tracking (call at episode start)."""
        self._episode_shaping = 0.0
        self._episode_chase = 0.0
        self._episode_attack = 0.0
        self._step_count = 0
        self._episode_sparse = 0.0
        self._episode_outcome = None
