"""
AI Usage Declaration:
This file was developed with assistance from AI tools.

Potential-Based Reward Shaping (PBRS) for DreamerV3 Air Hockey.

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

Why This Works for DreamerV3:
- World model can learn dense reward dynamics (not just "0 everywhere")
- Imagination produces meaningful reward predictions
- Lambda-returns become non-trivial
- Policy gets actual gradient signal

Component Design:
| Component   | Formula                              | Intuition                    |
|-------------|--------------------------------------|------------------------------|
| phi_chase   | -dist(agent, puck) / MAX_DIST        | Pull agent toward puck       |
| phi_attack  | -dist(puck, opponent_goal) / MAX_DIST| Pull puck toward goal        |

Weight Constraint: W_ATTACK > W_CHASE
- Ensures shooting toward goal is net positive
- When agent shoots: loses phi_chase (puck leaves), gains phi_attack
- Net change is positive for forward shots, negative for backward shots
"""

import numpy as np


# Environment constants for air hockey
TABLE_LENGTH = 9.0   # -4.5 to +4.5 in x
TABLE_WIDTH = 5.0    # -2.5 to +2.5 in y
MAX_DISTANCE = np.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)  # ~10.3
OPPONENT_GOAL_X = 4.5
OPPONENT_GOAL = np.array([OPPONENT_GOAL_X, 0.0])

# Default weights (V3.3 design)
# W_ATTACK > W_CHASE ensures forward shooting is net positive
DEFAULT_W_CHASE = 0.5
DEFAULT_W_ATTACK = 1.0

# Scaling factor for potential magnitude
POTENTIAL_SCALE = 1.0  # Keep potentials in reasonable range


def compute_potential(obs, w_chase=DEFAULT_W_CHASE, w_attack=DEFAULT_W_ATTACK):
    """
    Two-component potential function for air hockey.

    Components:
    1. phi_chase: Reward agent proximity to puck
       - Creates gradient pulling agent toward puck
       - Range: [-1, 0] (0 when agent is at puck)

    2. phi_attack: Reward puck proximity to opponent goal
       - Creates gradient pulling puck toward goal
       - Range: [-1, 0] (0 when puck is at goal)

    Combined: phi(s) = W_CHASE * phi_chase + W_ATTACK * phi_attack

    Arguments:
        obs: 18-dimensional observation
            [0:2]   - player position (x, y)
            [12:14] - puck position (x, y)
        w_chase: Weight for chase component (default: 0.5)
        w_attack: Weight for attack component (default: 1.0)

    Returns:
        phi: Potential value for this state
    """
    # Extract positions
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])

    # phi_chase: Agent proximity to puck (negative distance, normalized)
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase = -dist_to_puck / MAX_DISTANCE  # Range: [-1, 0]

    # phi_attack: Puck proximity to opponent goal (negative distance, normalized)
    dist_puck_to_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
    phi_attack = -dist_puck_to_goal / MAX_DISTANCE  # Range: [-1, 0]

    # Combine with weights
    phi = w_chase * phi_chase + w_attack * phi_attack

    return POTENTIAL_SCALE * phi


def compute_pbrs(obs, obs_next, done, gamma, w_chase=DEFAULT_W_CHASE, w_attack=DEFAULT_W_ATTACK):
    """
    Compute Potential-Based Reward Shaping.

    Formula: F(s,a,s') = gamma * phi(s') - phi(s)

    CRITICAL: Terminal state potential MUST be 0.
    This ensures the shaped return equals the true return plus a constant
    (the initial potential), preserving policy optimality.

    Arguments:
        obs: Current observation (18 dims)
        obs_next: Next observation (18 dims)
        done: Boolean, True if episode terminated
        gamma: Discount factor (MUST match training gamma for theory to hold)
        w_chase: Weight for chase component
        w_attack: Weight for attack component

    Returns:
        shaping_reward: F(s,a,s') to add to sparse reward
    """
    phi_current = compute_potential(obs, w_chase, w_attack)

    # CRITICAL: Terminal state has zero potential
    # This is required for policy invariance guarantee
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

        # Store shaped_reward in buffer for world model training
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
                   Tune based on sparse reward magnitude (±1 for hockey)
            w_chase: Weight for chase component (default: 0.5)
            w_attack: Weight for attack component (default: 1.0)
            clip: Optional clipping for per-step shaping (None = no clip)
        """
        self.gamma = gamma
        self.scale = scale
        self.w_chase = w_chase
        self.w_attack = w_attack
        self.clip = clip

        # Tracking for logging
        self._episode_shaping = 0.0
        self._step_count = 0

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
        raw_shaping = compute_pbrs(
            obs, obs_next, done,
            self.gamma, self.w_chase, self.w_attack
        )

        # Apply global scale
        shaped = raw_shaping * self.scale

        # Optional clipping for stability
        if self.clip is not None:
            shaped = np.clip(shaped, -self.clip, self.clip)

        # Track for logging
        self._episode_shaping += shaped
        self._step_count += 1

        return shaped

    def get_episode_stats(self):
        """Get episode statistics for logging."""
        stats = {
            'pbrs/episode_total': self._episode_shaping,
            'pbrs/episode_mean': self._episode_shaping / max(1, self._step_count),
        }
        return stats

    def reset(self):
        """Reset episode tracking (call at episode start)."""
        self._episode_shaping = 0.0
        self._step_count = 0


def get_potential_components(obs, w_chase=DEFAULT_W_CHASE, w_attack=DEFAULT_W_ATTACK):
    """
    Get individual potential components for debugging.

    Arguments:
        obs: 18-dimensional observation
        w_chase: Weight for chase component
        w_attack: Weight for attack component

    Returns:
        Dictionary with component values for analysis
    """
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])

    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase = -dist_to_puck / MAX_DISTANCE

    dist_puck_to_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
    phi_attack = -dist_puck_to_goal / MAX_DISTANCE

    phi_combined = w_chase * phi_chase + w_attack * phi_attack

    return {
        'dist_agent_to_puck': dist_to_puck,
        'dist_puck_to_goal': dist_puck_to_goal,
        'phi_chase': phi_chase,
        'phi_attack': phi_attack,
        'phi_chase_weighted': w_chase * phi_chase,
        'phi_attack_weighted': w_attack * phi_attack,
        'phi_total': phi_combined,
        'phi_scaled': POTENTIAL_SCALE * phi_combined,
    }
