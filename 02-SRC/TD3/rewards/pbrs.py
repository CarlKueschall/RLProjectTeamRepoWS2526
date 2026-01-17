"""
AI Usage Declaration:
This file was developed with assistance from AI tools.

Potential-Based Reward Shaping (PBRS) V3.1 for Air Hockey.

Based on research from:
- Ng et al. (1999): Policy invariance under reward transformations
- Wiewiora (2003): Potential-based shaping and Q-value initialization

Design Philosophy (V3.1 - Strong Chase, Simple Math):
- TWO components: φ_chase + φ_attack (simplified from V3)
- STRONG φ_chase (W=1.0): Agent always races toward the puck
- φ_attack (W=1.2): Slightly higher to ensure forward shooting is positive
- NO conditional logic: Both components always active everywhere
- NO stationary puck reduction: Always full chase incentive

Key Insight:
- Strong chase handles EVERYTHING: defense, interception, ready position
- The only constraint: W_ATTACK > W_CHASE so shooting forward is net positive

Shooting Math:
  Forward shot:  Δ = W_ATTACK × (+D) + W_CHASE × (-D) = D × (1.2 - 1.0) = +0.2D ✓
  Backward shot: Δ = W_ATTACK × (-D) + W_CHASE × (-D) = D × (-1.2 - 1.0) = -2.2D ✗

Reward Matrix (V3.1):
| Action              | φ_chase | φ_attack | Net    | Result            |
|---------------------|---------|----------|--------|-------------------|
| Chase puck          | +1.0    | 0        | +1.0   | STRONG encourage  |
| Shoot forward       | -1.0    | +1.2     | +0.2   | Encouraged        |
| Shoot backward      | -1.0    | -1.2     | -2.2   | Heavily penalized |
| Puck in our half    | active  | penalty  | chase  | Agent races to it |

Mathematical guarantee: F(s,s') = γφ(s') - φ(s) preserves optimal policy.
"""

import numpy as np


# Environment constants for air hockey
TABLE_LENGTH = 9.0   # -4.5 to +4.5 in x
TABLE_WIDTH = 5.0    # -2.5 to +2.5 in y
MAX_DISTANCE = np.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)  # ~10.3
OPPONENT_GOAL = np.array([4.5, 0.0])

# Component weights (V3.1 design - Strong Chase, Simple Math)
# Key constraint: W_ATTACK > W_CHASE ensures forward shooting is net positive
W_CHASE = 1.0    # Agent → Puck (STRONG, always active everywhere)
W_ATTACK = 1.2   # Puck → Opponent Goal (always active, slightly higher than chase)

# Scaling
EPISODE_SCALE = 100.0


def compute_potential(obs, gamma=0.99):
    """
    Two-component potential function for air hockey (V3.1 - Strong Chase, Simple Math).

    Components:
    1. φ_chase (W=1.0): Reward agent proximity to puck
       - Always active, always full strength
       - No conditional logic, no stationary reduction

    2. φ_attack (W=1.2): Reward puck proximity to opponent goal
       - Always active everywhere
       - W_ATTACK > W_CHASE ensures forward shooting is net positive

    Combined: φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack

    Shooting math:
      Forward shot (D distance):  Δ = 1.2×(+D) + 1.0×(-D) = +0.2D ✓
      Backward shot (D distance): Δ = 1.2×(-D) + 1.0×(-D) = -2.2D ✗

    Arguments:
        obs: 18-dimensional observation
            [0:2]   - player position (x, y)
            [12:14] - puck position (x, y)
        gamma: Discount factor (must match training gamma)

    Returns:
        phi: Potential value for this state
    """
    # === Extract state ===
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])

    # === φ_chase: Agent proximity to puck (STRONG, always active) ===
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase = -dist_to_puck / MAX_DISTANCE  # Range: [-1, 0]

    # === φ_attack: Puck proximity to opponent goal (always active) ===
    dist_puck_to_opp_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
    phi_attack = -dist_puck_to_opp_goal / MAX_DISTANCE  # Range: [-1, 0]

    # === Combine with weights ===
    phi_combined = W_CHASE * phi_chase + W_ATTACK * phi_attack

    # === Scale for episode ===
    phi = EPISODE_SCALE * phi_combined
    # Range: [-220, 0] (W_CHASE × 1 + W_ATTACK × 1 = 2.2, × 100 = 220)

    return phi


def compute_pbrs(obs, obs_next, done, gamma=0.99):
    """
    Compute Potential-Based Reward Shaping with episodic correction.

    Formula: F(s,a,s') = γ·φ(s') - φ(s)

    CRITICAL: Terminal state potential must be 0 to avoid bias.
    Without this, the shaped return includes γ^T·φ(s_T) - φ(s_0) bias.

    Arguments:
        obs: Current observation (18 dims)
        obs_next: Next observation (18 dims)
        done: Boolean, True if episode terminated/truncated
        gamma: Discount factor (must match training gamma)

    Returns:
        shaping_reward: F(s,a,s') = γ·φ(s') - φ(s) (with terminal correction)
    """
    phi_current = compute_potential(obs, gamma)

    # CRITICAL: Force terminal potential to 0
    if done:
        phi_next = 0.0
    else:
        phi_next = compute_potential(obs_next, gamma)

    shaping_reward = gamma * phi_next - phi_current

    return shaping_reward


class PBRSReward:
    """
    Potential-Based Reward Shaping wrapper with independent annealing support.

    V2 Features:
    - Two-component potential (chase + attack)
    - Minimum weight floor (never fully removes PBRS)
    - Slow annealing support

    Usage:
        # No annealing (constant PBRS)
        shaper = PBRSReward(gamma=0.99, pbrs_scale=0.02)

        # With slow annealing and minimum weight
        shaper = PBRSReward(gamma=0.99, pbrs_scale=0.02,
                           anneal_start=5000, anneal_episodes=15000,
                           min_weight=0.1)

        shaped_reward = sparse_reward + shaper.compute(obs, obs_next, done, episode)
    """

    def __init__(self, gamma=0.99, pbrs_scale=0.02,
                 anneal_start=0, anneal_episodes=15000,
                 min_weight=0.0,
                 constant_weight=True, annealing_episodes=5000):
        """
        Initialize PBRS reward shaper.

        Arguments:
            gamma: Discount factor
            pbrs_scale: Global scaling factor (default: 0.02)

            Independent annealing (recommended):
            anneal_start: Episode to start annealing (0=never anneal)
            anneal_episodes: Episodes over which to anneal (default: 15000)
            min_weight: Minimum weight floor (default: 0.0, set to 0.1 to retain attack incentive)

            Legacy self-play annealing (for compatibility):
            constant_weight: If True, disable self-play annealing
            annealing_episodes: Legacy param for self-play annealing duration
        """
        self.gamma = gamma
        self.pbrs_scale = pbrs_scale

        # Independent annealing (new, preferred)
        self.anneal_start = anneal_start
        self.anneal_episodes = anneal_episodes
        self.min_weight = min_weight

        # Legacy self-play annealing (kept for compatibility)
        self.constant_weight = constant_weight
        self.legacy_annealing_episodes = annealing_episodes
        self.self_play_start = None

    def set_self_play_start(self, episode):
        """Set the episode when self-play starts (for legacy annealing)."""
        self.self_play_start = episode

    def compute(self, obs_curr, obs_next, done, episode=None):
        """
        Compute PBRS reward with optional annealing.

        Annealing priority:
        1. Independent annealing (if anneal_start > 0)
        2. Legacy self-play annealing (if constant_weight=False)
        3. No annealing (constant weight = 1.0)

        Arguments:
            obs_curr: Current observation
            obs_next: Next observation
            done: Episode done flag
            episode: Current episode number (required for annealing)

        Returns:
            shaped_reward: Additional reward term (add to sparse reward)
        """
        shaped_reward = compute_pbrs(obs_curr, obs_next, done, self.gamma)

        # Apply global PBRS scaling
        shaped_reward *= self.pbrs_scale

        # Apply annealing weight
        weight = self.get_annealing_weight(episode)
        shaped_reward *= weight

        return shaped_reward

    def get_annealing_weight(self, episode):
        """
        Get current annealing weight (for logging and computation).

        Returns value in [min_weight, 1.0]:
        - 1.0 = full PBRS strength
        - min_weight = minimum retained (default 0.0, can be 0.1 for attack incentive)
        """
        if episode is None:
            return 1.0

        # Priority 1: Independent annealing (if configured)
        if self.anneal_start > 0:
            if episode < self.anneal_start:
                return 1.0  # Full strength before annealing starts
            else:
                episodes_since_start = episode - self.anneal_start
                # Anneal from 1.0 to min_weight over anneal_episodes
                progress = episodes_since_start / self.anneal_episodes
                weight = max(self.min_weight, 1.0 - progress * (1.0 - self.min_weight))
                return weight

        # Priority 2: Legacy self-play annealing
        if not self.constant_weight:
            if self.self_play_start is not None and episode > self.self_play_start:
                episodes_since_selfplay = episode - self.self_play_start
                weight = max(self.min_weight, 1.0 - (episodes_since_selfplay / self.legacy_annealing_episodes))
                return weight

        # Default: no annealing
        return 1.0

    def reset(self):
        """Reset episode state (no-op for stateless shaper)."""
        pass


# === Utility functions for debugging and analysis ===

def get_potential_components(obs):
    """
    Get individual potential components for debugging (V3.1 - simplified).

    Returns dict with each component's contribution.
    """
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])

    # Chase component (always full strength)
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase = -dist_to_puck / MAX_DISTANCE

    # Attack component (always active)
    dist_puck_to_opp_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
    phi_attack = -dist_puck_to_opp_goal / MAX_DISTANCE

    # Combined potential
    phi_combined = W_CHASE * phi_chase + W_ATTACK * phi_attack

    return {
        # Chase component
        'phi_chase': phi_chase,
        'phi_chase_weighted': W_CHASE * phi_chase,
        'dist_to_puck': dist_to_puck,
        # Attack component
        'phi_attack': phi_attack,
        'phi_attack_weighted': W_ATTACK * phi_attack,
        'dist_puck_to_opp_goal': dist_puck_to_opp_goal,
        # Combined
        'phi_combined': phi_combined,
        'phi_scaled': EPISODE_SCALE * phi_combined,
        # Weights
        'W_CHASE': W_CHASE,
        'W_ATTACK': W_ATTACK,
    }
