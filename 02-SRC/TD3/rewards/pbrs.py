"""
AI Usage Declaration:
This file was developed with assistance from AI tools.

Potential-Based Reward Shaping (PBRS) V3.2 for Air Hockey.

Based on research from:
- Ng et al. (1999): Policy invariance under reward transformations
- Wiewiora (2003): Potential-based shaping and Q-value initialization

Design Philosophy (V3.2 - Cross-Court Bonus):
- THREE components: φ_chase + φ_attack + φ_cross
- STRONG φ_chase (W=1.0): Agent always races toward the puck
- φ_attack (W=1.2): Ensures forward shooting is net positive
- φ_cross (W=0.4): NEW - Cross-court bonus for alternating shots

Key Insight (V3.2 - Cross-Court):
- When shooting, reward puck being far from opponent's y-position
- This encourages alternating shots (upper/lower corners)
- Tires out opponent by forcing them to move
- Similar to table tennis strategy

Cross-Court Math:
  φ_cross = (|puck_y - opponent_y| / TABLE_WIDTH) × tanh(puck_vel_x)
  - Only active when puck moving toward opponent goal (vel_x > 0)
  - Smooth activation via tanh (no discontinuity)
  - Higher reward when opponent is out of position

Shooting Math:
  Forward shot:  Δ = W_ATTACK × (+D) + W_CHASE × (-D) = D × (1.2 - 1.0) = +0.2D ✓
  Backward shot: Δ = W_ATTACK × (-D) + W_CHASE × (-D) = D × (-1.2 - 1.0) = -2.2D ✗

Reward Matrix (V3.2):
| Action              | φ_chase | φ_attack | φ_cross | Net      | Result              |
|---------------------|---------|----------|---------|----------|---------------------|
| Chase puck          | +1.0    | 0        | 0       | +1.0     | STRONG encourage    |
| Shoot forward       | -1.0    | +1.2     | +0.4*   | +0.6     | Encouraged          |
| Cross-court shot    | -1.0    | +1.2     | +0.4    | +0.6     | BONUS for cross     |
| Shoot at opponent   | -1.0    | +1.2     | 0       | +0.2     | Less rewarded       |
| Shoot backward      | -1.0    | -1.2     | 0       | -2.2     | Heavily penalized   |

* φ_cross bonus depends on opponent position relative to puck trajectory

Mathematical guarantee: F(s,s') = γφ(s') - φ(s) preserves optimal policy.
"""

import numpy as np


# Environment constants for air hockey
TABLE_LENGTH = 9.0   # -4.5 to +4.5 in x
TABLE_WIDTH = 5.0    # -2.5 to +2.5 in y
MAX_DISTANCE = np.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)  # ~10.3
OPPONENT_GOAL = np.array([4.5, 0.0])

# Component weights (V3.2 design - Strong Chase + Cross-Court)
# Key constraint: W_ATTACK > W_CHASE ensures forward shooting is net positive
W_CHASE = 1.0    # Agent → Puck (STRONG, always active everywhere)
W_ATTACK = 1.2   # Puck → Opponent Goal (always active, slightly higher than chase)
W_CROSS = 0.4    # Cross-court bonus (reward puck away from opponent's y-position)

# Scaling
EPISODE_SCALE = 100.0


def compute_potential(obs, gamma=0.99, w_cross=None):
    """
    Three-component potential function for air hockey (V3.2 - Cross-Court).

    Components:
    1. φ_chase (W=1.0): Reward agent proximity to puck
       - Always active, always full strength
       - No conditional logic, no stationary reduction

    2. φ_attack (W=1.2): Reward puck proximity to opponent goal
       - Always active everywhere
       - W_ATTACK > W_CHASE ensures forward shooting is net positive

    3. φ_cross (W=0.4): Cross-court bonus (NEW in V3.2)
       - Reward puck being far from opponent's y-position when moving toward goal
       - Uses smooth tanh activation based on puck velocity toward goal
       - Encourages alternating shots to tire out opponent

    Combined: φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack + W_CROSS × φ_cross

    Arguments:
        obs: 18-dimensional observation
            [0:2]   - player position (x, y)
            [6:8]   - opponent position (x, y)
            [12:14] - puck position (x, y)
            [14:16] - puck velocity (vx, vy)
        gamma: Discount factor (must match training gamma)
        w_cross: Override weight for cross-court component (None = use W_CROSS)

    Returns:
        phi: Potential value for this state
    """
    # === Extract state ===
    player_pos = np.array([obs[0], obs[1]])
    opponent_y = obs[7]  # Opponent's y-position
    puck_pos = np.array([obs[12], obs[13]])
    puck_vel_x = obs[14]  # Puck velocity toward opponent goal

    # === φ_chase: Agent proximity to puck (STRONG, always active) ===
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase = -dist_to_puck / MAX_DISTANCE  # Range: [-1, 0]

    # === φ_attack: Puck proximity to opponent goal (always active) ===
    dist_puck_to_opp_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
    phi_attack = -dist_puck_to_opp_goal / MAX_DISTANCE  # Range: [-1, 0]

    # === φ_cross: Cross-court bonus (smooth activation) ===
    # Only active when puck is moving toward opponent goal
    # Uses tanh for smooth activation (no discontinuity)
    puck_y = obs[13]
    y_separation = abs(puck_y - opponent_y)  # How far puck is from opponent's y
    y_separation_normalized = y_separation / TABLE_WIDTH  # Range: [0, 1]

    # Smooth activation: tanh(puck_vel_x) ranges from -1 to 1
    # When puck moving toward opponent goal (vel_x > 0): positive activation
    # When puck moving away (vel_x < 0): zero or negative (clamped to 0)
    velocity_activation = np.tanh(puck_vel_x)  # Smooth, bounded [-1, 1]
    velocity_activation = max(0.0, velocity_activation)  # Only reward forward motion

    # φ_cross: higher when puck is far from opponent AND moving toward goal
    phi_cross = y_separation_normalized * velocity_activation  # Range: [0, 1]

    # === Combine with weights ===
    cross_weight = w_cross if w_cross is not None else W_CROSS
    phi_combined = (W_CHASE * phi_chase +
                    W_ATTACK * phi_attack +
                    cross_weight * phi_cross)

    # === Scale for episode ===
    phi = EPISODE_SCALE * phi_combined
    # Range: approx [-220, +40] with cross-court bonus

    return phi


def compute_pbrs(obs, obs_next, done, gamma=0.99, w_cross=None):
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
        w_cross: Override weight for cross-court component (None = use W_CROSS)

    Returns:
        shaping_reward: F(s,a,s') = γ·φ(s') - φ(s) (with terminal correction)
    """
    phi_current = compute_potential(obs, gamma, w_cross=w_cross)

    # CRITICAL: Force terminal potential to 0
    if done:
        phi_next = 0.0
    else:
        phi_next = compute_potential(obs_next, gamma, w_cross=w_cross)

    shaping_reward = gamma * phi_next - phi_current

    return shaping_reward


class PBRSReward:
    """
    Potential-Based Reward Shaping wrapper with independent annealing support.

    V3.2 Features:
    - Three-component potential (chase + attack + cross-court)
    - Cross-court bonus to encourage alternating shots
    - Minimum weight floor (never fully removes PBRS)
    - Slow annealing support

    Usage:
        # No annealing (constant PBRS)
        shaper = PBRSReward(gamma=0.99, pbrs_scale=0.02)

        # With cross-court bonus
        shaper = PBRSReward(gamma=0.99, pbrs_scale=0.02, w_cross=0.4)

        # With slow annealing and minimum weight
        shaper = PBRSReward(gamma=0.99, pbrs_scale=0.02,
                           anneal_start=5000, anneal_episodes=15000,
                           min_weight=0.1)

        shaped_reward = sparse_reward + shaper.compute(obs, obs_next, done, episode)
    """

    def __init__(self, gamma=0.99, pbrs_scale=0.02,
                 anneal_start=0, anneal_episodes=15000,
                 min_weight=0.0, w_cross=None,
                 constant_weight=True, annealing_episodes=5000):
        """
        Initialize PBRS reward shaper.

        Arguments:
            gamma: Discount factor
            pbrs_scale: Global scaling factor (default: 0.02)
            w_cross: Weight for cross-court component (None = use default W_CROSS=0.4)

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
        self.w_cross = w_cross  # None means use default W_CROSS

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
        shaped_reward = compute_pbrs(obs_curr, obs_next, done, self.gamma,
                                     w_cross=self.w_cross)

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

def get_potential_components(obs, w_cross=None):
    """
    Get individual potential components for debugging (V3.2 - with cross-court).

    Returns dict with each component's contribution.
    """
    player_pos = np.array([obs[0], obs[1]])
    opponent_y = obs[7]  # Opponent's y-position
    puck_pos = np.array([obs[12], obs[13]])
    puck_y = obs[13]
    puck_vel_x = obs[14]

    # Chase component (always full strength)
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase = -dist_to_puck / MAX_DISTANCE

    # Attack component (always active)
    dist_puck_to_opp_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
    phi_attack = -dist_puck_to_opp_goal / MAX_DISTANCE

    # Cross-court component
    y_separation = abs(puck_y - opponent_y)
    y_separation_normalized = y_separation / TABLE_WIDTH
    velocity_activation = max(0.0, np.tanh(puck_vel_x))
    phi_cross = y_separation_normalized * velocity_activation

    # Combined potential
    cross_weight = w_cross if w_cross is not None else W_CROSS
    phi_combined = (W_CHASE * phi_chase +
                    W_ATTACK * phi_attack +
                    cross_weight * phi_cross)

    return {
        # Chase component
        'phi_chase': phi_chase,
        'phi_chase_weighted': W_CHASE * phi_chase,
        'dist_to_puck': dist_to_puck,
        # Attack component
        'phi_attack': phi_attack,
        'phi_attack_weighted': W_ATTACK * phi_attack,
        'dist_puck_to_opp_goal': dist_puck_to_opp_goal,
        # Cross-court component
        'phi_cross': phi_cross,
        'phi_cross_weighted': cross_weight * phi_cross,
        'y_separation': y_separation,
        'y_separation_normalized': y_separation_normalized,
        'velocity_activation': velocity_activation,
        'puck_vel_x': puck_vel_x,
        'opponent_y': opponent_y,
        'puck_y': puck_y,
        # Combined
        'phi_combined': phi_combined,
        'phi_scaled': EPISODE_SCALE * phi_combined,
        # Weights
        'W_CHASE': W_CHASE,
        'W_ATTACK': W_ATTACK,
        'W_CROSS': cross_weight,
    }
