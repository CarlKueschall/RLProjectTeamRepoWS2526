"""
AI Usage Declaration:
This file was developed with assistance from AI tools.

Potential-Based Reward Shaping (PBRS) for Air Hockey.

Based on research from:
- Ng et al. (1999): Policy invariance under reward transformations
- Wiewiora (2003): Potential-based shaping and Q-value initialization

Design Philosophy:
- MINIMAL shaping: Only guide behaviors hard to discover from sparse rewards
- AVOID encoding strategy: Let agent discover shooting angles, timing, etc.
- PREVENT exploits: No reward for standing near stationary puck

Components:
1. φ_chase: Reward being close to MOVING puck (defense/interception)
2. φ_defensive: Triangle defense positioning when puck in own half

Mathematical guarantee: F(s,s') = γφ(s') - φ(s) preserves optimal policy.
"""

import numpy as np


# Environment constants for air hockey
TABLE_LENGTH = 9.0   # -4.5 to +4.5 in x
TABLE_WIDTH = 5.0    # -2.5 to +2.5 in y
MAX_DISTANCE = np.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)  # ~10.3
OPPONENT_GOAL = np.array([4.5, 0.0])
OWN_GOAL = np.array([-4.5, 0.0])

# Thresholds
PUCK_MOVING_THRESHOLD = 0.3  # Minimum puck speed to be considered "in play"


def compute_potential(obs, gamma=0.99):
    """
    Minimal potential function for air hockey.

    Only two components:
    1. φ_chase: Reward proximity to MOVING puck only
       - Helps defense: intercept opponent shots
       - Helps offense: chase rebounds
       - PREVENTS: standing next to stationary puck (the "do nothing" exploit)

    2. φ_defensive: Triangle defense when puck in own half
       - Position 40% from goal to puck
       - Blocks direct shots while allowing reaction

    Arguments:
        obs: 18-dimensional observation
            [0:2]   - player position (x, y)
            [12:14] - puck position (x, y)
            [14:16] - puck velocity (vx, vy)
        gamma: Discount factor (must match training gamma)

    Returns:
        phi: Potential value for this state
    """
    # === Extract state ===
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])
    puck_vel = np.array([obs[14], obs[15]])
    puck_speed = np.linalg.norm(puck_vel)

    # === Component 1: Chase moving puck ===
    # CRITICAL: Only reward when puck is moving!
    # This prevents the "stand near stationary puck" exploit.
    # If puck is stationary, agent gets NO reward for being close.
    # Agent must HIT the puck to make it move, then can chase it.
    if puck_speed > PUCK_MOVING_THRESHOLD:
        dist_to_puck = np.linalg.norm(player_pos - puck_pos)
        phi_chase = -dist_to_puck / MAX_DISTANCE  # Range: [-1, 0]
    else:
        phi_chase = 0.0  # NO reward for stationary puck

    # === Component 2: Defensive positioning (own half only) ===
    # Triangle defense: position 40% of the way from goal to puck
    # This blocks direct shots while allowing reaction to banks
    puck_in_own_half = puck_pos[0] < 0

    if puck_in_own_half:
        ideal_defensive_pos = OWN_GOAL + 0.4 * (puck_pos - OWN_GOAL)
        defensive_error = np.linalg.norm(player_pos - ideal_defensive_pos)
        # Normalize by table width, cap at 1.0
        phi_defensive = -0.4 * min(defensive_error / TABLE_WIDTH, 1.0)
    else:
        phi_defensive = 0.0

    # === Combine components ===
    # Scale to episode-appropriate magnitude
    # With EPISODE_SCALE=100 and pbrs_scale=0.02:
    # - Max per-step shaping: ~0.02 * 100 * 0.01 = 0.02
    # - Max episode shaping: ~0.02 * 100 * 1.4 = 2.8 (< sparse reward 10)
    EPISODE_SCALE = 100.0

    phi = EPISODE_SCALE * (phi_chase + phi_defensive)
    # Total range: approximately [-140, 0]
    # φ_chase: [-100, 0] when puck moving, 0 when stationary
    # φ_defensive: [-40, 0] when puck in own half, 0 otherwise

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
    Potential-Based Reward Shaping wrapper.

    Usage:
        shaper = PBRSReward(gamma=0.99, pbrs_scale=0.02)
        shaped_reward = sparse_reward + shaper.compute(obs, obs_next, done)
    """

    def __init__(self, gamma=0.99, annealing_episodes=5000, pbrs_scale=0.02, constant_weight=True):
        """
        Initialize PBRS reward shaper.

        Arguments:
            gamma: Discount factor
            annealing_episodes: Episodes over which to anneal shaping during self-play
            pbrs_scale: Global scaling factor (default: 0.02, mathematically derived)
            constant_weight: If True, disable annealing (recommended for stability)
        """
        self.gamma = gamma
        self.annealing_episodes = annealing_episodes
        self.pbrs_scale = pbrs_scale
        self.constant_weight = constant_weight
        self.self_play_start = None
        self.episode = 0

    def set_self_play_start(self, episode):
        """Set the episode when self-play starts (for annealing)."""
        self.self_play_start = episode

    def compute(self, obs_curr, obs_next, done, episode=None):
        """
        Compute PBRS reward with optional annealing.

        Arguments:
            obs_curr: Current observation
            obs_next: Next observation
            done: Episode done flag
            episode: Current episode number (for annealing)

        Returns:
            shaped_reward: Additional reward term (add to sparse reward)
        """
        shaped_reward = compute_pbrs(obs_curr, obs_next, done, self.gamma)

        # Apply global PBRS scaling
        shaped_reward *= self.pbrs_scale

        # Optional annealing during self-play
        if not self.constant_weight:
            if self.self_play_start is not None and episode is not None:
                if episode > self.self_play_start:
                    episodes_since_selfplay = episode - self.self_play_start
                    w_shaping = max(0.0, 1.0 - (episodes_since_selfplay / self.annealing_episodes))
                    shaped_reward *= w_shaping

        return shaped_reward

    def get_annealing_weight(self, episode):
        """Get current annealing weight (for logging)."""
        if self.constant_weight:
            return 1.0
        if self.self_play_start is None or episode <= self.self_play_start:
            return 1.0
        else:
            episodes_since_selfplay = episode - self.self_play_start
            return max(0.0, 1.0 - (episodes_since_selfplay / self.annealing_episodes))

    def reset(self):
        """Reset episode state (no-op for stateless shaper)."""
        pass


# === Utility functions for debugging and analysis ===

def get_potential_components(obs):
    """
    Get individual potential components for debugging.

    Returns dict with each component's contribution.
    """
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])
    puck_vel = np.array([obs[14], obs[15]])
    puck_speed = np.linalg.norm(puck_vel)

    # Chase component
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    puck_is_moving = puck_speed > PUCK_MOVING_THRESHOLD

    if puck_is_moving:
        phi_chase = -dist_to_puck / MAX_DISTANCE
    else:
        phi_chase = 0.0

    # Defensive component
    puck_in_own_half = puck_pos[0] < 0
    phi_defensive = 0.0
    ideal_defensive_pos = None

    if puck_in_own_half:
        ideal_defensive_pos = OWN_GOAL + 0.4 * (puck_pos - OWN_GOAL)
        defensive_error = np.linalg.norm(player_pos - ideal_defensive_pos)
        phi_defensive = -0.4 * min(defensive_error / TABLE_WIDTH, 1.0)

    return {
        'phi_chase': phi_chase,
        'phi_defensive': phi_defensive,
        'phi_total': phi_chase + phi_defensive,
        'phi_scaled': 100.0 * (phi_chase + phi_defensive),
        'puck_speed': puck_speed,
        'puck_is_moving': puck_is_moving,
        'dist_to_puck': dist_to_puck,
        'puck_in_own_half': puck_in_own_half,
        'ideal_defensive_pos': ideal_defensive_pos,
    }
