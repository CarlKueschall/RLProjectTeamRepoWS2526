"""
AI Usage Declaration:
This file was developed with assistance from AI tools.

Potential-Based Reward Shaping (PBRS) V3 for Air Hockey.

Based on research from:
- Ng et al. (1999): Policy invariance under reward transformations
- Wiewiora (2003): Potential-based shaping and Q-value initialization

Design Philosophy (V3 - Balanced Offense/Defense):
- THREE components: φ_chase + φ_attack + φ_defense
- ASYMMETRIC φ_attack: Only active in opponent half (don't penalize defensive positions)
- DEFENSIVE φ_chase: Full strength when puck is in our half (encourage recovery)
- φ_defense: Reward being between puck and own goal during defensive situations

Components:
1. φ_chase: Reward agent being close to puck (W_CHASE = 0.5)
   - MOVING puck: Full magnitude
   - STATIONARY puck in OPPONENT half: 30% magnitude (prevent camping)
   - STATIONARY puck in OUR half: FULL magnitude (encourage defensive recovery!)

2. φ_attack: Reward puck being close to opponent goal (W_ATTACK = 0.7)
   - ONLY ACTIVE when puck is in opponent half (x > 0)
   - When puck in our half: φ_attack = 0 (no penalty for defensive situations)

3. φ_defense: Reward defensive positioning (W_DEFENSE = 0.3)
   - ONLY ACTIVE when puck is in our half (x < 0)
   - Rewards agent being between puck and own goal

Reward Matrix (V3):
| Situation              | φ_chase | φ_attack | φ_defense | Net | Result       |
|------------------------|---------|----------|-----------|-----|--------------|
| Chase puck (opp half)  | +       | 0        | 0         | +   | Encouraged   |
| Shoot toward opp goal  | -       | +        | 0         | +   | Encouraged   |
| Chase puck (our half)  | +       | 0        | +         | ++  | STRONGLY encouraged! |
| Defend (between puck)  | ~       | 0        | +         | +   | Encouraged   |
| Ignore puck (our half) | 0       | 0        | -         | -   | Penalized    |

Mathematical guarantee: F(s,s') = γφ(s') - φ(s) preserves optimal policy.
"""

import numpy as np


# Environment constants for air hockey
TABLE_LENGTH = 9.0   # -4.5 to +4.5 in x
TABLE_WIDTH = 5.0    # -2.5 to +2.5 in y
MAX_DISTANCE = np.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)  # ~10.3
OPPONENT_GOAL = np.array([4.5, 0.0])
OWN_GOAL = np.array([-4.5, 0.0])
CENTER_X = 0.0  # Center line for half detection

# Thresholds
PUCK_MOVING_THRESHOLD = 0.3  # Minimum puck speed to be considered "in play"
STATIONARY_PUCK_WEIGHT = 0.3  # Reduced weight for stationary puck in OPPONENT half only

# Component weights (V3 design - balanced offense/defense)
W_CHASE = 0.5    # Agent → Puck (always active)
W_ATTACK = 0.7   # Puck → Opponent Goal (only in opponent half)
W_DEFENSE = 0.3  # Defensive positioning (only in our half)

# Scaling
EPISODE_SCALE = 100.0


def compute_potential(obs, gamma=0.99):
    """
    Three-component potential function for air hockey (V3 - Balanced Offense/Defense).

    Components:
    1. φ_chase: Reward agent proximity to puck (W_CHASE = 0.5)
       - MOVING puck: Full magnitude
       - STATIONARY puck in OPPONENT half: 30% (prevent camping near opponent goal)
       - STATIONARY puck in OUR half: FULL magnitude (encourage defensive recovery!)

    2. φ_attack: Reward puck proximity to opponent goal (W_ATTACK = 0.7)
       - ONLY ACTIVE when puck is in opponent half (x > 0)
       - Disabled in our half to avoid penalizing defensive situations

    3. φ_defense: Reward defensive positioning (W_DEFENSE = 0.3)
       - ONLY ACTIVE when puck is in our half (x < 0)
       - Rewards agent being between puck and own goal (good defensive position)

    Combined: φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack + W_DEFENSE × φ_defense

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

    # Determine which half the puck is in
    puck_in_our_half = puck_pos[0] < CENTER_X
    puck_is_moving = puck_speed > PUCK_MOVING_THRESHOLD

    # === φ_chase: Agent proximity to puck ===
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase_base = -dist_to_puck / MAX_DISTANCE  # Range: [-1, 0]

    # Apply stationary weight ONLY in opponent half (to prevent camping)
    # In our half, always use full weight to encourage defensive recovery
    if puck_is_moving or puck_in_our_half:
        phi_chase = phi_chase_base  # Full magnitude
    else:
        # Stationary puck in opponent half - reduce to prevent camping
        phi_chase = STATIONARY_PUCK_WEIGHT * phi_chase_base  # 30%

    # === φ_attack: Puck proximity to opponent goal (ASYMMETRIC in V3) ===
    # Only active when puck is in opponent half - avoids penalizing defensive positions
    if not puck_in_our_half:
        dist_puck_to_opp_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
        phi_attack = -dist_puck_to_opp_goal / MAX_DISTANCE  # Range: [-1, 0]
    else:
        phi_attack = 0.0  # No penalty when puck is in our half

    # === φ_defense: Defensive positioning (NEW in V3) ===
    # Only active when puck is in our half - rewards being between puck and own goal
    if puck_in_our_half:
        # Calculate how well positioned the agent is for defense
        # Good defense = agent is between puck and own goal (closer to own goal than puck is)
        dist_agent_to_own_goal = np.linalg.norm(player_pos - OWN_GOAL)
        dist_puck_to_own_goal = np.linalg.norm(puck_pos - OWN_GOAL)

        # Defensive ratio: how much closer is agent to own goal than puck
        # If agent is between puck and goal: ratio > 0 (good)
        # If agent is behind puck (away from goal): ratio < 0 (bad)
        if dist_puck_to_own_goal > 0.1:  # Avoid division issues
            # Normalize: +1 when perfectly between, -1 when far behind puck
            defensive_ratio = (dist_puck_to_own_goal - dist_agent_to_own_goal) / dist_puck_to_own_goal
            defensive_ratio = np.clip(defensive_ratio, -1.0, 1.0)
            phi_defense = defensive_ratio  # Range: [-1, 1]
        else:
            phi_defense = 0.0  # Puck at goal, no defensive bonus
    else:
        phi_defense = 0.0  # Not in defensive situation

    # === Combine with weights ===
    phi_combined = W_CHASE * phi_chase + W_ATTACK * phi_attack + W_DEFENSE * phi_defense

    # === Scale for episode ===
    phi = EPISODE_SCALE * phi_combined
    # Range approximately: [-120, +30] depending on situation

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
    Get individual potential components for debugging (V3).

    Returns dict with each component's contribution.
    """
    player_pos = np.array([obs[0], obs[1]])
    puck_pos = np.array([obs[12], obs[13]])
    puck_vel = np.array([obs[14], obs[15]])
    puck_speed = np.linalg.norm(puck_vel)

    # Determine puck location
    puck_in_our_half = puck_pos[0] < CENTER_X
    puck_is_moving = puck_speed > PUCK_MOVING_THRESHOLD

    # Chase component with location-dependent weight
    dist_to_puck = np.linalg.norm(player_pos - puck_pos)
    phi_chase_base = -dist_to_puck / MAX_DISTANCE

    # Full weight if moving OR in our half (defensive recovery)
    if puck_is_moving or puck_in_our_half:
        phi_chase = phi_chase_base
        chase_weight_applied = 1.0
    else:
        phi_chase = STATIONARY_PUCK_WEIGHT * phi_chase_base
        chase_weight_applied = STATIONARY_PUCK_WEIGHT

    # Attack component (V3 - asymmetric)
    dist_puck_to_opp_goal = np.linalg.norm(puck_pos - OPPONENT_GOAL)
    if not puck_in_our_half:
        phi_attack = -dist_puck_to_opp_goal / MAX_DISTANCE
    else:
        phi_attack = 0.0  # No penalty in our half

    # Defense component (V3 - new)
    if puck_in_our_half:
        dist_agent_to_own_goal = np.linalg.norm(player_pos - OWN_GOAL)
        dist_puck_to_own_goal = np.linalg.norm(puck_pos - OWN_GOAL)
        if dist_puck_to_own_goal > 0.1:
            defensive_ratio = (dist_puck_to_own_goal - dist_agent_to_own_goal) / dist_puck_to_own_goal
            phi_defense = np.clip(defensive_ratio, -1.0, 1.0)
        else:
            phi_defense = 0.0
    else:
        phi_defense = 0.0
        dist_agent_to_own_goal = np.linalg.norm(player_pos - OWN_GOAL)
        dist_puck_to_own_goal = np.linalg.norm(puck_pos - OWN_GOAL)

    # Combined potential
    phi_combined = W_CHASE * phi_chase + W_ATTACK * phi_attack + W_DEFENSE * phi_defense

    return {
        # Chase component
        'phi_chase': phi_chase,
        'phi_chase_base': phi_chase_base,
        'phi_chase_weighted': W_CHASE * phi_chase,
        'chase_weight_applied': chase_weight_applied,
        'dist_to_puck': dist_to_puck,
        # Attack component (V3 - asymmetric)
        'phi_attack': phi_attack,
        'phi_attack_weighted': W_ATTACK * phi_attack,
        'phi_attack_active': not puck_in_our_half,
        'dist_puck_to_opp_goal': dist_puck_to_opp_goal,
        # Defense component (V3 - new)
        'phi_defense': phi_defense,
        'phi_defense_weighted': W_DEFENSE * phi_defense,
        'phi_defense_active': puck_in_our_half,
        'dist_agent_to_own_goal': dist_agent_to_own_goal,
        'dist_puck_to_own_goal': dist_puck_to_own_goal,
        # Combined
        'phi_combined': phi_combined,
        'phi_scaled': EPISODE_SCALE * phi_combined,
        # Puck state
        'puck_speed': puck_speed,
        'puck_is_moving': puck_is_moving,
        'puck_in_our_half': puck_in_our_half,
        # Weights
        'W_CHASE': W_CHASE,
        'W_ATTACK': W_ATTACK,
        'W_DEFENSE': W_DEFENSE,
    }
