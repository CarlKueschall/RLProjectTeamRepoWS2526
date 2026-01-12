"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import numpy as np


def compute_potential(obs, gamma=0.99):
    #########################################################
    # Compute potential function \[\Phi(s)\] for Potential-Based Reward Shaping (PBRS).
    #########################################################
    # HAD TO FIX THIS PARt: Had to scale this down because q-values were exploding
    # Based on research from Robot Air Hockey Challenge (NeurIPS 2023-2024).
    # This is the ONLY safe way to shape rewards without changing the optimal policy.
    # Formula: \[F(s,a,s') = \gamma \cdot \Phi(s') - \Phi(s)\]
    # The sum over a trajectory telescopes to: \[\Phi(s_T) - \Phi(s_0)\]
    # This means the agent CANNOT "hack" the reward by looping or doing weird behaviors.
    # Potential Function:
    # \[\Phi(s) = \text{SCALE} \cdot [w_1 \cdot (1 - \text{dist}_{\text{puck} \to \text{goal}}) - w_2 \cdot \text{dist}_{\text{agent} \to \text{puck}}]\]
    #Arguments:
    # obs: Observation (18 dims)
    # gamma: Discount factor (must match training gamma)
    #Returns:
    # phi: Potential value for this state
    p1_pos = np.array([obs[0], obs[1]])  # our position
    puck_pos = np.array([obs[12], obs[13]])  # puck position
    our_goal_pos = np.array([-4.5, 0.0])  # Constant anchor, our goal
    has_puck = obs[16] > 0  # do we have puck?
    puck_x = obs[12]  # puck x coord

    #########################################################
    # Offensive progress potential - reward for puck near opponent goal

    dist_puck_to_opp_goal = max(0, 5.0 - puck_x)  # how far from their goal
    phi_offense = 1.5 * (1.0 - (dist_puck_to_opp_goal / 10.0))  # closer = better

    #########################################################
    # CRITICAL FIX: Add puck velocity component - reward shooting toward goal
    #########################################################
    puck_velocity_x = obs[14]  # Puck x velocity (toward opponent goal is positive)
    # Tanh to saturate at high velocities, scale to ~[-1, 1] range
    # Positive velocity toward goal (x>0) is good, negative is bad
    phi_velocity = 2.0 * np.tanh(puck_velocity_x / 2.0)  # Reward puck moving toward goal

    #########################################################
    # Proximity potential - use tanh to saturate at dist=1.5
    #########################################################
    dist_p1_to_puck = np.linalg.norm(p1_pos - puck_pos)  # distance to puck
    phi_prox = -1.5 * np.tanh(dist_p1_to_puck / 1.5)  # closer is better, smooth curve

    #########################################################
    # Defensive lane potential - gaussian reward for staying in interception lane
    #########################################################
    line_vec = puck_pos - our_goal_pos
    line_unit = line_vec / (np.linalg.norm(line_vec) + 1e-6)
    proj_len = np.clip(np.dot(p1_pos - our_goal_pos, line_unit), 0, np.linalg.norm(line_vec))
    closest_point_on_line = our_goal_pos + proj_len * line_unit
    dist_to_lane = np.linalg.norm(p1_pos - closest_point_on_line)
    
    phi_lane = 1.0 * np.exp(-(dist_to_lane**2) / 0.5)  # stay in defensive lane, gaussian falloff

    #########################################################
    # Cushion potential - push agent back when too far forward
    #########################################################
    sigmoid_puck = 1.0 / (1.0 + np.exp(-2.0 * puck_x))
    # ReLU(p1_x + 2.0) pushes agent back toward -2.0 when too far forward
    # Condition on !has_puck so we don't penalize breakaways
    cushion_activation = max(0, p1_pos[0] + 2.0) * sigmoid_puck * (1.0 - float(has_puck))
    phi_cushion = -2.0 * np.tanh(cushion_activation)

    #########################################################
    # Combine all potentials with scaling
    #########################################################
    # CRITICAL FIX: Increased from 0.05 to 1.0 (20x) to make PBRS truly impactful
    # Previous 0.05 * pbrs_scale=0.5 = 0.025 total multiplier = only 0.25% of sparse rewards
    # New 1.0 * pbrs_scale=2.0 = 2.0 total multiplier = ~10-20% of sparse rewards (±10)
    # This provides meaningful dense guidance while preserving policy invariance
    SCALE = 1.0  # Was 0.05, increased 20x to make PBRS contribute 10-20% of total reward
    phi = SCALE * (phi_offense + phi_prox + phi_lane + phi_cushion + phi_velocity)

    return phi


def compute_pbrs(obs, obs_next, done, gamma=0.99):
    #########################################################
    # Compute Potential-Based Reward Shaping (PBRS) with EPISODIC CORRECTION.
    #Setting terminal potential to 0 to get rid of bias
    # \[F(s,a,s') = \gamma \cdot \Phi(s') - \Phi(s)\]
    # In episodic RL with truncation, we MUST set \[\Phi(s_{\text{terminal}}) = 0\], otherwise
    # the shaped return includes a bias term \[\gamma^N \cdot \Phi(s_N) - \Phi(s_0)\] that can
    # systematically distort the reward and destabilize learning.
    # This is the ONLY form of reward shaping that is guaranteed to preserve
    # the optimal policy (Ng et al., 1999).
    # Advantages over naive reward shaping:
    # - Cannot be "hacked" by agent (sum telescopes to \[\Phi(s_T) - \Phi(s_0)\])
    # - Preserves optimal policy (policy invariance)
    # - Provides dense learning signal without changing task objective
    #Arguments:
    # obs: Current observation (18 dims)
    # obs_next: Next observation (18 dims)
    # done: Boolean, True if episode terminated/truncated
    # gamma: Discount factor (must match training gamma)
    #Returns:
    # shaping_reward: \[F(s,a,s') = \gamma \cdot \Phi(s') - \Phi(s)\] (with terminal correction)
    phi_current = compute_potential(obs, gamma)

    #########################################################
    # IMPORTANT: Force terminal potential to 0
    if done:
        phi_next = 0.0  # terminal state = 0 potential, no bias
    else:
        phi_next = compute_potential(obs_next, gamma)

    shaping_reward = gamma * phi_next - phi_current  # \[F(s,a,s') = \gamma \cdot \Phi(s') - \Phi(s)\]

    return shaping_reward


class PBRSReward:
    ######################################################
    # Potential-Based Reward Shaping wrapper.

    def __init__(self, gamma=0.99, annealing_episodes=5000, pbrs_scale=1.0, constant_weight=False):
        #########################################################
        # Initialize PBRS reward shaper.
        #Arguments:
        # gamma: Discount factor
        # annealing_episodes: Episodes over which to anneal shaping during self-play
        # pbrs_scale: Global scaling factor for PBRS magnitude (default: 1.0)
        # constant_weight: If True, disable annealing and keep PBRS weight constant (default: False)
        self.gamma = gamma
        self.annealing_episodes = annealing_episodes
        self.pbrs_scale = pbrs_scale
        self.constant_weight = constant_weight
        self.self_play_start = None
        self.episode = 0

    def set_self_play_start(self, episode):
        #########################################################
        # Set the episode when self-play starts (for annealing).
        self.self_play_start = episode

    def compute(self, obs_curr, obs_next, done, episode=None):
        #########################################################
        #Compute PBRS reward with optional annealing.
        #Arguments:
        # obs_curr: Current observation
        # obs_next: Next observation
        # done: Episode done flag
        # episode: Current episode number (for annealing)
        #Returns:
        # shaped_reward: Additional reward term
        shaped_reward = compute_pbrs(obs_curr, obs_next, done, self.gamma)

        #########################################################
        # Apply global PBRS scaling (allows reducing PBRS magnitude)
        #########################################################
        shaped_reward *= self.pbrs_scale

        #########################################################
        # PBRS annealing - reduce shaping during self-play to prevent interference
        # CRITICAL FIX: Can be disabled with constant_weight flag to keep PBRS active
        #########################################################
        if not self.constant_weight:  # only anneal if constant_weight is False
            if self.self_play_start is not None and episode is not None:
                if episode > self.self_play_start:
                    # Linearly anneal from 1.0 → 0.0 over annealing_episodes
                    episodes_since_selfplay = episode - self.self_play_start
                    w_shaping = max(0.0, 1.0 - (episodes_since_selfplay / self.annealing_episodes))
                    shaped_reward *= w_shaping  # fade out shaping during self-play, let agent learn naturally

        return shaped_reward

    def get_annealing_weight(self, episode):
        #########################################################
        # Get current annealing weight (for logging).
        if self.constant_weight:  # constant weight mode: always 1.0
            return 1.0
        if self.self_play_start is None or episode <= self.self_play_start:
            return 1.0
        else:
            episodes_since_selfplay = episode - self.self_play_start
            return max(0.0, 1.0 - (episodes_since_selfplay / self.annealing_episodes))

    def reset(self):
        #########################################################
        # Reset episode state.
        pass
