import numpy as np


def compute_potential(obs, gamma=0.99):
    #########################################################
    # Compute potential function Φ(s) for Potential-Based Reward Shaping (PBRS).
    #########################################################
    # CRITICAL FIX: Had to scale this down because q-values were exploding
    # Based on research from Robot Air Hockey Challenge (NeurIPS 2023-2024).
    # This is the ONLY safe way to shape rewards without changing the optimal policy.
    # Formula: F(s,a,s') = γ·Φ(s') - Φ(s)
    # The sum over a trajectory telescopes to: Φ(s_T) - Φ(s_0)
    # This means the agent CANNOT "hack" the reward by looping or doing weird behaviors.
    # Potential Function:
    # Φ(s) = SCALE * [w1·(1 - dist_puck→goal_norm) - w2·dist_agent→puck_norm]
    #Arguments:
    # obs: Observation (18 dims)
    # gamma: Discount factor (must match training gamma)
    #Returns:
    # phi: Potential value for this state
    # === Extract Objects ===
    p1_pos = np.array([obs[0], obs[1]])  # our position
    puck_pos = np.array([obs[12], obs[13]])  # puck position
    our_goal_pos = np.array([-4.5, 0.0])  # Constant anchor, our goal
    has_puck = obs[16] > 0  # do we have puck?
    puck_x = obs[12]  # puck x coord

    #########################################################
    # === 1. Offensive Progress Potential ===
    # Progressive reward for puck being near opponent goal
    dist_puck_to_opp_goal = max(0, 5.0 - puck_x)  # how far from their goal
    phi_offense = 1.5 * (1.0 - (dist_puck_to_opp_goal / 10.0))  # closer = better

    #########################################################
    # === 2. Smooth Proximity Potential (Lipschitz Continuous) ===
    # Use tanh to saturate at dist=1.5. No "cliffs" or infinite gradients.
    dist_p1_to_puck = np.linalg.norm(p1_pos - puck_pos)  # distance to puck
    phi_prox = -1.5 * np.tanh(dist_p1_to_puck / 1.5)  # closer is better, smooth curve

    #########################################################
    # === 3. Defensive Lane Potential (Gaussian Interception) ===
    #########################################################
    # Line segment distance (approx) from P1 to the line between Our Goal and Puck
    line_vec = puck_pos - our_goal_pos
    line_unit = line_vec / (np.linalg.norm(line_vec) + 1e-6)
    proj_len = np.clip(np.dot(p1_pos - our_goal_pos, line_unit), 0, np.linalg.norm(line_vec))
    closest_point_on_line = our_goal_pos + proj_len * line_unit
    dist_to_lane = np.linalg.norm(p1_pos - closest_point_on_line)
    
    # Gaussian reward for staying in the interception lane
    phi_lane = 1.0 * np.exp(-(dist_to_lane**2) / 0.5)  # stay in defensive lane, gaussian falloff

    #########################################################
    # === 4. Continuous Cushion Potential (Homesickness) ===
    # Use Sigmoid to smoothly activate only when puck is in offensive half
    sigmoid_puck = 1.0 / (1.0 + np.exp(-2.0 * puck_x))
    # ReLU(p1_x + 2.0) pushes agent back toward -2.0 when too far forward
    # Condition on !has_puck so we don't penalize breakaways
    cushion_activation = max(0, p1_pos[0] + 2.0) * sigmoid_puck * (1.0 - float(has_puck))
    phi_cushion = -2.0 * np.tanh(cushion_activation)

    #########################################################
    # === Combine with Scaling ===
    # Since the manifold is now smooth, we can safely use a higher scale than 0.001
    SCALE = 0.005  # scale it down so q-values don't explode
    phi = SCALE * (phi_offense + phi_prox + phi_lane + phi_cushion)
    #########################################################

    return phi


def compute_pbrs(obs, obs_next, done, gamma=0.99):
    #########################################################
    # Compute Potential-Based Reward Shaping (PBRS) with EPISODIC CORRECTION.
    #########################################################
    # CRITICAL FIX: Setting terminal potential to 0 to get rid of bias
    # F(s,a,s') = γ·Φ(s') - Φ(s)
    # In episodic RL with truncation, we MUST set Φ(s_terminal) = 0, otherwise
    # the shaped return includes a bias term γ^N·Φ(s_N) - Φ(s_0) that can
    # systematically distort the reward and destabilize learning.
    # This is the ONLY form of reward shaping that is guaranteed to preserve
    # the optimal policy (Ng et al., 1999).
    # Advantages over naive reward shaping:
    # - Cannot be "hacked" by agent (sum telescopes to Φ(s_T) - Φ(s_0))
    # - Preserves optimal policy (policy invariance)
    # - Provides dense learning signal without changing task objective
    #Arguments:
    # obs: Current observation (18 dims)
    # obs_next: Next observation (18 dims)
    # done: Boolean, True if episode terminated/truncated
    # gamma: Discount factor (must match training gamma)
    #Returns:
    # shaping_reward: F(s,a,s') = γ·Φ(s') - Φ(s) (with terminal correction)
    phi_current = compute_potential(obs, gamma)

    #########################################################
    # CRITICAL: Force terminal potential to 0
    #########################################################
    if done:
        phi_next = 0.0  # terminal state = 0 potential, no bias
    else:
        phi_next = compute_potential(obs_next, gamma)

    shaping_reward = gamma * phi_next - phi_current  # F(s,a,s') = γ·Φ(s') - Φ(s)

    return shaping_reward


class PBRSReward:
    ######################################################
    # Potential-Based Reward Shaping wrapper.
    ######################################################
    def __init__(self, gamma=0.99, annealing_episodes=5000):
        #########################################################
        # Initialize PBRS reward shaper.
        #Arguments:
        # gamma: Discount factor
        # annealing_episodes: Episodes over which to anneal shaping during self-play
        self.gamma = gamma
        self.annealing_episodes = annealing_episodes
        self.self_play_start = None
        self.episode = 0

    def set_self_play_start(self, episode):
        #########################################################
        # Set the episode when self-play starts (for annealing).
        self.self_play_start = episode

    def compute(self, obs_curr, obs_next, done, episode=None):
        #########################################################
        # Compute PBRS reward with optional annealing.
        #Arguments:
        # obs_curr: Current observation
        # obs_next: Next observation
        # done: Episode done flag
        # episode: Current episode number (for annealing)
        #Returns:
        # shaped_reward: Additional reward term
        shaped_reward = compute_pbrs(obs_curr, obs_next, done, self.gamma)

        #########################################################
        # PBRS ANNEALING: Reduce shaping during self-play to prevent interference
        #########################################################
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
        if self.self_play_start is None or episode <= self.self_play_start:
            return 1.0
        else:
            episodes_since_selfplay = episode - self.self_play_start
            return max(0.0, 1.0 - (episodes_since_selfplay / self.annealing_episodes))

    def reset(self):
        #########################################################
        # Reset episode state.
        pass
