"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import numpy as np
from collections import deque
class MetricsTracker:
    #########################################################
    # Track training metrics across episodes
    # We specifically out-source all the metric logic to this script because metric management with wandb
    # is very capable, but to fully utilize it it's worth keeping things clean and organized, otherwise the other scripts would
    # get very cluttered and hard to maintain/read etc.
    #########################################################
    def __init__(self, log_interval=10, rolling_window=100):
        self.log_interval = log_interval
        self.rolling_window = rolling_window

        # Episode metrics
        self.rewards_p1 = []
        self.rewards_p2 = []
        self.sparse_rewards = []
        self.episode_lengths = []
        self.training_calls_per_episode = []
        # Loss metrics
        self.critic_losses = []
        self.actor_losses = []

        # CRITICAL FIX: Gradient norm tracking for monitoring learning health
        self.critic_grad_norms = []
        self.actor_grad_norms = []

        # Win/loss/tie tracking
        self.wins = 0
        self.losses = 0
        self.ties = 0

        # Rolling outcomes for sliding window metrics
        self.rolling_outcomes = deque(maxlen=rolling_window)

        # Behavioral metrics for lazy learning diagnosis
        self.goals_scored = 0
        self.goals_conceded = 0

        # PBRS tracking
        self.pbrs_totals = []
        # Strategic stats
        self.strategic_stats = {}
        # Per-episode tracking (reset each episode)
        self._episode_step_rewards = []
        self._episode_action_magnitudes = []
        self._episode_agent_positions = []
        self._episode_puck_distances = []
        self._episode_time_near_puck = 0

        # Behavior metrics per episode (accumulated across all episodes)
        self.behavior_action_magnitude_avg = []
        self.behavior_action_magnitude_max = []
        self.behavior_lazy_action_ratio = []
        self.behavior_dist_to_puck_avg = []
        self.behavior_dist_to_puck_min = []
        self.behavior_time_near_puck = []
        self.behavior_distance_traveled = []
        self.behavior_velocity_avg = []
        self.behavior_velocity_max = []
        self.behavior_puck_touches = []

        # Shoot/Keep action tracking (action[3])
        self._episode_shoot_actions = []  # Track action[3] values each step
        self._episode_possession_steps = 0  # Count steps with possession
        self.behavior_shoot_action_avg = []  # Average action[3] value per episode
        self.behavior_shoot_action_when_possess = []  # Average action[3] when having possession
        self.behavior_possession_ratio = []  # Fraction of episode with possession

        # Self-play tracking
        self._last_eval = {}  # {opponent_type: win_rate}
        self._peak_eval = {}  # {opponent_type: win_rate}

        # Reward Hacking Detection (research-based)
        # Track: possession hoarding, PBRS exploitation, shooting behavior
        self._episode_puck_speed_when_possess = []  # Track puck speed when agent has possession
        self._episode_alignment_when_possess = []  # Track velocity alignment when possessing
        self.hacking_possession_hoarding_ratio = []  # Ratio of possession without shooting
        self.hacking_pbrs_to_sparse_ratio = []  # PBRS magnitude vs sparse reward
        self.hacking_possession_no_shoot_count = []  # Steps possessing with slow/misaligned puck

    def reset(self):
        #########################################################
        # Reset all metrics
        #########################################################
        self.rewards_p1 = []
        self.rewards_p2 = []
        self.sparse_rewards = []
        self.episode_lengths = []
        self.training_calls_per_episode = []
        self.critic_losses = []
        self.actor_losses = []
        self.critic_grad_norms = []
        self.actor_grad_norms = []
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.rolling_outcomes = deque(maxlen=self.rolling_window)
        self.goals_scored = 0
        self.goals_conceded = 0
        self.pbrs_totals = []
        self.strategic_stats = {}
        self.behavior_action_magnitude_avg = []
        self.behavior_action_magnitude_max = []
        self.behavior_lazy_action_ratio = []
        self.behavior_dist_to_puck_avg = []
        self.behavior_dist_to_puck_min = []
        self.behavior_time_near_puck = []
        self.behavior_distance_traveled = []
        self.behavior_velocity_avg = []
        self.behavior_velocity_max = []
        self.behavior_puck_touches = []
        self._last_eval = {}
        self._peak_eval = {}
        self.hacking_possession_hoarding_ratio = []
        self.hacking_pbrs_to_sparse_ratio = []
        self.hacking_possession_no_shoot_count = []

    def reset_episode(self):
        # Reset per-episode tracking
        self._episode_step_rewards = []
        self._episode_action_magnitudes = []
        self._episode_agent_positions = []  # Track positions for distance/velocity metrics
        self._episode_puck_distances = []  # Track distances to puck
        self._episode_time_near_puck = 0  # Count steps near puck (< 0.5)
        self._episode_shoot_actions = []  # Track action[3] values
        self._episode_shoot_actions_when_possess = []  # Track action[3] when having possession
        self._episode_possession_steps = 0  # Count steps with possession
        # Hacking detection per-episode tracking
        self._episode_puck_speed_when_possess = []
        self._episode_alignment_when_possess = []
        self._episode_hoarding_steps = 0  # Steps with possession but not shooting

    def add_step_reward(self, reward):
        # Add reward for a single step
        self._episode_step_rewards.append(reward)

    def add_action_magnitude(self, magnitude):
        # Add action magnitude for a single step
        self._episode_action_magnitudes.append(magnitude)

    def add_agent_position(self, pos):
        # Add agent position [x, y] for a single step
        self._episode_agent_positions.append(pos)

    def add_puck_distance(self, distance):
        # Add distance to puck for a single step
        self._episode_puck_distances.append(distance)
        # Track time spent near puck
        if distance < 0.5:
            self._episode_time_near_puck += 1

    def add_shoot_action(self, shoot_action, has_possession, puck_speed=None, alignment=None):
        # Track action[3] (shoot/keep) and possession status
        # shoot_action: value of action[3], negative=keep, positive=shoot
        # has_possession: True if obs[16] > 0 (agent has puck)
        # puck_speed: speed of puck (for hacking detection)
        # alignment: velocity alignment toward goal (for hacking detection)
        self._episode_shoot_actions.append(shoot_action)
        if has_possession:
            self._episode_possession_steps += 1
            self._episode_shoot_actions_when_possess.append(shoot_action)
            # Hacking detection: track puck state when possessing
            if puck_speed is not None:
                self._episode_puck_speed_when_possess.append(puck_speed)
            if alignment is not None:
                self._episode_alignment_when_possess.append(alignment)
            # Detect hoarding: possessing but puck slow or misaligned
            SHOOTING_SPEED_THRESHOLD = 0.5
            ALIGNMENT_THRESHOLD = 0.3
            if puck_speed is not None and alignment is not None:
                if puck_speed < SHOOTING_SPEED_THRESHOLD or alignment < ALIGNMENT_THRESHOLD:
                    self._episode_hoarding_steps += 1

    def add_episode_result(self, reward_p1, length, winner, reward_p2=None, sparse_reward=None):
        #########################################################
        # Record metrics for a single episode
        # reward_p1: Shaped reward for player 1 (with PBRS + strategic bonuses)
        # reward_p2: Reward for player 2/opponent (negative of P1 sparse reward)
        # sparse_reward: Unshapen sparse reward (±1 for wins/losses)
        # winner: 1=win, -1=loss, 0=tie
        #########################################################
        self.rewards_p1.append(reward_p1)
        if reward_p2 is not None:
            self.rewards_p2.append(reward_p2)
        if sparse_reward is not None:
            self.sparse_rewards.append(sparse_reward)
        self.episode_lengths.append(length)

        # Track outcome
        if winner == 1:
            self.wins += 1
            self.rolling_outcomes.append(1)
            self.goals_scored += 1
        elif winner == -1:
            self.losses += 1
            self.rolling_outcomes.append(-1)
            self.goals_conceded += 1
        else:
            self.ties += 1
            self.rolling_outcomes.append(0)

    def add_losses(self, losses):
        #########################################################
        # Add loss metrics
        # losses: List of (critic_loss, actor_loss) tuples
        #########################################################
        for critic_loss, actor_loss in losses:
            self.critic_losses.append(critic_loss if isinstance(critic_loss, float) else critic_loss.item())
            if actor_loss != 0.0:  # Only track non-zero actor losses
                self.actor_losses.append(actor_loss)

    def add_grad_norms(self, grad_norms):
        #########################################################
        # CRITICAL FIX: Add gradient norm metrics for monitoring learning health
        # grad_norms: List of (critic_grad_norm, actor_grad_norm) tuples
        #########################################################
        for critic_grad_norm, actor_grad_norm in grad_norms:
            self.critic_grad_norms.append(critic_grad_norm if isinstance(critic_grad_norm, float) else critic_grad_norm)
            if actor_grad_norm != 0.0:  # Only track non-zero actor gradient norms
                self.actor_grad_norms.append(actor_grad_norm)

    # def get_win_rate_decisive_only(self):
    #     #########################################################
    #     decisive_games = self.wins + self.losses
    #     return self.wins / decisive_games if decisive_games > 0 else 0

    def add_strategic_stats(self, stats):
        # Add strategic reward shaping stats
        self.strategic_stats.update(stats)

    def add_pbrs_total(self, pbrs_total, sparse_reward=None):
        # Add total PBRS reward for an episode
        self.pbrs_totals.append(pbrs_total)
        # Track PBRS to sparse ratio for hacking detection
        if sparse_reward is not None and abs(sparse_reward) > 1e-6:
            ratio = abs(pbrs_total) / abs(sparse_reward)
            self.hacking_pbrs_to_sparse_ratio.append(float(ratio))
        elif pbrs_total != 0:
            # Sparse reward is 0 but PBRS is not - could indicate exploitation
            self.hacking_pbrs_to_sparse_ratio.append(float(abs(pbrs_total)))

    def get_win_rate(self):
        # Calculate overall win rate
        total = self.wins + self.losses + self.ties
        return self.wins / total if total > 0 else 0

    def get_rolling_win_rate(self):
        # Calculate rolling window win rate
        if len(self.rolling_outcomes) == 0:
            return 0.0
        rolling_total = len(self.rolling_outcomes)
        rolling_wins = 0
        for x in self.rolling_outcomes:
            if x == 1:
                rolling_wins += 1
        return rolling_wins / rolling_total

    def get_avg_reward(self, window=None):
        # Get average reward over specified window (or all)
        data = self.rewards_p1[-window:] if window else self.rewards_p1
        return np.mean(data) if data else 0.0

    def get_avg_pbrs(self):
        # Get average PBRS reward
        return np.mean(self.pbrs_totals) if self.pbrs_totals else 0.0

    def finalize_episode_behavior_metrics(self):
        #########################################################
        # Compute behavior metrics for the episode and add to accumulated lists
        # Called at the end of each episode
        #########################################################
        # Action magnitude metrics
        if self._episode_action_magnitudes:
            self.behavior_action_magnitude_avg.append(float(np.mean(self._episode_action_magnitudes)))
            self.behavior_action_magnitude_max.append(float(np.max(self._episode_action_magnitudes)))
            # Lazy action ratio: count actions with magnitude < 0.1
            lazy_count = sum(1 for mag in self._episode_action_magnitudes if mag < 0.1)
            self.behavior_lazy_action_ratio.append(float(lazy_count / len(self._episode_action_magnitudes)))
        else:
            self.behavior_action_magnitude_avg.append(0.0)
            self.behavior_action_magnitude_max.append(0.0)
            self.behavior_lazy_action_ratio.append(0.0)

        # Puck proximity metrics
        if self._episode_puck_distances:
            self.behavior_dist_to_puck_avg.append(float(np.mean(self._episode_puck_distances)))
            self.behavior_dist_to_puck_min.append(float(np.min(self._episode_puck_distances)))
            self.behavior_time_near_puck.append(float(self._episode_time_near_puck))
        else:
            self.behavior_dist_to_puck_avg.append(0.0)
            self.behavior_dist_to_puck_min.append(0.0)
            self.behavior_time_near_puck.append(0.0)

        # Movement metrics
        if len(self._episode_agent_positions) > 1:
            # Compute distance traveled
            positions = np.array(self._episode_agent_positions)
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            self.behavior_distance_traveled.append(float(np.sum(distances)))

            # Velocity metrics (distance per step)
            if len(distances) > 0:
                self.behavior_velocity_avg.append(float(np.mean(distances)))
                self.behavior_velocity_max.append(float(np.max(distances)))
            else:
                self.behavior_velocity_avg.append(0.0)
                self.behavior_velocity_max.append(0.0)
        else:
            self.behavior_distance_traveled.append(0.0)
            self.behavior_velocity_avg.append(0.0)
            self.behavior_velocity_max.append(0.0)

        # Puck touches from strategic stats
        if 'puck_touches' in self.strategic_stats:
            self.behavior_puck_touches.append(float(self.strategic_stats['puck_touches']))
        else:
            self.behavior_puck_touches.append(0.0)

        # Shoot/Keep action metrics (action[3])
        episode_length = len(self._episode_shoot_actions) if self._episode_shoot_actions else 1
        if self._episode_shoot_actions:
            self.behavior_shoot_action_avg.append(float(np.mean(self._episode_shoot_actions)))
        else:
            self.behavior_shoot_action_avg.append(0.0)

        if self._episode_shoot_actions_when_possess:
            self.behavior_shoot_action_when_possess.append(float(np.mean(self._episode_shoot_actions_when_possess)))
        else:
            self.behavior_shoot_action_when_possess.append(0.0)

        # Possession ratio: fraction of episode with puck possession
        self.behavior_possession_ratio.append(float(self._episode_possession_steps / episode_length))

        # Hacking detection metrics
        # 1. Possession hoarding ratio: % of possession time spent not shooting
        if self._episode_possession_steps > 0:
            hoarding_ratio = self._episode_hoarding_steps / self._episode_possession_steps
            self.hacking_possession_hoarding_ratio.append(float(hoarding_ratio))
            self.hacking_possession_no_shoot_count.append(float(self._episode_hoarding_steps))
        else:
            self.hacking_possession_hoarding_ratio.append(0.0)
            self.hacking_possession_no_shoot_count.append(0.0)

    def get_behavior_metrics(self):
        #########################################################
        # Get averaged behavior metrics for the last log_interval episodes
        # Should only be called during logging
        #########################################################
        metrics = {}

        # Average over last log_interval episodes
        metrics['behavior/action_magnitude_avg'] = np.mean(self.behavior_action_magnitude_avg[-self.log_interval:]) if self.behavior_action_magnitude_avg else 0.0
        metrics['behavior/action_magnitude_max'] = np.mean(self.behavior_action_magnitude_max[-self.log_interval:]) if self.behavior_action_magnitude_max else 0.0
        metrics['behavior/lazy_action_ratio'] = np.mean(self.behavior_lazy_action_ratio[-self.log_interval:]) if self.behavior_lazy_action_ratio else 0.0
        metrics['behavior/dist_to_puck_avg'] = np.mean(self.behavior_dist_to_puck_avg[-self.log_interval:]) if self.behavior_dist_to_puck_avg else 0.0
        metrics['behavior/dist_to_puck_min'] = np.mean(self.behavior_dist_to_puck_min[-self.log_interval:]) if self.behavior_dist_to_puck_min else 0.0
        metrics['behavior/time_near_puck'] = np.mean(self.behavior_time_near_puck[-self.log_interval:]) if self.behavior_time_near_puck else 0.0
        metrics['behavior/distance_traveled'] = np.mean(self.behavior_distance_traveled[-self.log_interval:]) if self.behavior_distance_traveled else 0.0
        metrics['behavior/velocity_avg'] = np.mean(self.behavior_velocity_avg[-self.log_interval:]) if self.behavior_velocity_avg else 0.0
        metrics['behavior/velocity_max'] = np.mean(self.behavior_velocity_max[-self.log_interval:]) if self.behavior_velocity_max else 0.0
        metrics['behavior/puck_touches'] = np.mean(self.behavior_puck_touches[-self.log_interval:]) if self.behavior_puck_touches else 0.0

        # Shoot/Keep action metrics
        metrics['behavior/shoot_action_avg'] = np.mean(self.behavior_shoot_action_avg[-self.log_interval:]) if self.behavior_shoot_action_avg else 0.0
        metrics['behavior/shoot_action_when_possess'] = np.mean(self.behavior_shoot_action_when_possess[-self.log_interval:]) if self.behavior_shoot_action_when_possess else 0.0
        metrics['behavior/possession_ratio'] = np.mean(self.behavior_possession_ratio[-self.log_interval:]) if self.behavior_possession_ratio else 0.0

        # Hacking detection metrics
        metrics['hacking/possession_hoarding_ratio'] = np.mean(self.hacking_possession_hoarding_ratio[-self.log_interval:]) if self.hacking_possession_hoarding_ratio else 0.0
        metrics['hacking/possession_no_shoot_count'] = np.mean(self.hacking_possession_no_shoot_count[-self.log_interval:]) if self.hacking_possession_no_shoot_count else 0.0
        metrics['hacking/pbrs_to_sparse_ratio'] = np.mean(self.hacking_pbrs_to_sparse_ratio[-self.log_interval:]) if self.hacking_pbrs_to_sparse_ratio else 0.0

        return metrics

    def get_log_metrics(self):
        #########################################################
        # Get metrics for logging
        #########################################################
        avg_reward_p1 = np.mean(self.rewards_p1[-self.log_interval:]) if self.rewards_p1 else 0.0
        avg_reward_p2 = np.mean(self.rewards_p2[-self.log_interval:]) if self.rewards_p2 else 0.0
        avg_sparse = np.mean(self.sparse_rewards[-self.log_interval:]) if self.sparse_rewards else 0.0

        total_reward_avg = avg_reward_p1
        sparse_ratio = (
            avg_sparse / total_reward_avg
            if abs(total_reward_avg) > 1e-6
            else (1.0 if abs(avg_sparse) < 1e-6 else 0.0)
        )

        return {
            "rewards/p1": avg_reward_p1,
            "rewards/p2": avg_reward_p2,
            "rewards/sparse_only": avg_sparse,
            "rewards/sparse_ratio": sparse_ratio,
            "losses/critic_loss": np.mean(self.critic_losses[-self.log_interval:]) if self.critic_losses else 0.0,
            "losses/actor_loss": np.mean(self.actor_losses[-self.log_interval:]) if self.actor_losses else 0.0,
            # CRITICAL FIX: Gradient norm logging for monitoring learning health
            "gradients/critic_grad_norm": np.mean(self.critic_grad_norms[-self.log_interval:]) if self.critic_grad_norms else 0.0,
            "gradients/actor_grad_norm": np.mean(self.actor_grad_norms[-self.log_interval:]) if self.actor_grad_norms else 0.0,
            "training/calls_per_episode": np.mean(self.training_calls_per_episode[-self.log_interval:]) if self.training_calls_per_episode else 0,
            "training/avg_episode_length": np.mean(self.episode_lengths[-self.log_interval:]) if self.episode_lengths else 0,
        }

        # def get_summary_with_percentiles(self, window=None):
    #     #########################################################
    #     # DEPRECATED: Summary with percentile statistics
    #     window = window or len(self.rewards_p1)
    #     reward_window = self.rewards_p1[-window:] if self.rewards_p1 else []
    #     if reward_window:
    #         p25, p50, p75 = np.percentile(reward_window, [25, 50, 75])
    #     else:
    #         p25, p50, p75 = 0.0, 0.0, 0.0
    #     return {
    #         'win_rate': self.get_win_rate(),
    #         'rolling_win_rate': self.get_rolling_win_rate(),
    #         'avg_reward_p1': np.mean(reward_window) if reward_window else 0.0,
    #         'reward_p25': p25,
    #         'reward_p50': p50,
    #         'reward_p75': p75,
    #         'wins': self.wins,
    #         'losses': self.losses,
    #         'ties': self.ties,
    #     }

    @property
    def total_games(self):
        # Total number of games played
        return self.wins + self.losses + self.ties

    def get_last_eval(self, opponent_type):
        # Get last evaluation win rate against opponent type
        return self._last_eval.get(opponent_type, 0.0)

    def get_peak_eval(self, opponent_type):
        # Get peak evaluation win rate against opponent type
        return self._peak_eval.get(opponent_type, 0.0)

    def set_last_eval(self, opponent_type, win_rate):
        # Set last evaluation win rate against opponent type
        self._last_eval[opponent_type] = win_rate

    def set_peak_eval(self, opponent_type, win_rate):
        #########################################################
        # Update peak evaluation win rate against opponent type if higher
        #########################################################
        if win_rate > self._peak_eval.get(opponent_type, 0.0):
            self._peak_eval[opponent_type] = win_rate

    def get_summary(self, window=None):
        #########################################################
        # Get summary of recent metrics
        # window: Number of recent episodes to summarize (None = all)
        #########################################################
        window = window or len(self.rewards_p1)

        return {
            'win_rate': self.get_win_rate(),
            'rolling_win_rate': self.get_rolling_win_rate(),
            'avg_reward_p1': np.mean(self.rewards_p1[-window:]) if self.rewards_p1 else 0.0,
            'avg_reward_p2': np.mean(self.rewards_p2[-window:]) if self.rewards_p2 else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths[-window:]) if self.episode_lengths else 0.0,
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'total_episodes': len(self.rewards_p1),
        }








