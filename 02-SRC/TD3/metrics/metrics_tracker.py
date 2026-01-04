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
        # Self-play tracking
        self._last_eval = {}  # {opponent_type: win_rate}
        self._peak_eval = {}  # {opponent_type: win_rate}

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
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.rolling_outcomes = deque(maxlen=self.rolling_window)
        self.goals_scored = 0
        self.goals_conceded = 0
        self.pbrs_totals = []
        self.strategic_stats = {}
        self._last_eval = {}
        self._peak_eval = {}

    def reset_episode(self):
        # Reset per-episode tracking
        self._episode_step_rewards = []
        self._episode_action_magnitudes = []

    def add_step_reward(self, reward):
        # Add reward for a single step
        self._episode_step_rewards.append(reward)

    def add_action_magnitude(self, magnitude):
        # Add action magnitude for a single step
        self._episode_action_magnitudes.append(magnitude)

    def add_episode_result(self, reward, length, winner):
        #########################################################
        # Record metrics for a single episode
        # winner: 1=win, -1=loss, 0=tie
        #########################################################
        self.rewards_p1.append(reward)
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

    # def get_win_rate_decisive_only(self):
    #     #########################################################
    #     decisive_games = self.wins + self.losses
    #     return self.wins / decisive_games if decisive_games > 0 else 0

    def add_strategic_stats(self, stats):
        # Add strategic reward shaping stats
        self.strategic_stats.update(stats)

    def add_pbrs_total(self, pbrs_total):
        # Add total PBRS reward for an episode
        self.pbrs_totals.append(pbrs_total)

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








