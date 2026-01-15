"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description='Train TD3 agent on Hockey environment')

    # Environment
    parser.add_argument('--mode', type=str, default='NORMAL',
                        choices=['NORMAL', 'TRAIN_SHOOTING', 'TRAIN_DEFENSE'],
                        help='Hockey game mode (default: NORMAL)')
    parser.add_argument('--opponent', type=str, default='weak',
                        choices=['weak', 'strong', 'self'],
                        help='Opponent type (default: weak)')
    parser.add_argument('--keep_mode', action='store_true', default=True,
                        help='Enable keep mode (allows puck holding, default: True)')
    parser.add_argument('--no_keep_mode', dest='keep_mode', action='store_false',
                        help='Disable keep mode (puck bounces immediately on contact)')

    # Training
    parser.add_argument('--max_episodes', type=int, default=5000,
                        help='Maximum training episodes (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # TD3 hyperparameters
    parser.add_argument('--eps', type=float, default=1.0,
                        help='Initial exploration noise (default: 1.0, START FULLY RANDOM)')
    parser.add_argument('--eps_min', type=float, default=0.1,
                        help='Minimum exploration noise (default: 0.1, maintain some exploration)')
    parser.add_argument('--eps_decay', type=float, default=0.9997,
                        help='Noise decay per episode (default: 0.9997, SLOW decay for extended exploration)')
    parser.add_argument('--warmup_episodes', type=int, default=2000,
                        help='Episodes to collect data before training starts (default: 2000, ~2%% of 100k)')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help='Actor learning rate (default: 3e-4)')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                        help='Critic learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor (default: 0.95, REDUCED from 0.99 to prevent Q-explosion)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient (default: 0.005, balances speed vs stability)')
    parser.add_argument('--policy_freq', type=int, default=2,
                        help='Delayed policy update frequency (default: 2)')
    parser.add_argument('--target_update_freq', type=int, default=2,
                        help='Target network update frequency in gradient steps (default: 2, update targets more frequently to prevent divergence)')
    parser.add_argument('--target_noise_std', type=float, default=0.2,
                        help='Target policy noise std (default: 0.2)')
    parser.add_argument('--target_noise_clip', type=float, default=0.5,
                        help='Target policy noise clip (default: 0.5)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size (default: 1024, INCREASED for stable Q-learning)')
    parser.add_argument('--buffer_size', type=int, default=int(1e6),
                        help='Replay buffer size (default: 1000000, ~4%% of 25M total transitions)')

    parser.add_argument('--train_freq', type=int, default=10,
                        help='Train every N steps (-1=after episode only, 10=every 10 steps, default: 10)')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1, no accumulation)')
    parser.add_argument('--reward_shaping', action='store_true', default=True,
                        help='Enable Potential-Based Reward Shaping (PBRS) - policy-invariant dense rewards (default: True)')
    parser.add_argument('--no_reward_shaping', dest='reward_shaping', action='store_false',
                        help='Disable reward shaping (use only sparse rewards)')
    parser.add_argument('--pbrs_scale', type=float, default=1.0,
                        help='PBRS magnitude scaling factor (default: 1.0). Use 0.5 to reduce PBRS influence, allowing sparse rewards to dominate')

    # Tie penalty (encourages decisive play over stalemates)
    parser.add_argument('--tie_penalty', type=float, default=-3.0,
                        help='Terminal penalty for tied games (default: -3.0, encourages decisive wins over stalemates)')
    parser.add_argument('--no_tie_penalty', action='store_true',
                        help='Disable tie penalty (ties give 0 reward)')

    # Reward scaling (fixes Q-explosion from sparse rewards)
    parser.add_argument('--reward_scale', type=float, default=0.1,
                        help='Scale factor for game rewards (default: 0.1, maps ±10 to ±1 range)')

    # Epsilon reset on self-play (re-enable exploration when facing new opponents)
    parser.add_argument('--epsilon_reset_on_selfplay', action='store_true', default=True,
                        help='Reset epsilon to 0.5 when self-play activates (default: True)')
    parser.add_argument('--no_epsilon_reset_on_selfplay', dest='epsilon_reset_on_selfplay', action='store_false',
                        help='Do not reset epsilon on self-play activation')
    parser.add_argument('--epsilon_reset_value', type=float, default=0.5,
                        help='Epsilon value to reset to when self-play starts (default: 0.5)')

    # PBRS constant weight (disable annealing during self-play)
    parser.add_argument('--pbrs_constant_weight', action='store_true', default=True,
                        help='Keep PBRS weight constant instead of annealing (default: True)')
    parser.add_argument('--no_pbrs_constant_weight', dest='pbrs_constant_weight', action='store_false',
                        help='Enable PBRS annealing during self-play')

    # Critic initialization (DISABLED: broke training with artificial 5.0 bias)
    parser.add_argument('--init_critic_bias_positive', action='store_true', default=False,
                        help='Initialize critic output bias to +5.0 (default: False, DISABLED - breaks learning)')
    parser.add_argument('--no_init_critic_bias_positive', dest='init_critic_bias_positive', action='store_false',
                        help='Use default (zero) critic bias initialization (default)')

    # LR decay (cosine annealing for long training runs)
    parser.add_argument('--lr_decay', action='store_true', default=False,
                        help='Enable cosine LR decay (default: False, DISABLED to prevent learning freeze)')
    parser.add_argument('--no_lr_decay', dest='lr_decay', action='store_false',
                        help='Disable LR decay (constant learning rate)')
    parser.add_argument('--lr_min_factor', type=float, default=0.1,
                        help='Minimum LR as fraction of initial (default: 0.1, so 3e-4 -> 3e-5)')

    parser.add_argument('--q_clip', type=float, default=25.0,
                        help='Maximum absolute Q-value (clip to [-q_clip, q_clip], default: 25.0, tighter bounds prevent explosion)')
    parser.add_argument('--q_clip_mode', type=str, default='soft', choices=['hard', 'soft'],
                        help='Q-value clipping mode: hard=clamp, soft=tanh scaling (default: soft, prevents negative spiral)')
    parser.add_argument('--q_warning_threshold', type=float, default=10.0,
                        help='Warn when Q-values exceed this threshold (default: 10.0, catch explosion earlier)')

    # Network architecture
    parser.add_argument('--hidden_actor', type=int, nargs='+', default=[256, 256],
                        help='Actor hidden sizes (default: [256, 256])')
    parser.add_argument('--hidden_critic', type=int, nargs='+', default=[256, 256, 128],
                        help='Critic hidden sizes (default: [256, 256, 128])')

    # Optimization
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')

    # Logging
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Episodes between logging (default: 10)')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='Episodes between checkpoints (default: 500)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    # GIF generation now happens automatically during evaluations (see --eval_interval)
    # Keeping gif_interval for backward compatibility but it's ignored
    parser.add_argument('--gif_interval', type=int, default=0,
                        help='DEPRECATED: GIFs now generate during evaluations (--eval_interval)')
    parser.add_argument('--gif_episodes', type=int, default=3,
                        help='Number of episodes to record per GIF (stitched horizontally, default: 3)')

    # Checkpoint loading
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to load and continue training')

    # Progressive Opponent Training (4-phase: warmup → weak → strong → self-play)
    parser.add_argument('--opponent_progression', action='store_true',
                        help='Enable progressive opponent difficulty (weak → strong → self-play)')
    parser.add_argument('--weak_until', type=int, default=3000,
                        help='Train against weak opponent until this episode (default: 3000)')
    parser.add_argument('--strong_until', type=int, default=8000,
                        help='Train against strong opponent until this episode, then start self-play (default: 8000)')

    # Self-Play Training (can be appended to any training run)
    parser.add_argument('--self_play_start', type=int, default=0,
                        help='Episode to start self-play (0=disabled, e.g. 5000 = after 5000 eps against weak)')
    parser.add_argument('--disable_selfplay', action='store_true', default=False,
                        help='Completely disable self-play even if self_play_start > 0 (forces weak opponent only)')
    parser.add_argument('--self_play_pool_size', type=int, default=25,
                        help='Number of past checkpoints to keep in opponent pool (default: 25, more diversity for long training)')
    parser.add_argument('--self_play_save_interval', type=int, default=500,
                        help='Save current agent to pool every N episodes during self-play (default: 500)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Evaluate against weak opponent every N episodes (applies to all phases, default: 1000)')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of episodes to run per evaluation (default: 100)')
    parser.add_argument('--self_play_weak_ratio', type=float, default=0.5,
                        help='Ratio of episodes to train against weak opponent during self-play (default: 0.5 = 50%%)')
    parser.add_argument('--episode_block_size', type=int, default=50,
                        help='Number of consecutive episodes per opponent before switching (default: 50, INCREASED for stable Q-learning)')

    # Advanced Self-Play Features (anti-forgetting, PFSP, performance-gating)
    parser.add_argument('--use_dual_buffers', action='store_true', default=False,
                        help='Use dual replay buffers (anchor vs pool) to prevent catastrophic forgetting')
    parser.add_argument('--use_pfsp', action='store_true', default=True,
                        help='Use PFSP (Prioritized Fictitious Self-Play) opponent selection (default: True)')
    parser.add_argument('--no_pfsp', dest='use_pfsp', action='store_false',
                        help='Disable PFSP opponent selection')
    parser.add_argument('--pfsp_mode', type=str, default='variance', choices=['variance', 'hard'],
                        help='PFSP mode: variance (focus ~50%%) or hard (focus hardest)')
    parser.add_argument('--dynamic_anchor_mixing', action='store_true', default=False,
                        help='Dynamically adjust anchor opponent ratio based on forgetting detection')
    parser.add_argument('--performance_gated_selfplay', action='store_true', default=False,
                        help='Gate self-play activation on performance (90%% vs weak + low variance) instead of episode count')
    parser.add_argument('--selfplay_gate_winrate', type=float, default=0.75,
                        help='Min eval win-rate vs strong to activate self-play (default: 0.75, LOWERED to be reachable)')
    parser.add_argument('--regression_rollback', action='store_true', default=False,
                        help='Enable automatic rollback on performance regression')
    parser.add_argument('--regression_threshold', type=float, default=0.15,
                        help='Rollback if eval vs weak drops >this from best (default: 0.15 = 15%%)')

    # Prioritized Experience Replay (PER) - oversample high-TD-error transitions
    parser.add_argument('--use_per', action='store_true', default=False,
                        help='Enable Prioritized Experience Replay to oversample rare winning experiences (default: False)')
    parser.add_argument('--no_per', dest='use_per', action='store_false',
                        help='Disable PER (use uniform sampling)')
    parser.add_argument('--per_alpha', type=float, default=0.6,
                        help='PER priority exponent: 0=uniform, 1=full prioritization (default: 0.6)')
    parser.add_argument('--per_beta_start', type=float, default=0.4,
                        help='PER initial beta for importance sampling correction (default: 0.4, anneals to 1.0)')
    parser.add_argument('--per_beta_frames', type=int, default=100000,
                        help='PER frames to anneal beta from start to 1.0 (default: 100000)')

    # Parallel Environment Data Collection
    parser.add_argument('--parallel_envs', type=int, default=1,
                        help='Number of parallel environments for data collection (default: 1, no parallelism). '
                             'Set to 4-8 for 2-4x speedup on multi-core systems.')
    parser.add_argument('--parallel_batch_size', type=int, default=0,
                        help='Episodes to collect per parallel batch (default: 0 = same as parallel_envs). '
                             'Set higher to collect more episodes before training.')

    return parser.parse_args()


def get_mode(mode_str):
    # just translating the string to the enum
    import hockey.hockey_env as h_env
    mode_map = {
        'NORMAL': h_env.Mode.NORMAL,
        'TRAIN_SHOOTING': h_env.Mode.TRAIN_SHOOTING,
        'TRAIN_DEFENSE': h_env.Mode.TRAIN_DEFENSE,
    }
    return mode_map.get(mode_str, h_env.Mode.NORMAL)


def get_max_timesteps(mode):
    # we need a way to get the max timesteps for individual modes, they differ greatly.
    import hockey.hockey_env as h_env
    mode_map = {
        h_env.Mode.NORMAL: 250,
        h_env.Mode.TRAIN_SHOOTING: 80,
        h_env.Mode.TRAIN_DEFENSE: 80,
    }
    return mode_map.get(mode, 250)
