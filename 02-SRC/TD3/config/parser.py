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
    parser.add_argument('--warmup_episodes', type=int, default=500,
                        help='Episodes to collect data before training starts (default: 500)')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help='Actor learning rate (default: 3e-4)')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                        help='Critic learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
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
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--buffer_size', type=int, default=int(5e5),
                        help='Replay buffer size (default: 500000)')

    parser.add_argument('--train_freq', type=int, default=10,
                        help='Train every N steps (-1=after episode only, 10=every 10 steps, default: 10)')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1, no accumulation)')
    parser.add_argument('--reward_shaping', action='store_true', default=True,
                        help='Enable Potential-Based Reward Shaping (PBRS) - policy-invariant dense rewards (default: True)')
    parser.add_argument('--no_reward_shaping', dest='reward_shaping', action='store_false',
                        help='Disable reward shaping (use only sparse rewards)')

    parser.add_argument('--q_clip', type=float, default=25.0,
                        help='Maximum absolute Q-value (clip to [-q_clip, q_clip], default: 25.0, tighter bounds prevent explosion)')
    parser.add_argument('--q_clip_mode', type=str, default='hard', choices=['hard', 'soft'],
                        help='Q-value clipping mode: hard=clamp, soft=tanh scaling (default: hard, proven more stable)')
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
    parser.add_argument('--self_play_pool_size', type=int, default=10,
                        help='Number of past checkpoints to keep in opponent pool (default: 10)')
    parser.add_argument('--self_play_save_interval', type=int, default=500,
                        help='Save current agent to pool every N episodes during self-play (default: 500)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Evaluate against weak opponent every N episodes (applies to all phases, default: 1000)')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of episodes to run per evaluation (default: 100)')
    parser.add_argument('--self_play_weak_ratio', type=float, default=0.5,
                        help='Ratio of episodes to train against weak opponent during self-play (default: 0.5 = 50%%)')

    # Advanced Self-Play Features (anti-forgetting, PFSP, performance-gating)
    parser.add_argument('--use_dual_buffers', action='store_true', default=False,
                        help='Use dual replay buffers (anchor vs pool) to prevent catastrophic forgetting')
    parser.add_argument('--use_pfsp', action='store_true', default=False,
                        help='Use PFSP (Prioritized Fictitious Self-Play) opponent selection')
    parser.add_argument('--pfsp_mode', type=str, default='variance', choices=['variance', 'hard'],
                        help='PFSP mode: variance (focus ~50%%) or hard (focus hardest)')
    parser.add_argument('--dynamic_anchor_mixing', action='store_true', default=False,
                        help='Dynamically adjust anchor opponent ratio based on forgetting detection')
    parser.add_argument('--performance_gated_selfplay', action='store_true', default=False,
                        help='Gate self-play activation on performance (90%% vs weak + low variance) instead of episode count')
    parser.add_argument('--selfplay_gate_winrate', type=float, default=0.90,
                        help='Min eval win-rate vs weak to activate self-play (default: 0.90)')
    parser.add_argument('--regression_rollback', action='store_true', default=False,
                        help='Enable automatic rollback on performance regression')
    parser.add_argument('--regression_threshold', type=float, default=0.15,
                        help='Rollback if eval vs weak drops >this from best (default: 0.15 = 15%%)')

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
