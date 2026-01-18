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
    # TD3 paper uses constant Gaussian noise N(0, 0.1), so eps=0.1 with no decay
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Exploration noise std (default: 0.1, TD3 paper N(0,0.1))')
    parser.add_argument('--eps_min', type=float, default=0.1,
                        help='Minimum exploration noise (default: 0.1, no decay per TD3 paper)')
    parser.add_argument('--eps_decay', type=float, default=1.0,
                        help='Noise decay per episode (default: 1.0, no decay per TD3 paper)')
    parser.add_argument('--warmup_episodes', type=int, default=2000,
                        help='Episodes to collect data before training starts (default: 2000, ~2%% of 100k)')
    parser.add_argument('--lr_actor', type=float, default=1e-3,
                        help='Actor learning rate (default: 1e-3, TD3 paper)')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                        help='Critic learning rate (default: 1e-3, TD3 paper)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99, standard for sparse rewards)')
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
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size (default: 100, TD3 paper)')
    parser.add_argument('--buffer_size', type=int, default=int(1e6),
                        help='Replay buffer size (default: 1000000, ~4%% of 25M total transitions)')

    parser.add_argument('--train_freq', type=int, default=-1,
                        help='Train every N steps (-1=after episode only [DEFAULT, standard TD3], N=every N steps)')
    parser.add_argument('--iter_fit', type=int, default=250,
                        help='Gradient updates per training call (default: 250, matches ~250 steps/episode for standard TD3)')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1, no accumulation)')
    parser.add_argument('--reward_shaping', action='store_true', default=True,
                        help='Enable Potential-Based Reward Shaping (PBRS) - policy-invariant dense rewards (default: True)')
    parser.add_argument('--no_reward_shaping', dest='reward_shaping', action='store_false',
                        help='Disable reward shaping (use only sparse rewards)')
    parser.add_argument('--pbrs_scale', type=float, default=0.02,
                        help='PBRS magnitude scaling factor (default: 0.02). Mathematically derived: ensures episode shaping < sparse reward')

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

    # PBRS Annealing (independent of self-play)
    parser.add_argument('--pbrs_anneal_start', type=int, default=0,
                        help='Episode to start PBRS annealing (0=never anneal, default: 0). '
                             'Recommended: 5000 for 100k training to let PBRS guide early learning.')
    parser.add_argument('--pbrs_anneal_episodes', type=int, default=15000,
                        help='Episodes over which to anneal PBRS (default: 15000, slow annealing). '
                             'After anneal_start + anneal_episodes, PBRS weight = min_weight.')
    parser.add_argument('--pbrs_min_weight', type=float, default=0.1,
                        help='Minimum PBRS weight after annealing (default: 0.1). '
                             'Retains attack incentive to prevent shooting backward exploit.')
    parser.add_argument('--pbrs_cross_weight', type=float, default=0.4,
                        help='Weight for cross-court bonus component (default: 0.4). '
                             'Rewards shooting away from opponent position to encourage alternating shots. '
                             'Set to 0 to disable cross-court bonus.')
    parser.add_argument('--pbrs_clip', type=float, default=1.0,
                        help='Max absolute value for per-step PBRS reward (default: 1.0). '
                             'Prevents reward signal inconsistencies from large potential changes. '
                             'Set to 0 to disable clipping.')

    # Epsilon reset at PBRS annealing (re-explore when reward landscape changes)
    parser.add_argument('--epsilon_reset_at_anneal', action='store_true', default=False,
                        help='Reset epsilon when PBRS annealing starts (default: False). '
                             'Forces re-exploration when reward landscape changes.')
    parser.add_argument('--no_epsilon_reset_at_anneal', dest='epsilon_reset_at_anneal', action='store_false',
                        help='Do not reset epsilon when PBRS annealing starts')
    parser.add_argument('--epsilon_anneal_reset_value', type=float, default=0.4,
                        help='Epsilon value to reset to when PBRS annealing starts (default: 0.4)')

    # Legacy: PBRS constant weight (for self-play annealing, kept for compatibility)
    parser.add_argument('--pbrs_constant_weight', action='store_true', default=True,
                        help='Keep PBRS weight constant during self-play (default: True). '
                             'Note: Use --pbrs_anneal_start for independent annealing.')
    parser.add_argument('--no_pbrs_constant_weight', dest='pbrs_constant_weight', action='store_false',
                        help='Enable PBRS annealing during self-play (legacy, prefer --pbrs_anneal_start)')

    parser.add_argument('--q_clip', type=float, default=25.0,
                        help='Maximum absolute Q-value (clip to [-q_clip, q_clip], default: 25.0, tighter bounds prevent explosion)')
    parser.add_argument('--q_clip_mode', type=str, default='soft', choices=['hard', 'soft'],
                        help='Q-value clipping mode: hard=clamp, soft=tanh scaling (default: soft, prevents negative spiral)')
    parser.add_argument('--q_warning_threshold', type=float, default=10.0,
                        help='Warn when Q-values exceed this threshold (default: 10.0, catch explosion earlier)')

    # Value Function Regularization (anti-lazy learning)
    parser.add_argument('--vf_reg_lambda', type=float, default=0.1,
                        help='VF regularization strength (default: 0.1). Penalizes Q(passive) > Q(active) to prevent lazy agents. Set to 0 to disable.')

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
    parser.add_argument('--gif_episodes', type=int, default=3,
                        help='Number of episodes to record per GIF (stitched horizontally, default: 3)')

    # Checkpoint loading
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to load and continue training')

    # Self-Play Training
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

    # PFSP (Prioritized Fictitious Self-Play) opponent selection
    parser.add_argument('--use_pfsp', action='store_true', default=True,
                        help='Use PFSP (Prioritized Fictitious Self-Play) opponent selection (default: True)')
    parser.add_argument('--no_pfsp', dest='use_pfsp', action='store_false',
                        help='Disable PFSP opponent selection')
    parser.add_argument('--pfsp_mode', type=str, default='variance', choices=['variance', 'hard'],
                        help='PFSP mode: variance (focus ~50%%) or hard (focus hardest)')

    # Self-play activation gate (required for activation check logging)
    parser.add_argument('--selfplay_gate_winrate', type=float, default=0.7,
                        help='Win rate threshold vs weak for self-play activation logging (default: 0.7, informational only)')

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
