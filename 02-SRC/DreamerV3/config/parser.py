"""
Command-line argument parser for DreamerV3 hockey training.

Based on hyperparameters from:
- DreamerV3 paper (Hafner et al., 2023)
- Robot Air Hockey Challenge 2023 (Orsula et al., 2024)

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DreamerV3 agent on Hockey environment'
    )

    # ===================
    # Environment
    # ===================
    parser.add_argument('--mode', type=str, default='NORMAL',
                        choices=['NORMAL', 'TRAIN_SHOOTING', 'TRAIN_DEFENSE'],
                        help='Hockey game mode (default: NORMAL)')
    parser.add_argument('--opponent', type=str, default='weak',
                        choices=['weak', 'strong', 'self'],
                        help='Opponent type (default: weak)')
    parser.add_argument('--include_fault_penalty', action='store_true', default=True,
                        help='Include fault penalty in rewards (default: True)')
    parser.add_argument('--no_fault_penalty', dest='include_fault_penalty', action='store_false',
                        help='Disable fault penalty')
    parser.add_argument('--fault_penalty', type=float, default=-0.33,
                        help='Penalty for faults (default: -0.33, from Robot Air Hockey paper)')

    # ===================
    # Training
    # ===================
    parser.add_argument('--max_steps', type=int, default=10_000_000,
                        help='Maximum training steps (default: 10M)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of parallel environments (default: 1)')

    # ===================
    # World Model (RSSM)
    # ===================
    parser.add_argument('--stoch_size', type=int, default=32,
                        help='Stochastic state size (default: 32)')
    parser.add_argument('--deter_size', type=int, default=512,
                        help='Deterministic state size (default: 512, reduce to 256 for 2080ti)')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden layer size (default: 512, reduce to 256 for 2080ti)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of MLP layers (default: 2)')

    # ===================
    # Imagination
    # ===================
    parser.add_argument('--imagination_horizon', type=int, default=50,
                        help='Imagination horizon for actor-critic training (default: 50). '
                             'Critical for sparse reward propagation - do not reduce below 25.')

    # ===================
    # Training Hyperparameters
    # ===================
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16, reduce to 8 for 2080ti)')
    parser.add_argument('--batch_length', type=int, default=64,
                        help='Sequence length for training (default: 64, reduce to 50 for 2080ti)')
    parser.add_argument('--lr_world', type=float, default=3e-4,
                        help='World model learning rate (default: 3e-4)')
    parser.add_argument('--lr_actor', type=float, default=3e-5,
                        help='Actor learning rate (default: 3e-5)')
    parser.add_argument('--lr_critic', type=float, default=3e-5,
                        help='Critic learning rate (default: 3e-5)')
    parser.add_argument('--gamma', type=float, default=0.997,
                        help='Discount factor (default: 0.997, higher than TD3 for longer horizons)')
    parser.add_argument('--lambda_gae', type=float, default=0.95,
                        help='GAE lambda for advantage estimation (default: 0.95)')
    parser.add_argument('--grad_clip', type=float, default=100.0,
                        help='Gradient clipping norm (default: 100.0)')

    # ===================
    # Replay Buffer
    # ===================
    parser.add_argument('--buffer_size', type=int, default=1_000_000,
                        help='Replay buffer size in transitions (default: 1M)')
    parser.add_argument('--min_buffer_size', type=int, default=1000,
                        help='Minimum buffer size before training starts (default: 1000)')

    # ===================
    # Exploration
    # ===================
    parser.add_argument('--expl_noise', type=float, default=0.3,
                        help='Exploration noise std (default: 0.3)')
    parser.add_argument('--expl_decay', type=float, default=0.9999,
                        help='Exploration noise decay per step (default: 0.9999)')
    parser.add_argument('--expl_min', type=float, default=0.1,
                        help='Minimum exploration noise (default: 0.1)')

    # ===================
    # Self-Play
    # ===================
    parser.add_argument('--self_play_start', type=int, default=0,
                        help='Step to start self-play (0=disabled, default: 0). '
                             'Recommended: start after agent can beat weak opponent.')
    parser.add_argument('--self_play_pool_size', type=int, default=25,
                        help='Number of past checkpoints in opponent pool (default: 25)')
    parser.add_argument('--self_play_save_interval', type=int, default=100000,
                        help='Save to opponent pool every N steps (default: 100000)')
    parser.add_argument('--self_play_weak_ratio', type=float, default=0.3,
                        help='Ratio of episodes against weak opponent during self-play (default: 0.3)')
    parser.add_argument('--use_pfsp', action='store_true', default=True,
                        help='Use Prioritized Fictitious Self-Play (default: True)')
    parser.add_argument('--no_pfsp', dest='use_pfsp', action='store_false',
                        help='Disable PFSP opponent selection')
    parser.add_argument('--pfsp_mode', type=str, default='variance',
                        choices=['variance', 'hard'],
                        help='PFSP mode: variance (focus ~50%% win rate) or hard (focus hardest)')

    # ===================
    # Logging
    # ===================
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Steps between logging (default: 1000)')
    parser.add_argument('--eval_interval', type=int, default=50000,
                        help='Steps between evaluation (default: 50000)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help='Episodes per evaluation (default: 50)')
    parser.add_argument('--save_interval', type=int, default=100000,
                        help='Steps between checkpoint saves (default: 100000)')
    parser.add_argument('--gif_interval', type=int, default=100000,
                        help='Steps between GIF recordings (default: 100000)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='dreamer-hockey',
                        help='W&B project name (default: dreamer-hockey)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')

    # ===================
    # Checkpointing
    # ===================
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory for saving checkpoints (default: results)')

    # ===================
    # Hardware
    # ===================
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use mixed precision training (reduces memory, may affect stability)')

    # ===================
    # DreamerV3 Specific
    # ===================
    parser.add_argument('--free_nats', type=float, default=1.0,
                        help='Free nats for KL loss (default: 1.0)')
    parser.add_argument('--kl_balance', type=float, default=0.8,
                        help='KL balance between prior and posterior (default: 0.8)')
    parser.add_argument('--entropy_scale', type=float, default=3e-4,
                        help='Entropy regularization scale (default: 3e-4)')

    return parser.parse_args()


def get_config_dict(args):
    """Convert args to config dictionary for agent initialization."""
    return {
        # World Model
        'stoch_size': args.stoch_size,
        'deter_size': args.deter_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,

        # Imagination
        'imagination_horizon': args.imagination_horizon,

        # Training
        'batch_size': args.batch_size,
        'batch_length': args.batch_length,
        'lr_world': args.lr_world,
        'lr_actor': args.lr_actor,
        'lr_critic': args.lr_critic,
        'gamma': args.gamma,
        'lambda_gae': args.lambda_gae,
        'grad_clip': args.grad_clip,

        # Buffer
        'buffer_size': args.buffer_size,
        'min_buffer_size': args.min_buffer_size,

        # Exploration
        'expl_noise': args.expl_noise,
        'expl_decay': args.expl_decay,
        'expl_min': args.expl_min,

        # DreamerV3 Specific
        'free_nats': args.free_nats,
        'kl_balance': args.kl_balance,
        'entropy_scale': args.entropy_scale,

        # Hardware
        'fp16': args.fp16,
    }
