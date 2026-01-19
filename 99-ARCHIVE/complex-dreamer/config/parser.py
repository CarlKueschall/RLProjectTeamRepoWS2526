"""
Command-line argument parser for DreamerV3 hockey training.

Based on hyperparameters from:
- DreamerV3 paper (Hafner et al., 2023)
- Robot Air Hockey Challenge 2023 (Orsula et al., 2024)

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import argparse


# Model size presets (approximate parameter counts)
# Paper sizes: XS=7M, S=18M, M=50M, L=200M, XL=1B
MODEL_SIZES = {
    'xs': {  # ~2M params - for quick testing
        'deter_size': 256,
        'hidden_size': 256,
        'num_categories': 16,
        'num_classes': 16,
        'batch_size': 8,
        'batch_length': 32,
    },
    'small': {  # ~8M params - for consumer GPUs (RTX 2080, 3060, etc.)
        'deter_size': 512,
        'hidden_size': 512,
        'num_categories': 32,
        'num_classes': 32,
        'batch_size': 16,
        'batch_length': 50,
    },
    'medium': {  # ~20M params - for RTX 3080/3090/4080
        'deter_size': 1024,
        'hidden_size': 1024,
        'num_categories': 32,
        'num_classes': 32,
        'batch_size': 16,
        'batch_length': 64,
    },
    'large': {  # ~50M params - paper's default, needs A100/H100
        'deter_size': 2048,
        'hidden_size': 2048,
        'num_categories': 32,
        'num_classes': 32,
        'batch_size': 16,
        'batch_length': 64,
    },
    'xlarge': {  # ~100M+ params - for multi-GPU / H100
        'deter_size': 4096,
        'hidden_size': 4096,
        'num_categories': 32,
        'num_classes': 32,
        'batch_size': 16,
        'batch_length': 64,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DreamerV3 agent on Hockey environment'
    )

    # ===================
    # Model Size Preset
    # ===================
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['xs', 'small', 'medium', 'large', 'xlarge'],
                        help='Model size preset. Overrides individual size params. '
                             'xs=~2M (testing), small=~8M (consumer GPU), '
                             'medium=~20M (3090), large=~50M (A100), xlarge=~100M+ (H100)')

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
    # World Model (RSSM with Categorical Latents)
    # ===================
    parser.add_argument('--num_categories', type=int, default=32,
                        help='Number of categorical variables for latent (default: 32)')
    parser.add_argument('--num_classes', type=int, default=32,
                        help='Classes per categorical variable (default: 32)')
    parser.add_argument('--deter_size', type=int, default=512,
                        help='Deterministic state size (default: 512, reduce to 256 for 2080ti)')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden layer size (default: 512, reduce to 256 for 2080ti)')
    parser.add_argument('--unimix', type=float, default=0.01,
                        help='Uniform mixing ratio for categorical latents (default: 0.01 = 1%%)')

    # ===================
    # Imagination
    # ===================
    parser.add_argument('--imagination_horizon', type=int, default=15,
                        help='Imagination horizon for actor-critic training (default: 15)')
    parser.add_argument('--imagine_batch_size', type=int, default=256,
                        help='Number of starting states for imagination (default: 256). '
                             'Subsampled from batch_size * batch_length positions.')

    # ===================
    # Training Hyperparameters
    # ===================
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16, reduce to 8 for 2080ti)')
    parser.add_argument('--batch_length', type=int, default=64,
                        help='Sequence length for training (default: 64, reduce to 50 for 2080ti)')
    parser.add_argument('--lr_world', type=float, default=4e-5,
                        help='World model learning rate (default: 4e-5, from DreamerV3 paper)')
    parser.add_argument('--lr_actor', type=float, default=4e-5,
                        help='Actor learning rate (default: 4e-5, from DreamerV3 paper)')
    parser.add_argument('--lr_critic', type=float, default=4e-5,
                        help='Critic learning rate (default: 4e-5, from DreamerV3 paper)')
    parser.add_argument('--gamma', type=float, default=0.997,
                        help='Discount factor (default: 0.997, higher than TD3 for longer horizons)')
    parser.add_argument('--lambda_gae', type=float, default=0.95,
                        help='GAE lambda for advantage estimation (default: 0.95)')
    parser.add_argument('--grad_clip', type=float, default=10.0,
                        help='Gradient clipping norm for standard clipping (default: 10.0, only used if --no_agc)')
    parser.add_argument('--use_agc', action='store_true', default=True,
                        help='Use Adaptive Gradient Clipping (default: True, from DreamerV3 paper)')
    parser.add_argument('--no_agc', dest='use_agc', action='store_false',
                        help='Use standard gradient norm clipping instead of AGC')
    parser.add_argument('--agc_clip', type=float, default=0.3,
                        help='AGC clip factor (default: 0.3, from DreamerV3 paper)')

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
    parser.add_argument('--eval_frequency', type=int, default=100,
                        help='Training episodes between evaluations (default: 100)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help='Number of evaluation episodes per eval run (default: 50)')
    parser.add_argument('--save_frequency', type=int, default=500,
                        help='Training episodes between checkpoint saves (default: 500)')
    parser.add_argument('--gif_frequency', type=int, default=200,
                        help='Training episodes between GIF recordings (default: 200)')
    parser.add_argument('--gif_episodes', type=int, default=3,
                        help='Number of episodes to record per GIF (default: 3)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='rl-hockey',
                        help='W&B project name (default: rl-hockey)')
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
    # PBRS (Potential-Based Reward Shaping)
    # ===================
    parser.add_argument('--use_pbrs', action='store_true', default=True,
                        help='Use PBRS for dense reward signal (default: True)')
    parser.add_argument('--no_pbrs', dest='use_pbrs', action='store_false',
                        help='Disable PBRS reward shaping')
    parser.add_argument('--pbrs_scale', type=float, default=0.03,
                        help='Global scaling for PBRS rewards (default: 0.03). '
                             'Keeps shaping small relative to sparse ±1 terminal rewards.')
    parser.add_argument('--pbrs_w_chase', type=float, default=0.5,
                        help='Weight for chase component (agent->puck distance, default: 0.5)')
    parser.add_argument('--pbrs_w_attack', type=float, default=1.0,
                        help='Weight for attack component (puck->goal distance, default: 1.0). '
                             'Must be > w_chase for forward shooting to be net positive.')
    parser.add_argument('--pbrs_clip', type=float, default=None,
                        help='Optional per-step clipping for PBRS rewards (default: None)')

    # ===================
    # DreamerV3 Specific
    # ===================
    parser.add_argument('--free_nats', type=float, default=1.0,
                        help='Free nats for KL loss (default: 1.0)')
    parser.add_argument('--kl_balance', type=float, default=0.8,
                        help='KL balance between prior and posterior (default: 0.8)')
    parser.add_argument('--entropy_scale', type=float, default=3e-3,
                        help='Entropy regularization scale (default: 3e-3). '
                             'Higher than DreamerV3 paper (3e-4) to prevent entropy collapse.')
    parser.add_argument('--terminal_reward_weight', type=float, default=200.0,
                        help='Weight multiplier for non-zero (terminal) rewards in loss (default: 200). '
                             'Handles class imbalance. Lower than 1000 since PBRS provides dense signal.')

    return parser.parse_args()


def apply_model_size_preset(args):
    """Apply model size preset if specified, overriding individual params."""
    if args.model_size is not None:
        preset = MODEL_SIZES[args.model_size]
        # Only override if user didn't explicitly set the value
        # (we can't detect this perfectly, so preset always wins)
        args.deter_size = preset['deter_size']
        args.hidden_size = preset['hidden_size']
        args.num_categories = preset['num_categories']
        args.num_classes = preset['num_classes']
        args.batch_size = preset['batch_size']
        args.batch_length = preset['batch_length']
    return args


def get_config_dict(args):
    """Convert args to config dictionary for agent initialization."""
    # Apply model size preset first
    args = apply_model_size_preset(args)

    return {
        # World Model (Categorical Latents)
        'num_categories': args.num_categories,
        'num_classes': args.num_classes,
        'deter_size': args.deter_size,
        'recurrent_size': args.deter_size,  # Alias for our implementation
        'hidden_size': args.hidden_size,
        'unimix': args.unimix,

        # Imagination
        'imagination_horizon': args.imagination_horizon,
        'imagine_batch_size': args.imagine_batch_size,

        # Training
        'batch_size': args.batch_size,
        'batch_length': args.batch_length,
        'lr_world': args.lr_world,
        'lr_actor': args.lr_actor,
        'lr_critic': args.lr_critic,
        'gamma': args.gamma,
        'lambda_gae': args.lambda_gae,
        'grad_clip': args.grad_clip,

        # Gradient Clipping
        'use_agc': args.use_agc,
        'agc_clip': args.agc_clip,

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
        'terminal_reward_weight': args.terminal_reward_weight,

        # Hardware
        'fp16': args.fp16,

        # PBRS
        'use_pbrs': args.use_pbrs,
        'pbrs_scale': args.pbrs_scale,
        'pbrs_w_chase': args.pbrs_w_chase,
        'pbrs_w_attack': args.pbrs_w_attack,
        'pbrs_clip': args.pbrs_clip,

        # Model size (for logging)
        'model_size': args.model_size,
    }
