#!/usr/bin/env python3
"""
Async Training Script for Hockey TD3 Agent.

This script uses fully asynchronous training where:
- Multiple collector processes continuously run episodes
- A trainer thread continuously updates the network
- A shared buffer receives transitions from all collectors
- The main thread orchestrates logging, evaluation, and checkpointing

This architecture provides 2-3x speedup over sequential training by
keeping the GPU continuously busy while collectors run in parallel.

Usage:
    python train_hockey_async.py --mode NORMAL --opponent weak --num_workers 4

Key differences from train_hockey.py:
- Uses multiprocessing for true parallelism (no GIL)
- Collectors run continuously without blocking training
- Training happens in a background thread
- Weight syncs are periodic, not per-episode
"""

import argparse
import numpy as np
import torch
from pathlib import Path

import hockey.hockey_env as h_env
from hockey.hockey_env import Mode


def parse_async_args():
    """Parse command-line arguments for async training."""
    parser = argparse.ArgumentParser(description='Async TD3 Hockey Training')

    # Environment settings
    parser.add_argument('--mode', type=str, default='NORMAL',
                        choices=['NORMAL', 'TRAIN_SHOOTING', 'TRAIN_DEFENSE'],
                        help='Hockey environment mode')
    parser.add_argument('--opponent', type=str, default='weak',
                        choices=['weak', 'strong'],
                        help='Opponent type for training')
    parser.add_argument('--keep_mode', type=bool, default=True,
                        help='Enable keep mode (puck possession)')

    # Async-specific settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel collector workers')
    parser.add_argument('--weight_sync_interval', type=int, default=50,
                        help='Sync weights to collectors every N training steps')
    parser.add_argument('--warmup_episodes', type=int, default=100,
                        help='Episodes to collect before training starts')

    # Training settings
    parser.add_argument('--max_episodes', type=int, default=100000,
                        help='Maximum number of episodes')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--buffer_size', type=int, default=1000000,
                        help='Replay buffer capacity')

    # TD3 hyperparameters
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                        help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient')
    parser.add_argument('--policy_freq', type=int, default=2,
                        help='Policy update frequency')
    parser.add_argument('--target_update_freq', type=int, default=2,
                        help='Target network update frequency')
    parser.add_argument('--target_noise_std', type=float, default=0.2,
                        help='Target policy smoothing noise std')
    parser.add_argument('--target_noise_clip', type=float, default=0.5,
                        help='Target policy noise clip')

    # Network architecture
    parser.add_argument('--hidden_actor', type=int, nargs='+', default=[400, 300],
                        help='Actor hidden layer sizes')
    parser.add_argument('--hidden_critic', type=int, nargs='+', default=[400, 300, 128],
                        help='Critic hidden layer sizes')

    # Exploration
    parser.add_argument('--eps', type=float, default=1.0,
                        help='Initial exploration noise')
    parser.add_argument('--eps_min', type=float, default=0.05,
                        help='Minimum exploration noise')
    parser.add_argument('--eps_decay', type=float, default=0.9999,
                        help='Exploration noise decay rate')

    # Q-value handling
    parser.add_argument('--q_clip', type=float, default=25.0,
                        help='Q-value clipping threshold')
    parser.add_argument('--q_clip_mode', type=str, default='soft',
                        choices=['soft', 'hard'],
                        help='Q-value clipping mode')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')

    # Reward shaping
    parser.add_argument('--reward_scale', type=float, default=1.0,
                        help='Sparse reward scaling factor')
    parser.add_argument('--reward_shaping', action='store_true',
                        help='Enable PBRS reward shaping')
    parser.add_argument('--pbrs_scale', type=float, default=0.5,
                        help='PBRS reward scale')
    parser.add_argument('--pbrs_constant_weight', action='store_true',
                        help='Keep PBRS weight constant (no annealing)')

    # Prioritized Experience Replay
    parser.add_argument('--use_per', action='store_true',
                        help='Enable Prioritized Experience Replay')
    parser.add_argument('--per_alpha', type=float, default=0.6,
                        help='PER priority exponent')
    parser.add_argument('--per_beta_start', type=float, default=0.4,
                        help='PER initial IS correction')
    parser.add_argument('--per_beta_frames', type=int, default=100000,
                        help='PER beta annealing frames')

    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Evaluation interval (episodes)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help='Episodes per evaluation')

    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval (episodes)')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Checkpoint save interval (episodes)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU (disable GPU/MPS)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def get_mode(mode_str: str) -> int:
    """Convert mode string to Mode enum value."""
    mode_map = {
        'NORMAL': Mode.NORMAL,
        'TRAIN_SHOOTING': Mode.TRAIN_SHOOTING,
        'TRAIN_DEFENSE': Mode.TRAIN_DEFENSE,
    }
    return mode_map.get(mode_str, Mode.NORMAL)


def get_max_timesteps(mode: int) -> int:
    """Get max timesteps based on mode."""
    if mode == Mode.NORMAL:
        return 250
    else:
        return 150  # Training modes are shorter


def main():
    args = parse_async_args()

    # Set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Get environment info
    mode = get_mode(args.mode)
    max_timesteps = get_max_timesteps(mode)

    # Create a temporary environment to get observation dimensions
    temp_env = h_env.HockeyEnv(mode=mode, keep_mode=args.keep_mode)
    test_obs, _ = temp_env.reset()
    obs_dim = len(test_obs)
    action_dim = 4  # Agent outputs 4D actions
    temp_env.close()

    print("=" * 60)
    print("ASYNC TD3 HOCKEY TRAINING")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Opponent: {args.opponent}")
    print(f"Workers: {args.num_workers}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Buffer size: {args.buffer_size:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"PER: {'Enabled' if args.use_per else 'Disabled'}")
    print(f"PBRS: {'Enabled' if args.reward_shaping else 'Disabled'}")
    print(f"Weight sync interval: {args.weight_sync_interval}")
    print("=" * 60)

    # Create environment config for workers
    env_config = {
        'mode': mode,
        'keep_mode': args.keep_mode,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'hidden_actor': args.hidden_actor,
        'max_timesteps': max_timesteps,
        'reward_scale': args.reward_scale,
        'seed': args.seed,
        'default_opponent': args.opponent,
        'pbrs_enabled': args.reward_shaping,
        'pbrs_scale': args.pbrs_scale,
        'pbrs_constant_weight': args.pbrs_constant_weight,
        'gamma': args.gamma,
    }

    # Create and run orchestrator
    from async_orchestrator import AsyncTrainingOrchestrator

    orchestrator = AsyncTrainingOrchestrator(args, env_config)

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'agent_state' in checkpoint:
                orchestrator.trainer.restore_agent_state(checkpoint['agent_state'])
                print(f"Restored from episode {checkpoint.get('episode', 0)}")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")

    # Run training
    orchestrator.run()


if __name__ == '__main__':
    main()
