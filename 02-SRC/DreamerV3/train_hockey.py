"""
DreamerV3 Training Script for Hockey Environment.

DreamerV3 with PBRS (Potential-Based Reward Shaping):
- PBRS provides dense reward signal during data collection
- Mathematically proven to preserve optimal policy (Ng et al., 1999)
- World model learns from shaped rewards, enabling meaningful imagination
- Self-play with PFSP for generalization

Based on:
- DreamerV3 paper (Hafner et al., 2023)
- Robot Air Hockey Challenge 2023 (Orsula et al., 2024)
- PBRS theory (Ng et al., 1999)

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

# Local imports
from config.parser import parse_args, get_config_dict
from envs.hockey_wrapper import HockeyEnvDreamer
from agents.hockey_dreamer import HockeyDreamer
from utils.buffer import EpisodeBuffer
from opponents.self_play import SelfPlayManager
from visualization.gif_recorder import create_gif_dreamer, save_gif_dreamer
from rewards.pbrs import PBRSRewardShaper


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device for training."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_arg)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode(env, agent, buffer, reward_shaper=None, explore: bool = True):
    """
    Run one episode and collect experience.

    Args:
        env: Hockey environment
        agent: DreamerV3 agent
        buffer: Episode buffer
        reward_shaper: Optional PBRS reward shaper
        explore: Whether to use exploration

    Returns:
        episode_reward: Total sparse reward (before shaping)
        episode_shaped_reward: Total shaped reward (stored in buffer)
        episode_length: Number of steps
        info: Final step info
        pbrs_stats: PBRS statistics if shaper provided, else None
    """
    obs, _ = env.reset()
    agent.reset()
    if reward_shaper is not None:
        reward_shaper.reset()

    episode_reward = 0.0  # Sparse reward only
    episode_shaped_reward = 0.0  # Reward stored in buffer
    episode_length = 0
    done = False
    is_first = True

    while not done:
        # Get action
        action = agent.act(obs, deterministic=not explore)

        # Step environment
        next_obs, sparse_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Compute shaped reward for buffer
        if reward_shaper is not None:
            shaping = reward_shaper.shape(obs, next_obs, done)
            shaped_reward = sparse_reward + shaping
        else:
            shaped_reward = sparse_reward

        # Store transition with shaped reward
        buffer.add(obs, action, shaped_reward, done, is_first)
        is_first = False

        # Update
        obs = next_obs
        episode_reward += sparse_reward
        episode_shaped_reward += shaped_reward
        episode_length += 1

    # Get PBRS stats if shaper is active
    pbrs_stats = reward_shaper.get_episode_stats() if reward_shaper is not None else None

    return episode_reward, episode_shaped_reward, episode_length, info, pbrs_stats


def evaluate(env, agent, num_episodes: int = 50, prefix: str = 'eval') -> dict:
    """
    Evaluate agent without exploration.

    Args:
        env: Evaluation environment
        agent: Agent to evaluate
        num_episodes: Number of evaluation episodes
        prefix: Prefix for metric names

    Returns:
        Dictionary with evaluation metrics
    """
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    total_length = 0
    goals_scored = 0
    goals_conceded = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        agent.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        total_reward += episode_reward
        total_length += episode_length

        winner = info.get('winner', 0)
        if winner == 1:
            wins += 1
            goals_scored += 1
        elif winner == -1:
            losses += 1
            goals_conceded += 1
        else:
            draws += 1

    return {
        f'{prefix}/win_rate': wins / num_episodes,
        f'{prefix}/loss_rate': losses / num_episodes,
        f'{prefix}/draw_rate': draws / num_episodes,
        f'{prefix}/mean_reward': total_reward / num_episodes,
        f'{prefix}/mean_length': total_length / num_episodes,
        f'{prefix}/goals_scored': goals_scored,
        f'{prefix}/goals_conceded': goals_conceded,
    }


def main():
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("DreamerV3 Hockey Training (Simplified)")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Opponent: {args.opponent}")
    print(f"Max steps: {args.max_steps:,}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nUsing device: {device}")

    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create environment
    env = HockeyEnvDreamer(
        mode=args.mode,
        opponent=args.opponent if args.opponent != 'self' else 'weak',
        include_fault_penalty=args.include_fault_penalty,
        fault_penalty=args.fault_penalty,
        seed=args.seed,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Create agent
    config = get_config_dict(args)
    agent = HockeyDreamer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=config['hidden_size'],
        num_categories=config.get('num_categories', 32),
        num_classes=config.get('num_classes', 32),
        recurrent_size=config.get('deter_size', 256),
        embed_dim=config.get('hidden_size', 256),
        horizon=config['imagination_horizon'],
        imagine_batch_size=config.get('imagine_batch_size', 256),
        gamma=config['gamma'],
        lambda_gae=config['lambda_gae'],
        kl_free=config.get('free_nats', 1.0),
        unimix=config.get('unimix', 0.01),
        entropy_scale=config.get('entropy_scale', 3e-4),
        lr_world=config['lr_world'],
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic'],
        grad_clip=config['grad_clip'],
        use_agc=config.get('use_agc', True),
        agc_clip=config.get('agc_clip', 0.3),
        terminal_reward_weight=config.get('terminal_reward_weight', 100.0),
        device=str(device),
    )
    latent_size = config.get('num_categories', 32) * config.get('num_classes', 32)
    print(f"\nCreated HockeyDreamer agent")
    print(f"  Model size: {config.get('model_size', 'custom')} (deter={config['deter_size']}, hidden={config['hidden_size']})")
    print(f"  Categorical latent: {config.get('num_categories', 32)} categories × {config.get('num_classes', 32)} classes = {latent_size}")
    print(f"  Gradient clipping: {'AGC(' + str(config.get('agc_clip', 0.3)) + ')' if config.get('use_agc', True) else 'norm(' + str(config['grad_clip']) + ')'}")
    print(f"  Imagination: horizon={config['imagination_horizon']}, batch={config.get('imagine_batch_size', 256)}")

    # Create PBRS reward shaper if enabled
    reward_shaper = None
    if config.get('use_pbrs', True):
        reward_shaper = PBRSRewardShaper(
            gamma=config['gamma'],
            scale=config.get('pbrs_scale', 0.03),
            w_chase=config.get('pbrs_w_chase', 0.5),
            w_attack=config.get('pbrs_w_attack', 1.0),
            clip=config.get('pbrs_clip', None),
        )
        print(f"\nPBRS enabled:")
        print(f"  scale={config.get('pbrs_scale', 0.03)}, w_chase={config.get('pbrs_w_chase', 0.5)}, w_attack={config.get('pbrs_w_attack', 1.0)}")
    else:
        print(f"\nPBRS disabled - using sparse rewards only")

    # Create replay buffer
    buffer = EpisodeBuffer(
        capacity=args.buffer_size,
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
    )

    # Self-play setup
    self_play_manager = None
    if args.self_play_start > 0:
        self_play_manager = SelfPlayManager(
            pool_size=args.self_play_pool_size,
            save_interval=args.self_play_save_interval,
            use_pfsp=args.use_pfsp,
            pfsp_mode=args.pfsp_mode,
        )
        print(f"Self-play enabled at step {args.self_play_start:,}")

    # W&B setup
    if not args.no_wandb:
        try:
            import wandb
            run_name = args.run_name or f"DreamerV3-{args.mode}-{args.opponent}-seed{args.seed}"

            # Comprehensive config for W&B
            wandb_config = {
                # Environment
                'env/mode': args.mode,
                'env/opponent': args.opponent,
                'env/obs_dim': obs_dim,
                'env/action_dim': action_dim,

                # Architecture
                'arch/hidden_size': config['hidden_size'],
                'arch/latent_size': config.get('latent_size', 64),
                'arch/recurrent_size': config.get('deter_size', 256),

                # Imagination
                'imagination/horizon': config['imagination_horizon'],
                'imagination/batch_size': config.get('imagine_batch_size', 256),

                # Training
                'train/batch_size': args.batch_size,
                'train/batch_length': args.batch_length,
                'train/lr_world': config['lr_world'],
                'train/lr_actor': config['lr_actor'],
                'train/lr_critic': config['lr_critic'],
                'train/gamma': config['gamma'],
                'train/lambda_gae': config['lambda_gae'],
                'train/grad_clip': config['grad_clip'],

                # DreamerV3 specific
                'dreamer/kl_free': config.get('free_nats', 1.0),
                'dreamer/entropy_scale': config.get('entropy_scale', 3e-4),
                'dreamer/terminal_reward_weight': config.get('terminal_reward_weight', 200.0),

                # PBRS
                'pbrs/enabled': config.get('use_pbrs', True),
                'pbrs/scale': config.get('pbrs_scale', 0.03),
                'pbrs/w_chase': config.get('pbrs_w_chase', 0.5),
                'pbrs/w_attack': config.get('pbrs_w_attack', 1.0),
                'pbrs/clip': config.get('pbrs_clip', None),

                # Buffer
                'buffer/capacity': args.buffer_size,
                'buffer/min_size': args.min_buffer_size,

                # Logging
                'logging/eval_frequency': args.eval_frequency,
                'logging/eval_episodes': args.eval_episodes,
                'logging/gif_frequency': args.gif_frequency,
                'logging/gif_episodes': args.gif_episodes,
                'logging/save_frequency': args.save_frequency,

                # Meta
                'seed': args.seed,
                'max_steps': args.max_steps,
                'device': str(device),
            }

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=wandb_config,
                tags=['dreamerv3', args.mode, args.opponent],
            )
            print(f"W&B run: {run_name}")
        except ImportError:
            print("W&B not installed, logging disabled")
            args.no_wandb = True

    # Training loop
    total_steps = 0
    episode_count = 0
    best_eval_winrate = 0.0
    wins, losses, draws = 0, 0, 0
    train_metrics = {}
    last_log_time = time.time()

    print("\nStarting training...")
    print("Legend: W=Win, L=Loss, D=Draw | Training starts after buffer fills\n")
    start_time = time.time()

    while total_steps < args.max_steps:
        episode_start = time.time()

        # Run episode
        episode_reward, episode_shaped_reward, episode_length, info, pbrs_stats = run_episode(
            env, agent, buffer, reward_shaper=reward_shaper, explore=True
        )
        episode_time = time.time() - episode_start

        total_steps += episode_length
        episode_count += 1

        # Track outcomes
        winner = info.get('winner', 0)
        if winner == 1:
            wins += 1
            outcome = 'W'
        elif winner == -1:
            losses += 1
            outcome = 'L'
        else:
            draws += 1
            outcome = 'D'

        # Train agent
        train_time = 0
        training_active = len(buffer) >= args.min_buffer_size
        if training_active:
            train_start = time.time()
            batch = buffer.sample(args.batch_size, args.batch_length)
            if batch is not None:
                train_metrics = agent.train_step(batch)
            train_time = time.time() - train_start

        # Brief per-episode logging (console)
        elapsed = time.time() - start_time
        win_rate = wins / episode_count if episode_count > 0 else 0

        if training_active:
            pbrs_str = f" | pbrs: {pbrs_stats['pbrs/episode_total']:+.2f}" if pbrs_stats else ""
            print(f"Ep {episode_count:4d} [{outcome}] | "
                  f"steps: {episode_length:3d} | "
                  f"r: {episode_reward:+.1f} | "
                  f"shaped: {episode_shaped_reward:+.2f}{pbrs_str} | "
                  f"world_loss: {train_metrics.get('world/loss', 0):.3f}")
        else:
            # Before training starts, show buffer fill progress
            fill_pct = 100 * len(buffer) / args.min_buffer_size
            print(f"Ep {episode_count:4d} [{outcome}] | "
                  f"steps: {episode_length:3d} | "
                  f"r: {episode_reward:+.1f} | "
                  f"buffer: {len(buffer):,}/{args.min_buffer_size:,} ({fill_pct:.0f}%)")

        # W&B logging every episode (for smooth graphs)
        if not args.no_wandb and training_active:
            import wandb

            # Build log dict
            log_dict = {
                # Episode metrics
                'episode/reward': episode_reward,
                'episode/shaped_reward': episode_shaped_reward,
                'episode/length': episode_length,
                'episode/outcome': {'W': 1, 'L': -1, 'D': 0}[outcome],
                'episode/time': episode_time,
                'episode/train_time': train_time,

                # Running stats
                'stats/win_rate': win_rate,
                'stats/total_steps': total_steps,
                'stats/buffer_size': len(buffer),
                'stats/buffer_episodes': buffer.num_episodes,

                # All training metrics (world model + behavior)
                **train_metrics,

                # Step for x-axis
                'step': total_steps,
                'episode': episode_count,
            }

            # Add PBRS stats if available
            if pbrs_stats is not None:
                log_dict.update(pbrs_stats)

            wandb.log(log_dict)

        # Self-play opponent switching
        if self_play_manager is not None and total_steps >= args.self_play_start:
            # Save to pool periodically
            if total_steps % args.self_play_save_interval == 0:
                self_play_manager.update_pool(
                    episode_count, agent, checkpoint_dir
                )

        # Detailed summary every 10 episodes (console only)
        if episode_count % 10 == 0:
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0

            print(f"\n{'─'*60}")
            print(f"Summary at Episode {episode_count} | Step {total_steps:,}")
            print(f"  Win Rate: {win_rate:.1%} (W:{wins} L:{losses} D:{draws})")
            print(f"  Buffer: {len(buffer):,} transitions | {buffer.num_episodes} episodes")
            print(f"  Speed: {steps_per_sec:.1f} steps/sec | Elapsed: {elapsed/60:.1f} min")
            if train_metrics:
                print(f"  Losses: world={train_metrics.get('world/loss', 0):.3f} "
                      f"actor={train_metrics.get('behavior/actor_loss', 0):.3f} "
                      f"critic={train_metrics.get('behavior/critic_loss', 0):.3f}")
            print(f"{'─'*60}\n")

        # Evaluation
        if episode_count % args.eval_frequency == 0 and episode_count > 0:
            print(f"\n{'='*60}")
            print(f"EVALUATION at episode {episode_count:,} (step {total_steps:,})")
            print(f"{'='*60}")

            # Evaluate vs weak opponent
            eval_env_weak = HockeyEnvDreamer(
                mode=args.mode,
                opponent='weak',
                seed=args.seed + 1000
            )
            eval_weak = evaluate(eval_env_weak, agent, args.eval_episodes, prefix='eval_weak')
            eval_env_weak.close()
            print(f"  vs Weak:   {eval_weak['eval_weak/win_rate']:.1%} win | "
                  f"{eval_weak['eval_weak/draw_rate']:.1%} draw | "
                  f"{eval_weak['eval_weak/loss_rate']:.1%} loss")

            # Evaluate vs strong opponent
            eval_env_strong = HockeyEnvDreamer(
                mode=args.mode,
                opponent='strong',
                seed=args.seed + 2000
            )
            eval_strong = evaluate(eval_env_strong, agent, args.eval_episodes, prefix='eval_strong')
            eval_env_strong.close()
            print(f"  vs Strong: {eval_strong['eval_strong/win_rate']:.1%} win | "
                  f"{eval_strong['eval_strong/draw_rate']:.1%} draw | "
                  f"{eval_strong['eval_strong/loss_rate']:.1%} loss")

            print(f"{'='*60}\n")

            # W&B logging for evaluation
            if not args.no_wandb:
                import wandb
                wandb.log({
                    **eval_weak,
                    **eval_strong,
                    'eval/step': total_steps,
                })

            # Save best model (based on weak opponent performance)
            if eval_weak['eval_weak/win_rate'] > best_eval_winrate:
                best_eval_winrate = eval_weak['eval_weak/win_rate']
                agent.save(checkpoint_dir / 'best_model.pth')
                print(f"New best model saved! Win rate vs weak: {best_eval_winrate:.1%}\n")

        # GIF recording (independent of main evaluation - runs more frequently)
        if not args.no_wandb and episode_count % args.gif_frequency == 0 and episode_count > 0:
            print(f"\n{'─'*60}")
            print(f"GIF RECORDING at episode {episode_count:,}")
            print(f"{'─'*60}")

            # Record GIF vs weak opponent
            gif_env_weak = HockeyEnvDreamer(
                mode=args.mode,
                opponent='weak',
                seed=args.seed + 3000
            )
            frames_weak, results_weak = create_gif_dreamer(
                gif_env_weak, agent,
                num_episodes=args.gif_episodes,
                max_timesteps=250
            )
            gif_env_weak.close()
            save_gif_dreamer(frames_weak, results_weak, episode_count, 'weak')

            # Record GIF vs strong opponent
            gif_env_strong = HockeyEnvDreamer(
                mode=args.mode,
                opponent='strong',
                seed=args.seed + 4000
            )
            frames_strong, results_strong = create_gif_dreamer(
                gif_env_strong, agent,
                num_episodes=args.gif_episodes,
                max_timesteps=250
            )
            gif_env_strong.close()
            save_gif_dreamer(frames_strong, results_strong, episode_count, 'strong')
            print(f"{'─'*60}\n")

        # Periodic checkpoint
        if episode_count % args.save_frequency == 0:
            agent.save(checkpoint_dir / f'checkpoint_ep{episode_count}.pth')

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total steps: {total_steps:,}")
    print(f"Total episodes: {episode_count}")
    print(f"Final win rate: {wins/episode_count:.1%}")
    print(f"Best eval win rate: {best_eval_winrate:.1%}")
    print(f"Time elapsed: {(time.time() - start_time)/3600:.2f} hours")

    # Cleanup
    env.close()
    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
