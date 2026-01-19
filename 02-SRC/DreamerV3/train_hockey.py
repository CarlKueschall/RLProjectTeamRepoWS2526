"""
DreamerV3 Training Script for Hockey Environment.

Phase 1: Uses external DreamerV3 PyTorch implementation for initial testing.
Phase 2: Will integrate custom implementation.

Based on:
- DreamerV3 paper (Hafner et al., 2023)
- Robot Air Hockey Challenge 2023 (Orsula et al., 2024)

Key design choices:
- SPARSE REWARDS ONLY - no reward shaping needed
- Imagination-based credit assignment (horizon=50)
- Self-play with PFSP for generalization

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Local imports
from config.parser import parse_args, get_config_dict
from envs.hockey_wrapper import HockeyEnvDreamer, HockeyVecEnv
from opponents.self_play import SelfPlayManager
from opponents.fixed import FixedOpponent

# Optional imports - gracefully handle if not available
try:
    from metrics.metrics_tracker import MetricsTracker
except ImportError:
    MetricsTracker = None

try:
    from visualization.gif_recorder import GIFRecorder
except ImportError:
    GIFRecorder = None

# Import our custom DreamerV3 implementation
from agents.dreamer_agent import DreamerV3Agent
DREAMER_BACKEND = "custom"


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


class SimpleReplayBuffer:
    """
    Simple sequence-based replay buffer for DreamerV3.

    Stores complete episodes and samples sequences for training.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Storage
        self.episodes = []
        self.current_episode = {
            'obs': [],
            'action': [],
            'reward': [],
            'done': [],
        }
        self.total_steps = 0

    def add(self, obs, action, reward, done):
        """Add a transition to the current episode."""
        self.current_episode['obs'].append(obs)
        self.current_episode['action'].append(action)
        self.current_episode['reward'].append(reward)
        self.current_episode['done'].append(done)
        self.total_steps += 1

        if done:
            self._finalize_episode()

    def _finalize_episode(self):
        """Store completed episode."""
        if len(self.current_episode['obs']) > 0:
            episode = {
                'obs': np.array(self.current_episode['obs'], dtype=np.float32),
                'action': np.array(self.current_episode['action'], dtype=np.float32),
                'reward': np.array(self.current_episode['reward'], dtype=np.float32),
                'done': np.array(self.current_episode['done'], dtype=bool),
            }
            self.episodes.append(episode)

            # FIFO eviction based on total transitions
            while self._total_transitions() > self.capacity and len(self.episodes) > 1:
                self.episodes.pop(0)

        self.current_episode = {'obs': [], 'action': [], 'reward': [], 'done': []}

    def _total_transitions(self):
        return sum(len(ep['obs']) for ep in self.episodes)

    def sample_sequences(self, batch_size: int, sequence_length: int):
        """Sample batch of sequences for training."""
        if len(self.episodes) == 0:
            return None

        batch = {
            'obs': [],
            'action': [],
            'reward': [],
            'done': [],
        }

        for _ in range(batch_size):
            # Sample random episode
            ep_idx = np.random.randint(len(self.episodes))
            ep = self.episodes[ep_idx]

            # Sample random start position
            max_start = max(0, len(ep['obs']) - sequence_length)
            start = np.random.randint(max_start + 1) if max_start > 0 else 0
            end = min(start + sequence_length, len(ep['obs']))

            # Extract sequence (may be shorter than sequence_length)
            seq_len = end - start

            # Pad if necessary
            obs_seq = np.zeros((sequence_length, self.obs_dim), dtype=np.float32)
            action_seq = np.zeros((sequence_length, self.action_dim), dtype=np.float32)
            reward_seq = np.zeros(sequence_length, dtype=np.float32)
            done_seq = np.ones(sequence_length, dtype=bool)  # Pad with done=True

            obs_seq[:seq_len] = ep['obs'][start:end]
            action_seq[:seq_len] = ep['action'][start:end]
            reward_seq[:seq_len] = ep['reward'][start:end]
            done_seq[:seq_len] = ep['done'][start:end]

            batch['obs'].append(obs_seq)
            batch['action'].append(action_seq)
            batch['reward'].append(reward_seq)
            batch['done'].append(done_seq)

        return {
            'obs': np.stack(batch['obs']),
            'action': np.stack(batch['action']),
            'reward': np.stack(batch['reward']),
            'done': np.stack(batch['done']),
        }

    def __len__(self):
        return self._total_transitions()


def train_episode(env, agent, buffer, exploration_noise, device):
    """
    Run one training episode.

    Returns:
        episode_reward: Total sparse reward
        episode_length: Number of steps
        info: Final step info (contains winner)
    """
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    done = False

    # Reset agent state for new episode (important for RSSM)
    if agent is not None and hasattr(agent, 'reset'):
        agent.reset()

    while not done:
        # Get action from agent
        with torch.no_grad():
            if hasattr(agent, 'act'):
                # Our DreamerV3Agent expects numpy array and returns numpy array
                action = agent.act(obs, explore=True)
            else:
                # Fallback: random action during initial implementation
                action = env.action_space.sample()

        # Add exploration noise
        if exploration_noise > 0:
            noise = np.random.normal(0, exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        buffer.add(obs, action, reward, done)

        # Update
        obs = next_obs
        episode_reward += reward
        episode_length += 1

    return episode_reward, episode_length, info


def main():
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("DreamerV3 Hockey Training")
    print("=" * 60)
    print(f"Backend: {DREAMER_BACKEND or 'Custom (Phase 2)'}")
    print(f"Mode: {args.mode}")
    print(f"Opponent: {args.opponent}")
    print(f"Max steps: {args.max_steps:,}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Check for DreamerV3 implementation
    if DREAMER_BACKEND is None:
        print("\nWARNING: No DreamerV3 implementation found!")
        print("Install one of:")
        print("  pip install dreamer-pytorch")
        print("  pip install sheeprl")
        print("\nOr wait for Phase 2 custom implementation.")
        print("\nRunning with RANDOM ACTIONS for testing infrastructure...")

    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nUsing device: {device}")

    # Create save directory
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

    # Create replay buffer
    buffer = SimpleReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # Create agent
    config = get_config_dict(args)

    if DREAMER_BACKEND == "custom":
        # Use our custom DreamerV3 implementation
        agent = DreamerV3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            stoch_size=config.get('stoch_size', 32),
            deter_size=config.get('deter_size', 256),
            hidden_size=config.get('hidden_size', 256),
            imagination_horizon=config.get('imagination_horizon', 50),
            gamma=config.get('gamma', 0.997),
            lr_world=config.get('lr_world', 3e-4),
            lr_actor=config.get('lr_actor', 3e-5),
            lr_critic=config.get('lr_critic', 3e-5),
            device=str(device),
        )
        print(f"\nCreated DreamerV3Agent (custom implementation)")
    elif DREAMER_BACKEND == "dreamer-pytorch":
        agent = DreamerAgent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            config=config,
        ).to(device)
    elif DREAMER_BACKEND == "sheeprl":
        agent = DreamerAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            **config,
        ).to(device)
    else:
        # Placeholder agent for infrastructure testing
        agent = None
        print("\nUsing placeholder agent (random actions)")

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
            run_name = args.run_name or f"DreamerV3-{args.mode}-{args.opponent}-{args.seed}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
            )
            print(f"W&B run: {run_name}")
        except ImportError:
            print("W&B not installed, logging disabled")
            args.no_wandb = True

    # Training loop
    total_steps = 0
    episode_count = 0
    best_eval_winrate = 0.0
    exploration_noise = args.expl_noise

    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    wins, losses, draws = 0, 0, 0

    print("\nStarting training...")
    start_time = time.time()

    while total_steps < args.max_steps:
        # Run episode
        episode_reward, episode_length, info = train_episode(
            env, agent, buffer, exploration_noise, device
        )

        total_steps += episode_length
        episode_count += 1

        # Track outcomes
        winner = info.get('winner', 0)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Decay exploration
        exploration_noise = max(args.expl_min, exploration_noise * args.expl_decay)

        # Train agent
        if agent is not None and len(buffer) >= args.min_buffer_size:
            # Sample batch and train
            batch = buffer.sample_sequences(args.batch_size, args.batch_length)
            if batch is not None:
                if hasattr(agent, 'train_step'):
                    train_metrics = agent.train_step(batch)
                elif hasattr(agent, 'update'):
                    train_metrics = agent.update(batch)
                else:
                    train_metrics = {}

        # Self-play opponent switching
        if self_play_manager is not None and total_steps >= args.self_play_start:
            # Decide opponent for next episode
            if np.random.random() > args.self_play_weak_ratio:
                # Play against pool opponent
                opponent_state = self_play_manager.get_opponent()
                if opponent_state is not None and agent is not None:
                    # Create opponent from saved state
                    pass  # TODO: Implement opponent loading

            # Save current agent to pool periodically
            if total_steps % args.self_play_save_interval == 0 and agent is not None:
                if hasattr(agent, 'state_dict'):
                    self_play_manager.add_to_pool(agent.state_dict(), total_steps)

        # Logging
        if episode_count % 10 == 0:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths

            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0

            win_rate = wins / episode_count if episode_count > 0 else 0

            print(f"Step {total_steps:,} | Ep {episode_count} | "
                  f"Reward: {np.mean(recent_rewards):.2f} | "
                  f"WinRate: {win_rate:.1%} | "
                  f"Eps: {exploration_noise:.3f} | "
                  f"SPS: {steps_per_sec:.0f}")

            if not args.no_wandb:
                import wandb
                wandb.log({
                    'step': total_steps,
                    'episode': episode_count,
                    'reward/mean': np.mean(recent_rewards),
                    'reward/episode': episode_reward,
                    'episode_length': np.mean(recent_lengths),
                    'win_rate': win_rate,
                    'exploration_noise': exploration_noise,
                    'buffer_size': len(buffer),
                    'steps_per_second': steps_per_sec,
                })

        # Evaluation
        if total_steps % args.eval_interval == 0 and total_steps > 0:
            print(f"\n--- Evaluation at step {total_steps:,} ---")

            # Evaluate against weak opponent
            eval_env = HockeyEnvDreamer(mode=args.mode, opponent='weak', seed=args.seed + 1000)
            eval_wins = 0

            for ep in range(args.eval_episodes):
                obs, _ = eval_env.reset()
                done = False
                # Reset agent state for new episode
                if agent is not None and hasattr(agent, 'reset'):
                    agent.reset()
                while not done:
                    if agent is not None and hasattr(agent, 'act'):
                        with torch.no_grad():
                            action = agent.act(obs, explore=False)
                    else:
                        action = eval_env.action_space.sample()

                    obs, _, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated

                if info.get('winner', 0) == 1:
                    eval_wins += 1

            eval_winrate = eval_wins / args.eval_episodes
            print(f"Eval vs Weak: {eval_winrate:.1%} ({eval_wins}/{args.eval_episodes})")

            if not args.no_wandb:
                import wandb
                wandb.log({
                    'eval/win_rate_weak': eval_winrate,
                    'eval/step': total_steps,
                })

            # Save best model
            if eval_winrate > best_eval_winrate and agent is not None:
                best_eval_winrate = eval_winrate
                if hasattr(agent, 'state_dict'):
                    torch.save(
                        agent.state_dict(),
                        checkpoint_dir / 'best_model.pth'
                    )
                    print(f"New best model saved! Win rate: {eval_winrate:.1%}")

            eval_env.close()
            print()

        # Periodic checkpoint
        if total_steps % args.save_interval == 0 and agent is not None:
            if hasattr(agent, 'state_dict'):
                torch.save(
                    agent.state_dict(),
                    checkpoint_dir / f'checkpoint_{total_steps}.pth'
                )

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
