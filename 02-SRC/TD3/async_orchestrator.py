"""
Async Training Orchestrator.

Coordinates collector processes, trainer thread, buffer management,
logging, evaluation, and checkpointing for fully async training.
"""

import numpy as np
import torch
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import deque
import multiprocessing as mp

import wandb
from tqdm import tqdm


class AsyncMetricsTracker:
    """
    Thread-safe metrics tracker for async training.

    Tracks episode results, training metrics, and provides
    aggregated statistics for logging.
    """

    def __init__(self, rolling_window: int = 100, log_interval: int = 50):
        self.rolling_window = rolling_window
        self.log_interval = log_interval
        # Use RLock (reentrant lock) because get_metrics_dict() calls get_win_rate()
        # and get_rolling_win_rate() which also acquire the lock
        self._lock = threading.RLock()

        # Episode metrics
        self.episode_rewards = deque(maxlen=rolling_window)
        self.episode_lengths = deque(maxlen=rolling_window)
        self.sparse_rewards = deque(maxlen=rolling_window)

        # Outcomes
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.total_episodes = 0

        # Rolling outcomes
        self.rolling_outcomes = deque(maxlen=rolling_window)

        # Training metrics
        self.critic_losses = deque(maxlen=rolling_window)
        self.actor_losses = deque(maxlen=rolling_window)
        self.q_values = deque(maxlen=rolling_window)
        self.critic_grad_norms = deque(maxlen=rolling_window)
        self.actor_grad_norms = deque(maxlen=rolling_window)

        # Behavior metrics
        self.action_magnitudes = deque(maxlen=rolling_window)
        self.puck_distances = deque(maxlen=rolling_window)
        self.puck_touches = deque(maxlen=rolling_window)

        # PER stats
        self.per_beta = 0.0
        self.per_max_priority = 1.0

        # Evaluation results
        self.eval_vs_weak = 0.0
        self.eval_vs_strong = 0.0

        # Goals
        self.goals_scored = 0
        self.goals_conceded = 0

    def add_episode(self, stats):
        """Add episode statistics."""
        with self._lock:
            self.episode_rewards.append(stats.episode_reward)
            self.episode_lengths.append(stats.episode_length)
            self.sparse_rewards.append(stats.sparse_reward)

            # Track outcome
            self.total_episodes += 1
            if stats.winner == 1:
                self.wins += 1
                self.rolling_outcomes.append(1)
                self.goals_scored += 1
            elif stats.winner == -1:
                self.losses += 1
                self.rolling_outcomes.append(-1)
                self.goals_conceded += 1
            else:
                self.ties += 1
                self.rolling_outcomes.append(0)

            # Behavior metrics
            if stats.action_magnitudes:
                self.action_magnitudes.append(np.mean(stats.action_magnitudes))
            if stats.puck_distances:
                self.puck_distances.append(np.mean(stats.puck_distances))
            self.puck_touches.append(stats.puck_touches)

    def add_training_metrics(self, metrics):
        """Add training metrics."""
        with self._lock:
            self.critic_losses.append(metrics.critic_loss)
            if metrics.actor_loss != 0:
                self.actor_losses.append(metrics.actor_loss)
            self.q_values.append(metrics.q_value_mean)
            self.critic_grad_norms.append(metrics.critic_grad_norm)
            if metrics.actor_grad_norm != 0:
                self.actor_grad_norms.append(metrics.actor_grad_norm)

            # PER stats
            if metrics.per_stats:
                self.per_beta = metrics.per_stats.get('beta', 0.0)
                self.per_max_priority = metrics.per_stats.get('max_priority', 1.0)

    def set_eval_result(self, opponent: str, win_rate: float):
        """Set evaluation result."""
        with self._lock:
            if opponent == 'weak':
                self.eval_vs_weak = win_rate
            elif opponent == 'strong':
                self.eval_vs_strong = win_rate

    def get_win_rate(self) -> float:
        """Get overall win rate."""
        with self._lock:
            total = self.wins + self.losses + self.ties
            return self.wins / total if total > 0 else 0.0

    def get_rolling_win_rate(self) -> float:
        """Get rolling window win rate."""
        with self._lock:
            if len(self.rolling_outcomes) == 0:
                return 0.0
            wins = sum(1 for x in self.rolling_outcomes if x == 1)
            return wins / len(self.rolling_outcomes)

    def get_metrics_dict(self) -> Dict[str, float]:
        """Get all metrics as a dictionary for logging."""
        with self._lock:
            metrics = {
                # Performance
                'performance/win_rate': self.get_win_rate(),
                'performance/rolling_win_rate': self.get_rolling_win_rate(),
                'performance/wins': self.wins,
                'performance/losses': self.losses,
                'performance/ties': self.ties,
                'performance/total_episodes': self.total_episodes,

                # Rewards
                'rewards/episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
                'rewards/sparse_reward': np.mean(self.sparse_rewards) if self.sparse_rewards else 0.0,

                # Episode length
                'training/episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,

                # Training metrics
                'losses/critic_loss': np.mean(self.critic_losses) if self.critic_losses else 0.0,
                'losses/actor_loss': np.mean(self.actor_losses) if self.actor_losses else 0.0,

                # Q-values
                'values/Q_mean': np.mean(self.q_values) if self.q_values else 0.0,
                'values/Q_std': np.std(self.q_values) if len(self.q_values) > 1 else 0.0,

                # Gradients
                'gradients/critic_grad_norm': np.mean(self.critic_grad_norms) if self.critic_grad_norms else 0.0,
                'gradients/actor_grad_norm': np.mean(self.actor_grad_norms) if self.actor_grad_norms else 0.0,

                # Behavior
                'behavior/action_magnitude': np.mean(self.action_magnitudes) if self.action_magnitudes else 0.0,
                'behavior/puck_distance': np.mean(self.puck_distances) if self.puck_distances else 0.0,
                'behavior/puck_touches': np.mean(self.puck_touches) if self.puck_touches else 0.0,

                # Evaluation
                'eval/win_rate_vs_weak': self.eval_vs_weak,
                'eval/win_rate_vs_strong': self.eval_vs_strong,

                # Scoring
                'scoring/goals_scored': self.goals_scored,
                'scoring/goals_conceded': self.goals_conceded,

                # PER
                'per/beta': self.per_beta,
                'per/max_priority': self.per_max_priority,
            }
            return metrics


class AsyncTrainingOrchestrator:
    """
    Main orchestrator for fully async training.

    Coordinates:
    - Multiple collector processes (continuous episode collection)
    - Trainer thread (continuous network updates)
    - Buffer management (thread-safe transitions)
    - W&B logging
    - Periodic evaluation
    - Checkpointing
    """

    def __init__(self, args, env_config: Dict[str, Any]):
        """
        Args:
            args: Command-line arguments
            env_config: Environment configuration
        """
        self.args = args
        self.env_config = env_config

        # Create results directories
        self.results_dir = Path('./results')
        self.checkpoints_dir = self.results_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Import components
        from agents import TD3Agent, get_device
        from agents.async_memory import create_async_buffer
        from async_collector import AsyncCollector
        from async_trainer import AsyncTrainer
        from opponents import FixedOpponent
        from evaluation import evaluate_vs_opponent

        # Get device
        self.device = get_device(force_cpu=args.cpu)
        print(f"[Orchestrator] Using device: {self.device}")

        # Create async buffer
        # IMPORTANT: Scale per_beta_frames based on batch_size
        # per_beta_frames is specified in "samples to process", convert to training steps
        # This ensures beta anneals properly regardless of batch_size
        # Example: per_beta_frames=100000 with batch_size=256 -> 100000/256 ≈ 390 training steps
        # But that's too few! So we interpret per_beta_frames as "training steps", not samples.
        # With async training doing ~0.5-1 train step per episode, we need many episodes.
        # Keep per_beta_frames as training steps, but warn if it seems misconfigured.
        scaled_beta_frames = args.per_beta_frames if args.use_per else 100000
        expected_episodes = args.max_episodes
        expected_train_steps = expected_episodes  # Rough estimate: ~1 train step per episode with good throughput
        if args.use_per and scaled_beta_frames > expected_train_steps * 2:
            print(f"[Orchestrator] WARNING: per_beta_frames={scaled_beta_frames} may be too high!")
            print(f"  With max_episodes={expected_episodes}, expect ~{expected_train_steps} train steps")
            print(f"  Beta may not reach 1.0. Consider reducing per_beta_frames or increasing max_episodes.")

        self.buffer = create_async_buffer(
            use_per=args.use_per,
            max_size=args.buffer_size,
            alpha=args.per_alpha if args.use_per else 0.6,
            beta_start=args.per_beta_start if args.use_per else 0.4,
            beta_frames=scaled_beta_frames,
        )
        print(f"[Orchestrator] Created {'PER' if args.use_per else 'uniform'} buffer (size: {args.buffer_size})")
        if args.use_per:
            print(f"  PER: alpha={args.per_alpha}, beta_start={args.per_beta_start}, beta_frames={scaled_beta_frames}")

        # Create agent
        from gymnasium import spaces
        single_player_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(env_config['obs_dim'],),
            dtype=np.float32
        )

        self.agent = TD3Agent(
            obs_space,
            single_player_action_space,
            eps=args.eps,
            eps_min=args.eps_min,
            eps_decay=args.eps_decay,
            learning_rate_actor=args.lr_actor,
            learning_rate_critic=args.lr_critic,
            discount=args.gamma,
            tau=args.tau,
            policy_freq=args.policy_freq,
            target_update_freq=args.target_update_freq,
            target_noise_std=args.target_noise_std,
            target_noise_clip=args.target_noise_clip,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            hidden_sizes_actor=args.hidden_actor,
            hidden_sizes_critic=args.hidden_critic,
            grad_clip=args.grad_clip,
            force_cpu=args.cpu,
            q_clip=args.q_clip,
            q_clip_mode=args.q_clip_mode,
            use_per=False,  # Buffer handles PER, not agent
        )
        print(f"[Orchestrator] Created TD3 agent")

        # Create transition queue for collectors -> buffer
        # Note: macOS has small semaphore limits, but we try to use reasonable size
        # If queue fills up, we'll see drops in the debug output
        self._ctx = mp.get_context('spawn')
        import sys
        if sys.platform == 'darwin':
            # macOS: Use moderately sized queue (semaphore limited)
            queue_size = 5000  # Increased from 1000 to handle burst collection
        else:
            queue_size = 50000
        print(f"[Orchestrator] Transition queue size: {queue_size} (platform: {sys.platform})")
        self.transition_queue = self._ctx.Queue(maxsize=queue_size)
        self._transition_queue_drops = 0  # Track how many transitions are dropped

        # Create collector
        # IMPORTANT: Convert weights to CPU for pickling (MPS tensors can't be pickled)
        initial_weights_cpu = {k: v.cpu() for k, v in self.agent.policy.state_dict().items()}
        self.collector = AsyncCollector(
            num_workers=args.num_workers,
            env_config=env_config,
            transition_queue=self.transition_queue,
            initial_weights=initial_weights_cpu,
            initial_eps=args.eps,
        )

        # Create trainer with weight callback
        def weight_callback(state_dict, eps, obs_mean, obs_std):
            self.collector.sync_weights(state_dict, eps, obs_mean, obs_std)

        self.trainer = AsyncTrainer(
            agent=self.agent,
            buffer=self.buffer,
            weight_callback=weight_callback,
            config={
                'batch_size': args.batch_size,
                'warmup_transitions': args.warmup_episodes * 100,  # Approximate transitions
                'weight_sync_interval': args.weight_sync_interval,
            }
        )

        # Evaluation opponents
        self.weak_eval_bot = FixedOpponent(weak=True)
        self.strong_eval_bot = FixedOpponent(weak=False)

        # Metrics tracker
        self.metrics = AsyncMetricsTracker(
            rolling_window=100,
            log_interval=args.log_interval
        )

        # State
        self.start_time = None
        self._running = False
        self._buffer_filler_thread = None

        # Best checkpoint tracking
        self.best_win_rate = 0.0
        self.best_checkpoint_path = None

    def _start_buffer_filler(self):
        """Start thread that moves transitions from queue to buffer's internal queue."""
        def filler_loop():
            print("[BufferFiller] Thread started")
            batch = []
            batch_size = 100
            total_filled = 0
            filled_in_interval = 0
            last_print = time.time()
            add_errors = 0
            loop_count = 0

            while self._running:
                loop_count += 1
                # Print every 500 loops (~5ms * 500 = 2.5s)
                if loop_count % 500 == 0:
                    print(f"[BufferFiller] Loop #{loop_count}: batch_len={len(batch)}, total={total_filled}, buf={self.buffer.size}")
                try:
                    # Batch transitions for efficiency
                    while len(batch) < batch_size:
                        try:
                            transition = self.transition_queue.get(timeout=0.01)
                            batch.append(transition)
                        except queue.Empty:
                            break
                        except Exception as e:
                            print(f"[BufferFiller] Error getting from queue: {e}")
                            break

                    if batch:
                        # Add transitions using buffer's non-blocking queue
                        batch_count = 0
                        for t in batch:
                            try:
                                self.buffer.add_transition(t)
                                batch_count += 1
                            except Exception as add_err:
                                add_errors += 1
                                if add_errors <= 3:
                                    print(f"[BufferFiller] Error: {add_err}")

                        total_filled += batch_count
                        filled_in_interval += batch_count
                        batch.clear()

                        # Debug print every 2 seconds (not 1)
                        now = time.time()
                        if now - last_print > 2.0:
                            try:
                                stats = self.buffer.get_stats() if hasattr(self.buffer, 'get_stats') else {}
                            except:
                                stats = {}
                            print(f"[BufferFiller] +{filled_in_interval:.0f} trans/2s | Total: {total_filled} | Buffer.size: {self.buffer.size} | Stats: {stats}")
                            filled_in_interval = 0
                            last_print = now
                    else:
                        # No transitions in queue, sleep briefly
                        time.sleep(0.01)

                except Exception as e:
                    print(f"[BufferFiller] CRITICAL ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    batch.clear()
                    time.sleep(0.1)

            print("[BufferFiller] Thread exiting")

        print("[BufferFiller] Starting buffer filler thread")
        self._buffer_filler_thread = threading.Thread(target=filler_loop, daemon=True)
        self._buffer_filler_thread.start()

    def start(self):
        """Start all components."""
        if self._running:
            return

        self._running = True
        self.start_time = time.time()

        # Start buffer (background drain thread)
        self.buffer.start()

        # Start collectors (will send to transition_queue)
        self.collector.start()

        # Start buffer filler (transition_queue -> buffer's _add_queue -> buffer.start() drain thread)
        self._start_buffer_filler()

        # Start trainer
        self.trainer.start()

        print("[Orchestrator] All components started")

    def stop(self):
        """Stop all components."""
        if not self._running:
            return

        print("[Orchestrator] Stopping components...")
        self._running = False

        # Stop in reverse order
        self.trainer.stop()
        self.collector.stop()
        self.buffer.stop()

        print("[Orchestrator] All components stopped")

    def run(self):
        """Main orchestration loop."""
        try:
            self.start()

            # Initialize W&B
            if not self.args.no_wandb:
                run_name = f"TD3-ASYNC-{self.args.mode}-{self.args.opponent}-seed{self.args.seed}"
                wandb.init(
                    project="rl-hockey",
                    name=run_name,
                    config=vars(self.args),
                    tags=["TD3", "Hockey", "async", self.args.mode, self.args.opponent]
                )

            last_log_time = time.time()

            # Track episode-based intervals
            last_eval_episode = 0
            last_save_episode = 0
            last_eps_decay_episode = 0

            # Progress bar
            pbar = tqdm(total=self.args.max_episodes, desc="Async Training", unit="ep")

            while self.metrics.total_episodes < self.args.max_episodes:
                # Process episode stats from collectors
                episode_stats = self.collector.collect_stats(timeout=0.1)
                for stats in episode_stats:
                    self.metrics.add_episode(stats)
                    pbar.update(1)

                # Process training metrics
                training_metrics = self.trainer.collect_metrics(max_items=50)
                for tm in training_metrics:
                    self.metrics.add_training_metrics(tm)

                current_episode = self.metrics.total_episodes

                # Update progress bar
                pbar.set_postfix({
                    'reward': f'{np.mean(list(self.metrics.episode_rewards)[-10:]):.1f}' if self.metrics.episode_rewards else '0.0',
                    'win_rate': f'{self.metrics.get_rolling_win_rate():.2%}',
                    'buffer': f'{self.buffer.size}',
                    'train': f'{self.trainer.train_step}',
                    'eps': f'{self.agent._eps:.3f}',
                })

                # Periodic logging (time-based, every 10 seconds)
                current_time = time.time()
                if current_time - last_log_time >= 10:
                    self._log_metrics()
                    last_log_time = current_time

                # Periodic epsilon decay (episode-based, decay once per episode)
                # This matches sequential training behavior: eps *= eps_decay after each episode
                episodes_since_decay = current_episode - last_eps_decay_episode
                if episodes_since_decay > 0:
                    # Decay for each episode that passed
                    for _ in range(episodes_since_decay):
                        self.trainer.decay_epsilon()
                    last_eps_decay_episode = current_episode
                    # Also sync epsilon to collectors
                    self.collector.update_epsilon(self.agent._eps)

                # Periodic evaluation (episode-based)
                if current_episode - last_eval_episode >= self.args.eval_interval:
                    self._run_evaluation()
                    last_eval_episode = current_episode

                # Periodic checkpoint (episode-based)
                if current_episode - last_save_episode >= self.args.save_interval:
                    self._save_checkpoint()
                    last_save_episode = current_episode

                time.sleep(0.01)  # Prevent busy-waiting

            pbar.close()

        except KeyboardInterrupt:
            print("\n[Orchestrator] Received interrupt, shutting down...")

        except Exception as e:
            print(f"\n[Orchestrator] Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Only save/log if we actually started
            if self._running:
                self._save_final_checkpoint()
                self._log_final_summary()
            self.stop()

            if not self.args.no_wandb and wandb.run is not None:
                wandb.finish()

    def _log_metrics(self):
        """Log metrics to W&B and console."""
        elapsed = time.time() - self.start_time
        eps_per_sec = self.metrics.total_episodes / elapsed if elapsed > 0 else 0

        metrics_dict = self.metrics.get_metrics_dict()
        metrics_dict['training/eps_per_sec'] = eps_per_sec
        metrics_dict['training/elapsed_time'] = elapsed
        metrics_dict['training/epsilon'] = self.agent._eps
        metrics_dict['training/buffer_size'] = self.buffer.size
        metrics_dict['training/train_step'] = self.trainer.train_step
        metrics_dict['async/num_workers'] = self.args.num_workers

        # Collector stats
        collector_stats = self.collector.get_stats()
        metrics_dict['async/collector_episodes'] = collector_stats['episodes_collected']

        # Trainer stats
        trainer_stats = self.trainer.get_stats()
        metrics_dict['async/train_steps'] = trainer_stats['train_step']
        metrics_dict['async/avg_train_time'] = trainer_stats['avg_train_time']

        # Buffer stats
        buffer_stats = self.buffer.get_stats()
        metrics_dict['async/buffer_queue_size'] = buffer_stats.get('queue_size', 0)

        if not self.args.no_wandb:
            wandb.log(metrics_dict, step=self.metrics.total_episodes)

        # Console output
        print(f"\n[Progress] Episodes: {self.metrics.total_episodes}, "
              f"Train steps: {self.trainer.train_step}, "
              f"Win rate: {self.metrics.get_rolling_win_rate():.2%}, "
              f"Speed: {eps_per_sec:.2f} ep/s")

    def _run_evaluation(self):
        """Run evaluation against fixed opponents."""
        from evaluation import evaluate_vs_opponent
        import hockey.hockey_env as h_env

        print("\n[Evaluation] Starting evaluation...")

        # Pause trainer during evaluation
        self.trainer.pause()
        time.sleep(0.1)  # Brief pause to let current training step complete

        try:
            # Create temporary environment
            mode = self.env_config['mode']
            max_timesteps = self.env_config.get('max_timesteps', 250)
            keep_mode = self.env_config.get('keep_mode', True)

            # Evaluate vs weak
            eval_weak = evaluate_vs_opponent(
                self.agent, self.weak_eval_bot,
                mode=mode,
                num_episodes=self.args.eval_episodes,
                max_timesteps=max_timesteps,
                eval_seed=self.args.seed,
                keep_mode=keep_mode
            )
            self.metrics.set_eval_result('weak', eval_weak['win_rate'])

            # Evaluate vs strong
            eval_strong = evaluate_vs_opponent(
                self.agent, self.strong_eval_bot,
                mode=mode,
                num_episodes=self.args.eval_episodes,
                max_timesteps=max_timesteps,
                eval_seed=self.args.seed,
                keep_mode=keep_mode
            )
            self.metrics.set_eval_result('strong', eval_strong['win_rate'])

            print(f"[Evaluation] vs Weak: {eval_weak['win_rate']:.1%} "
                  f"({eval_weak['wins']}W/{eval_weak['ties']}T/{eval_weak['losses']}L)")
            print(f"[Evaluation] vs Strong: {eval_strong['win_rate']:.1%} "
                  f"({eval_strong['wins']}W/{eval_strong['ties']}T/{eval_strong['losses']}L)")

            # Update best checkpoint
            if eval_weak['win_rate'] > self.best_win_rate:
                self.best_win_rate = eval_weak['win_rate']
                self._save_best_checkpoint()

            if not self.args.no_wandb:
                wandb.log({
                    'eval/win_rate_vs_weak': eval_weak['win_rate'],
                    'eval/win_rate_vs_strong': eval_strong['win_rate'],
                    'eval/avg_reward_weak': eval_weak['avg_reward'],
                    'eval/avg_reward_strong': eval_strong['avg_reward'],
                }, step=self.metrics.total_episodes)

        finally:
            # Resume trainer
            self.trainer.resume()

    def _save_checkpoint(self):
        """Save periodic checkpoint."""
        episode = self.metrics.total_episodes
        checkpoint_path = self.checkpoints_dir / f'TD3_ASYNC_{self.args.mode}_{episode}_seed{self.args.seed}.pth'

        checkpoint_data = {
            'agent_state': self.trainer.get_agent_state(),
            'episode': episode,
            'train_step': self.trainer.train_step,
            'win_rate': self.metrics.get_rolling_win_rate(),
            'config': vars(self.args),
        }

        torch.save(checkpoint_data, checkpoint_path)
        print(f"[Checkpoint] Saved: {checkpoint_path.name}")

        if not self.args.no_wandb and wandb.run is not None:
            wandb.save(str(checkpoint_path))

    def _save_best_checkpoint(self):
        """Save best checkpoint based on evaluation."""
        checkpoint_path = self.checkpoints_dir / f'TD3_ASYNC_{self.args.mode}_best_seed{self.args.seed}.pth'

        checkpoint_data = {
            'agent_state': self.trainer.get_agent_state(),
            'episode': self.metrics.total_episodes,
            'train_step': self.trainer.train_step,
            'win_rate': self.best_win_rate,
            'config': vars(self.args),
        }

        torch.save(checkpoint_data, checkpoint_path)
        self.best_checkpoint_path = checkpoint_path
        print(f"[Checkpoint] New best ({self.best_win_rate:.1%}): {checkpoint_path.name}")

        if not self.args.no_wandb and wandb.run is not None:
            wandb.save(str(checkpoint_path))

    def _save_final_checkpoint(self):
        """Save final checkpoint."""
        checkpoint_path = self.checkpoints_dir / f'TD3_ASYNC_{self.args.mode}_final_seed{self.args.seed}.pth'

        checkpoint_data = {
            'agent_state': self.trainer.get_agent_state(),
            'episode': self.metrics.total_episodes,
            'train_step': self.trainer.train_step,
            'win_rate': self.metrics.get_rolling_win_rate(),
            'config': vars(self.args),
        }

        torch.save(checkpoint_data, checkpoint_path)
        print(f"[Checkpoint] Final saved: {checkpoint_path.name}")

        if not self.args.no_wandb and wandb.run is not None:
            wandb.save(str(checkpoint_path))

    def _log_final_summary(self):
        """Log final training summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        final_win_rate = self.metrics.get_rolling_win_rate()
        eps_per_sec = self.metrics.total_episodes / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("ASYNC TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Episodes: {self.metrics.total_episodes}")
        print(f"Training steps: {self.trainer.train_step}")
        print(f"Final win rate: {final_win_rate:.1%}")
        print(f"Best win rate: {self.best_win_rate:.1%}")
        print(f"Wins: {self.metrics.wins}, Losses: {self.metrics.losses}, Ties: {self.metrics.ties}")
        print(f"Speed: {eps_per_sec:.2f} ep/s")
        print("=" * 60)

        if not self.args.no_wandb and wandb.run is not None:
            wandb.summary['final_win_rate'] = final_win_rate
            wandb.summary['best_win_rate'] = self.best_win_rate
            wandb.summary['total_episodes'] = self.metrics.total_episodes
            wandb.summary['total_train_steps'] = self.trainer.train_step
            wandb.summary['training_time_seconds'] = elapsed
            wandb.summary['eps_per_sec'] = eps_per_sec
