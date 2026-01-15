"""
Async Trainer Process for continuous training.

Runs on a separate thread (not process due to PyTorch GPU constraints),
continuously sampling from the buffer and updating networks.
"""

import numpy as np
import torch
import time
import threading
import queue
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Metrics from a training batch."""
    train_step: int
    critic_loss: float
    actor_loss: float
    q_value_mean: float
    q_value_min: float
    q_value_max: float
    critic_grad_norm: float
    actor_grad_norm: float
    per_stats: Optional[Dict] = None
    timestamp: float = field(default_factory=time.time)


class AsyncTrainer:
    """
    Continuous training loop that runs in a background thread.

    Samples from an async buffer and updates networks continuously.
    Broadcasts weight updates periodically to collectors.
    """

    def __init__(
        self,
        agent,  # TD3Agent instance
        buffer,  # AsyncMemory or AsyncPrioritizedMemory instance
        weight_callback=None,  # Callback for weight broadcasts
        config: Optional[Dict] = None,
    ):
        """
        Args:
            agent: TD3Agent instance (already on GPU if available)
            buffer: Async replay buffer instance
            weight_callback: Optional callback(state_dict, eps, obs_mean, obs_std) for weight sync
            config: Training configuration
        """
        self.agent = agent
        self.buffer = buffer
        self.weight_callback = weight_callback
        self.config = config or {}

        # Training parameters
        self.batch_size = self.config.get('batch_size', 256)
        self.warmup_transitions = self.config.get('warmup_transitions', 10000)
        self.weight_sync_interval = self.config.get('weight_sync_interval', 50)
        self.iterations_per_step = self.config.get('iterations_per_step', 1)

        # State
        self.train_step = 0
        self._running = False
        self._paused = False
        self._thread = None

        # Metrics queue (for main thread to consume)
        self.metrics_queue = queue.Queue(maxsize=1000)

        # Checkpoint queue (trainer signals when to save)
        self.checkpoint_queue = queue.Queue(maxsize=10)

        # Statistics
        self.total_train_steps = 0
        self.total_train_time = 0.0
        self._lock = threading.Lock()

    def start(self):
        """Start the training thread."""
        if self._running:
            return

        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()
        print("[AsyncTrainer] Started training thread")

    def stop(self):
        """Stop the training thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        print("[AsyncTrainer] Stopped training thread")

    def pause(self):
        """Pause training (for evaluation)."""
        self._paused = True

    def resume(self):
        """Resume training."""
        self._paused = False

    def _train_loop(self):
        """Main training loop."""
        print(f"[AsyncTrainer] Waiting for buffer warmup ({self.warmup_transitions} transitions)")

        # Wait for buffer to have enough samples
        while self._running and self.buffer.size < self.warmup_transitions:
            time.sleep(0.5)
            if self.buffer.size > 0 and self.buffer.size % 1000 == 0:
                print(f"[AsyncTrainer] Buffer warmup: {self.buffer.size}/{self.warmup_transitions}")

        if not self._running:
            return

        print(f"[AsyncTrainer] Warmup complete, starting training (buffer: {self.buffer.size})")

        last_weight_sync = 0
        last_metrics_time = time.time()
        error_count = 0
        loop_count = 0
        last_debug_time = time.time()

        while self._running:
            loop_count += 1

            # Debug output every second
            if time.time() - last_debug_time > 1.0:
                print(f"[AsyncTrainer] Loop #{loop_count}: buffer={self.buffer.size}, paused={self._paused}")
                last_debug_time = time.time()

            # Check if paused
            if self._paused:
                time.sleep(0.1)
                continue

            # Check buffer has enough samples
            if self.buffer.size < self.batch_size:
                # time.sleep(0.01)
                continue

            train_start = time.time()

            # Sample batch
            try:
                batch = self.buffer.sample(self.batch_size)
            except Exception as e:
                print(f"[AsyncTrainer] Sampling error: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                if error_count > 5:
                    print(f"[AsyncTrainer] Too many sampling errors, stopping")
                    self._running = False
                time.sleep(0.1)
                continue

            if batch is None:
                print(f"[AsyncTrainer] Warning: batch is None (buffer size: {self.buffer.size})")
                time.sleep(0.01)
                continue

            # Debug: Check batch format on first few samples
            if self.train_step < 3:
                print(f"[AsyncTrainer] Batch shape: {batch.shape if hasattr(batch, 'shape') else 'N/A'}, type: {type(batch)}")

            # Perform training step
            try:
                metrics = self._train_step(batch)
                self.train_step += 1
                error_count = 0  # Reset error count on success

                with self._lock:
                    self.total_train_steps += 1
                    self.total_train_time += time.time() - train_start

                # Push metrics to queue
                try:
                    self.metrics_queue.put_nowait(metrics)
                except queue.Full:
                    # Drop oldest metrics if full
                    try:
                        self.metrics_queue.get_nowait()
                        self.metrics_queue.put_nowait(metrics)
                    except:
                        pass

                # Sync weights periodically
                if self.train_step - last_weight_sync >= self.weight_sync_interval:
                    self._broadcast_weights()
                    last_weight_sync = self.train_step

            except Exception as e:
                import traceback
                print(f"[AsyncTrainer] Training error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def _train_step(self, batch) -> TrainingMetrics:
        """
        Perform a single training step.

        Args:
            batch: Sampled batch from buffer (may include PER indices/weights)

        Returns:
            TrainingMetrics with loss and gradient information
        """
        # Debug logging removed - was causing 3x slowdown (called on every step)

        try:
            # Check if PER batch (has indices and weights)
            if isinstance(batch, tuple) and len(batch) == 3:
                data, indices, is_weights = batch
                use_per = True
            else:
                data = batch
                indices = None
                is_weights = None
                use_per = False

            # Extract data components (data is 2D: data[:, i] gives i-th component of all transitions)
            s_raw = np.stack(data[:, 0]).astype(np.float32)
            s_prime_raw = np.stack(data[:, 3]).astype(np.float32)
        except Exception as e:
            print(f"[AsyncTrainer] Error extracting data from batch: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            # Update observation statistics
            self.agent._update_obs_stats(s_raw)
            self.agent._update_obs_stats(s_prime_raw)

            # Normalize observations
            s = torch.from_numpy(self.agent._normalize_obs(s_raw)).to(self.agent.device)
            s_prime = torch.from_numpy(self.agent._normalize_obs(s_prime_raw)).to(self.agent.device)

            a_full = torch.from_numpy(np.stack(data[:, 1]).astype(np.float32)).to(self.agent.device)
            rew = torch.from_numpy(np.stack(data[:, 2]).astype(np.float32)[:, None]).to(self.agent.device)
            done = torch.from_numpy(np.stack(data[:, 4]).astype(np.float32)[:, None]).to(self.agent.device)
        except Exception as e:
            print(f"[AsyncTrainer] Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Compute target Q-values
        with torch.no_grad():
            a_next_agent = self.agent.policy_target(s_prime)
            noise = torch.randn_like(a_next_agent) * self.agent._config["target_noise_std"]
            noise = noise.clamp(
                -self.agent._config["target_noise_clip"],
                self.agent._config["target_noise_clip"]
            )
            a_next_agent_smooth = a_next_agent + noise
            a_next_agent_smooth = a_next_agent_smooth.clamp(
                torch.from_numpy(self.agent._action_space.low).to(self.agent.device),
                torch.from_numpy(self.agent._action_space.high).to(self.agent.device)
            )

            a_next_opponent = a_full[:, 4:]
            a_next_full = torch.cat([a_next_agent_smooth, a_next_opponent], dim=1)

            q1_target_next = self.agent.Q1_target.Q_value(s_prime, a_next_full)
            q2_target_next = self.agent.Q2_target.Q_value(s_prime, a_next_full)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            target_q = rew + self.agent._config['discount'] * q_target_next * (1 - done)

            q_clip = self.agent._config.get('q_clip', 100.0)
            q_clip_mode = self.agent._config.get('q_clip_mode', 'soft')

            if q_clip_mode == 'soft':
                target_q = q_clip * torch.tanh(target_q / q_clip)
            else:
                target_q = torch.clamp(target_q, -q_clip, q_clip)

        # Update critics
        q1_loss, q1_grad_norm = self.agent.Q1.fit(
            s, a_full, target_q,
            weights=is_weights if use_per else None
        )
        q2_loss, q2_grad_norm = self.agent.Q2.fit(
            s, a_full, target_q.detach(),
            weights=is_weights if use_per else None
        )

        avg_critic_loss = (q1_loss + q2_loss) / 2
        avg_critic_grad_norm = (q1_grad_norm + q2_grad_norm) / 2

        # Update priorities if PER
        if use_per and indices is not None:
            with torch.no_grad():
                q1_current = self.agent.Q1.Q_value(s, a_full)
                q2_current = self.agent.Q2.Q_value(s, a_full)
                q_current = torch.min(q1_current, q2_current)
                td_errors = torch.abs(q_current - target_q).cpu().numpy().flatten()

            self.buffer.update_priorities(indices, td_errors)

        # Update actor (with policy frequency delay)
        actor_loss = 0.0
        actor_grad_norm = 0.0

        self.agent.total_steps += 1
        if self.agent.total_steps % self.agent._config["policy_freq"] == 0:
            self.agent.optimizer.zero_grad()

            a_current_agent = self.agent.policy(s)
            a_current_opponent = a_full[:, 4:]
            a_current_full = torch.cat([a_current_agent, a_current_opponent], dim=1)

            q1_val = self.agent.Q1.Q_value(s, a_current_full)
            q2_val = self.agent.Q2.Q_value(s, a_current_full)
            q_val = torch.min(q1_val, q2_val)

            actor_loss_tensor = -q_val.mean()
            actor_loss_tensor.backward()

            # Compute gradient norm
            actor_grad_norm = 0.0
            for p in self.agent.policy.parameters():
                if p.grad is not None:
                    actor_grad_norm += p.grad.data.norm(2).item() ** 2
            actor_grad_norm = actor_grad_norm ** 0.5

            if "grad_clip" in self.agent._config:
                torch.nn.utils.clip_grad_norm_(
                    self.agent.policy.parameters(),
                    self.agent._config["grad_clip"]
                )

            self.agent.optimizer.step()
            actor_loss = actor_loss_tensor.item()

        # Update target networks
        if self.agent.total_steps % self.agent._config["target_update_freq"] == 0:
            if self.agent._config["use_target_net"]:
                self.agent._soft_update_targets()

        # Compute Q-value statistics
        with torch.no_grad():
            q_values = self.agent.Q1.Q_value(s, a_full)
            q_mean = q_values.mean().item()
            q_min = q_values.min().item()
            q_max = q_values.max().item()

        # Collect PER stats if available
        per_stats = None
        if use_per and hasattr(self.buffer, 'get_stats'):
            per_stats = self.buffer.get_stats()

        metrics = TrainingMetrics(
            train_step=self.train_step,
            critic_loss=avg_critic_loss,
            actor_loss=actor_loss,
            q_value_mean=q_mean,
            q_value_min=q_min,
            q_value_max=q_max,
            critic_grad_norm=avg_critic_grad_norm,
            actor_grad_norm=actor_grad_norm,
            per_stats=per_stats,
        )
        return metrics

    def _broadcast_weights(self):
        """Broadcast current weights to collectors."""
        if self.weight_callback is None:
            return

        try:
            state_dict = self.agent.policy.state_dict()
            eps = self.agent._eps
            obs_mean = self.agent.obs_mean if self.agent.normalize_obs else None
            obs_std = self.agent.obs_std if self.agent.normalize_obs else None

            self.weight_callback(state_dict, eps, obs_mean, obs_std)
        except Exception as e:
            print(f"[AsyncTrainer] Weight broadcast error: {e}")

    def collect_metrics(self, max_items: int = 100) -> list:
        """
        Collect recent metrics from the queue.

        Args:
            max_items: Maximum number of metrics to collect

        Returns:
            List of TrainingMetrics
        """
        metrics_list = []
        for _ in range(max_items):
            try:
                metrics = self.metrics_queue.get_nowait()
                metrics_list.append(metrics)
            except queue.Empty:
                break
        return metrics_list

    def get_latest_weights(self) -> Tuple[Dict, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the latest network weights.

        Returns:
            Tuple of (state_dict, epsilon, obs_mean, obs_std)
        """
        with self._lock:
            state_dict = {k: v.cpu().clone() for k, v in self.agent.policy.state_dict().items()}
            eps = self.agent._eps
            obs_mean = self.agent.obs_mean.copy() if self.agent.normalize_obs else None
            obs_std = self.agent.obs_std.copy() if self.agent.normalize_obs else None
        return state_dict, eps, obs_mean, obs_std

    def get_agent_state(self) -> Tuple:
        """Get full agent state for checkpointing."""
        with self._lock:
            return self.agent.state()

    def restore_agent_state(self, state: Tuple):
        """Restore agent state from checkpoint."""
        with self._lock:
            self.agent.restore_state(state)

    def decay_epsilon(self):
        """Decay exploration epsilon."""
        with self._lock:
            self.agent.decay_epsilon()

    def get_stats(self) -> Dict[str, Any]:
        """Get trainer statistics."""
        with self._lock:
            return {
                'train_step': self.train_step,
                'total_train_steps': self.total_train_steps,
                'total_train_time': self.total_train_time,
                'avg_train_time': self.total_train_time / max(1, self.total_train_steps),
                'epsilon': self.agent._eps,
                'running': self._running,
                'paused': self._paused,
            }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused
