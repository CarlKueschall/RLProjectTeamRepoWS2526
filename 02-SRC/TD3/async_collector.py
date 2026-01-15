"""
Async Collector Process for continuous episode collection.

Runs as a separate process to collect episodes continuously while
the trainer process runs training updates. Communicates via queues.
"""

import numpy as np
import time
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class EpisodeStats:
    """Statistics for a completed episode."""
    episode_num: int
    episode_reward: float
    sparse_reward: float
    winner: int  # -1 = loss, 0 = tie, 1 = win
    episode_length: int
    puck_touches: int
    action_magnitudes: List[float]
    puck_distances: List[float]
    agent_positions: List[List[float]]
    shoot_actions: List[Tuple[float, bool]]
    opponent_type: str
    worker_id: int


def collector_worker(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    transition_queue: Queue,
    env_config: Dict[str, Any],
    initial_weights: Optional[Dict] = None,
    initial_eps: float = 1.0,
):
    """
    Worker process that runs hockey episodes continuously.

    Each worker:
    1. Creates its own environment and lightweight policy network
    2. Receives weight updates from main process
    3. Runs complete episodes and sends transitions to buffer
    4. Sends episode stats back to main process

    Args:
        worker_id: Unique identifier for this worker
        task_queue: Queue for receiving tasks (weight updates, shutdown)
        result_queue: Queue for sending episode stats back
        transition_queue: Queue for sending transitions to buffer
        env_config: Configuration for environment and agent
        initial_weights: Initial actor weights (optional)
        initial_eps: Initial exploration epsilon
    """
    # Import here to ensure each process has its own imports
    import torch
    import hockey.hockey_env as h_env
    from hockey.hockey_env import BasicOpponent

    # Create environment in worker process
    env = h_env.HockeyEnv(
        mode=env_config['mode'],
        keep_mode=env_config.get('keep_mode', True)
    )

    # Create opponents
    weak_opponent = BasicOpponent(weak=True)
    strong_opponent = BasicOpponent(weak=False)

    # Create lightweight actor network (CPU-only, for action selection)
    # CRITICAL: Must match TD3Agent architecture exactly!
    # - activation_fun=ReLU for hidden layers (matches td3_agent.py)
    # - output_activation=Tanh for output (action bounds are [-1, 1])
    from agents.model import Model
    from agents.noise import OUNoise

    actor = Model(
        input_size=env_config['obs_dim'],
        output_size=env_config['action_dim'],
        hidden_sizes=env_config['hidden_actor'],
        activation_fun=torch.nn.ReLU(),  # MUST match TD3Agent!
        output_activation=torch.nn.Tanh()
    )

    # Load initial weights if provided
    if initial_weights is not None:
        actor.load_state_dict(initial_weights)

    actor.eval()  # Set to eval mode

    # Current exploration epsilon
    current_eps = initial_eps

    # Create OUNoise for exploration (MUST match TD3Agent exploration strategy!)
    # TD3 uses OUNoise, not epsilon-greedy. Using epsilon-greedy causes
    # distribution mismatch between collected experience and policy behavior.
    action_noise = OUNoise(shape=(env_config['action_dim'],))

    # Observation normalization (will be updated with weights)
    obs_mean = np.zeros(env_config['obs_dim'], dtype=np.float32)
    obs_std = np.ones(env_config['obs_dim'], dtype=np.float32)

    # Create PBRS shaper if enabled
    pbrs_shaper = None
    if env_config.get('pbrs_enabled', False):
        from rewards import PBRSReward
        pbrs_shaper = PBRSReward(
            gamma=env_config.get('gamma', 0.99),
            pbrs_scale=env_config.get('pbrs_scale', 0.5),
            constant_weight=env_config.get('pbrs_constant_weight', True),
        )

    def normalize_obs(obs):
        """Normalize observation using running mean and std."""
        return ((obs - obs_mean) / (obs_std + 1e-8)).astype(np.float32)

    def select_action(obs, eps):
        """Select action with OUNoise exploration (matches TD3Agent.act())."""
        # CRITICAL: Use OUNoise like TD3Agent, NOT epsilon-greedy!
        # TD3 always takes policy action + scaled noise
        with torch.no_grad():
            obs_normalized = normalize_obs(obs)
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)
            action = actor(obs_tensor).squeeze(0).numpy()

        # Add OUNoise scaled by epsilon (same as TD3Agent.act())
        if eps > 0:
            noise = action_noise()
            action = action + noise * eps

        # Clip to action bounds
        action = np.clip(action, -1.0, 1.0)
        return action

    # Episode counter for this worker
    episode_counter = 0
    global_episode_offset = worker_id * 1000000  # Unique episode IDs per worker
    transitions_sent = 0  # Debug counter
    transitions_dropped = 0  # Track dropped transitions
    last_queue_warning = time.time()

    max_timesteps = env_config.get('max_timesteps', 250)
    reward_scale = env_config.get('reward_scale', 1.0)

    # Log to both stdout and file (daemon processes may not capture stdout)
    log_msg = f"[Worker {worker_id}] Started, ready for episodes"
    print(log_msg)
    try:
        with open(f"worker_{worker_id}.log", "a") as f:
            f.write(log_msg + "\n")
    except:
        pass

    # Main worker loop - runs continuously
    running = True
    while running:
        # Check for control messages (non-blocking)
        try:
            task = task_queue.get_nowait()
            task_type = task[0]

            if task_type == 'shutdown':
                running = False
                break

            elif task_type == 'update_weights':
                # Update actor weights and normalization stats
                _, state_dict, eps, new_obs_mean, new_obs_std = task
                actor.load_state_dict(state_dict)
                current_eps = eps
                if new_obs_mean is not None:
                    obs_mean[:] = new_obs_mean
                if new_obs_std is not None:
                    obs_std[:] = new_obs_std

            elif task_type == 'update_eps':
                _, eps = task
                current_eps = eps

            elif task_type == 'ping':
                result_queue.put(('pong', worker_id))

        except queue.Empty:
            pass  # No control message, continue running episode

        # Run an episode
        opponent_type = env_config.get('default_opponent', 'weak')

        # Select opponent
        if opponent_type == 'weak':
            opponent = weak_opponent
        elif opponent_type == 'strong':
            opponent = strong_opponent
        else:
            opponent = weak_opponent

        # Reset environment
        seed = env_config.get('seed')
        if seed is not None:
            np.random.seed(seed + episode_counter + worker_id * 10000)
            reset_seed = np.random.randint(0, 1000000)
        else:
            reset_seed = None

        obs, info = env.reset(seed=reset_seed)
        obs_agent2 = env.obs_agent_two()

        # Reset PBRS shaper
        if pbrs_shaper:
            pbrs_shaper.reset()

        # Reset OUNoise for new episode (important for temporally correlated noise)
        action_noise.reset()

        # Episode tracking
        episode_reward = 0.0
        sparse_reward = 0.0
        puck_touches = 0
        action_magnitudes = []
        puck_distances = []
        agent_positions = []
        shoot_actions = []

        episode_num = global_episode_offset + episode_counter

        for t in range(max_timesteps):
            obs_curr = obs.copy()

            # Get agent action
            action1 = select_action(obs, current_eps)

            # Get opponent action
            action2 = opponent.act(obs_agent2)

            # Combine actions (8D: 4D agent + 4D opponent)
            action_combined = np.hstack([action1[:4], action2[:4]])

            # Step environment
            obs_next, r1, done, truncated, info = env.step(action_combined)

            # Scale reward
            r1_scaled = r1 * reward_scale
            sparse_reward += r1_scaled

            # Apply PBRS if enabled
            if pbrs_shaper:
                pbrs_bonus = pbrs_shaper.compute(
                    obs_curr, obs_next,
                    done=(done or truncated),
                    episode=episode_num
                )
                r1_shaped = r1_scaled + pbrs_bonus
            else:
                r1_shaped = r1_scaled

            # Compute metrics
            dist_to_puck = np.sqrt(
                (obs_next[0] - obs_next[12])**2 +
                (obs_next[1] - obs_next[13])**2
            )

            # Track puck touches
            env_touch_reward = info.get('reward_touch_puck', 0.0)
            if env_touch_reward > 0 or dist_to_puck < 0.3:
                puck_touches += 1

            # Send transition to buffer queue
            transition = (
                obs_curr,
                action_combined.copy(),
                r1_shaped,
                obs_next.copy(),
                float(done or truncated)
            )
            try:
                transition_queue.put_nowait(transition)
                transitions_sent += 1
                # Debug print every 1000 transitions per worker
                # Note: Don't call qsize() - it's not implemented on macOS
                if transitions_sent % 1000 == 0:
                    print(f"[Worker {worker_id}] Sent {transitions_sent} transitions")
            except queue.Full:
                # Buffer queue full, drop oldest
                transitions_dropped += 1
                try:
                    old = transition_queue.get_nowait()
                    transition_queue.put_nowait(transition)
                    transitions_sent += 1
                    # Warn every 100 drops
                    if transitions_dropped % 100 == 0:
                        print(f"[Worker {worker_id}] WARNING: Dropped {transitions_dropped} transitions (queue overflow)")
                except:
                    pass

            # Track metrics
            episode_reward += r1_shaped
            action_magnitudes.append(float(np.linalg.norm(action1[:2])))
            puck_distances.append(float(dist_to_puck))
            agent_positions.append([float(obs_next[0]), float(obs_next[1])])

            # Track shoot action (action[3]) and possession
            has_possession = len(obs_next) > 16 and obs_next[16] > 0
            shoot_actions.append((float(action1[3]) if len(action1) > 3 else 0.0, has_possession))

            # Update observations
            obs = obs_next
            obs_agent2 = env.obs_agent_two()

            if done or truncated:
                break

        # Episode complete - get winner
        winner = info.get('winner', 0)
        if winner not in [-1, 0, 1]:
            winner = 0

        # Create episode stats
        stats = EpisodeStats(
            episode_num=episode_num,
            episode_reward=episode_reward,
            sparse_reward=sparse_reward,
            winner=winner,
            episode_length=t + 1,
            puck_touches=puck_touches,
            action_magnitudes=action_magnitudes,
            puck_distances=puck_distances,
            agent_positions=agent_positions,
            shoot_actions=shoot_actions,
            opponent_type=opponent_type,
            worker_id=worker_id,
        )

        # Send stats to main process
        try:
            result_queue.put_nowait(('episode_done', stats))
        except queue.Full:
            # Stats queue full, skip this stat
            pass

        episode_counter += 1

    # Cleanup
    env.close()
    print(f"[Worker {worker_id}] Shutdown complete")


class AsyncCollector:
    """
    Manages multiple collector workers for continuous episode collection.

    Workers run episodes autonomously with periodic weight syncs.
    Transitions are pushed to a shared queue for the buffer.
    """

    def __init__(
        self,
        num_workers: int,
        env_config: Dict[str, Any],
        transition_queue: Queue,
        initial_weights: Optional[Dict] = None,
        initial_eps: float = 1.0,
    ):
        """
        Args:
            num_workers: Number of parallel collector workers
            env_config: Environment configuration
            transition_queue: Shared queue for transitions (goes to buffer)
            initial_weights: Initial actor weights
            initial_eps: Initial exploration epsilon
        """
        self.num_workers = num_workers
        self.env_config = env_config
        self.transition_queue = transition_queue

        # Use 'spawn' method for multiprocessing (safer on all platforms)
        self._ctx = mp.get_context('spawn')

        # Communication queues
        # Note: macOS has small semaphore limits, use smaller queue sizes
        import sys
        result_queue_size = 100 if sys.platform == 'darwin' else 1000
        self.task_queues: List[Queue] = []
        self.result_queue = self._ctx.Queue(maxsize=result_queue_size)
        self.workers: List[Process] = []

        self._running = False
        self._initial_weights = initial_weights
        self._initial_eps = initial_eps

        # Statistics
        self.episodes_collected = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0

    def start(self):
        """Start all collector workers."""
        if self._running:
            return

        self._running = True

        # Small queue size for task queues (just need a few messages)
        task_queue_size = 10

        for i in range(self.num_workers):
            task_queue = self._ctx.Queue(maxsize=task_queue_size)
            self.task_queues.append(task_queue)

            worker = self._ctx.Process(
                target=collector_worker,
                args=(
                    i,
                    task_queue,
                    self.result_queue,
                    self.transition_queue,
                    self.env_config,
                    self._initial_weights,
                    self._initial_eps,
                ),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        print(f"[AsyncCollector] Started {self.num_workers} workers")

    def stop(self):
        """Stop all collector workers."""
        if not self._running:
            return

        self._running = False

        # Send shutdown signal to all workers
        for task_queue in self.task_queues:
            try:
                task_queue.put(('shutdown',), timeout=1.0)
            except:
                pass

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()

        self.workers.clear()
        self.task_queues.clear()

        print(f"[AsyncCollector] All workers stopped")

    def sync_weights(
        self,
        actor_state_dict: Dict,
        eps: float,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
    ):
        """
        Broadcast weight update to all workers.

        Args:
            actor_state_dict: Actor network state dict (CPU tensors)
            eps: Current exploration epsilon
            obs_mean: Observation normalization mean
            obs_std: Observation normalization std
        """
        # Convert to CPU if needed
        cpu_state_dict = {k: v.cpu() for k, v in actor_state_dict.items()}

        for task_queue in self.task_queues:
            try:
                # Clear old weight updates first
                while not task_queue.empty():
                    try:
                        task_queue.get_nowait()
                    except:
                        break

                task_queue.put_nowait((
                    'update_weights',
                    cpu_state_dict,
                    eps,
                    obs_mean.copy() if obs_mean is not None else None,
                    obs_std.copy() if obs_std is not None else None,
                ))
            except:
                pass  # Best effort

    def update_epsilon(self, eps: float):
        """Update epsilon for all workers."""
        for task_queue in self.task_queues:
            try:
                task_queue.put_nowait(('update_eps', eps))
            except:
                pass

    def collect_stats(self, timeout: float = 0.1) -> List[EpisodeStats]:
        """
        Collect episode statistics from workers (non-blocking).

        Args:
            timeout: Maximum time to wait for stats

        Returns:
            List of EpisodeStats from completed episodes
        """
        stats_list = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                result = self.result_queue.get_nowait()
                if result[0] == 'episode_done':
                    stats = result[1]
                    stats_list.append(stats)

                    # Update counters
                    self.episodes_collected += 1
                    if stats.winner == 1:
                        self.wins += 1
                    elif stats.winner == -1:
                        self.losses += 1
                    else:
                        self.ties += 1

            except queue.Empty:
                break

        return stats_list

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            'episodes_collected': self.episodes_collected,
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'num_workers': self.num_workers,
            'running': self._running,
        }

    @property
    def is_running(self) -> bool:
        return self._running
