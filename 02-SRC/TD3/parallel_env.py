"""
Parallel Hockey Environment for faster data collection.

Uses multiprocessing to run multiple hockey environments in parallel,
collecting episodes concurrently and adding transitions to the replay buffer.

Design:
- Each worker has its own copy of the agent policy (for action selection only)
- Workers run complete episodes autonomously (no step-by-step communication)
- Main process periodically syncs weights to workers
- Workers send back complete episode data
- Main process adds transitions to buffer and trains

This provides 2-4x speedup for data collection without changing the
training algorithm (which still runs on the main process with GPU).
"""

import numpy as np
from multiprocessing import Process, Queue, Event
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Any
import time
import pickle
import torch


class EpisodeData:
    """Container for all data from a completed episode."""

    def __init__(self):
        self.transitions: List[Tuple] = []  # (obs, action, reward, obs_next, done)
        self.episode_reward: float = 0.0
        self.sparse_reward: float = 0.0
        self.winner: int = 0  # -1 = loss, 0 = tie, 1 = win
        self.episode_length: int = 0
        self.puck_touches: int = 0
        self.action_magnitudes: List[float] = []
        self.puck_distances: List[float] = []
        self.shoot_actions: List[Tuple[float, bool]] = []  # (action_value, has_possession)
        self.agent_positions: List[List[float]] = []


def worker_process(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    mode: int,
    keep_mode: bool,
    max_timesteps: int,
    reward_scale: float,
    pbrs_config: Dict[str, Any],
    obs_dim: int,
    action_dim: int,
    hidden_actor: List[int],
):
    """
    Worker process that runs hockey episodes autonomously.

    Each worker:
    1. Creates its own environment and lightweight policy network
    2. Receives weight updates from main process
    3. Runs complete episodes and sends back transitions

    Args:
        worker_id: Unique identifier for this worker
        task_queue: Queue for receiving tasks (episodes to run, weight updates)
        result_queue: Queue for sending results back
        mode: Hockey environment mode
        keep_mode: Whether keep mode is enabled
        max_timesteps: Maximum timesteps per episode
        reward_scale: Scaling factor for sparse rewards
        pbrs_config: Configuration for PBRS reward shaping
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_actor: Hidden layer sizes for actor network
    """
    # Import here to ensure each process has its own imports
    import torch
    import hockey.hockey_env as h_env
    from hockey.hockey_env import BasicOpponent

    # Create environment in worker process
    env = h_env.HockeyEnv(mode=mode, keep_mode=keep_mode)

    # Create opponents
    weak_opponent = BasicOpponent(weak=True)
    strong_opponent = BasicOpponent(weak=False)

    # Create lightweight actor network (CPU-only, for action selection)
    from agents.model import Model
    actor = Model(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_actor,
        output_activation=torch.nn.Tanh()
    )
    actor.eval()  # Set to eval mode

    # Current exploration epsilon
    current_eps = 1.0

    # Create PBRS shaper if enabled
    pbrs_shaper = None
    if pbrs_config.get('enabled', False):
        from rewards import PBRSReward
        pbrs_shaper = PBRSReward(
            gamma=pbrs_config.get('gamma', 0.99),
            pbrs_scale=pbrs_config.get('scale', 0.5),
            constant_weight=pbrs_config.get('constant_weight', True),
        )

    def select_action(obs, eps):
        """Select action with epsilon-greedy exploration."""
        if np.random.random() < eps:
            # Random action
            return np.random.uniform(-1, 1, action_dim)
        else:
            # Policy action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = actor(obs_tensor).squeeze(0).numpy()
            return action

    # Main worker loop
    while True:
        try:
            task = task_queue.get(timeout=60)
        except:
            continue

        if task is None or task[0] == 'shutdown':
            break

        task_type = task[0]

        if task_type == 'update_weights':
            # Update actor weights
            _, state_dict, eps = task
            actor.load_state_dict(state_dict)
            current_eps = eps

        elif task_type == 'run_episode':
            # Run a complete episode
            _, episode_num, seed, opponent_type = task

            # Select opponent
            if opponent_type == 'weak':
                opponent = weak_opponent
            elif opponent_type == 'strong':
                opponent = strong_opponent
            else:
                opponent = weak_opponent  # Default

            # Reset environment
            if seed is not None:
                np.random.seed(seed + episode_num)
                reset_seed = np.random.randint(0, 1000000)
            else:
                reset_seed = None

            obs, info = env.reset(seed=reset_seed)
            obs_agent2 = env.obs_agent_two()

            # Reset PBRS shaper
            if pbrs_shaper:
                pbrs_shaper.reset()

            # Episode data container
            episode_data = EpisodeData()
            puck_touches = 0

            for t in range(max_timesteps):
                # Store current observation
                obs_curr = obs.copy()

                # Get agent action
                action1 = select_action(obs, current_eps)

                # Get opponent action
                action2 = opponent.act(obs_agent2)

                # Combine actions
                action_combined = np.hstack([action1[:4], action2[:4]])

                # Step environment
                obs_next, r1, done, truncated, info = env.step(action_combined)

                # Scale reward
                r1_scaled = r1 * reward_scale
                episode_data.sparse_reward += r1_scaled

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

                # Store transition
                episode_data.transitions.append((
                    obs_curr,
                    action_combined.copy(),
                    r1_shaped,
                    obs_next.copy(),
                    float(done or truncated)
                ))

                # Track metrics
                episode_data.episode_reward += r1_shaped
                episode_data.action_magnitudes.append(np.linalg.norm(action1[:2]))
                episode_data.puck_distances.append(dist_to_puck)
                if len(obs_next) > 16:
                    episode_data.shoot_actions.append((action1[3], obs_next[16] > 0))
                else:
                    episode_data.shoot_actions.append((action1[3], False))
                episode_data.agent_positions.append([obs_next[0], obs_next[1]])

                # Update observations
                obs = obs_next
                obs_agent2 = env.obs_agent_two()

                if done or truncated:
                    episode_data.winner = info.get('winner', 0)
                    break

            # Episode complete
            episode_data.episode_length = len(episode_data.transitions)
            episode_data.puck_touches = puck_touches

            # Send completed episode data
            result_queue.put(('episode_done', worker_id, episode_num, episode_data))

        elif task_type == 'ping':
            result_queue.put(('pong', worker_id))

    # Cleanup
    env.close()


class ParallelHockeyEnv:
    """
    Manages multiple parallel hockey environment workers.

    Workers run episodes autonomously with periodic weight syncs from main process.
    This provides true parallelism without step-by-step communication overhead.

    Usage:
        parallel_env = ParallelHockeyEnv(
            num_workers=4,
            mode=mode,
            obs_dim=18,
            action_dim=4,
            hidden_actor=[256, 256],
            ...
        )

        # Sync weights before collection
        parallel_env.sync_weights(agent.policy.state_dict(), agent._eps)

        # Collect episodes in parallel
        episodes = parallel_env.collect_episodes(
            num_episodes=8,
            opponent_type='weak',
            episode_start=100,
            seed=42
        )

        # Add transitions to buffer
        for episode_data in episodes:
            for transition in episode_data.transitions:
                agent.buffer.add_transition(transition)
    """

    def __init__(
        self,
        num_workers: int,
        mode: int,
        keep_mode: bool,
        max_timesteps: int,
        obs_dim: int,
        action_dim: int,
        hidden_actor: List[int],
        reward_scale: float = 1.0,
        pbrs_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize parallel environment workers.

        Args:
            num_workers: Number of parallel workers
            mode: Hockey environment mode
            keep_mode: Whether keep mode is enabled
            max_timesteps: Maximum timesteps per episode
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_actor: Hidden layer sizes for actor network
            reward_scale: Scaling factor for sparse rewards
            pbrs_config: Configuration for PBRS reward shaping
        """
        self.num_workers = num_workers
        self.mode = mode
        self.keep_mode = keep_mode
        self.max_timesteps = max_timesteps
        self.reward_scale = reward_scale
        self.pbrs_config = pbrs_config or {'enabled': False}

        # Use 'spawn' method for multiprocessing (safer on all platforms)
        ctx = mp.get_context('spawn')

        # Create communication queues
        self.task_queues: List[mp.Queue] = []
        self.result_queue = ctx.Queue()
        self.workers: List[Process] = []

        # Start workers
        for i in range(num_workers):
            task_queue = ctx.Queue()
            self.task_queues.append(task_queue)

            p = ctx.Process(
                target=worker_process,
                args=(
                    i,
                    task_queue,
                    self.result_queue,
                    mode,
                    keep_mode,
                    max_timesteps,
                    reward_scale,
                    self.pbrs_config,
                    obs_dim,
                    action_dim,
                    hidden_actor,
                )
            )
            p.start()
            self.workers.append(p)

        print(f"[ParallelEnv] Started {num_workers} worker processes")

    def sync_weights(self, actor_state_dict: Dict, eps: float):
        """
        Synchronize actor weights to all workers.

        Call this before collecting episodes to ensure workers use current policy.

        Args:
            actor_state_dict: Actor network state dict (CPU tensors)
            eps: Current exploration epsilon
        """
        # Convert to CPU if needed
        cpu_state_dict = {k: v.cpu() for k, v in actor_state_dict.items()}

        for task_queue in self.task_queues:
            task_queue.put(('update_weights', cpu_state_dict, eps))

    def collect_episodes(
        self,
        num_episodes: int,
        opponent_type: str = 'weak',
        episode_start: int = 0,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, EpisodeData]]:
        """
        Collect multiple episodes in parallel.

        Workers run episodes autonomously using their local policy copy.
        Returns list of (episode_num, episode_data) tuples.

        Args:
            num_episodes: Number of episodes to collect
            opponent_type: Type of opponent ('weak' or 'strong')
            episode_start: Starting episode number
            seed: Base random seed

        Returns:
            List of (episode_num, EpisodeData) tuples
        """
        completed_episodes: List[Tuple[int, EpisodeData]] = []
        episodes_started = 0
        active_workers = set()

        # Start initial episodes on all workers
        for i in range(min(num_episodes, self.num_workers)):
            episode_num = episode_start + i
            self.task_queues[i].put((
                'run_episode',
                episode_num,
                seed,
                opponent_type
            ))
            active_workers.add(i)
            episodes_started += 1

        # Collect results and dispatch new episodes
        while len(completed_episodes) < num_episodes:
            try:
                result = self.result_queue.get(timeout=120)
            except:
                print("[ParallelEnv] Warning: Timeout waiting for worker result")
                break

            result_type = result[0]

            if result_type == 'episode_done':
                _, worker_id, episode_num, episode_data = result
                completed_episodes.append((episode_num, episode_data))

                # Start new episode if more needed
                if episodes_started < num_episodes:
                    next_episode_num = episode_start + episodes_started
                    self.task_queues[worker_id].put((
                        'run_episode',
                        next_episode_num,
                        seed,
                        opponent_type
                    ))
                    episodes_started += 1
                else:
                    active_workers.discard(worker_id)

        # Sort by episode number
        completed_episodes.sort(key=lambda x: x[0])
        return completed_episodes

    def close(self):
        """Shutdown all worker processes."""
        for task_queue in self.task_queues:
            task_queue.put(('shutdown',))

        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        print("[ParallelEnv] All workers shut down")


def create_parallel_env(args, mode: int, max_timesteps: int, obs_dim: int, action_dim: int) -> Optional[ParallelHockeyEnv]:
    """
    Create a parallel environment from command-line arguments.

    Args:
        args: Parsed command-line arguments
        mode: Hockey environment mode
        max_timesteps: Maximum timesteps per episode
        obs_dim: Observation dimension
        action_dim: Action dimension

    Returns:
        ParallelHockeyEnv if parallel mode enabled (parallel_envs > 1), None otherwise
    """
    if not getattr(args, 'parallel_envs', 1) > 1:
        return None

    pbrs_config = {
        'enabled': args.reward_shaping,
        'scale': args.pbrs_scale,
        'gamma': args.gamma,
        'components': ['distance', 'velocity', 'possession'],
        'constant_weight': args.pbrs_constant_weight,
    }

    return ParallelHockeyEnv(
        num_workers=args.parallel_envs,
        mode=mode,
        keep_mode=args.keep_mode,
        max_timesteps=max_timesteps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_actor=args.hidden_actor,
        reward_scale=args.reward_scale,
        pbrs_config=pbrs_config,
    )
