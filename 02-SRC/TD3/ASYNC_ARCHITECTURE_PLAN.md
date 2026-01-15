# Async Collection + Training Architecture Plan

## Problem Statement

The current parallel implementation achieves ~1.07 ep/s vs ~0.8 ep/s serial - only a **20-30% improvement** instead of the expected 2-4x. The reason: **collection and training are sequential, not overlapping**.

```
Current Flow (Blocking):

Time →
Workers:  [Collect 4 eps]  [IDLE........]  [Collect 4 eps]  [IDLE........]
Main:     [BLOCKING.......]  [Process+Train]  [BLOCKING.......]  [Process+Train]
GPU:      [IDLE............]  [Training....]  [IDLE............]  [Training....]

The GPU sits idle 50%+ of the time waiting for workers to finish collecting episodes!
```

## Target Architecture

```
Async Flow (Overlapping):

Time →
Workers:  [Collect] [Collect] [Collect] [Collect] [Collect] [Collect] ...
Buffer:   <── transitions constantly flowing in ────────────────────────>
GPU:              [Train] [Train] [Train] [Train] [Train] [Train] ...

Both collection and training run continuously in parallel.
Expected speedup: 1.5-2x overall throughput.
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MAIN PROCESS                                     │
│  - Orchestration, logging, evaluation, checkpointing                     │
│  - Spawns and monitors child processes                                   │
│  - Handles W&B logging and graceful shutdown                             │
└────────────────────────────────────────────────────────────────────────┬┘
                                                                          │
         ┌────────────────────────────────────────────────────────────────┴─┐
         │                                                                  │
         ▼                                                                  ▼
┌─────────────────────────┐                              ┌─────────────────────────┐
│   COLLECTOR PROCESS     │                              │   TRAINER PROCESS       │
│   (CPU-bound)           │                              │   (GPU-bound)           │
├─────────────────────────┤                              ├─────────────────────────┤
│ - Manages N workers     │                              │ - Owns TD3Agent         │
│ - Runs episodes         │                              │ - Continuous training   │
│ - Pushes transitions    │                              │ - Pulls from buffer     │
│ - Receives weight syncs │                              │ - Pushes weight updates │
└─────────┬───────────────┘                              └───────────┬─────────────┘
          │                                                          │
          │ transitions                                    batches   │
          │ (mp.Queue)                                     (read)    │
          ▼                                                          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SHARED REPLAY BUFFER                             │
    │  - Thread-safe queue for incoming transitions                       │
    │  - RLock-protected for sampling                                     │
    │  - Supports PER (Prioritized Experience Replay)                     │
    │  - Dual buffer support (anchor + pool)                              │
    └─────────────────────────────────────────────────────────────────────┘
          ▲                                                          ▲
          │                                                          │
          │ weights                                      priority    │
          │ (mp.Queue)                                   updates     │
          │                                              (mp.Queue)  │
┌─────────┴───────────────┐                              ┌───────────┴─────────────┐
│   WEIGHT SYNC QUEUE     │                              │  PRIORITY UPDATE QUEUE  │
│   (Trainer → Collector) │                              │   (Trainer → Buffer)    │
└─────────────────────────┘                              └─────────────────────────┘
```

---

## Component Specifications

### 1. AsyncReplayBuffer (New Class)

**Location**: `02-SRC/TD3/agents/async_memory.py`

**Purpose**: Thread-safe replay buffer that supports concurrent reads (sampling) and writes (adding transitions).

```python
class AsyncReplayBuffer:
    """
    Thread-safe replay buffer for async training.

    Design:
    - Uses RLock to allow same thread to acquire lock multiple times
    - Separates add_queue (producer) from internal storage (consumer)
    - Batches incoming transitions to reduce lock contention
    - Supports both standard and prioritized sampling
    """

    def __init__(self, max_size, alpha=0.6, use_per=False):
        self.max_size = max_size
        self.use_per = use_per

        # Internal storage (protected by lock)
        self._lock = threading.RLock()
        self._transitions = np.empty(max_size, dtype=object)
        self._size = 0
        self._idx = 0

        # PER components (if enabled)
        if use_per:
            self._priorities = np.zeros(max_size, dtype=np.float32)
            self._max_priority = 1.0
            self._alpha = alpha

        # Incoming queue (lock-free for producers)
        self._add_queue = mp.Queue(maxsize=10000)

        # Background thread to drain queue into storage
        self._drain_thread = None
        self._running = False

    def start(self):
        """Start background drain thread."""
        self._running = True
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def stop(self):
        """Stop background drain thread."""
        self._running = False
        if self._drain_thread:
            self._drain_thread.join(timeout=5.0)

    def add_transition(self, transition):
        """
        Add transition (non-blocking).
        Called by collector process - puts on queue, returns immediately.
        """
        try:
            self._add_queue.put_nowait(transition)
        except queue.Full:
            # Buffer overwhelmed - drop oldest in queue
            try:
                self._add_queue.get_nowait()
                self._add_queue.put_nowait(transition)
            except:
                pass  # Best effort

    def _drain_loop(self):
        """Background thread: drain queue into storage."""
        batch = []
        while self._running:
            try:
                # Batch up to 100 transitions
                while len(batch) < 100:
                    transition = self._add_queue.get(timeout=0.01)
                    batch.append(transition)
            except queue.Empty:
                pass

            if batch:
                with self._lock:
                    for t in batch:
                        self._add_single(t)
                batch.clear()

    def _add_single(self, transition):
        """Add single transition (must hold lock)."""
        self._transitions[self._idx] = transition
        if self.use_per:
            self._priorities[self._idx] = self._max_priority ** self._alpha
        self._idx = (self._idx + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample batch (blocking if needed).
        Called by trainer process.
        """
        with self._lock:
            if self._size < batch_size:
                return None

            if self.use_per:
                return self._sample_prioritized(batch_size)
            else:
                indices = np.random.randint(0, self._size, size=batch_size)
                return self._transitions[indices]

    def update_priorities(self, indices, priorities):
        """Update priorities for PER (called by trainer)."""
        if not self.use_per:
            return
        with self._lock:
            for idx, priority in zip(indices, priorities):
                self._priorities[idx] = (priority + 1e-6) ** self._alpha
                self._max_priority = max(self._max_priority, priority)

    @property
    def size(self):
        with self._lock:
            return self._size
```

**Key Design Decisions**:
- **Queue-based adds**: Collectors push to lock-free queue, background thread drains
- **Lock-protected storage**: Only one thread modifies the array at a time
- **Batched draining**: Reduces lock acquisition overhead (100 transitions per lock)
- **Non-blocking adds**: Collectors never wait; if queue full, drop oldest

---

### 2. CollectorProcess (New Class)

**Location**: `02-SRC/TD3/parallel_env.py` (extend existing)

**Purpose**: Background process that continuously runs episodes and feeds transitions to the buffer.

```python
class CollectorProcess:
    """
    Continuously collects episodes in background.

    Runs as separate process to avoid Python GIL.
    Communicates via multiprocessing queues.
    """

    def __init__(self, num_workers, env_config, buffer_queue, weight_queue, stats_queue):
        self.num_workers = num_workers
        self.env_config = env_config

        # Communication queues
        self.buffer_queue = buffer_queue      # Outgoing transitions
        self.weight_queue = weight_queue      # Incoming weight updates
        self.stats_queue = stats_queue        # Outgoing episode stats

        # Process handle
        self._process = None
        self._running = mp.Event()

    def start(self):
        """Start collector process."""
        self._running.set()
        self._process = mp.Process(
            target=self._collector_main,
            args=(
                self.num_workers,
                self.env_config,
                self.buffer_queue,
                self.weight_queue,
                self.stats_queue,
                self._running
            ),
            daemon=True
        )
        self._process.start()

    def stop(self):
        """Stop collector process."""
        self._running.clear()
        if self._process:
            self._process.join(timeout=10.0)
            if self._process.is_alive():
                self._process.terminate()

    def send_weights(self, state_dict, eps):
        """Send weight update to collector (non-blocking)."""
        try:
            # Clear old weights first
            while not self.weight_queue.empty():
                try:
                    self.weight_queue.get_nowait()
                except:
                    break
            self.weight_queue.put_nowait((state_dict, eps))
        except:
            pass

    @staticmethod
    def _collector_main(num_workers, env_config, buffer_queue, weight_queue, stats_queue, running_event):
        """
        Main loop for collector process.
        Spawns workers, receives episodes, pushes transitions.
        """
        import torch
        from agents.model import Model

        # Create lightweight actor for workers
        actor = Model(
            input_dim=env_config['obs_dim'],
            output_dim=env_config['action_dim'],
            hidden_layers=env_config['hidden_actor']
        )
        current_eps = 1.0

        # Create worker pool (reuse existing ParallelEnv worker logic)
        workers = []
        task_queues = []
        result_queue = mp.Queue()

        for i in range(num_workers):
            task_q = mp.Queue()
            worker = mp.Process(
                target=episode_worker,  # Existing worker function
                args=(i, task_q, result_queue, env_config, actor.state_dict(), current_eps)
            )
            worker.start()
            workers.append(worker)
            task_queues.append(task_q)

        episode_counter = 0
        active_tasks = 0

        # Dispatch initial tasks
        for i in range(num_workers):
            task_queues[i].put(('run_episode', env_config['opponent'], episode_counter + i, None))
            active_tasks += 1
        episode_counter += num_workers

        # Main collection loop
        while running_event.is_set():
            # Check for weight updates (non-blocking)
            try:
                state_dict, eps = weight_queue.get_nowait()
                current_eps = eps
                # Broadcast to all workers
                for tq in task_queues:
                    tq.put(('update_weights', state_dict, eps))
            except queue.Empty:
                pass

            # Collect completed episodes
            try:
                result = result_queue.get(timeout=0.1)
                if result[0] == 'episode_done':
                    _, ep_num, transitions, metrics, winner = result

                    # Push transitions to buffer
                    for t in transitions:
                        buffer_queue.put(t)

                    # Push stats
                    stats_queue.put((ep_num, metrics, winner))

                    # Dispatch new task to idle worker
                    worker_id = result[1] if len(result) > 5 else ep_num % num_workers
                    task_queues[worker_id % num_workers].put(
                        ('run_episode', env_config['opponent'], episode_counter, None)
                    )
                    episode_counter += 1

            except queue.Empty:
                continue

        # Cleanup
        for tq in task_queues:
            tq.put(('shutdown', None, None))
        for w in workers:
            w.join(timeout=5.0)
            if w.is_alive():
                w.terminate()
```

**Key Design Decisions**:
- **Separate process**: Avoids Python GIL, true parallelism
- **Continuous loop**: Workers always running, no blocking waits
- **Non-blocking weight updates**: Collectors check for updates but don't wait
- **Push-based transitions**: Each transition immediately pushed to buffer queue

---

### 3. TrainerProcess (New Class)

**Location**: `02-SRC/TD3/agents/async_trainer.py` (new file)

**Purpose**: Continuous training loop that samples from buffer and updates networks.

```python
class TrainerProcess:
    """
    Continuous training process.

    Samples from shared buffer, updates networks, broadcasts weights.
    Runs on GPU (or CPU if unavailable).
    """

    def __init__(self, agent_config, buffer, weight_broadcast_queue,
                 priority_queue, checkpoint_queue, metrics_queue):
        self.agent_config = agent_config
        self.buffer = buffer  # AsyncReplayBuffer

        # Communication
        self.weight_broadcast_queue = weight_broadcast_queue
        self.priority_queue = priority_queue
        self.checkpoint_queue = checkpoint_queue
        self.metrics_queue = metrics_queue

        # Process control
        self._process = None
        self._running = mp.Event()

        # Training parameters
        self.weight_sync_interval = 10  # Sync weights every N training steps
        self.checkpoint_interval = 1000

    def start(self):
        """Start trainer process."""
        self._running.set()
        self._process = mp.Process(
            target=self._trainer_main,
            args=(
                self.agent_config,
                self.buffer,
                self.weight_broadcast_queue,
                self.priority_queue,
                self.checkpoint_queue,
                self.metrics_queue,
                self._running,
                self.weight_sync_interval,
                self.checkpoint_interval
            ),
            daemon=True
        )
        self._process.start()

    def stop(self):
        """Stop trainer process."""
        self._running.clear()
        if self._process:
            self._process.join(timeout=30.0)
            if self._process.is_alive():
                self._process.terminate()

    @staticmethod
    def _trainer_main(agent_config, buffer, weight_queue, priority_queue,
                      checkpoint_queue, metrics_queue, running_event,
                      weight_sync_interval, checkpoint_interval):
        """
        Main training loop.
        """
        import torch
        from agents.td3_agent import TD3Agent

        # Create agent with GPU
        agent = TD3Agent(**agent_config)

        train_step = 0
        warmup_steps = agent_config.get('warmup_steps', 10000)
        batch_size = agent_config.get('batch_size', 256)

        # Wait for buffer to have enough samples
        while buffer.size < warmup_steps and running_event.is_set():
            time.sleep(0.5)

        print(f"[Trainer] Warmup complete, starting training (buffer size: {buffer.size})")

        # Main training loop
        while running_event.is_set():
            # Sample batch
            batch = buffer.sample(batch_size)
            if batch is None:
                time.sleep(0.01)
                continue

            # Train step
            if buffer.use_per:
                data, indices, is_weights = batch
                losses, grad_norms, _ = agent.train_step(data, is_weights)

                # Compute new priorities (TD errors)
                td_errors = agent.compute_td_errors(data)
                priority_queue.put((indices, td_errors))
            else:
                losses, grad_norms, _ = agent.train_step(batch)

            train_step += 1

            # Broadcast weights periodically
            if train_step % weight_sync_interval == 0:
                state_dict = {k: v.cpu() for k, v in agent.policy.state_dict().items()}
                eps = agent._eps
                weight_queue.put((state_dict, eps))

            # Send metrics
            if train_step % 10 == 0:
                metrics_queue.put({
                    'train_step': train_step,
                    'critic_loss': losses.get('critic_loss', 0),
                    'actor_loss': losses.get('actor_loss', 0),
                    'q_value_mean': losses.get('q_mean', 0),
                })

            # Checkpoint
            if train_step % checkpoint_interval == 0:
                checkpoint_queue.put({
                    'train_step': train_step,
                    'agent_state': agent.state(),
                    'buffer_size': buffer.size
                })

        # Final checkpoint
        checkpoint_queue.put({
            'train_step': train_step,
            'agent_state': agent.state(),
            'buffer_size': buffer.size,
            'final': True
        })
```

**Key Design Decisions**:
- **Continuous loop**: No waiting for episodes, just continuous sampling and training
- **Periodic weight sync**: Every N steps, not every episode
- **Async metrics**: Push metrics to queue, main process handles logging
- **Checkpoint queue**: Trainer doesn't save directly, signals main process

---

### 4. AsyncTrainingOrchestrator (Main Integration)

**Location**: `02-SRC/TD3/async_training.py` (new file)

**Purpose**: Main process that orchestrates collector, trainer, logging, and evaluation.

```python
class AsyncTrainingOrchestrator:
    """
    Main orchestrator for async training.

    Responsibilities:
    - Spawn and manage collector/trainer processes
    - Handle W&B logging
    - Run periodic evaluations
    - Save checkpoints
    - Graceful shutdown
    """

    def __init__(self, args):
        self.args = args

        # Shared components
        self.buffer = AsyncReplayBuffer(
            max_size=args.buffer_size,
            alpha=args.per_alpha if args.use_per else 0.6,
            use_per=args.use_per
        )

        # Communication queues
        self.transition_queue = mp.Queue(maxsize=50000)
        self.weight_queue = mp.Queue(maxsize=10)
        self.stats_queue = mp.Queue(maxsize=1000)
        self.priority_queue = mp.Queue(maxsize=1000)
        self.checkpoint_queue = mp.Queue(maxsize=10)
        self.metrics_queue = mp.Queue(maxsize=1000)

        # Components
        self.collector = None
        self.trainer = None

        # State
        self.episode_count = 0
        self.train_steps = 0
        self.start_time = None

    def start(self):
        """Start all components."""
        self.start_time = time.time()

        # Start buffer drain thread
        self.buffer.start()

        # Start buffer filler (drains transition_queue into buffer)
        self._start_buffer_filler()

        # Start collector
        self.collector = CollectorProcess(
            num_workers=self.args.num_workers,
            env_config=self._make_env_config(),
            buffer_queue=self.transition_queue,
            weight_queue=self.weight_queue,
            stats_queue=self.stats_queue
        )
        self.collector.start()

        # Start trainer
        self.trainer = TrainerProcess(
            agent_config=self._make_agent_config(),
            buffer=self.buffer,
            weight_broadcast_queue=self.weight_queue,
            priority_queue=self.priority_queue,
            checkpoint_queue=self.checkpoint_queue,
            metrics_queue=self.metrics_queue
        )
        self.trainer.start()

    def _start_buffer_filler(self):
        """Background thread to move transitions from queue to buffer."""
        def filler_loop():
            while self._running:
                try:
                    transition = self.transition_queue.get(timeout=0.1)
                    self.buffer.add_transition(transition)
                except queue.Empty:
                    continue

        self._running = True
        self._filler_thread = threading.Thread(target=filler_loop, daemon=True)
        self._filler_thread.start()

    def run(self):
        """Main orchestration loop."""
        try:
            self.start()

            last_log_time = time.time()
            last_eval_time = time.time()

            while self.episode_count < self.args.max_episodes:
                # Process episode stats
                self._process_stats()

                # Process training metrics
                self._process_metrics()

                # Process priority updates (for PER)
                self._process_priorities()

                # Process checkpoints
                self._process_checkpoints()

                # Periodic logging
                if time.time() - last_log_time > 10:  # Every 10 seconds
                    self._log_progress()
                    last_log_time = time.time()

                # Periodic evaluation
                if time.time() - last_eval_time > self.args.eval_interval_seconds:
                    self._run_evaluation()
                    last_eval_time = time.time()

                time.sleep(0.1)  # Don't spin too fast

        except KeyboardInterrupt:
            print("\n[Orchestrator] Received interrupt, shutting down...")
        finally:
            self.stop()

    def _process_stats(self):
        """Process episode completion stats."""
        while True:
            try:
                ep_num, metrics, winner = self.stats_queue.get_nowait()
                self.episode_count += 1

                # Track metrics
                self._update_metrics(metrics, winner)

            except queue.Empty:
                break

    def _process_metrics(self):
        """Process training metrics from trainer."""
        while True:
            try:
                metrics = self.metrics_queue.get_nowait()
                self.train_steps = metrics.get('train_step', self.train_steps)

                # Log to W&B if enabled
                if self.args.use_wandb:
                    wandb.log(metrics, step=self.train_steps)

            except queue.Empty:
                break

    def _process_checkpoints(self):
        """Process checkpoint requests from trainer."""
        while True:
            try:
                checkpoint = self.checkpoint_queue.get_nowait()
                self._save_checkpoint(checkpoint)
            except queue.Empty:
                break

    def _log_progress(self):
        """Log current progress."""
        elapsed = time.time() - self.start_time
        eps_per_sec = self.episode_count / elapsed if elapsed > 0 else 0

        print(f"[Progress] Episodes: {self.episode_count}, "
              f"Train steps: {self.train_steps}, "
              f"Buffer: {self.buffer.size}, "
              f"Speed: {eps_per_sec:.2f} ep/s")

    def stop(self):
        """Stop all components."""
        print("[Orchestrator] Stopping components...")

        self._running = False

        if self.collector:
            self.collector.stop()

        if self.trainer:
            self.trainer.stop()

        self.buffer.stop()

        print("[Orchestrator] Shutdown complete")
```

---

## Implementation Steps

### Phase 1: AsyncReplayBuffer (Estimated: foundation)
1. Create `agents/async_memory.py` with `AsyncReplayBuffer` class
2. Implement lock-protected storage with queue-based adds
3. Add PER support with thread-safe priority updates
4. Write unit tests for concurrent access patterns
5. Benchmark: verify add/sample throughput under load

### Phase 2: CollectorProcess (Estimated: core parallelism)
1. Refactor `parallel_env.py` to support continuous collection mode
2. Create `CollectorProcess` wrapper class
3. Implement non-blocking weight synchronization
4. Add episode stats reporting via queue
5. Test: verify continuous episode generation

### Phase 3: TrainerProcess (Estimated: training loop)
1. Create `agents/async_trainer.py` with `TrainerProcess` class
2. Implement continuous training loop
3. Add periodic weight broadcasting
4. Implement checkpoint signaling
5. Test: verify training converges with async buffer

### Phase 4: Orchestrator Integration (Estimated: glue code)
1. Create `async_training.py` with `AsyncTrainingOrchestrator`
2. Implement queue-based communication
3. Add W&B logging integration
4. Implement evaluation (pause training or use separate evaluator)
5. Add graceful shutdown handling

### Phase 5: Integration & Testing (Estimated: validation)
1. Create `train_hockey_async.py` entry point
2. Add CLI flags: `--async`, `--num_workers`, `--weight_sync_interval`
3. Run convergence tests vs serial training
4. Profile and optimize bottlenecks
5. Document usage and performance characteristics

---

## Risk Mitigation

### 1. Buffer Corruption
**Risk**: Race conditions corrupt buffer state
**Mitigation**:
- Use RLock for all buffer access
- Batch writes to reduce lock contention
- Add integrity checks in debug mode

### 2. Stale Weights
**Risk**: Workers use old policy for too long
**Mitigation**:
- Tune `weight_sync_interval` (start with 10, adjust based on results)
- Monitor policy lag metrics
- Add staleness threshold that forces immediate sync

### 3. Deadlocks
**Risk**: Processes wait for each other forever
**Mitigation**:
- Use timeouts on all queue operations
- Implement watchdog that detects hung processes
- Add graceful degradation (fall back to serial if async fails)

### 4. Memory Pressure
**Risk**: Queues grow unbounded
**Mitigation**:
- Set maxsize on all queues
- Implement back-pressure (slow down producer if consumer overwhelmed)
- Monitor queue depths

### 5. Observation Normalization Drift
**Risk**: Collector and trainer have different normalization stats
**Mitigation**:
- Option A: Freeze normalization after warmup
- Option B: Include normalization stats in weight sync
- Option C: Compute normalization from buffer (not running stats)

---

## Expected Performance

| Metric | Current (Blocking) | Expected (Async) |
|--------|-------------------|------------------|
| Episodes/sec | 1.07 | 2.5-3.5 |
| GPU utilization | ~30-40% | ~80-90% |
| Time to 100K episodes | ~26 hours | ~10-12 hours |
| Code complexity | Low | Medium-High |

**Conservative estimate**: 2x speedup
**Optimistic estimate**: 3x speedup (with tuned parameters)

---

## Fallback Plan

If async implementation proves too complex or unstable:

1. **Simpler improvement**: Increase batch size and reduce sync frequency
   - Collect 16-32 episodes per batch (not 4-8)
   - Train more iterations per batch
   - Expected: 30-50% improvement over current

2. **Hybrid approach**: Async collection, sync training
   - Workers continuously push to buffer (async)
   - Training happens in main thread when buffer has enough
   - Simpler than full async, still improves throughput

3. **Process-based parallelism only**:
   - Multiple independent training runs (different seeds)
   - Aggregate best models
   - No code changes, just run multiple jobs

---

## Configuration Parameters

New arguments for `train_hockey.py`:

```python
# Async training options
parser.add_argument('--async_training', action='store_true',
                    help='Enable fully async collection + training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of parallel collection workers')
parser.add_argument('--weight_sync_interval', type=int, default=10,
                    help='Sync weights every N training steps')
parser.add_argument('--buffer_queue_size', type=int, default=50000,
                    help='Max size of transition queue')
parser.add_argument('--warmup_transitions', type=int, default=10000,
                    help='Transitions to collect before training starts')
```

---

## Testing Strategy

### Unit Tests
1. `test_async_buffer.py`: Concurrent add/sample correctness
2. `test_collector_process.py`: Episode generation, weight sync
3. `test_trainer_process.py`: Training convergence, checkpoint handling

### Integration Tests
1. Short training run (1000 episodes): verify no crashes
2. Compare learning curves: async vs serial (same seed)
3. Stress test: high worker count, small buffer

### Performance Tests
1. Throughput benchmark: episodes/second vs worker count
2. GPU utilization measurement
3. Memory usage profiling

---

## Next Steps

1. **Approve this plan** or request modifications
2. **Implement Phase 1** (AsyncReplayBuffer) first
3. **Test buffer independently** before integrating
4. **Proceed incrementally** through remaining phases
