"""
Async-safe Replay Buffer for parallel training.

Thread-safe implementation supporting concurrent reads (sampling) and writes (adding transitions).
Uses a queue-based producer pattern with lock-protected internal storage.
"""

import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, List, Any
import multiprocessing as mp


class SumTreeAsync:
    """
    Thread-safe Sum tree for prioritized sampling.
    Uses fine-grained locking to allow concurrent operations.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0
        self._lock = threading.RLock()

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree (must hold lock)."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node given a value s (must hold lock)."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total priority sum (thread-safe)."""
        with self._lock:
            return self.tree[0]

    def add(self, priority: float, data: Any):
        """Add data with given priority (thread-safe)."""
        with self._lock:
            idx = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer] = data
            self._update_internal(idx, priority)
            self.data_pointer = (self.data_pointer + 1) % self.capacity
            self.n_entries = min(self.n_entries + 1, self.capacity)

    def _update_internal(self, idx: int, priority: float):
        """Update priority (must hold lock)."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def update(self, idx: int, priority: float):
        """Update priority of existing node (thread-safe)."""
        with self._lock:
            self._update_internal(idx, priority)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get data, priority, and tree index for a value s (thread-safe)."""
        with self._lock:
            idx = self._retrieve(0, s)
            data_idx = idx - self.capacity + 1
            return idx, self.tree[idx], self.data[data_idx]

    @property
    def size(self) -> int:
        """Get current number of entries (thread-safe)."""
        with self._lock:
            return self.n_entries


class AsyncPrioritizedMemory:
    """
    Thread-safe Prioritized Experience Replay (PER) buffer.

    Design:
    - Uses queue for incoming transitions (non-blocking for producers)
    - Background thread drains queue into storage
    - RLock protects sampling to allow re-entrant locking
    - Supports importance sampling weight computation
    """

    def __init__(
        self,
        max_size: int = 500000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-5,
        batch_drain_size: int = 100,
    ):
        """
        Args:
            max_size: Maximum buffer capacity
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial IS correction factor (anneals to 1.0)
            beta_frames: Frames over which to anneal beta
            epsilon: Small constant for non-zero priority
            batch_drain_size: Number of transitions to drain per batch
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.batch_drain_size = batch_drain_size

        # Thread-safe storage
        self.tree = SumTreeAsync(max_size)
        self.max_priority = 1.0
        self.frame = 0

        # Lock for non-tree operations
        self._lock = threading.RLock()

        # Incoming transition queue (producers push here)
        # Note: This is a threading.Queue (not mp.Queue) so larger sizes are fine
        self._add_queue = queue.Queue(maxsize=10000)

        # Background drain thread
        self._running = False
        self._drain_thread = None

        # Statistics
        self._transitions_added = 0
        self._transitions_dropped = 0

    def start(self):
        """Start the background drain thread."""
        if self._running:
            return
        self._running = True
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def stop(self):
        """Stop the background drain thread."""
        self._running = False
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=5.0)
            self._drain_thread = None

    def add_transition(self, transition: Tuple):
        """
        Add a transition to the buffer (non-blocking).

        Called by collector process/thread. Puts on queue and returns immediately.
        If queue is full, drops the oldest item.
        """
        try:
            self._add_queue.put_nowait(transition)
        except queue.Full:
            # Buffer overwhelmed - drop oldest in queue
            try:
                self._add_queue.get_nowait()
                self._add_queue.put_nowait(transition)
                self._transitions_dropped += 1
            except:
                pass  # Best effort

    def add_transition_direct(self, transition: Tuple):
        """
        Add a transition directly to storage (blocking).

        Use this for single-threaded operation or when you need
        immediate addition without going through the queue.
        """
        with self._lock:
            priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)
        with self._lock:
            self._transitions_added += 1

    def _drain_loop(self):
        """Background thread: drain queue into storage."""
        batch = []
        while self._running:
            try:
                # Batch up transitions
                while len(batch) < self.batch_drain_size:
                    try:
                        transition = self._add_queue.get(timeout=0.01)
                        batch.append(transition)
                    except queue.Empty:
                        break

                if batch:
                    # Add all batched transitions
                    with self._lock:
                        priority = (self.max_priority + self.epsilon) ** self.alpha

                    for t in batch:
                        self.tree.add(priority, t)

                    with self._lock:
                        self._transitions_added += len(batch)

                    batch.clear()
                else:
                    # No transitions available, sleep briefly
                    time.sleep(0.001)

            except Exception as e:
                print(f"[AsyncPrioritizedMemory] Drain error: {e}")
                batch.clear()
                time.sleep(0.1)

    def _get_beta(self) -> float:
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        with self._lock:
            beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame / self.beta_frames)
            return beta

    def sample(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample a batch of transitions with priority-based probability.

        Returns:
            If buffer has enough samples:
                (transitions, indices, is_weights)
            Otherwise:
                None
        """
        current_size = self.tree.size
        if current_size < batch_size:
            return None

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)
        transitions = []

        # Get total priority and compute segment size
        total = self.tree.total()
        if total <= 0:
            return None

        segment = total / batch_size

        # Get beta for IS weights
        beta = self._get_beta()
        with self._lock:
            self.frame += 1

        # Stratified sampling
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            # Handle edge case where priority is 0
            if priority == 0:
                priority = self.epsilon

            priorities[i] = priority
            indices[i] = idx

            # Convert stored data to list format
            if isinstance(data, tuple):
                transitions.append(list(data))
            else:
                transitions.append(data)

        # Compute importance sampling weights
        sampling_probs = priorities / total
        is_weights = np.power(current_size * sampling_probs, -beta)
        is_weights /= is_weights.max()  # Normalize by max weight

        return np.asarray(transitions, dtype=object), indices, is_weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Args:
            indices: Tree indices from sample()
            td_errors: Absolute TD errors from training
        """
        with self._lock:
            for idx, td_error in zip(indices, td_errors):
                priority = (np.abs(td_error) + self.epsilon) ** self.alpha
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, np.abs(td_error) + self.epsilon)

    def __len__(self) -> int:
        return self.tree.size

    @property
    def size(self) -> int:
        return self.tree.size

    def get_stats(self) -> dict:
        """Get buffer statistics for monitoring."""
        with self._lock:
            return {
                'beta': self._get_beta(),
                'max_priority': self.max_priority,
                'frame': self.frame,
                'total_priority': self.tree.total(),
                'n_entries': self.tree.size,
                'alpha': self.alpha,
                'transitions_added': self._transitions_added,
                'transitions_dropped': self._transitions_dropped,
                'queue_size': self._add_queue.qsize(),
            }


class AsyncMemory:
    """
    Thread-safe uniform replay buffer.

    Simpler version without prioritization for faster operation.
    """

    def __init__(self, max_size: int = 500000, batch_drain_size: int = 100):
        self.max_size = max_size
        self.batch_drain_size = batch_drain_size

        # Storage
        self._lock = threading.RLock()
        self._transitions = np.empty(max_size, dtype=object)
        self._size = 0
        self._idx = 0
        self._initialized = False

        # Incoming queue
        self._add_queue = queue.Queue(maxsize=10000)

        # Background thread
        self._running = False
        self._drain_thread = None

        # Statistics
        self._transitions_added = 0
        self._transitions_dropped = 0

    def start(self):
        """Start the background drain thread."""
        if self._running:
            return
        self._running = True
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def stop(self):
        """Stop the background drain thread."""
        self._running = False
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=5.0)
            self._drain_thread = None

    def add_transition(self, transition: Tuple):
        """Add a transition (non-blocking)."""
        try:
            self._add_queue.put_nowait(transition)
        except queue.Full:
            try:
                self._add_queue.get_nowait()
                self._add_queue.put_nowait(transition)
                self._transitions_dropped += 1
            except:
                pass

    def add_transition_direct(self, transition: Tuple):
        """Add a transition directly (blocking)."""
        with self._lock:
            if not self._initialized:
                # Initialize with proper shape
                self._transitions = np.empty(self.max_size, dtype=object)
                self._initialized = True

            self._transitions[self._idx] = transition
            self._idx = (self._idx + 1) % self.max_size
            self._size = min(self._size + 1, self.max_size)
            self._transitions_added += 1

    def _drain_loop(self):
        """Background thread: drain queue into storage."""
        batch = []
        while self._running:
            try:
                # Batch up transitions
                while len(batch) < self.batch_drain_size:
                    try:
                        transition = self._add_queue.get(timeout=0.01)
                        batch.append(transition)
                    except queue.Empty:
                        break

                if batch:
                    with self._lock:
                        if not self._initialized:
                            self._transitions = np.empty(self.max_size, dtype=object)
                            self._initialized = True

                        for t in batch:
                            self._transitions[self._idx] = t
                            self._idx = (self._idx + 1) % self.max_size
                            self._size = min(self._size + 1, self.max_size)

                        self._transitions_added += len(batch)

                    batch.clear()
                else:
                    time.sleep(0.001)

            except Exception as e:
                print(f"[AsyncMemory] Drain error: {e}")
                batch.clear()
                time.sleep(0.1)

    def sample(self, batch_size: int) -> Optional[np.ndarray]:
        """
        Sample a batch uniformly.

        Returns data in same format as original Memory class:
        2D object array where data[:, i] gives the i-th component of all transitions.
        """
        with self._lock:
            if self._size < batch_size:
                return None

            indices = np.random.randint(0, self._size, size=batch_size)
            sampled = self._transitions[indices]

            # Convert to 2D format: list of lists -> 2D array
            # This makes data[:, 0] work as expected
            result = []
            for t in sampled:
                if isinstance(t, (tuple, list)):
                    result.append(list(t))
                else:
                    result.append(t)
            return np.asarray(result, dtype=object)

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                'n_entries': self._size,
                'transitions_added': self._transitions_added,
                'transitions_dropped': self._transitions_dropped,
                'queue_size': self._add_queue.qsize(),
            }


def create_async_buffer(
    use_per: bool = False,
    max_size: int = 500000,
    alpha: float = 0.6,
    beta_start: float = 0.4,
    beta_frames: int = 100000,
):
    """
    Factory function to create appropriate async buffer.

    Args:
        use_per: Whether to use Prioritized Experience Replay
        max_size: Buffer capacity
        alpha: PER priority exponent
        beta_start: PER initial IS correction
        beta_frames: PER beta annealing frames

    Returns:
        AsyncPrioritizedMemory or AsyncMemory instance
    """
    if use_per:
        return AsyncPrioritizedMemory(
            max_size=max_size,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
        )
    else:
        return AsyncMemory(max_size=max_size)
