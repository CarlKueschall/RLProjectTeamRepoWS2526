# DreamerV3 + Self-Play Implementation Plan

**Authors:** Serhat Alpay, Carl Kueschall
**Date:** January 2025
**Status:** Planning Phase

---

## Executive Summary

This document outlines the phased implementation of DreamerV3 for the laser-hockey-gym environment, based on the successful approach from the Robot Air Hockey Challenge 2023 (2nd place, best pure learning-based approach).

**Key Configuration (from winning paper):**
- Algorithm: DreamerV3 (world model + actor + critic)
- Rewards: **Sparse only** (+2 goal, -1 conceded, -1/3 fault)
- Imagination horizon: 50 timesteps
- Self-play: Opponent pool expanding from 1 baseline to 25 agents
- Training: 100M simulation steps

---

## Phase 0: Repository Cleanup

### Objective
Archive all TD3-specific code to `99-ARCHIVE/` before starting DreamerV3 implementation. This keeps the repository clean and preserves all historical work.

### Files to ARCHIVE (move to 99-ARCHIVE/TD3/)

#### Core TD3 Implementation
```
02-SRC/TD3/agents/td3_agent.py          → 99-ARCHIVE/TD3/agents/td3_agent.py
02-SRC/TD3/agents/model.py              → 99-ARCHIVE/TD3/agents/model.py
02-SRC/TD3/agents/memory.py             → 99-ARCHIVE/TD3/agents/memory.py
02-SRC/TD3/agents/noise.py              → 99-ARCHIVE/TD3/agents/noise.py
02-SRC/TD3/agents/device.py             → 99-ARCHIVE/TD3/agents/device.py
02-SRC/TD3/agents/__init__.py           → 99-ARCHIVE/TD3/agents/__init__.py
```

#### PBRS Reward Shaping (No longer needed)
```
02-SRC/TD3/rewards/pbrs.py              → 99-ARCHIVE/TD3/rewards/pbrs.py
02-SRC/TD3/rewards/base.py              → 99-ARCHIVE/TD3/rewards/base.py
02-SRC/TD3/rewards/__init__.py          → 99-ARCHIVE/TD3/rewards/__init__.py
```

#### Training Scripts
```
02-SRC/TD3/train_hockey.py              → 99-ARCHIVE/TD3/train_hockey.py
02-SRC/TD3/config/parser.py             → 99-ARCHIVE/TD3/config/parser.py
02-SRC/TD3/config/__init__.py           → 99-ARCHIVE/TD3/config/__init__.py
```

#### All sbatch Files
```
02-SRC/TD3/train_hockey_*.sbatch        → 99-ARCHIVE/TD3/sbatch/
Root level: train_hockey*.sbatch        → 99-ARCHIVE/TD3/sbatch/
```

#### Analysis and Debug Files
```
02-SRC/TD3/ANALYSIS_*.md                → 99-ARCHIVE/TD3/analysis/
02-SRC/TD3/*_GUIDE.md                   → 99-ARCHIVE/TD3/guides/
02-SRC/TD3/*_PLAN.md                    → 99-ARCHIVE/TD3/plans/
02-SRC/TD3/PBRS_*.md                    → 99-ARCHIVE/TD3/pbrs_docs/
02-SRC/TD3/PERFORMANCE_REPORT_*.md      → 99-ARCHIVE/TD3/reports/
02-SRC/TD3/REWARD_SHAPING_*.md          → 99-ARCHIVE/TD3/docs/
```

#### W&B Exports and Logs
```
02-SRC/TD3/wandb_run_*.txt              → 99-ARCHIVE/TD3/wandb_exports/
02-SRC/TD3/training.log                 → 99-ARCHIVE/TD3/logs/
02-SRC/TD3/worker_*.log                 → 99-ARCHIVE/TD3/logs/
02-SRC/TD3/wandb/                       → 99-ARCHIVE/TD3/wandb/
02-SRC/TD3/wandb_analysis/              → 99-ARCHIVE/TD3/wandb_analysis/
```

#### Saved Checkpoints
```
02-SRC/TD3/results_checkpoints_*.pth    → 99-ARCHIVE/TD3/checkpoints/
02-SRC/TD3/results/                     → 99-ARCHIVE/TD3/results/
```

#### Test Videos
```
02-SRC/TD3/test_gameplay_*.mp4          → 99-ARCHIVE/TD3/videos/
```

#### Tests (TD3-specific)
```
02-SRC/TD3/tests/test_pbrs_potential.py → 99-ARCHIVE/TD3/tests/
```

### Files to KEEP (Reusable Infrastructure)

#### Self-Play System (Adapt for DreamerV3)
```
02-SRC/TD3/opponents/self_play.py       → KEEP (refactor)
02-SRC/TD3/opponents/pfsp.py            → KEEP (reuse)
02-SRC/TD3/opponents/fixed.py           → KEEP (reuse)
02-SRC/TD3/opponents/base.py            → KEEP (reuse)
02-SRC/TD3/opponents/__init__.py        → KEEP (update)
```

#### Evaluation Framework
```
02-SRC/TD3/evaluation/evaluator.py      → KEEP (adapt)
02-SRC/TD3/evaluation/__init__.py       → KEEP
```

#### Metrics and Visualization
```
02-SRC/TD3/metrics/metrics_tracker.py   → KEEP (adapt)
02-SRC/TD3/metrics/__init__.py          → KEEP
02-SRC/TD3/visualization/gif_recorder.py → KEEP (reuse)
02-SRC/TD3/visualization/frame_capture.py → KEEP (reuse)
02-SRC/TD3/visualization/__init__.py    → KEEP
```

#### Testing Scripts
```
02-SRC/TD3/test_hockey.py               → KEEP (adapt)
02-SRC/TD3/test_player2_perspective.py  → KEEP
```

#### Utility Scripts
```
02-SRC/TD3/download_wandb_run.py        → KEEP (reuse)
02-SRC/TD3/requirements.txt             → KEEP (update)
02-SRC/TD3/__init__.py                  → KEEP
```

#### Competition Client
```
02-SRC/comprl-hockey-agent/             → KEEP (entire folder, adapt later)
```

### Archive Structure After Cleanup

```
99-ARCHIVE/
├── TD3/
│   ├── agents/
│   │   ├── td3_agent.py
│   │   ├── model.py
│   │   ├── memory.py
│   │   ├── noise.py
│   │   ├── device.py
│   │   └── __init__.py
│   ├── rewards/
│   │   ├── pbrs.py
│   │   ├── base.py
│   │   └── __init__.py
│   ├── config/
│   │   ├── parser.py
│   │   └── __init__.py
│   ├── train_hockey.py
│   ├── sbatch/
│   │   └── (all .sbatch files)
│   ├── analysis/
│   │   └── (all ANALYSIS_*.md files)
│   ├── guides/
│   │   └── (all *_GUIDE.md files)
│   ├── plans/
│   │   └── (all *_PLAN.md files)
│   ├── pbrs_docs/
│   │   └── (all PBRS_*.md files)
│   ├── reports/
│   │   └── (all PERFORMANCE_REPORT_*.md files)
│   ├── docs/
│   │   └── (other documentation)
│   ├── wandb_exports/
│   │   └── (all wandb_run_*.txt files)
│   ├── logs/
│   │   └── (training.log, worker_*.log)
│   ├── wandb/
│   │   └── (full wandb directory)
│   ├── wandb_analysis/
│   │   └── (analysis scripts)
│   ├── checkpoints/
│   │   └── (all .pth files)
│   ├── results/
│   │   └── (results subdirectories)
│   ├── videos/
│   │   └── (test videos)
│   └── tests/
│       └── (TD3-specific tests)
└── sbatch_archive/
    └── (existing archived sbatch files)
```

---

## Phase 1: Initial Testing with PyTorch Port (Days 1-3)

### Objective
Validate that DreamerV3 can learn on hockey environment before building custom implementation.

### Step 1.1: Environment Setup

```bash
# Create new conda environment
conda create -n dreamer-hockey python=3.10
conda activate dreamer-hockey

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install hockey environment
pip install hockey

# Install DreamerV3 PyTorch port
pip install dreamerv3-torch  # or clone from GitHub

# Alternative: Use official JAX implementation
# pip install dreamerv3  # Official (requires JAX)
```

### Step 1.2: Create Minimal Hockey Wrapper

Create `02-SRC/DreamerV3/envs/hockey_wrapper.py`:

```python
"""
Minimal hockey environment wrapper for DreamerV3.
"""
import gymnasium as gym
import numpy as np
import hockey.hockey_env as h_env


class HockeyEnvDreamer(gym.Env):
    """Hockey environment wrapper for DreamerV3."""

    def __init__(self, mode="NORMAL", opponent="weak"):
        self.env = h_env.HockeyEnv(mode=getattr(h_env.Mode, mode))
        self.opponent = h_env.BasicOpponent(weak=(opponent == "weak"))

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None):
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        # Get opponent action
        obs_opponent = self.env.obs_agent_two()
        action_opponent = self.opponent.act(obs_opponent)

        # Combined action
        action_combined = np.hstack([action, action_opponent])

        # Step environment
        obs, reward, done, truncated, info = self.env.step(action_combined)

        # Sparse reward only: +1 win, -1 loss, 0 tie
        if info.get('winner') == 1:
            reward = 1.0
        elif info.get('winner') == -1:
            reward = -1.0
        else:
            reward = 0.0

        return obs.astype(np.float32), reward, done, truncated, info

    def render(self):
        return self.env.render(mode='rgb_array')
```

### Step 1.3: Minimal Training Script

Create `02-SRC/DreamerV3/train_minimal.py`:

```python
"""
Minimal DreamerV3 training script for hockey.
Tests that the algorithm can learn on sparse rewards.
"""
import argparse
from envs.hockey_wrapper import HockeyEnvDreamer

# Import based on available library
try:
    from dreamerv3_torch import Dreamer
    BACKEND = "torch"
except ImportError:
    from dreamerv3 import Dreamer
    BACKEND = "jax"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--opponent', type=str, default='weak')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Create environment
    env = HockeyEnvDreamer(mode="NORMAL", opponent=args.opponent)

    # Create DreamerV3 agent
    config = {
        'imagination_horizon': 50,
        'batch_size': 16,
        'batch_length': 64,
        # Add more config as needed
    }

    agent = Dreamer(env.observation_space, env.action_space, config)

    # Training loop
    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0
    episode_count = 0

    for step in range(args.steps):
        action = agent.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        agent.store(obs, action, reward, next_obs, done)
        agent.train()

        episode_reward += reward
        obs = next_obs

        if done or truncated:
            episode_count += 1
            print(f"Episode {episode_count}: reward={episode_reward}, winner={info.get('winner', 0)}")
            episode_reward = 0
            obs, _ = env.reset()

    print(f"Training complete. Backend: {BACKEND}")


if __name__ == "__main__":
    main()
```

### Step 1.4: Validation Criteria

**Success criteria for Phase 1:**
- [ ] Agent wins >30% of games against weak opponent after 1M steps
- [ ] No reward hacking / passive positioning observed
- [ ] Training is stable (no NaN, no crashes)
- [ ] W&B logging works

**If Phase 1 succeeds:** Proceed to Phase 2 (full implementation)
**If Phase 1 fails:** Debug, try official JAX implementation, or reconsider approach

---

## Phase 2: Full Custom Implementation (Days 4-14)

### Objective
Build production-quality DreamerV3 implementation with self-play.

### New Directory Structure

```
02-SRC/DreamerV3/
├── __init__.py
├── train_hockey.py              # Main training script
├── test_hockey.py               # Evaluation script (adapted)
├── config/
│   ├── __init__.py
│   └── parser.py                # New argument parser
├── agents/
│   ├── __init__.py
│   ├── dreamer_agent.py         # DreamerV3 agent wrapper
│   ├── world_model.py           # RSSM world model
│   ├── actor_critic.py          # Actor and critic networks
│   └── replay_buffer.py         # Sequence replay buffer
├── envs/
│   ├── __init__.py
│   └── hockey_wrapper.py        # Hockey environment wrapper
├── opponents/                   # Copied from TD3, adapted
│   ├── __init__.py
│   ├── base.py
│   ├── fixed.py
│   ├── self_play.py
│   └── pfsp.py
├── evaluation/                  # Copied from TD3, adapted
│   ├── __init__.py
│   └── evaluator.py
├── metrics/                     # Copied from TD3, adapted
│   ├── __init__.py
│   └── metrics_tracker.py
├── visualization/               # Copied from TD3
│   ├── __init__.py
│   ├── gif_recorder.py
│   └── frame_capture.py
├── utils/
│   ├── __init__.py
│   └── device.py                # Device management
├── requirements.txt
└── train_hockey.sbatch          # SLURM script
```

### Step 2.1: World Model (RSSM)

The Recurrent State Space Model is the core of DreamerV3.

```python
# agents/world_model.py

class RSSM(nn.Module):
    """
    Recurrent State Space Model for DreamerV3.

    Components:
    - Encoder: obs -> embedded
    - Sequence model: (h, z, a) -> h'
    - Representation: (h, obs) -> z (posterior)
    - Transition: h -> z (prior)
    - Decoder: (h, z) -> obs_pred
    - Reward predictor: (h, z) -> reward_pred
    - Continue predictor: (h, z) -> continue_pred
    """

    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.stoch_size = config.get('stoch_size', 32)
        self.deter_size = config.get('deter_size', 512)
        self.hidden_size = config.get('hidden_size', 512)

        # Encoder
        self.encoder = MLP(obs_dim, self.hidden_size, [256, 256])

        # Sequence model (GRU)
        self.gru = nn.GRUCell(
            self.stoch_size + action_dim,
            self.deter_size
        )

        # Representation model (posterior)
        self.posterior = MLP(
            self.deter_size + self.hidden_size,
            self.stoch_size * 2,  # mean and std
            [256]
        )

        # Transition model (prior)
        self.prior = MLP(
            self.deter_size,
            self.stoch_size * 2,
            [256]
        )

        # Decoder
        self.decoder = MLP(
            self.deter_size + self.stoch_size,
            obs_dim,
            [256, 256]
        )

        # Reward predictor
        self.reward_pred = MLP(
            self.deter_size + self.stoch_size,
            1,
            [256]
        )

        # Continue predictor
        self.continue_pred = MLP(
            self.deter_size + self.stoch_size,
            1,
            [256]
        )
```

### Step 2.2: Actor-Critic

```python
# agents/actor_critic.py

class Actor(nn.Module):
    """Policy network for DreamerV3."""

    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.net = MLP(state_dim, action_dim * 2, [256, 256])

    def forward(self, state):
        out = self.net(state)
        mean, std = out.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return mean, std


class Critic(nn.Module):
    """Value network for DreamerV3."""

    def __init__(self, state_dim, config):
        super().__init__()
        self.net = MLP(state_dim, 1, [256, 256])

    def forward(self, state):
        return self.net(state)
```

### Step 2.3: Sequence Replay Buffer

```python
# agents/replay_buffer.py

class SequenceReplayBuffer:
    """
    Replay buffer storing full episode sequences.
    DreamerV3 trains on sequences, not single transitions.
    """

    def __init__(self, capacity, sequence_length, obs_dim, action_dim):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.episodes = []
        self.current_episode = {
            'obs': [],
            'action': [],
            'reward': [],
            'done': []
        }

    def add(self, obs, action, reward, done):
        self.current_episode['obs'].append(obs)
        self.current_episode['action'].append(action)
        self.current_episode['reward'].append(reward)
        self.current_episode['done'].append(done)

        if done:
            self._finalize_episode()

    def _finalize_episode(self):
        # Convert to numpy and store
        episode = {
            'obs': np.array(self.current_episode['obs']),
            'action': np.array(self.current_episode['action']),
            'reward': np.array(self.current_episode['reward']),
            'done': np.array(self.current_episode['done'])
        }
        self.episodes.append(episode)

        # FIFO eviction
        while len(self.episodes) > self.capacity:
            self.episodes.pop(0)

        self.current_episode = {'obs': [], 'action': [], 'reward': [], 'done': []}

    def sample(self, batch_size):
        """Sample batch of sequences."""
        sequences = []
        for _ in range(batch_size):
            # Random episode
            ep_idx = np.random.randint(len(self.episodes))
            ep = self.episodes[ep_idx]

            # Random start position
            max_start = max(0, len(ep['obs']) - self.sequence_length)
            start = np.random.randint(max_start + 1)

            seq = {
                'obs': ep['obs'][start:start+self.sequence_length],
                'action': ep['action'][start:start+self.sequence_length],
                'reward': ep['reward'][start:start+self.sequence_length],
                'done': ep['done'][start:start+self.sequence_length]
            }
            sequences.append(seq)

        return self._stack_sequences(sequences)
```

### Step 2.4: DreamerV3 Agent

```python
# agents/dreamer_agent.py

class DreamerV3Agent:
    """
    Full DreamerV3 agent implementation.
    """

    def __init__(self, obs_dim, action_dim, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # World model
        self.world_model = RSSM(obs_dim, action_dim, config).to(self.device)

        # Actor-critic
        state_dim = config['deter_size'] + config['stoch_size']
        self.actor = Actor(state_dim, action_dim, config).to(self.device)
        self.critic = Critic(state_dim, config).to(self.device)

        # Target critic
        self.critic_target = Critic(state_dim, config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.world_opt = torch.optim.Adam(self.world_model.parameters(), lr=3e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-5)

        # Replay buffer
        self.buffer = SequenceReplayBuffer(
            capacity=config.get('buffer_episodes', 1000),
            sequence_length=config.get('batch_length', 64),
            obs_dim=obs_dim,
            action_dim=action_dim
        )

        # State for acting
        self.h = None
        self.z = None

    def act(self, obs, explore=True):
        """Select action given observation."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Update latent state
            if self.h is None:
                self.h = torch.zeros(1, self.config['deter_size']).to(self.device)
                self.z = torch.zeros(1, self.config['stoch_size']).to(self.device)

            # Get action from policy
            state = torch.cat([self.h, self.z], dim=-1)
            mean, std = self.actor(state)

            if explore:
                action = torch.normal(mean, std)
            else:
                action = mean

            action = torch.tanh(action)

            # Update latent state with action
            embedded = self.world_model.encoder(obs_t)
            self.h = self.world_model.gru(
                torch.cat([self.z, action], dim=-1),
                self.h
            )
            posterior_params = self.world_model.posterior(
                torch.cat([self.h, embedded], dim=-1)
            )
            self.z = self._sample_stoch(posterior_params)

            return action.squeeze(0).cpu().numpy()

    def reset(self):
        """Reset latent state for new episode."""
        self.h = None
        self.z = None

    def train(self):
        """Train all components."""
        if len(self.buffer.episodes) < 2:
            return {}

        # Sample batch of sequences
        batch = self.buffer.sample(self.config.get('batch_size', 16))

        # World model training
        wm_loss, wm_metrics = self._train_world_model(batch)

        # Actor-critic training (imagination)
        ac_loss, ac_metrics = self._train_actor_critic()

        return {**wm_metrics, **ac_metrics}

    def _train_world_model(self, batch):
        """Train world model on real sequences."""
        # ... implementation
        pass

    def _train_actor_critic(self):
        """Train actor-critic in imagination."""
        # Imagine trajectories
        horizon = self.config.get('imagination_horizon', 50)
        # ... imagination loop
        # ... actor and critic updates
        pass
```

### Step 2.5: Self-Play Integration

Adapt existing self-play system for DreamerV3:

```python
# In train_hockey.py

def main():
    # ... setup ...

    # Self-play manager (reuse existing)
    self_play_manager = SelfPlayManager(
        pool_size=args.self_play_pool_size,
        save_interval=args.self_play_save_interval,
        use_pfsp=args.use_pfsp,
        pfsp_mode=args.pfsp_mode
    )

    for episode in range(args.max_episodes):
        # Get opponent
        if episode < args.self_play_start:
            opponent = weak_opponent
        else:
            opponent = self_play_manager.get_opponent()

        # Run episode
        obs, _ = env.reset()
        agent.reset()

        while not done:
            action = agent.act(obs)
            obs_opp = env.obs_agent_two()
            action_opp = opponent.act(obs_opp)

            obs_next, reward, done, truncated, info = env.step(
                np.hstack([action, action_opp])
            )

            agent.buffer.add(obs, action, reward, done)
            obs = obs_next

        # Train
        metrics = agent.train()

        # Update self-play stats
        if episode >= args.self_play_start:
            self_play_manager.update_stats(opponent, info.get('winner', 0))

        # Save to pool periodically
        if episode >= args.self_play_start and episode % args.self_play_save_interval == 0:
            self_play_manager.add_to_pool(agent.state_dict(), episode)
```

---

## Phase 3: Training and Evaluation (Days 15-21)

### Step 3.1: Hyperparameter Configuration

Based on Robot Air Hockey Challenge paper:

```python
DEFAULT_CONFIG = {
    # World model
    'stoch_size': 32,
    'deter_size': 512,
    'hidden_size': 512,

    # Training
    'batch_size': 16,
    'batch_length': 64,
    'imagination_horizon': 50,
    'learning_rate_wm': 3e-4,
    'learning_rate_actor': 3e-5,
    'learning_rate_critic': 3e-5,

    # Buffer
    'buffer_episodes': 1000,

    # Self-play
    'self_play_start': 5000,  # episodes
    'self_play_pool_size': 25,
    'self_play_save_interval': 1000,
    'use_pfsp': True,
    'pfsp_mode': 'variance',

    # Rewards (SPARSE ONLY)
    'reward_win': 1.0,
    'reward_loss': -1.0,
    'reward_tie': 0.0,

    # Total training
    'max_steps': 100_000_000,  # 100M steps
    'eval_interval': 10000,
    'save_interval': 50000,
}
```

### Step 3.2: SLURM Training Script

Create `02-SRC/DreamerV3/train_hockey.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=dreamer-hockey
#SBATCH --output=logs/dreamer_%j.out
#SBATCH --error=logs/dreamer_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G

# Load modules
module load cuda/12.1
module load python/3.10

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer-hockey

# Run training
python train_hockey.py \
    --max_steps 100000000 \
    --imagination_horizon 50 \
    --self_play_start 5000 \
    --self_play_pool_size 25 \
    --use_pfsp \
    --pfsp_mode variance \
    --seed 42 \
    --wandb_project dreamer-hockey
```

### Step 3.3: Evaluation Protocol

Test against:
1. Weak opponent (built-in)
2. Strong opponent (built-in)
3. TD3 best checkpoint (from archive)
4. Self-play pool opponents

Report metrics:
- Win rate
- Goal difference
- Episode length distribution
- No passive positioning (verify with GIFs)

---

## Phase 4: Competition Integration (Days 22-28)

### Step 4.1: Update COMPRL Client

Adapt `02-SRC/comprl-hockey-agent/run_client.py` for DreamerV3:

```python
class DreamerHockeyAgent:
    """DreamerV3 agent for COMPRL competition."""

    def __init__(self, checkpoint_path):
        self.agent = DreamerV3Agent.load(checkpoint_path)
        self.agent.eval()

    def act(self, obs):
        return self.agent.act(obs, explore=False)

    def reset(self):
        self.agent.reset()
```

### Step 4.2: Tournament Testing

- Run against competition server
- Monitor for edge cases
- Ensure no crashes during long sessions

---

## Timeline Summary

| Phase | Days | Description |
|-------|------|-------------|
| 0 | 0.5 | Repository cleanup, archive TD3 |
| 1 | 1-3 | Initial testing with PyTorch port |
| 2 | 4-14 | Full custom implementation |
| 3 | 15-21 | Training and evaluation |
| 4 | 22-28 | Competition integration |

**Total: ~4 weeks**

---

## Risk Mitigation

### Risk 1: PyTorch DreamerV3 port doesn't work
**Mitigation:** Fall back to official JAX implementation

### Risk 2: Training is unstable
**Mitigation:** Start with shorter imagination horizon (25), increase gradually

### Risk 3: Self-play causes collapse
**Mitigation:** Use higher weak_ratio (0.7) initially, reduce over time

### Risk 4: Computation budget exceeded
**Mitigation:** Use fewer parallel workers, reduce batch size

---

## Success Criteria

1. **Phase 1:** Agent wins >30% vs weak opponent (1M steps)
2. **Phase 2:** Clean, documented implementation compiles and runs
3. **Phase 3:** Agent wins >70% vs weak, >50% vs strong (100M steps)
4. **Phase 4:** Stable competition client, no crashes

---

## References

1. Hafner et al. (2023) - "Mastering Diverse Domains through World Models" (DreamerV3)
2. Orsula et al. (2024) - "Learning to Play Air Hockey with Model-Based Deep RL"
3. `01-DOCS/DreamerV3/RESEARCH_RESULTS.md` - Comprehensive research document

---

*This plan will be updated as implementation progresses.*
