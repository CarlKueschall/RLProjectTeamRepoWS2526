# RL Hockey Project

Authors: Serhat Alpay, Carl Kueschall

Reinforcement Learning project for training agents to play hockey using DreamerV3 (world-model based) and TD3 algorithms with self-play.

## Quick Start

### DreamerV3 (Primary - Active Development)

```bash
conda activate py310
cd 02-SRC/DreamerV3

# Basic training against weak opponent
python train_hockey.py --opponent weak --seed 42

# Training with self-play (activates at episode 50)
python train_hockey.py \
    --opponent weak \
    --seed 42 \
    --self_play_start 50 \
    --self_play_pool_size 10 \
    --use_pfsp

# Training with DreamSmooth (temporal reward smoothing for sparse rewards)
python train_hockey.py --opponent weak --seed 42 --use_dreamsmooth

# Full configuration example
python train_hockey.py \
    --opponent weak \
    --seed 42 \
    --gradient_steps 1000000 \
    --replay_ratio 32 \
    --warmup_episodes 10 \
    --batch_size 32 \
    --batch_length 32 \
    --imagination_horizon 15 \
    --recurrent_size 256 \
    --latent_length 16 \
    --latent_classes 16 \
    --lr_world 0.0003 \
    --lr_actor 0.00008 \
    --lr_critic 0.0001 \
    --discount 0.997 \
    --entropy_scale 0.0003 \
    --gradient_clip 100 \
    --checkpoint_interval 5000 \
    --eval_interval 1000 \
    --gif_interval 10000 \
    --wandb_project rl-hockey
```

### TD3 (Archived)

```bash
conda activate py310
cd 99-ARCHIVE/TD3

# Basic training against weak opponent
python train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 25000 \
    --seed 42

# Training with self-play (activates at episode 8000)
python train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 100000 \
    --seed 42 \
    --self_play_start 8000 \
    --self_play_pool_size 25 \
    --use_pfsp
```

## Key Features

### DreamerV3 (Primary)

- **World Model**: RSSM with categorical latents (16x16 = 256 dim stochastic state)
- **Imagination Training**: Actor-critic trained entirely in latent space rollouts
- **Two-Hot Symlog**: Discretized reward/value prediction for sparse reward handling
- **Auxiliary Tasks**: Goal prediction, distance, and shot quality heads improve representations
- **Slow Critic (EMA)**: Exponential moving average for stable bootstrap targets
- **DreamSmooth**: Optional temporal reward smoothing (arXiv:2311.01450)
- **Self-Play**: PFSP opponent selection from checkpoint pool
- **MLP Architecture**: Simple encoder/decoder for 18-dim observations
- **GIF Recording**: Periodic gameplay GIFs logged to W&B

### TD3 (Archived)

- **TD3 Algorithm**: Twin Delayed DDPG with target smoothing and Q-value clipping
- **PBRS**: Potential-Based Reward Shaping for dense exploration guidance
- **Self-Play**: Opponent pool management with PFSP (Prioritized Fictitious Self-Play) selection
- **VF Regularization**: Anti-lazy learning to prevent passive agents

## File Structure

```
02-SRC/DreamerV3/           # PRIMARY - Active development
├── train_hockey.py          # Main training script
├── dreamer.py               # Dreamer agent (world model + behavior)
├── networks.py              # Neural network components (incl. auxiliary task heads)
├── buffer.py                # Replay buffer for sequences (with DreamSmooth support)
├── utils.py                 # Helpers (lambda returns, moments, TwoHotSymlog)
├── opponents/               # Opponent management
│   ├── self_play.py         # Self-play manager with PFSP
│   ├── pfsp.py              # PFSP opponent selection weights
│   ├── fixed.py             # Fixed opponent wrapper
│   └── base.py              # Base opponent class
├── configs/
│   └── hockey.yml           # Default configuration
└── visualization/
    ├── gif_recorder.py      # GIF recording for W&B
    └── frame_capture.py     # Frame capture utilities

02-SRC/comprl-hockey-agent/  # Competition client
├── run_client.py            # Tournament client
└── autorestart.sh           # Auto-restart wrapper

99-ARCHIVE/TD3/              # ARCHIVED - TD3 implementation
├── train_hockey.py          # Main training script
├── agents/
│   ├── td3_agent.py         # TD3 agent implementation
│   ├── model.py             # Neural network models (MLP)
│   ├── memory.py            # Replay buffer
│   └── noise.py             # Gaussian noise for exploration
├── rewards/
│   └── pbrs.py              # Potential-Based Reward Shaping
└── opponents/
    ├── self_play.py         # Self-play manager with PFSP
    └── pfsp.py              # PFSP opponent selection
```

---

## DreamerV3 Agent

World-model based RL agent that learns entirely in imagination. Based on DreamerV3 paper, simplified for hockey with MLP encoder/decoder.

### Architecture Overview

**World Model (RSSM)**:
- **Encoder**: MLP mapping 18-dim observations to embeddings
- **Recurrent Model**: GRU for deterministic state evolution
- **Prior/Posterior**: Categorical latent prediction (16 vars × 16 classes)
- **Decoder**: MLP reconstructing observations from full state
- **Reward Model**: Two-Hot Symlog distribution for reward prediction
- **Continue Model**: Bernoulli for episode termination

**Behavior (Actor-Critic)**:
- **Actor**: Tanh-squashed Gaussian policy
- **Critic**: Two-Hot Symlog distribution for value estimation
- **Slow Critic**: EMA target network for stable bootstrapping
- **Training**: Lambda returns with percentile-based value normalization

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--imagination_horizon` | 15 | Steps to imagine for actor-critic training |
| `--batch_size` | 32 | Sequences per batch |
| `--batch_length` | 32 | Timesteps per sequence |
| `--entropy_scale` | 0.0003 | Entropy bonus for exploration (DreamerV3 paper default) |
| `--lr_world` | 3e-4 | World model learning rate |
| `--lr_actor` | 8e-5 | Actor learning rate |
| `--lr_critic` | 1e-4 | Critic learning rate |
| `--discount` | 0.997 | Discount factor γ |
| `--free_nats` | 1.0 | KL free nats threshold |
| `--slow_critic_decay` | 0.98 | EMA decay for slow critic target |
| `--use_dreamsmooth` | False | Enable DreamSmooth temporal reward smoothing |
| `--gif_interval` | 10000 | GIF recording interval (0=disabled) |

### Self-Play Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--self_play_start` | 0 | Episode to start self-play (0=disabled) |
| `--self_play_pool_size` | 10 | Max opponents in pool |
| `--use_pfsp` | False | Enable Prioritized Fictitious Self-Play |
| `--pfsp_mode` | variance | PFSP mode: "variance" or "hard" |
| `--weak_ratio` | 0.3 | Probability of facing anchor vs pool |

### W&B Metrics

Training logs to Weights & Biases with:
- **World Model**: reconstruction loss, reward loss, KL divergence
- **Auxiliary Tasks**: goal prediction loss, distance loss, shot quality loss
- **Behavior**: actor loss, critic loss, entropy, advantages
- **Stats**: win rate, episode rewards, buffer size
- **Visualization**: Periodic gameplay GIFs

---

## TD3 Agent (Archived)

Implementation follows the TD3 paper (Fujimoto et al., 2018):
- Twin critics to reduce overestimation bias
- Delayed policy updates (every 2 critic updates)
- Target policy smoothing with clipped noise
- Gaussian exploration noise N(0, 0.1)
- Q-value clipping (soft/hard) to prevent explosion

### Reward Shaping (PBRS V3.1)

Potential-Based Reward Shaping V3.1 uses **strong chase + simple math** while preserving optimal policy (Ng et al., 1999).

**Design Philosophy:**
- **STRONG φ_chase (W=1.0)**: Agent always races toward the puck
- **φ_attack (W=1.2)**: Slightly higher weight ensures forward shooting is net positive
- **NO conditional logic**: Both components always active everywhere
- **Simple math**: W_ATTACK > W_CHASE guarantees correct shooting incentive

#### Sparse Environment Rewards

| Signal | Raw Value | After `reward_scale=0.1` | Description |
|--------|-----------|--------------------------|-------------|
| **Win** | +10 | +1.0 | Terminal reward for scoring a goal |
| **Loss** | -10 | -1.0 | Terminal penalty for conceding a goal |
| **Tie** | 0 | 0 | No reward for draws |

#### PBRS V3.1 Potential Components

| Component | Weight | Range | Description |
|-----------|--------|-------|-------------|
| **φ_chase** | W=1.0 | [-1, 0] | Reward agent proximity to puck. **STRONG**, always active everywhere. |
| **φ_attack** | W=1.2 | [-1, 0] | Reward puck proximity to opponent goal. Always active. |

**Combined Potential:**
```
φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack
     = 1.0 × (-dist_to_puck/MAX) + 1.2 × (-dist_puck_to_goal/MAX)
```

#### Shooting Math (The Key Insight)

When shooting the puck distance D:
```
Forward shot:  Δ = W_ATTACK × (+D/MAX) + W_CHASE × (-D/MAX) = D/MAX × (1.2 - 1.0) = +0.2D ✓
Backward shot: Δ = W_ATTACK × (-D/MAX) + W_CHASE × (-D/MAX) = D/MAX × (-1.2 - 1.0) = -2.2D ✗
```

**W_ATTACK > W_CHASE guarantees forward shooting is always positive!**

### Self-Play (TD3)

Automatic curriculum learning through self-play:
- **Opponent Pool**: Circular buffer of past checkpoints (FIFO)
- **PFSP Selection**: Prioritizes opponents with ~50% win rate (variance mode) or hardest opponents (hard mode)
- **Anchor Mixing**: Balances training against weak/strong anchors vs self-play pool
- Activates at configurable episode threshold (`--self_play_start`)

## W&B Integration

Training logs to Weights & Biases:
- Episode rewards and win rates
- Loss statistics and gradient norms
- Goal scored/conceded metrics
- Model checkpoints (best and periodic)
- Gameplay GIFs (DreamerV3)
