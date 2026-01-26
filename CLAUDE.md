# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Python Environment

**ALWAYS USE `conda activate py310` WHEN TRYING TO EXECUTE PYTHON CODE.**

All Python commands should be run in the py310 conda environment. If you encounter import errors or missing packages, ensure the correct environment is activated first.

## Project Overview

This repository contains three main components:

1. **DreamerV3 Training System** (`02-SRC/DreamerV3/`): World-model based RL agent that learns entirely in imagination (ACTIVE)
2. **COMPRL Client** (`02-SRC/comprl-hockey-agent/`): Client that connects trained agents to a competition server
3. **TD3 Training System** (`99-ARCHIVE/TD3/`): TD3 + PBRS + Self-Play with PFSP (ARCHIVED)

The project implements:
- **DreamerV3**: World model + imagination-based actor-critic training with self-play support
- **TD3** (archived): Pure TD3 + PBRS + Self-Play with PFSP

## Benchmark Results

**Best Checkpoint: 266k gradient steps**

| Opponent | Win Rate |
|----------|----------|
| Weak Bot | 87% |
| Strong Bot | 90% |
| **Combined** | **88.5%** |

Achieved using a **3-phase training approach** (~54 hours total):
1. **Phase 1**: Mixed opponents + self-play (30h, seed 42, 0→268k steps)
2. **Phase 2**: Mixed opponents only (8h, seed 43, 192k→270k steps)
3. **Phase 3**: Fine-tuning with lower LRs (16h, seed 43, 260k→340k steps)

## Common Commands

### DreamerV3 Training (Primary)

```bash
# Activate environment first!
conda activate py310
cd 02-SRC/DreamerV3

# Basic training with weak opponent
python3 train_hockey.py --opponent weak --seed 42

# Training with self-play (activates at episode 50)
python3 train_hockey.py --opponent weak --seed 42 \
    --self_play_start 50 --self_play_pool_size 10 --use_pfsp

# Training with DreamSmooth (temporal reward smoothing for sparse rewards)
python3 train_hockey.py --opponent weak --seed 42 --use_dreamsmooth

# Training with custom hyperparameters
python3 train_hockey.py --opponent weak \
    --lr_world 3e-4 --lr_actor 8e-5 --lr_critic 1e-4 \
    --entropy_scale 0.0003 --imagination_horizon 15

# Disable W&B for local testing
python3 train_hockey.py --opponent weak --no_wandb --gradient_steps 10000

# Full configuration example
python3 train_hockey.py \
    --opponent weak \
    --seed 42 \
    --gradient_steps 1000000 \
    --batch_size 32 \
    --batch_length 32 \
    --imagination_horizon 15 \
    --lr_world 0.0003 \
    --lr_actor 0.00008 \
    --lr_critic 0.0001 \
    --entropy_scale 0.0003 \
    --gradient_clip 100 \
    --gif_interval 10000
```

### Running Competition Client

```bash
conda activate py310
cd 02-SRC/comprl-hockey-agent
pip install -r requirements.txt

# Run with environment variables
export COMPRL_SERVER_URL=<URL>
export COMPRL_SERVER_PORT=<PORT>
export COMPRL_ACCESS_TOKEN=<TOKEN>
python3 run_client.py --args --agent=strong

# Or with command-line arguments
python3 run_client.py --server-url <URL> --server-port <PORT> \
    --token <TOKEN> \
    --args --agent=strong

# Auto-restart wrapper (handles connection issues)
bash autorestart.sh --server-url <URL> --server-port <PORT> \
    --token <TOKEN> \
    --args --agent=strong
```

### TD3 Training (Archived)

```bash
conda activate py310
cd 99-ARCHIVE/TD3

# Basic training with weak opponent
python3 train_hockey.py --mode NORMAL --opponent weak

# Training with self-play (activates at episode 8000)
python3 train_hockey.py --mode NORMAL --opponent weak --self_play_start 8000 --use_pfsp
```

## Architecture Overview

### DreamerV3 Training System (Primary)

**Philosophy**: DreamerV3 learns a world model and trains the policy entirely in "imagination" (simulated rollouts in the learned latent space). This enables efficient credit assignment for sparse rewards without dense reward shaping.

**Core Components** (`02-SRC/DreamerV3/`):

- **`train_hockey.py`**: Main training loop
  - Collects experience in real environment
  - Trains world model on real data sequences
  - Trains actor-critic entirely in imagination
  - Full CLI argument support for all hyperparameters
  - W&B logging with GIF recording

- **`dreamer.py`**: Dreamer agent combining world model and behavior
  - `worldModelTraining()`: Train encoder, decoder, RSSM, reward/continue predictors
  - `behaviorTraining()`: Train actor-critic in imagined rollouts
  - `act()`: Select actions with recurrent state tracking
  - `saveCheckpoint()` / `loadCheckpoint()`: Model persistence

- **`networks.py`**: Neural network components
  - `EncoderMLP`: 18-dim observation → embedding
  - `DecoderMLP`: Full state → observation reconstruction
  - `RecurrentModel`: GRU for deterministic state
  - `PriorNet` / `PosteriorNet`: Categorical latent prediction
  - `RewardModel`: Two-Hot Symlog distribution for rewards
  - `ContinueModel`: Bernoulli for episode termination
  - `Actor`: Tanh-squashed Gaussian policy
  - `Critic`: Two-Hot Symlog distribution for values (with slow EMA target)

- **`buffer.py`**: Replay buffer for sequence sampling
  - Stores (obs, action, reward, next_obs, done) tuples
  - Samples contiguous sequences for world model training
  - DreamSmooth support for temporal reward smoothing

- **`utils.py`**: Helper functions
  - `computeLambdaValues()`: TD(λ) returns
  - `Moments`: Percentile-based value normalization
  - `sequentialModel1D()`: MLP builder
  - `TwoHotSymlog`: Discretized reward/value prediction for sparse signals

**Opponent Management** (`opponents/`):

- **`self_play.py`**: SelfPlayManager for training against previous checkpoints
  - Pool of past agents maintained (FIFO rotation)
  - PFSP (Prioritized Fictitious Self-Play) for smart opponent selection
  - Balances training against weak/strong anchors vs self-play pool
  - Selects opponents based on win-rate statistics

- **`fixed.py`**: FixedOpponent wraps built-in BasicOpponent (weak/strong)

- **`pfsp.py`**: Prioritized Fictitious Self-Play implementation for opponent selection weights

- **`base.py`**: BaseOpponent abstract class

**Auxiliary Tasks** (in `networks.py` and `dreamer.py`):

- **`GoalPredictionHead`**: Binary classification predicting if goal will occur in next K steps
- **`DistanceHead`**: Regression predicting puck-to-goal distance
- **`ShotQualityHead`**: Regression predicting offensive opportunity quality
- These help the world model learn goal-relevant representations without corrupting the reward signal

**Configuration** (`configs/`):

- **`hockey.yml`**: Default configuration with all hyperparameters
  - All values can be overridden via CLI arguments

### COMPRL Client System

**Components** (`02-SRC/comprl-hockey-agent/`):

- **`run_client.py`**: Main tournament client connecting trained agents to server
  - Wraps agents for remote competition
  - HockeyAgent wrapper for built-in opponents
  - Async communication with COMPRL server

- **`autorestart.sh`**: Wrapper script auto-restarting client on connection loss
  - Tracks restart frequency to prevent restart loops
  - Optional ntfy.sh notifications

### TD3 Training System (Archived)

**Core Components** (`99-ARCHIVE/TD3/`):

- **`train_hockey.py`**: Main training loop with PBRS and self-play
- **`agents/td3_agent.py`**: TD3Agent with twin critics, delayed policy updates
- **`opponents/self_play.py`**: Self-play with PFSP opponent selection
- **`rewards/pbrs.py`**: Potential-Based Reward Shaping

## Key Design Patterns

### DreamerV3 Training Pipeline

1. **Warmup**: Collect initial episodes into replay buffer
2. **Main Loop** (per iteration):
   - **Gradient Updates** (replay_ratio times):
     - Sample sequence batch from buffer
     - Train world model (reconstruction, reward, KL, auxiliary tasks)
     - Train actor-critic in imagination (lambda returns)
   - **Environment Interaction**: Run episode, add to buffer
3. **Evaluation**: Periodic eval episodes
4. **GIF Recording**: Periodic gameplay visualization for W&B

### Observation Handling

Hockey environment provides 18-dim observations (per player):
- Player position (2D), angle (1D), velocity (2D), angular velocity (1D)
- Opponent position (2D), angle (1D), velocity (2D), angular velocity (1D)
- Puck position (2D), velocity (2D)

Keep-mode affects observation dimension (OFF=16, ON=18). The training code detects and handles mismatches.

### Action Space

- **Agent outputs**: 4-dim continuous actions (agent's own movements)
- **Environment expects**: 8-dim combined actions (4 per player)
- **Critic training**: Sees combined 8-dim actions for full game context

## Important Implementation Details

### DreamerV3-Specific Features

- **RSSM World Model**: Recurrent state space model with deterministic (GRU) + stochastic (categorical) states
- **Categorical Latents**: 16 variables × 16 classes = 256-dim stochastic state (vs Gaussian in v1/v2)
- **Two-Hot Symlog**: Discretized reward/value prediction that handles sparse rewards (goals)
- **Auxiliary Tasks**: Goal prediction, distance, and shot quality heads improve latent representations
- **Free Nats**: KL loss has threshold below which it's not penalized (prevents posterior collapse)
- **Lambda Returns**: TD(λ) for value targets in imagination
- **Value Normalization**: Percentile-based moments for stable advantage computation
- **Imagination Training**: Actor-critic trained entirely in latent space rollouts (no real env gradients)
- **Slow Critic (EMA)**: Exponential moving average of critic weights for stable bootstrap targets (decay=0.98)
- **DreamSmooth**: Optional temporal reward smoothing for sparse rewards (arXiv:2311.01450)

### Self-Play Management (DreamerV3)

- **Opponent Pool**: Circular buffer of N past checkpoints (FIFO rotation)
- **Save Interval**: New opponent added periodically during self-play
- **Selection Strategy**:
  - `weak_ratio` defines probability of facing anchor (weak/strong) vs. pool
  - PFSP mode selects from pool based on win rates:
    - `variance`: Prioritizes opponents with ~50% win rate (most learning signal)
    - `hard`: Prioritizes hardest opponents (lowest win rate)
- **Activation**: Self-play activates at configurable episode threshold (`--self_play_start`)

### TD3-Specific Features (Archived)

- **Delayed Policy Updates**: Policy updates every N critic updates (policy_freq=2)
- **Twin Critic Networks**: Two Q-functions to reduce overestimation
- **Target Policy Smoothing**: Adds clipped noise to target actions for stability
- **Gaussian Exploration**: N(0, sigma) noise, constant per TD3 paper (no decay)
- **Q-Value Clipping**: Hard or soft (tanh) clipping prevents Q-value explosion
- **VF Regularization**: Penalizes near-zero Q-values to prevent passive agents
- **PBRS**: Potential-Based Reward Shaping for dense exploration guidance

## W&B Integration

**DreamerV3 Metrics**:
- World model: `world/loss`, `world/recon_loss`, `world/reward_loss`, `world/kl_loss`
- Auxiliary tasks: `world/aux_goal_loss`, `world/aux_distance_loss`, `world/aux_quality_loss`
- Behavior: `behavior/actor_loss`, `behavior/critic_loss`, `behavior/entropy`, `behavior/advantages`
- Stats: `stats/win_rate`, `stats/mean_reward`, `stats/buffer_size`
- Evaluation: `eval/win_rate`, `eval/gif_*` (gameplay GIFs)

## Performance Monitoring

**DreamerV3 Key Metrics**:
- `world/loss`: Should decrease over training
- `behavior/entropy`: Should stay **positive** throughout training (see entropy guide below)
- `stats/win_rate`: Should increase over training
- `world/kl_loss`: Should stabilize around free_nats threshold
- `diagnostics/return_range_S`: Should grow as returns become more variable; stuck at 1.0 = problem

### Entropy & Exploration Guide

**Entropy scale (η = 3e-4) is FIXED** - no annealing, no domain-specific tuning. Return normalization handles domain variation.

**Expected entropy values for 4-dim continuous actions** (std bounds: [0.368, 1.649]):
| Training Phase | Entropy Range | Notes |
|----------------|---------------|-------|
| Early | +5 to +7.7 | High exploration (std near max 1.649) |
| Mid | +3 to +5 | Learning proceeds |
| Converged | +1.7 to +3 | Focused but stochastic |
| **Negative** | **BUG!** | Fix: ensure σ_min > 0.242 |

**Red flags**:
- Entropy < 0: Policy std collapsed too low (bug in logStd bounds)
- Entropy stuck at 7.7 (max): No learning signal (check advantages, world model)
- Entropy > 8: IMPOSSIBLE with current bounds, indicates code bug
- Return range S always at floor (1.0): Reward signal weak or pathological

**Key insight**: Entropy-advantage balance evolves AUTOMATICALLY through percentile-based return normalization. Do NOT manually target specific ratios.

## Key Hyperparameters (DreamerV3)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--replay_ratio` | 32 | Gradient steps per env step. Use 32 for initial training, can reduce for fine-tuning. |
| `--imagination_horizon` | 15 | Steps to imagine for actor-critic training |
| `--batch_size` | 32 | Sequences per batch |
| `--batch_length` | 32 | Timesteps per sequence |
| `--entropy_scale` | 0.0003 | Entropy bonus (FIXED, no annealing) |
| `--lr_world` | 3e-4 | World model learning rate |
| `--lr_actor` | 1e-4 | Actor learning rate. **Must be ≤ lr_critic!** |
| `--lr_critic` | 1e-4 | Critic learning rate |
| `--discount` | 0.997 | Discount factor γ |
| `--free_nats` | 1.0 | KL free nats threshold |
| `--slow_critic_decay` | 0.98 | EMA decay for slow critic target |
| `--use_dreamsmooth` | True | Enable DreamSmooth for sparse rewards |
| `--warmup_episodes` | 100 | Episodes before training starts |
| `--buffer_capacity` | 250000 | Replay buffer size (~1000 episodes) |

### Critical Hyperparameter Warnings

**Do NOT override these with bad values:**
- `--lr_actor 0.0005` → Inverts actor/critic hierarchy, causes instability. Use ≤0.0001.
- Skipping DreamSmooth → Sparse reward signal makes training unstable.

**Note on replay ratio**: While 32 is optimal for initial training, Phase 3 fine-tuning successfully used replay_ratio=4 with lower learning rates.

## Recommended Training Approach (3-Phase)

The benchmark performance (88.5% combined win rate) was achieved using this 3-phase approach:

### Phase 1: Mixed Opponents + Self-Play
```bash
cd 02-SRC/DreamerV3
python3 train_hockey.py \
    --seed 42 \
    --replay_ratio 32 \
    --warmup_episodes 200 \
    --lr_world 0.0003 --lr_actor 0.0001 --lr_critic 0.0001 \
    --entropy_scale 0.0003 \
    --use_dreamsmooth --dreamsmooth_alpha 0.5 \
    --mixed_opponents --mixed_weak_prob 0.5 \
    --self_play_start 1000 --self_play_pool_size 15 --use_pfsp --pfsp_mode variance
```

### Phase 2: Mixed Opponents Only (resume from Phase 1)
```bash
python3 train_hockey.py \
    --seed 43 \
    --resume <phase1_checkpoint.pth> \
    --replay_ratio 16 \
    --lr_world 0.0002 --lr_actor 0.0001 --lr_critic 0.0001 \
    --use_dreamsmooth \
    --mixed_opponents --mixed_weak_prob 0.5
```

### Phase 3: Fine-Tuning (resume from Phase 2)
```bash
python3 train_hockey.py \
    --seed 43 \
    --resume <phase2_checkpoint.pth> \
    --replay_ratio 4 \
    --lr_world 0.0002 --lr_actor 0.00005 --lr_critic 0.00005 \
    --use_dreamsmooth \
    --mixed_opponents --mixed_weak_prob 0.5
```

**Key insights:**
1. Self-play bootstraps diverse play styles, then remove it to focus on target opponents
2. Gradually reduce learning rates and replay ratio across phases
3. DreamSmooth essential throughout for sparse reward handling
4. Mixed opponents (weak+strong) prevents overfitting to single opponent type
