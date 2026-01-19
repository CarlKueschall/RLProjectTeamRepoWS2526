# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains three main components:

1. **TD3 Training System** (`TD3/`): A reinforcement learning framework for training TD3 (Twin Delayed DDPG) agents to play air hockey
2. **DreamerV3 Training System** (`DreamerV3/`): A world-model based RL agent that learns entirely in imagination
3. **COMPRL Client** (`comprl-hockey-agent/`): A client that connects trained agents to a competition server

The project implements:
- **TD3**: Pure TD3 + PBRS + Self-Play with PFSP
- **DreamerV3**: World model + imagination-based actor-critic training

## Common Commands

### TD3 Training

```bash
# Basic training with weak opponent
cd TD3
python3 train_hockey.py --mode NORMAL --opponent weak

# Training with self-play (activates at episode 8000)
python3 train_hockey.py --mode NORMAL --opponent weak --self_play_start 8000 --use_pfsp

# Training with custom hyperparameters
python3 train_hockey.py --mode NORMAL --opponent weak --lr_actor 3e-4 --lr_critic 3e-4 --max_episodes 5000

# Training with specific seed for reproducibility
python3 train_hockey.py --mode NORMAL --opponent weak --seed 42
```

### DreamerV3 Training

```bash
# Basic training with weak opponent
cd DreamerV3
python3 train_hockey.py --opponent weak --seed 42

# Training with custom hyperparameters
python3 train_hockey.py --opponent weak \
    --lr_world 3e-4 --lr_actor 8e-5 --lr_critic 1e-4 \
    --entropy_scale 0.003 --imagination_horizon 15

# Disable W&B for local testing
python3 train_hockey.py --opponent weak --no_wandb --gradient_steps 10000

# Full configuration example (with auxiliary tasks - enabled by default)
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
    --entropy_scale 0.003 \
    --gradient_clip 100 \
    --gif_interval 10000
```

### Testing

```bash
# Test a checkpoint against weak opponent
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent weak --episodes 100

# Test against strong opponent
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent strong --episodes 100

# Test in tournament mode (no position alternation)
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent weak --no-alternation

# Generate video of test episodes
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent weak --render rgb_array --save_video test_video.mp4

# Verbose output showing per-episode results
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent weak --verbose --episodes 20
```

### Running Competition Client

```bash
# Install dependencies
cd comprl-hockey-agent
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

## Architecture Overview

### TD3 Training System

**Core Components** (`TD3/`):

- **`train_hockey.py`**: Main training loop orchestrating the entire RL pipeline
  - Creates hockey environment, opponent, and agent
  - Manages training loop, replay buffer, and checkpoint saving
  - Integrates PBRS reward shaping, metrics tracking, and W&B logging
  - Handles self-play opponent pool management

- **`test_hockey.py`**: Evaluation script for testing trained checkpoints
  - Loads checkpoints in multiple formats
  - Tests against fixed opponents (weak/strong) or self-play
  - Generates metrics and optional video output
  - Tournament mode testing with position fixing

**Agent Implementation** (`agents/`):

- **`td3_agent.py`**: TD3Agent class implementing the Twin Delayed DDPG algorithm
  - Actor network (policy) and two critic networks (Q-functions)
  - Gaussian exploration noise N(0, sigma)
  - Delayed policy updates and target network smoothing
  - VF regularization to prevent passive/lazy agents
  - Methods: `act()`, `train()`, `remember()`, `state()`, `restore_state()`

- **`model.py`**: Generic MLP model for actor/critic networks

- **`memory.py`**: ReplayBuffer class managing experience storage and uniform sampling

- **`noise.py`**: GaussianNoise class for continuous action space exploration

- **`device.py`**: Device management (CPU/GPU/MPS detection)

**Opponent Management** (`opponents/`):

- **`self_play.py`**: SelfPlayManager orchestrates training against a pool of previous checkpoints
  - Pool of past agents maintained (FIFO rotation)
  - PFSP (Prioritized Fictitious Self-Play) for smart opponent selection
  - Balances training against weak/strong anchors vs self-play pool
  - Selects opponents based on win-rate statistics

- **`fixed.py`**: FixedOpponent wraps built-in BasicOpponent (weak/strong)

- **`pfsp.py`**: Prioritized Fictitious Self-Play implementation for opponent selection weights

- **`base.py`**: BaseOpponent abstract class

**Reward Shaping** (`rewards/`):

- **`pbrs.py`**: Potential-Based Reward Shaping (PBRS) for dense exploration guidance
  - Computes potential functions from puck/player positions
  - Policy-invariant reward modification that doesn't change optimal policy
  - Configurable scaling via `--pbrs_scale`

**Evaluation & Metrics** (`evaluation/`, `metrics/`):

- **`evaluation/evaluator.py`**: Evaluation loops comparing agents against different opponents

- **`metrics/metrics_tracker.py`**: Tracks training metrics (win rates, goal differences, Q-values)

**Visualization** (`visualization/`):

- **`gif_recorder.py`**: Records game frames for GIF generation

- **`frame_capture.py`**: Frame capture utilities for video output

**Configuration** (`config/`):

- **`parser.py`**: Command-line argument parser with TD3 hyperparameters
  - Environment settings (mode, opponent, keep_mode)
  - Training settings (episodes, batch size, buffer size)
  - TD3-specific hyperparameters (learning rates, tau, policy frequency)
  - Network architecture (hidden layer sizes)
  - PBRS reward shaping and Q-value clipping settings
  - Self-play settings (pool size, save interval, PFSP mode)

### DreamerV3 Training System

**Core Components** (`DreamerV3/`):

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
  - `RewardModel`: Normal distribution for rewards
  - `ContinueModel`: Bernoulli for episode termination
  - `Actor`: Tanh-squashed Gaussian policy
  - `Critic`: Normal distribution for values

- **`buffer.py`**: Replay buffer for sequence sampling
  - Stores (obs, action, reward, next_obs, done) tuples
  - Samples contiguous sequences for world model training

- **`utils.py`**: Helper functions
  - `computeLambdaValues()`: TD(λ) returns
  - `Moments`: Percentile-based value normalization
  - `sequentialModel1D()`: MLP builder
  - `TwoHotSymlog`: Discretized reward/value prediction for sparse signals

**Auxiliary Tasks** (in `networks.py` and `dreamer.py`):

- **`GoalPredictionHead`**: Binary classification predicting if goal will occur in next K steps
- **`DistanceHead`**: Regression predicting puck-to-goal distance
- **`ShotQualityHead`**: Regression predicting offensive opportunity quality
- These help the world model learn goal-relevant representations without corrupting the reward signal

**Visualization** (`visualization/`):

- **`gif_recorder.py`**: Records gameplay GIFs for W&B
- **`frame_capture.py`**: Frame capture utilities

**Configuration** (`configs/`):

- **`hockey.yml`**: Default configuration with all hyperparameters
  - All values can be overridden via CLI arguments

### COMPRL Client System

**Components** (`comprl-hockey-agent/`):

- **`run_client.py`**: Main tournament client connecting trained agents to server
  - Wraps TD3Agent for remote competition
  - HockeyAgent wrapper for built-in opponents
  - TD3HockeyAgent class integrating trained models
  - Async communication with COMPRL server

- **`autorestart.sh`**: Wrapper script auto-restarting client on connection loss
  - Tracks restart frequency to prevent restart loops
  - Optional ntfy.sh notifications

## Key Design Patterns

### TD3 Training Pipeline

1. **Initialization**: Create environment, opponent(s), agent, metrics tracker
2. **Warmup Phase**: Collect random transitions before training begins
3. **Data Collection**: Run episodes, store transitions in replay buffer
4. **Training Phase**: Sample batches and update agent networks
5. **Evaluation**: Periodically test against benchmarks, save best checkpoints
6. **Self-Play**: Maintain opponent pool and select challengers via PFSP

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

### Checkpoint Format Compatibility

Multiple checkpoint formats supported:
- Tuple format: Direct agent state (Q1, Q2, policy state dicts)
- Dict with 'agent_state' key: Wrapped state
- Dict with network keys ('policy', 'Q1', 'Q2'): Individual components + optimizers

## Important Implementation Details

### TD3-Specific Features

- **Delayed Policy Updates**: Policy updates every N critic updates (policy_freq=2)
- **Twin Critic Networks**: Two Q-functions to reduce overestimation
- **Target Policy Smoothing**: Adds clipped noise to target actions for stability
- **Gaussian Exploration**: N(0, sigma) noise, constant per TD3 paper (no decay)
- **Gradient Clipping**: Prevents exploding gradients (grad_clip=1.0)
- **Q-Value Clipping**: Hard or soft (tanh) clipping prevents Q-value explosion (q_clip=25.0)
- **VF Regularization**: Penalizes near-zero Q-values to prevent passive agents

### DreamerV3-Specific Features

- **RSSM World Model**: Recurrent state space model with deterministic (GRU) + stochastic (categorical) states
- **Categorical Latents**: 16 variables × 16 classes = 256-dim stochastic state (vs Gaussian in v1/v2)
- **Two-Hot Symlog**: Discretized reward/value prediction that handles sparse rewards (goals)
- **Auxiliary Tasks**: Goal prediction, distance, and shot quality heads improve latent representations
- **Free Nats**: KL loss has threshold below which it's not penalized (prevents posterior collapse)
- **Lambda Returns**: TD(λ) for value targets in imagination
- **Value Normalization**: Percentile-based moments for stable advantage computation
- **Imagination Training**: Actor-critic trained entirely in latent space rollouts (no real env gradients)

### Reward Shaping Strategy

- **PBRS Integration**: Computes potential function based on puck distance/velocity
- **Policy Invariance**: Shaped rewards don't change optimal policy (provably)
- **Exploration Guidance**: Denser reward signal guides learning in early phases
- **Configurable Weight**: `--pbrs_scale` controls PBRS influence

### Self-Play Management

- **Opponent Pool**: Circular buffer of N past checkpoints (FIFO rotation)
- **Save Interval**: New opponent added every M episodes during self-play
- **Selection Strategy**:
  - `weak_ratio` defines probability of facing anchor (weak/strong) vs. pool
  - PFSP mode selects from pool based on win rates:
    - `variance`: Prioritizes opponents with ~50% win rate (most learning signal)
    - `hard`: Prioritizes hardest opponents (lowest win rate)
- **Activation**: Self-play activates at configurable episode threshold

## Testing & Validation

Key scenarios to test:

- Single mode tests: NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE
- Opponent types: weak, strong, self
- Position alternation: with/without (tournament mode)
- Checkpoint loading: various format conversions
- Self-play convergence: pool size, save intervals, PFSP selection

## W&B Integration

**TD3 Metrics**:
- Episode rewards and win rates
- Q-value statistics and gradient norms
- Goal scored/conceded metrics
- Model checkpoints (best and periodic saves)

**DreamerV3 Metrics**:
- World model: `world/loss`, `world/recon_loss`, `world/reward_loss`, `world/kl_loss`
- Auxiliary tasks: `world/aux_goal_loss`, `world/aux_distance_loss`, `world/aux_quality_loss`
- Behavior: `behavior/actor_loss`, `behavior/critic_loss`, `behavior/entropy`, `behavior/advantages`
- Stats: `stats/win_rate`, `stats/mean_reward`, `stats/buffer_size`
- Evaluation: `eval/win_rate`, `eval/gif_*` (gameplay GIFs)

## Performance Monitoring

**TD3 Key Metrics**:
- Win rate vs. opponent
- Mean episode reward
- Q-value distribution (min, max, mean)
- Gradient norms

**DreamerV3 Key Metrics**:
- `world/loss`: Should decrease over training
- `behavior/entropy`: Should stay positive (>0) - negative means policy collapsed
- `stats/win_rate`: Should increase over training
- `world/kl_loss`: Should stabilize around free_nats threshold
