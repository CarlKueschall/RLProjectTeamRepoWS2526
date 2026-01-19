# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Python Environment

**ALWAYS USE `conda activate py310` WHEN TRYING TO EXECUTE PYTHON CODE.**

All Python commands should be run in the py310 conda environment. If you encounter import errors or missing packages, ensure the correct environment is activated first.

## Project Overview

This repository contains three main components:

1. **TD3 Training System** (`TD3/`): A reinforcement learning framework for training TD3 (Twin Delayed DDPG) agents to play air hockey
2. **DreamerV3 Training System** (`DreamerV3/`): A world-model-based RL agent that learns entirely in imagination
3. **COMPRL Client** (`comprl-hockey-agent/`): A client that connects trained agents to a competition server

The project implements: **Pure TD3 + PBRS + Self-Play with PFSP** and **DreamerV3 with imagination-based training**.

## Common Commands

### TD3 Training

```bash
# Activate environment first!
conda activate py310

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
# Activate environment first!
conda activate py310

# Basic training with weak opponent
cd DreamerV3
python3 train_hockey.py --mode NORMAL --opponent weak --max_steps 1000000

# Training with custom hyperparameters
python3 train_hockey.py --mode NORMAL --opponent weak \
    --lr_world 3e-4 --lr_actor 8e-5 --lr_critic 1e-4 \
    --entropy_scale 3e-3 --imagination_horizon 15

# Training on GPU cluster
python3 train_hockey.py --mode NORMAL --opponent weak --device cuda --max_steps 10000000
```

### Testing

```bash
conda activate py310

# Test a TD3 checkpoint against weak opponent
cd TD3
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent weak --episodes 100

# Test against strong opponent
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent strong --episodes 100

# Test in tournament mode (no position alternation)
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent weak --no-alternation

# Verbose output showing per-episode results
python3 test_hockey.py --checkpoint ./results/checkpoints/best_model.pth --opponent weak --verbose --episodes 20
```

### Running Competition Client

```bash
conda activate py310

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

**Configuration** (`config/`):

- **`parser.py`**: Command-line argument parser with TD3 hyperparameters
  - Environment settings (mode, opponent, keep_mode)
  - Training settings (episodes, batch size, buffer size)
  - TD3-specific hyperparameters (learning rates, tau, policy frequency)
  - Network architecture (hidden layer sizes)
  - PBRS reward shaping and Q-value clipping settings
  - Self-play settings (pool size, save interval, PFSP mode)

### DreamerV3 Training System

**Philosophy**: DreamerV3 learns a world model and trains the policy entirely in "imagination" (simulated rollouts in the learned latent space). This enables efficient credit assignment for sparse rewards without dense reward shaping.

**Core Components** (`DreamerV3/`):

- **`train_hockey.py`**: Main training loop
  - Collects real experience in replay buffer
  - Trains world model on real data
  - Trains actor-critic entirely in imagined trajectories
  - Logs to W&B with comprehensive metrics

- **`agents/hockey_dreamer.py`**: HockeyDreamer agent class
  - Combines WorldModel and Behavior (actor-critic)
  - Methods: `act()`, `train_step()`, `reset()`, `state()`, `restore_state()`
  - Separate critic and actor training to prevent memory leaks

**World Model** (`models/`):

- **`world_model.py`**: Complete world model
  - **Encoder**: Observation → embedding (with symlog preprocessing)
  - **RSSM Dynamics**: Recurrent State Space Model
    - Deterministic state `h` (GRU hidden state)
    - Stochastic state `z` (Gaussian latent)
    - Prior: p(z|h) - prediction without observation
    - Posterior: q(z|h, embed) - inference with observation
  - **Decoder**: Latent → observation reconstruction
  - **Reward Head**: Latent → reward prediction
  - **Continue Head**: Latent → episode continuation probability

- **`sequence_model.py`**: RSSM implementation
  - `prior_step()`: Predict next state without observation
  - `posterior_step()`: Infer state with observation
  - `observe_sequence()`: Process batch of sequences
  - `imagine_sequence()`: Rollout policy in latent space

- **`behavior.py`**: Actor-Critic for imagination training
  - **Policy**: TanhNormal distribution for bounded actions
  - **Value Network**: Estimates expected return from latent features
  - **Lambda Returns**: TD(λ) for value targets
  - **Entropy Regularization**: Prevents policy collapse

**Utilities** (`utils/`):

- **`buffer.py`**: EpisodeBuffer for sequence sampling
- **`distributions.py`**: TanhNormal distribution implementation
- **`math_ops.py`**: Symlog transform, lambda returns, return normalization

**Key Hyperparameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--imagination_horizon` | 15 | Steps to imagine for actor-critic training |
| `--imagine_batch_size` | 256 | Starting states subsampled for imagination |
| `--entropy_scale` | 3e-3 | Entropy bonus to prevent policy collapse |
| `--lr_world` | 3e-4 | World model learning rate |
| `--lr_actor` | 8e-5 | Actor learning rate |
| `--lr_critic` | 1e-4 | Critic learning rate |
| `--free_nats` | 1.0 | KL loss free bits threshold |

**Known Issues & Solutions**:

1. **Entropy Collapse**: Policy becomes deterministic, stops exploring
   - Solution: Increase `--entropy_scale` (try 3e-3 or higher)
   - Policy min_std is set to 0.2 to provide exploration floor

2. **Memory Leaks**: MPS/CUDA OOM after ~350 episodes
   - Solution: Separate `train_critic()` and `train_actor()` methods
   - Explicit tensor deletion and gc.collect() after each phase

3. **Slow Training**: Imagination too expensive
   - Solution: Reduce `--imagination_horizon` to 15, subsample with `--imagine_batch_size`

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

1. **Data Collection**: Run episode in real environment, store in episode buffer
2. **World Model Training**: Sample sequence batch, compute losses:
   - Reconstruction loss (predict observations)
   - Reward prediction loss
   - Continue prediction loss
   - KL divergence (prior vs posterior)
3. **Imagination**: Sample starting states, rollout policy in latent space
4. **Critic Training**: Compute TD(λ) targets, update value network
5. **Actor Training**: Re-imagine with gradients through policy, update actor

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

- **RSSM World Model**: Recurrent state space model with deterministic + stochastic states
- **Symlog Preprocessing**: Symmetric log transform for observations and rewards
- **Free Nats**: KL loss has a threshold below which it's not penalized
- **TanhNormal Policy**: Bounded actions with entropy regularization
- **Lambda Returns**: TD(λ) for value targets with configurable λ
- **Return Normalization**: Normalizes returns for stable actor training

### Reward Shaping Strategy (TD3)

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

Training logs to Weights & Biases (`rl-hockey` project) with:

**TD3 Metrics:**
- Episode rewards and win rates
- Q-value statistics and gradient norms
- Goal scored/conceded metrics
- PBRS reward components

**DreamerV3 Metrics:**
- World model losses (reconstruction, reward, continue, KL)
- Behavior metrics (actor loss, critic loss, entropy, advantage)
- Imagination statistics (reward mean/std, continue probability)
- Gradient norms for all components

## Performance Monitoring

**TD3 Key Metrics:**
- Win rate vs. opponent
- Mean episode reward
- Q-value distribution (min, max, mean)
- Gradient norms

**DreamerV3 Key Metrics:**
- `behavior/entropy`: Should stay positive (>0), negative means policy collapsed
- `world/loss`: Should decrease over training
- `stats/win_rate`: Should increase over training
- `imagine/reward_mean`: Predicted rewards in imagination
