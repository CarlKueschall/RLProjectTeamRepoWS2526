# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two main components:

1. **TD3 Training System** (`TD3/`): A reinforcement learning framework for training TD3 (Twin Delayed DDPG) agents to play air hockey
2. **COMPRL Client** (`comprl-hockey-agent/`): A client that connects trained agents to a competition server

The project implements: **Pure TD3 + PBRS + Self-Play with PFSP**

## Common Commands

### Training

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

### Training Pipeline

1. **Initialization**: Create environment, opponent(s), agent, metrics tracker
2. **Warmup Phase**: Collect random transitions before training begins
3. **Data Collection**: Run episodes, store transitions in replay buffer
4. **Training Phase**: Sample batches and update agent networks
5. **Evaluation**: Periodically test against benchmarks, save best checkpoints
6. **Self-Play**: Maintain opponent pool and select challengers via PFSP

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

Training logs to Weights & Biases with:
- Episode rewards and win rates
- Q-value statistics and gradient norms
- Goal scored/conceded metrics
- Model checkpoints (best and periodic saves)
- Training hyperparameters for experiment tracking

## Performance Monitoring

Key metrics tracked in `metrics_tracker.py`:
- Win rate vs. opponent
- Mean episode reward
- Q-value distribution (min, max, mean)
- Gradient norms
- Training efficiency (steps per episode)
