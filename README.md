# RL Hockey Project

Authors: Serhat Alpay, Carl Kueschall

Reinforcement Learning project for training agents to play hockey using TD3 algorithm with self-play.

## Quick Start

```bash
cd 02-SRC/TD3

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

# Full configuration example
python train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 25000 \
    --seed 42 \
    --eps 0.1 \
    --eps_min 0.1 \
    --warmup_episodes 2000 \
    --batch_size 100 \
    --iter_fit 250 \
    --lr_actor 0.001 \
    --lr_critic 0.001 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    --buffer_size 1000000 \
    --reward_shaping \
    --pbrs_scale 0.02 \
    --q_clip 25.0 \
    --q_clip_mode soft \
    --hidden_actor 256 256 \
    --hidden_critic 256 256 128 \
    --log_interval 10 \
    --save_interval 500 \
    --eval_interval 1000
```

## Testing

```bash
cd 02-SRC/TD3

# Test against weak opponent
python test_hockey.py \
    --checkpoint results/checkpoints/best_model.pth \
    --opponent weak \
    --episodes 100

# Test with visualization
python test_hockey.py \
    --checkpoint results/checkpoints/best_model.pth \
    --opponent weak \
    --episodes 10 \
    --render human

# Tournament mode (no position alternation)
python test_hockey.py \
    --checkpoint results/checkpoints/best_model.pth \
    --opponent strong \
    --episodes 100 \
    --no-alternation
```

## Key Features

- **TD3 Algorithm**: Twin Delayed DDPG with target smoothing and Q-value clipping
- **PBRS**: Potential-Based Reward Shaping for dense exploration guidance
- **Self-Play**: Opponent pool management with PFSP (Prioritized Fictitious Self-Play) selection
- **VF Regularization**: Anti-lazy learning to prevent passive agents

## File Structure

```
02-SRC/TD3/
├── train_hockey.py          # Main training script
├── test_hockey.py           # Testing script
├── config/
│   └── parser.py            # Command line arguments
├── agents/
│   ├── td3_agent.py         # TD3 agent implementation
│   ├── model.py             # Neural network models (MLP)
│   ├── memory.py            # Replay buffer
│   ├── noise.py             # Gaussian noise for exploration
│   └── device.py            # Device management (CPU/GPU/MPS)
├── rewards/
│   └── pbrs.py              # Potential-Based Reward Shaping
├── opponents/
│   ├── self_play.py         # Self-play manager with PFSP
│   ├── pfsp.py              # PFSP opponent selection weights
│   ├── fixed.py             # Fixed opponent wrapper
│   └── base.py              # Base opponent class
├── evaluation/
│   └── evaluator.py         # Evaluation functions
├── metrics/
│   └── metrics_tracker.py   # Metrics tracking for W&B
└── visualization/
    ├── gif_recorder.py      # GIF recording for W&B
    └── frame_capture.py     # Frame capture utilities
```

## Main Components

### TD3 Agent

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

#### Reward Matrix (V3.1 Behavior)

| Action | φ_chase | φ_attack | Net PBRS | Result |
|--------|---------|----------|----------|--------|
| **Chase puck** | +1.0 | 0 | **+1.0** | STRONG encourage |
| **Shoot forward** | -1.0 | +1.2 | **+0.2** | Encouraged |
| **Shoot backward** | -1.0 | -1.2 | **-2.2** | Heavily penalized |
| **Puck in our half** | chase active | penalty active | chase dominates | Agent races to puck |

#### Why This Works

1. **Strong chase handles everything**: Defense, interception, ready position
2. **Puck in our half**: φ_attack penalty creates urgency, strong φ_chase drives agent to puck
3. **Shooting preserved**: W_ATTACK > W_CHASE ensures forward shots are always net positive
4. **Simple and robust**: No conditional logic, no edge cases

#### Annealing Strategy

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `pbrs_anneal_start` | 5000 | Episode to start annealing |
| `pbrs_anneal_episodes` | 15000 | Duration of annealing phase |
| `pbrs_min_weight` | 0.1 | Minimum weight (never fully turns off) |

**Timeline (100k training):**
```
Episode 0-warmup:    epsilon = initial (no decay during warmup)
Episode warmup+:     epsilon starts decaying
Episode 0-5000:      PBRS weight = 1.0 (full guidance)
Episode 5000-20000:  PBRS weight = 1.0 → 0.1 (gradual fade)
Episode 20000+:      PBRS weight = 0.1 (minimal guidance retained)
```

#### Scaling

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `EPISODE_SCALE` | 100 | Internal scaling for numerical stability |
| `pbrs_scale` | 0.02 | Ensures episode PBRS < sparse reward |
| `reward_scale` | 0.1 | Maps sparse +/-10 to +/-1 |

**Max potential range:** 220 (W_CHASE × 1 + W_ATTACK × 1 = 2.2, × 100)
**Max episode PBRS:** 220 × 0.02 = 4.4 (less than sparse reward of 10)

### Self-Play

Automatic curriculum learning through self-play:
- **Opponent Pool**: Circular buffer of past checkpoints (FIFO)
- **PFSP Selection**: Prioritizes opponents with ~50% win rate (variance mode) or hardest opponents (hard mode)
- **Anchor Mixing**: Balances training against weak/strong anchors vs self-play pool
- Activates at configurable episode threshold (`--self_play_start`)

## W&B Integration

Training logs to Weights & Biases:
- Episode rewards and win rates
- Q-value statistics (mean, min, max)
- Gradient norms (actor/critic)
- Goal scored/conceded metrics
- Model checkpoints (best and periodic)

---

## DreamerV3 Agent

World-model based RL agent that learns entirely in imagination. Based on NaturalDreamer, simplified for hockey with MLP encoder/decoder.

### Quick Start

```bash
cd 02-SRC/DreamerV3

# Basic training against weak opponent
python train_hockey.py --opponent weak --seed 42

# Training with PBRS disabled
python train_hockey.py --opponent weak --no_pbrs

# Training with custom hyperparameters
python train_hockey.py \
    --opponent weak \
    --seed 42 \
    --lr_world 3e-4 \
    --lr_actor 8e-5 \
    --lr_critic 1e-4 \
    --entropy_scale 0.003 \
    --imagination_horizon 15 \
    --batch_size 32 \
    --batch_length 32

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
    --entropy_scale 0.003 \
    --use_pbrs \
    --pbrs_scale 0.03 \
    --checkpoint_interval 5000 \
    --eval_interval 1000 \
    --gif_interval 10000 \
    --wandb_project rl-hockey
```

### Key Features

- **World Model**: RSSM with categorical latents (16x16 = 256 dim stochastic state)
- **Imagination Training**: Actor-critic trained entirely in latent space rollouts
- **MLP Architecture**: Simple encoder/decoder for 18-dim observations
- **PBRS Integration**: Optional dense reward shaping for faster exploration
- **GIF Recording**: Periodic gameplay GIFs logged to W&B

### File Structure

```
02-SRC/DreamerV3/
├── train_hockey.py          # Main training script
├── dreamer.py               # Dreamer agent (world model + behavior)
├── networks.py              # Neural network components
├── buffer.py                # Replay buffer for sequences
├── utils.py                 # Helpers (lambda returns, moments, etc.)
├── configs/
│   └── hockey.yml           # Default configuration
├── rewards/
│   └── pbrs.py              # Potential-Based Reward Shaping
└── visualization/
    ├── gif_recorder.py      # GIF recording for W&B
    └── frame_capture.py     # Frame capture utilities
```

### Architecture Overview

**World Model (RSSM)**:
- **Encoder**: MLP mapping 18-dim observations to embeddings
- **Recurrent Model**: GRU for deterministic state evolution
- **Prior/Posterior**: Categorical latent prediction (16 vars × 16 classes)
- **Decoder**: MLP reconstructing observations from full state
- **Reward Model**: Normal distribution for reward prediction
- **Continue Model**: Bernoulli for episode termination

**Behavior (Actor-Critic)**:
- **Actor**: Tanh-squashed Gaussian policy
- **Critic**: Normal distribution for value estimation
- **Training**: Lambda returns with value normalization

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--imagination_horizon` | 15 | Steps to imagine for actor-critic training |
| `--batch_size` | 32 | Sequences per batch |
| `--batch_length` | 32 | Timesteps per sequence |
| `--entropy_scale` | 0.003 | Entropy bonus for exploration |
| `--lr_world` | 3e-4 | World model learning rate |
| `--lr_actor` | 8e-5 | Actor learning rate |
| `--lr_critic` | 1e-4 | Critic learning rate |
| `--discount` | 0.997 | Discount factor γ |
| `--free_nats` | 1.0 | KL free nats threshold |
| `--gif_interval` | 10000 | GIF recording interval (0=disabled) |

### W&B Metrics

Training logs to Weights & Biases with:
- **World Model**: reconstruction loss, reward loss, KL divergence
- **Behavior**: actor loss, critic loss, entropy, advantages
- **Stats**: win rate, episode rewards, buffer size
- **Visualization**: Periodic gameplay GIFs

## SAC Agent - Serhat Alpay

#TODO
