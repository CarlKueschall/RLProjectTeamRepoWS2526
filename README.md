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

### Reward Shaping (PBRS)

Potential-Based Reward Shaping provides dense rewards while preserving optimal policy.

**Design Philosophy:**
- MINIMAL shaping: Only guide behaviors hard to discover from sparse rewards
- AVOID encoding strategy: Let agent discover shooting angles, timing, etc.
- PREVENT exploits: No reward for standing near stationary puck

#### Sparse Environment Rewards

| Signal | Raw Value | After `reward_scale=0.1` | Description |
|--------|-----------|--------------------------|-------------|
| **Win** | +10 | +1.0 | Terminal reward for scoring a goal |
| **Loss** | -10 | -1.0 | Terminal penalty for conceding a goal |
| **Tie** | 0 | 0 | No reward for draws |

#### PBRS Potential Components

| Component | Range | Active When | Intention |
|-----------|-------|-------------|-----------|
| **φ_chase** | [-100, 0] | Puck moving (speed > 0.3) | Reward being close to MOVING puck. Helps defense/interception. **PREVENTS**: standing next to stationary puck |
| **φ_defensive** | [-40, 0] | Puck in own half (x < 0) | Triangle defense positioning: 40% from goal to puck |

**What We Intentionally DO NOT Encode:**
- Shot direction (let agent discover bank shots, angles)
- Puck-to-goal distance (not directly actionable)
- Possession rewards (would encourage hoarding)

#### Scaling Derivation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `EPISODE_SCALE` | 100 | Internal scaling for numerical stability |
| `pbrs_scale` | 0.02 | Ensures episode PBRS < sparse reward |
| `reward_scale` | 0.1 | Maps sparse ±10 to ±1 |

**Mathematical Guarantee:**
```
Max potential range: ~105 (worst to best state)
Max episode PBRS: 105 × 0.02 = 2.1
Sparse reward (win): 10
Ratio: 2.1 / 10 = 0.21 < 1 ✓
```

#### Complete Reward Formula

```
r_total = r_sparse × reward_scale + pbrs_scale × (γ·φ(s') - φ(s))
        = r_sparse × 0.1          + 0.02       × (0.99·φ(s') - φ(s))
```

| Term | Typical Magnitude | Role |
|------|-------------------|------|
| Sparse (win/loss) | ±1.0 | Primary optimization target |
| PBRS per-step | ±0.02 | Dense learning signal |
| PBRS per-episode | ±0.5 to ±2.0 | Guides exploration |

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

## SAC Agent - Serhat Alpay

#TODO
