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

### Reward Shaping (PBRS V3)

Potential-Based Reward Shaping V3 provides **balanced offense/defense incentives** while preserving optimal policy (Ng et al., 1999).

**Design Philosophy:**
- THREE components: φ_chase + φ_attack + φ_defense
- **ASYMMETRIC φ_attack**: Only active in opponent half (no penalty for defensive positions)
- **FULL φ_chase in our half**: Stationary puck reduction only in opponent half (encourage defensive recovery)
- **NEW φ_defense**: Reward being between puck and own goal during defensive situations

#### Sparse Environment Rewards

| Signal | Raw Value | After `reward_scale=0.1` | Description |
|--------|-----------|--------------------------|-------------|
| **Win** | +10 | +1.0 | Terminal reward for scoring a goal |
| **Loss** | -10 | -1.0 | Terminal penalty for conceding a goal |
| **Tie** | 0 | 0 | No reward for draws |

#### PBRS V3 Potential Components

| Component | Weight | Range | Active When | Description |
|-----------|--------|-------|-------------|-------------|
| **φ_chase** | W=0.5 | [-1, 0] | Always | Reward agent proximity to puck. Full strength when moving OR in our half. 30% when stationary in opponent half. |
| **φ_attack** | W=0.7 | [-1, 0] | Puck in opponent half | Reward puck proximity to opponent goal. **Disabled in our half** to avoid penalizing defense. |
| **φ_defense** | W=0.3 | [-1, +1] | Puck in our half | Reward being between puck and own goal. Encourages proper defensive positioning. |

**Combined Potential:**
```
φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack + W_DEFENSE × φ_defense
     = 0.5 × chase + 0.7 × attack + 0.3 × defense
```

#### Reward Matrix (V3 Behavior)

| Situation | φ_chase | φ_attack | φ_defense | Net | Result |
|-----------|---------|----------|-----------|-----|--------|
| **Chase puck (opponent half)** | + | 0 | 0 | **+** | Encouraged |
| **Shoot toward opponent goal** | - | + | 0 | **+** | Encouraged |
| **Chase puck (our half)** | + | 0 | + | **++** | STRONGLY encouraged! |
| **Good defensive position** | ~ | 0 | + | **+** | Encouraged |
| **Ignore puck in our half** | 0 | 0 | - | **-** | Penalized |
| **Camp near stationary puck** | weak | 0 | 0 | **~** | Not rewarded |

#### Key V3 Improvements

1. **Defensive Recovery**: When opponent shoots into our half, φ_chase is FULL strength (not 30%) to encourage chasing
2. **No Defensive Penalty**: φ_attack = 0 when puck is in our half, so agent isn't penalized for being in defensive situations
3. **Defensive Positioning**: φ_defense rewards getting between puck and own goal when under attack

#### Annealing Strategy

PBRS anneals slowly from 1.0 to minimum weight over training:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `pbrs_anneal_start` | 5000 | Episode to start annealing |
| `pbrs_anneal_episodes` | 15000 | Duration of annealing phase |
| `pbrs_min_weight` | 0.1 | Minimum weight (never fully turns off) |
| `epsilon_reset_at_anneal` | True | Re-enable exploration when annealing starts |

**Timeline (100k training):**
```
Episode 0-warmup:    epsilon = initial (no decay during warmup)
Episode warmup+:     epsilon starts decaying
Episode 0-5000:      PBRS weight = 1.0 (full guidance)
Episode 5000-20000:  PBRS weight = 1.0 → 0.1 (gradual fade)
Episode 20000+:      PBRS weight = 0.1 (minimal guidance retained)
```

#### Scaling Derivation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `EPISODE_SCALE` | 100 | Internal scaling for numerical stability |
| `pbrs_scale` | 0.02 | Ensures episode PBRS < sparse reward |
| `reward_scale` | 0.1 | Maps sparse +/-10 to +/-1 |

#### Complete Reward Formula

```
r_total = r_sparse × reward_scale + pbrs_scale × weight(episode) × (γ·φ(s') - φ(s))
        = r_sparse × 0.1          + 0.02       × weight          × (0.99·φ(s') - φ(s))
```

| Term | Typical Magnitude | Role |
|------|-------------------|------|
| Sparse (win/loss) | +/-1.0 | Primary optimization target |
| PBRS per-step | +/-0.03 | Dense learning signal |
| PBRS per-episode | +/-1.0 to +/-2.0 | Guides exploration |

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
