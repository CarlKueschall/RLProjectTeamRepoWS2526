# Simplified DreamerV3 Implementation Plan for Hockey

## Executive Summary

This plan outlines a **minimal, focused DreamerV3 implementation** specifically designed for the hockey environment. We prioritize:
- **Simplicity**: Remove complexity that doesn't benefit our use case
- **Clarity**: Code that's explainable in a report
- **Self-Play Integration**: Clean interface for opponent management

---

## Analysis: What We Need vs. What We Can Remove

### Reference Implementation Complexity (dreamerv3-torch)
| Component | Lines | Purpose | **Keep?** |
|-----------|-------|---------|-----------|
| CNN Encoder/Decoder | ~200 | Image processing | **NO** - Hockey is 18-dim vectors |
| Discrete Categorical Latent (32x32) | ~100 | Robust latent space | **SIMPLIFY** - Use Gaussian |
| Two-Hot Bucket Encoding | ~150 | Handle extreme rewards | **NO** - Rewards bounded [-1, +1] |
| Multi-task decoder | ~100 | Image reconstruction | **NO** - No images |
| Block-diagonal GRU | ~50 | Parameter efficiency | **NO** - Use standard GRU |
| LaProp optimizer | ~80 | Stability | **NO** - Adam works fine |
| Symlog transforms | ~20 | Handle extreme values | **OPTIONAL** - Keep for robustness |
| Return EMA percentile | ~30 | Normalize returns | **YES** - Important for stability |

### What We Keep (Core DreamerV3)
1. **RSSM World Model** - The core innovation
2. **Imagination-based Actor-Critic** - Credit assignment through dreaming
3. **KL Balancing + Free Bits** - Latent space regularization
4. **Continue Predictor** - Discount-aware planning
5. **Lambda Returns** - TD(λ) value targets

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HockeyDreamer Agent                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    WORLD MODEL                            │  │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────────────────────┐ │  │
│  │  │ Encoder │──▶│  RSSM   │──▶│ Predictors (R, γ, obs)  │ │  │
│  │  │  (MLP)  │   │(GRU+VAE)│   │                         │ │  │
│  │  └─────────┘   └─────────┘   └─────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  BEHAVIOR (Actor-Critic)                  │  │
│  │  ┌─────────────┐            ┌─────────────┐              │  │
│  │  │   Policy    │            │   Value     │              │  │
│  │  │  (Tanh Gaussian)         │   (MLP)     │              │  │
│  │  └─────────────┘            └─────────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure (Our Design)

```
DreamerV3/
├── agents/
│   ├── __init__.py
│   └── hockey_dreamer.py      # Main agent class
├── models/
│   ├── __init__.py
│   ├── world_model.py         # RSSM + predictors
│   ├── sequence_model.py      # GRU dynamics
│   └── behavior.py            # Actor + Critic
├── utils/
│   ├── __init__.py
│   ├── distributions.py       # TanhNormal, etc.
│   ├── math_ops.py            # symlog, lambda_returns
│   └── buffer.py              # Episode replay buffer
├── config/
│   └── parser.py              # Arguments
├── envs/
│   └── hockey_wrapper.py      # Sparse reward wrapper
├── opponents/
│   └── self_play.py           # Self-play manager
└── train_hockey.py            # Training loop
```

---

## Component Specifications

### 1. World Model (`models/world_model.py`)

**Class: `WorldModel`**

```python
class WorldModel(nn.Module):
    """
    Learns environment dynamics via RSSM.

    State = (h, z) where:
      - h: deterministic recurrent state (GRU hidden)
      - z: stochastic latent state (Gaussian)
    """

    def __init__(self, obs_dim, action_dim, config):
        # Components:
        self.encoder      # obs -> embedding
        self.dynamics     # SequenceModel (RSSM)
        self.obs_decoder  # state -> obs prediction
        self.reward_head  # state -> reward prediction
        self.continue_head # state -> continue probability
```

**Key Methods:**
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `observe(obs, action, state)` | Real data | posterior, prior | Encode real sequence |
| `imagine(policy, start_state, horizon)` | Policy function | imagined trajectory | Dream future states |
| `compute_loss(batch)` | Sequence batch | world_loss, metrics | Train world model |

**Hyperparameters (Hockey-specific):**
```python
hidden_size = 256      # MLP hidden dimension
latent_size = 64       # Stochastic state z dimension
recurrent_size = 256   # GRU hidden h dimension
kl_free_bits = 1.0     # Free bits threshold
kl_dyn_scale = 0.5     # Dynamics KL weight
kl_rep_scale = 0.1     # Representation KL weight
```

---

### 2. Sequence Model (`models/sequence_model.py`)

**Class: `SequenceModel`** (Our name for RSSM)

```python
class SequenceModel(nn.Module):
    """
    Recurrent State-Space Model with separate prior/posterior.

    Prior:     p(z_t | h_t) - predict from dynamics only
    Posterior: q(z_t | h_t, obs_t) - infer from dynamics + observation
    """

    def __init__(self, embed_dim, action_dim, hidden_size, latent_size, recurrent_size):
        self.gru_cell          # Standard PyTorch GRUCell
        self.prior_net         # h -> (mean, std) for z
        self.posterior_net     # (h, embed) -> (mean, std) for z
```

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `initial_state(batch)` | Get zero initial (h, z) |
| `prior_step(state, action)` | Predict next state without observation |
| `posterior_step(state, action, embed)` | Infer next state with observation |
| `get_features(state)` | Concatenate [h, z] for downstream heads |

---

### 3. Behavior Model (`models/behavior.py`)

**Class: `Behavior`**

```python
class Behavior(nn.Module):
    """
    Actor-Critic trained entirely in imagination.
    """

    def __init__(self, feature_dim, action_dim, config):
        self.policy    # MLP -> TanhNormal distribution
        self.value     # MLP -> scalar value estimate
        self.value_target  # Slow-moving target for stability
```

**Policy Output:** TanhNormal distribution (bounded [-1, 1])
- Outputs mean and std
- Sample with reparameterization trick
- Squash through tanh

**Value Training:**
- Use lambda-returns as targets
- MSE loss (no two-hot needed for bounded rewards)
- Slow target update (τ = 0.02)

**Actor Training:**
- Maximize expected value in imagination
- Add entropy bonus for exploration
- Return normalization with EMA percentiles

---

### 4. Main Agent (`agents/hockey_dreamer.py`)

**Class: `HockeyDreamer`**

```python
class HockeyDreamer(nn.Module):
    """
    Complete DreamerV3 agent for hockey.

    Clean interface for:
    - Acting in environment
    - Training on batches
    - Self-play opponent management
    """

    # === SELF-PLAY INTERFACE ===
    def state(self) -> dict:
        """Serialize agent for opponent pool."""

    def restore_state(self, state: dict):
        """Load agent state for evaluation."""

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation."""

    # === TRAINING INTERFACE ===
    def train_step(self, batch: dict) -> dict:
        """One training iteration. Returns metrics."""

    def reset(self):
        """Reset recurrent state for new episode."""
```

---

### 5. Replay Buffer (`utils/buffer.py`)

**Class: `EpisodeBuffer`**

Store complete episodes, sample sequences:

```python
class EpisodeBuffer:
    """
    Episode-based replay buffer for DreamerV3.

    Stores complete episodes and samples fixed-length sequences.
    """

    def add_episode(self, episode: dict):
        """Add completed episode {obs, action, reward, done}."""

    def sample(self, batch_size: int, seq_length: int) -> dict:
        """Sample batch of sequences for training."""
```

---

## Training Loop

```python
def train_step(agent, buffer, config):
    """One training iteration."""

    # 1. Sample batch of sequences
    batch = buffer.sample(config.batch_size, config.seq_length)

    # 2. Train world model on real sequences
    world_metrics = agent.world_model.compute_loss(batch)

    # 3. Get posterior states as imagination starting points
    with torch.no_grad():
        states = agent.world_model.observe(batch)

    # 4. Imagine trajectories using current policy
    imagined = agent.world_model.imagine(
        policy=agent.behavior.policy,
        start_states=states,
        horizon=config.horizon
    )

    # 5. Train actor-critic in imagination
    behavior_metrics = agent.behavior.train(imagined)

    return {**world_metrics, **behavior_metrics}
```

---

## Self-Play Integration

The agent interface is designed for seamless self-play:

```python
# Save current agent to opponent pool
opponent_state = agent.state()
pool.add_opponent(opponent_state, episode_number)

# Load opponent from pool
opponent_state = pool.get_opponent()
opponent_agent = HockeyDreamer(config)
opponent_agent.restore_state(opponent_state)

# Use opponent in environment
def get_opponent_action(obs):
    return opponent_agent.act(obs, deterministic=True)

env.set_opponent(get_opponent_action)
```

---

## Key Simplifications vs. Reference

| Reference (dreamerv3-torch) | Our Version | Reasoning |
|-----------------------------|-------------|-----------|
| Discrete categorical z (32x32=1024) | Gaussian z (64-dim) | Simpler, works for continuous control |
| Two-hot bucket encoding (255 bins) | MSE regression | Rewards bounded [-1, +1] |
| CNN encoder/decoder | MLP only | Vector observations (18-dim) |
| Block-diagonal GRU | Standard GRUCell | Simpler, sufficient for small state |
| LaProp optimizer | Adam | Simpler, widely understood |
| Multiple observation heads | Single obs decoder | No images, just vectors |
| Complex config YAML | Simple argparse | Easier to modify |

---

## Hyperparameters (Hockey-Optimized)

```python
# World Model
hidden_size = 256
latent_size = 64
recurrent_size = 256
seq_length = 50
kl_free = 1.0
kl_dyn_scale = 0.5
kl_rep_scale = 0.1

# Behavior
horizon = 15            # Imagination rollout length
gamma = 0.997           # Discount factor
lambda_gae = 0.95       # GAE lambda
entropy_scale = 3e-4    # Entropy bonus
return_norm_decay = 0.99  # EMA for return normalization

# Training
batch_size = 16
lr_world = 3e-4
lr_actor = 3e-5
lr_critic = 3e-5
grad_clip = 100.0
target_update_tau = 0.02

# Buffer
buffer_size = 1_000_000
min_buffer_size = 1000
```

---

## Implementation Order

### Phase 1: Core Components (Priority)
1. `utils/math_ops.py` - symlog, lambda_returns
2. `utils/distributions.py` - TanhNormal
3. `models/sequence_model.py` - RSSM core
4. `models/world_model.py` - Full world model
5. `models/behavior.py` - Actor-Critic

### Phase 2: Agent & Training
6. `agents/hockey_dreamer.py` - Main agent class
7. `utils/buffer.py` - Episode buffer
8. `train_hockey.py` - Training loop

### Phase 3: Integration
9. Self-play integration with existing `opponents/self_play.py`
10. W&B logging integration
11. Evaluation scripts

---

## Testing Strategy

1. **Unit tests**: Each component in isolation
2. **Gradient flow**: Verify gradients propagate correctly
3. **Sanity checks**:
   - World model reconstruction improves
   - KL loss stays bounded
   - Value predictions are reasonable
4. **Integration**: Full training on hockey

---

## Report-Ready Explanations

The implementation should be explainable as:

> "We implemented a simplified DreamerV3 agent adapted for the hockey environment. The core innovation is the **Recurrent State-Space Model (RSSM)** which learns to predict future states, enabling the agent to **learn behavior entirely in imagination**. This allows efficient credit assignment even with sparse rewards (win/loss only).
>
> Key simplifications from the original:
> 1. Gaussian latent states instead of categorical (simpler, works for continuous control)
> 2. MSE value regression instead of two-hot encoding (rewards are bounded)
> 3. MLP encoder instead of CNN (vector observations)
>
> The agent is trained in two phases each iteration:
> 1. **World Model**: Learn dynamics from real experience
> 2. **Behavior**: Learn policy by imagining 15-step rollouts"

---

## References

- **Code Base**: NM512/dreamerv3-torch (MIT License)
- **Paper**: Hafner et al., "Mastering Diverse Domains through World Models" (2023)
- **Robot Air Hockey**: Orsula et al., 2nd place with DreamerV3 + sparse rewards

---

## AI Usage Declaration

This implementation plan and subsequent code was developed with assistance from Claude Code (Anthropic). The plan synthesizes concepts from:
- The official DreamerV3 paper
- The dreamerv3-torch reference implementation
- Our specific requirements for the hockey environment

The actual implementation will be our own design with original class/variable naming.
