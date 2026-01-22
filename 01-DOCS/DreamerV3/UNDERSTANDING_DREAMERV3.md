# Understanding DreamerV3: A Complete Guide

**Authors**: Carl Kueschall, Serhat Alpay
**Last Updated**: January 2026
**Project**: Air Hockey RL Agent

---

## Table of Contents

1. [Philosophy: Why World Models?](#1-philosophy-why-world-models)
2. [Architecture Overview](#2-architecture-overview)
3. [The World Model (RSSM)](#3-the-world-model-rssm)
4. [Training the World Model](#4-training-the-world-model)
5. [Imagination: Training in Latent Space](#5-imagination-training-in-latent-space)
6. [The Actor (Policy)](#6-the-actor-policy)
7. [The Critic (Value Function)](#7-the-critic-value-function)
8. [Entropy & Exploration](#8-entropy--exploration)
9. [Value Normalization (The Moments Class)](#9-value-normalization-the-moments-class)
10. [Two-Hot Symlog Encoding](#10-two-hot-symlog-encoding)
11. [DreamSmooth: Handling Sparse Rewards](#11-dreamsmooth-handling-sparse-rewards)
12. [The Complete Training Loop](#12-the-complete-training-loop)
13. [Hyperparameters Explained](#13-hyperparameters-explained)
14. [Common Pitfalls & Debugging](#14-common-pitfalls--debugging)
15. [Our Hockey-Specific Adaptations](#15-our-hockey-specific-adaptations)

---

## 1. Philosophy: Why World Models?

### The Problem with Model-Free RL

Traditional model-free methods (PPO, SAC, TD3) learn policies through trial and error:
- Agent takes action → Environment returns reward → Update policy
- **Problem**: Sparse rewards (like ±10 for goals in hockey) provide almost no learning signal
- **Problem**: Each environment step is "expensive" (can't reuse data efficiently)

### DreamerV3's Solution: Learn the World, Then Imagine

DreamerV3 takes a different approach:

1. **Learn a World Model**: Predict what happens next in a compressed "latent" space
2. **Imagine Trajectories**: Simulate thousands of possible futures without touching the real environment
3. **Train on Imagination**: Actor-critic learns entirely from imagined rollouts

**Key Insight**: By learning to predict the future, the agent can:
- Do credit assignment over long horizons (connect goal to actions 100 steps earlier)
- Train efficiently from limited real-world data (high replay ratio)
- Handle sparse rewards (world model learns structure, not just rewards)

### The DreamerV3 Mantra

> "Don't just react to rewards. Understand the world, then plan."

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        REAL ENVIRONMENT                          │
│   obs_t ──────────────────────────────────────────────► obs_t+1 │
│              │                                    ▲              │
│              │ action_t                           │              │
│              │                                    │              │
└──────────────┼────────────────────────────────────┼──────────────┘
               │                                    │
               ▼                                    │
┌──────────────────────────────────────────────────────────────────┐
│                         WORLD MODEL                               │
│                                                                   │
│   ┌─────────┐    ┌──────────────────────────────────────┐        │
│   │ ENCODER │───►│              RSSM                     │        │
│   │  (MLP)  │    │  ┌───────────┐    ┌──────────────┐   │        │
│   └─────────┘    │  │ GRU       │───►│ Categorical  │   │        │
│        │         │  │ (determ.) │    │ (stochastic) │   │        │
│        │         │  └───────────┘    └──────────────┘   │        │
│        │         │       h_t              z_t           │        │
│   obs_t│         └──────────────────────────────────────┘        │
│        │                        │                                 │
│        │                        ▼ full_state = [h_t, z_t]        │
│        │         ┌──────────────────────────────────────┐        │
│        │         │           PREDICTION HEADS            │        │
│        │         │  ┌─────────┐ ┌────────┐ ┌─────────┐  │        │
│        │         │  │ DECODER │ │ REWARD │ │CONTINUE │  │        │
│        │         │  │reconstruct│ │ r_hat │ │  c_hat │  │        │
│        │         │  └─────────┘ └────────┘ └─────────┘  │        │
│        │         └──────────────────────────────────────┘        │
└────────┼─────────────────────────────────────────────────────────┘
         │
         │ (During Imagination - No Real Observations!)
         │
┌────────┼─────────────────────────────────────────────────────────┐
│        │                ACTOR-CRITIC (Behavior)                   │
│        │                                                          │
│        │    full_state ──────────────────────────────────►       │
│        │         │                                    │           │
│        │         ▼                                    ▼           │
│        │    ┌─────────┐                         ┌─────────┐      │
│        │    │  ACTOR  │                         │ CRITIC  │      │
│        │    │ (Policy)│                         │ (Value) │      │
│        │    └─────────┘                         └─────────┘      │
│        │         │                                    │           │
│        │         ▼                                    ▼           │
│        │    action ~ π(a|s)                      V(s)            │
│        │                                                          │
└──────────────────────────────────────────────────────────────────┘
```

### Components Summary

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Encoder** | obs (18-dim) | embed (256-dim) | Compress observation |
| **RSSM** | (h, z, action, embed) | (h', z') | Predict next state |
| **Decoder** | full_state (512-dim) | obs_hat (18-dim) | Reconstruct observation |
| **Reward Head** | full_state | reward (Two-Hot) | Predict reward |
| **Continue Head** | full_state | prob (Bernoulli) | Predict episode end |
| **Actor** | full_state | action dist (TanhNormal) | Policy |
| **Critic** | full_state | value (Two-Hot) | Value estimation |

---

## 3. The World Model (RSSM)

### What is RSSM?

**Recurrent State Space Model** - The heart of DreamerV3's world model.

The state has two components:
1. **Deterministic state `h`**: GRU hidden state (256-dim)
   - Captures temporal dependencies (memory)
   - Updated recurrently: `h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])`

2. **Stochastic state `z`**: Categorical latent (16 vars × 16 classes = 256-dim)
   - Captures uncertainty about the world
   - Sampled from learned distribution

**Full State**: `s_t = concat(h_t, z_t)` → 512-dim vector used for all predictions.

### Why Categorical Latents? (DreamerV3 vs v1/v2)

| Version | Stochastic Type | Issue |
|---------|----------------|-------|
| DreamerV1/V2 | Gaussian | Posterior collapse, training instability |
| **DreamerV3** | **Categorical** | More robust, prevents collapse |

**Categorical Advantages**:
- Can't collapse to a single point (always has 16 classes)
- Natural discretization helps world model learn distinct states
- 1% uniform mixing (`uniformMix=0.01`) ensures all categories stay active

### Prior vs Posterior

**Posterior** (used during world model training):
```
z_t ~ Posterior(h_t, embed_t)
```
Has access to current observation → can "cheat" by looking at what actually happened.

**Prior** (used during imagination):
```
z_t ~ Prior(h_t)
```
No observation → must predict purely from internal state.

**KL Divergence Loss**: Forces prior to match posterior, so imagination is accurate.

---

## 4. Training the World Model

### What the World Model Learns

The world model is trained to predict:
1. **Observations** (reconstruction loss)
2. **Rewards** (reward prediction loss)
3. **Episode termination** (continue loss)
4. **Consistent dynamics** (KL divergence loss)

### Loss Function

```python
world_model_loss = reconstruction_loss + reward_loss + continue_loss + kl_loss
```

**Reconstruction Loss** (Decoder):
```python
# Predict observation from latent state
obs_hat = decoder(full_state)
recon_loss = MSE(obs_hat, obs)
```

**Reward Loss** (Two-Hot Symlog):
```python
# Predict reward as categorical distribution
reward_logits = reward_head(full_state)
reward_loss = two_hot_symlog.loss(reward_logits, actual_reward)
```

**Continue Loss** (Bernoulli):
```python
# Predict probability of episode continuing
continue_logit = continue_head(full_state)
continue_loss = BCE(continue_logit, ~done)
```

**KL Divergence Loss** (Prior ↔ Posterior):
```python
# Force prior to match posterior
kl_prior = KL(posterior || prior)      # Train prior
kl_posterior = KL(prior || posterior)  # Regularize posterior

kl_loss = beta_prior * max(kl_prior, free_nats) +
          beta_posterior * max(kl_posterior, free_nats)
```

### Free Nats: Preventing Posterior Collapse

**Problem**: If KL loss pushes too hard, posterior collapses to prior (loses information).

**Solution**: Don't penalize KL below `free_nats` threshold (default: 1.0).

```python
kl_loss = max(kl, free_nats)  # KL must exceed 1.0 to be penalized
```

This allows the posterior to maintain some "private" information not in the prior.

---

## 5. Imagination: Training in Latent Space

### The Magic of Imagination

After training the world model, we can **simulate entire trajectories without touching the environment**:

```python
def imagine_trajectory(initial_state, horizon=15):
    states = [initial_state]
    actions = []
    rewards = []
    continues = []

    state = initial_state
    for t in range(horizon):
        # Actor chooses action
        action = actor(state)

        # World model predicts next state (using PRIOR, not posterior)
        next_h = GRU(state.h, [state.z, action])
        next_z = Prior(next_h).sample()
        next_state = concat(next_h, next_z)

        # Predict reward and continue
        reward = reward_head(next_state).decode()
        cont = continue_head(next_state).sigmoid()

        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        continues.append(cont)

        state = next_state

    return states, actions, rewards, continues
```

### Why This Works

1. **World model learned accurate dynamics** from real experience
2. **Prior learned to match posterior** through KL training
3. **Actor-critic trained on imagined data** generalizes to real world

### Imagination Horizon

**Default: 15 steps**

- Too short (5): Can't do long-term credit assignment
- Too long (50): Compounding errors degrade trajectory quality
- 15: Good balance for hockey (250-step episodes)

---

## 6. The Actor (Policy)

### Architecture

```python
class Actor(nn.Module):
    def __init__(self):
        self.network = MLP(512 → 256 → 256 → 8)  # 8 = 2 * action_dim

    def forward(self, full_state, training=False):
        output = self.network(full_state)
        mean, logStd = output.chunk(2, dim=-1)

        # CRITICAL: Bound logStd to ensure positive entropy
        logStd = -0.5 + 2.5 * sigmoid(logStd)  # logStd ∈ [-0.5, 2]
        std = exp(logStd)                       # std ∈ [0.606, 7.39]

        # Sample from Gaussian
        dist = Normal(mean, std)
        sample = dist.sample()

        # Squash to [-1, 1] via tanh
        action = tanh(sample)

        if training:
            # Log probability with Jacobian correction
            logprob = dist.log_prob(sample)
            logprob -= log(1 - action² + 1e-6)  # Jacobian of tanh

            entropy = dist.entropy()
            return action, logprob.sum(-1), entropy.sum(-1)

        return action
```

### Why TanhNormal?

1. **Bounded Actions**: Hockey requires actions in [-1, 1]
2. **Smooth Exploration**: Gaussian gives smooth exploration around mean
3. **Reparameterization**: Gradients flow through sampling

### The LogStd Bounds (Critical!)

**Our Fix**: `logStdMin = -0.5` (not -2!)

| logStdMin | σ_min | Min Entropy/dim | Total (4-dim) | Status |
|-----------|-------|-----------------|---------------|--------|
| -5 | 0.0067 | -3.0 | -12.0 | **CATASTROPHIC** |
| -2 | 0.135 | -0.84 | **-3.4** | **BUG** |
| **-0.5** | **0.606** | **+0.5** | **+2.0** | **CORRECT** |
| 0 | 1.0 | +1.42 | +5.7 | OK |

**Why Negative Entropy is Catastrophic**:
```python
actor_loss = -mean(advantages * logprobs + entropyScale * entropy)
```
If entropy < 0, the entropy term **penalizes** exploration instead of encouraging it!

### Actor Loss Function

```python
# Reinforce with entropy bonus
actor_loss = -mean(
    sg(advantages) * logprobs +  # Policy gradient (sg = stop_gradient)
    entropyScale * entropy       # Exploration bonus
)
```

**Components**:
- `advantages * logprobs`: Increase probability of actions with positive advantage
- `entropyScale * entropy`: Encourage stochasticity (exploration)

---

## 7. The Critic (Value Function)

### Architecture

```python
class Critic(nn.Module):
    def __init__(self):
        self.network = MLP(512 → 256 → 256 → 255)  # 255 = Two-Hot bins
        self.twohot = TwoHotSymlog(bins=255)

    def forward(self, full_state):
        logits = self.network(full_state)
        return logits  # Decode later with twohot.decode(logits)
```

### Why Two-Hot Symlog for Values?

Same reason as rewards: sparse returns create multi-modal distributions.

- Normal distribution predicts mean → regresses to average
- Two-Hot captures "usually 0 OR sometimes ±10" structure

### Lambda Returns (TD-λ)

The critic learns to predict **lambda returns** (mix of TD and Monte Carlo):

```python
def compute_lambda_returns(rewards, values, continues, lambda_=0.95):
    """
    G_t = r_t + γ * ((1-λ) * V_{t+1} + λ * G_{t+1})
    """
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]

    for t in reversed(range(len(rewards))):
        returns[t] = rewards[t] + continues[t] * (
            (1 - lambda_) * values[t+1] + lambda_ * bootstrap
        )
        bootstrap = returns[t]

    return returns
```

**Lambda Controls Bias-Variance Tradeoff**:
- λ=0: Pure TD (low variance, high bias)
- λ=1: Pure Monte Carlo (high variance, low bias)
- λ=0.95: Good balance (paper default)

**Important**: We changed from λ=0.99 to λ=0.95. High lambda with sparse rewards = chasing noisy rollouts.

### Slow Critic (EMA Target)

To stabilize training, we use an exponential moving average of critic weights:

```python
# After each update
slow_critic_weights = decay * slow_critic_weights + (1 - decay) * critic_weights
```

**Default decay**: 0.98 (~50 updates to track changes)

Lambda returns bootstrap from **slow critic**, not main critic, reducing value estimation variance.

---

## 8. Entropy & Exploration

### The Core Insight

> **Entropy scale (η = 3e-4) is FIXED across all domains. No annealing. No tuning.**

This was our biggest misunderstanding. We thought:
- ❌ Entropy should decrease over training (annealing)
- ❌ entropyScale needs domain-specific tuning
- ❌ We should target a specific entropy level like SAC

**Reality**:
- ✅ Fixed η = 3e-4 works everywhere
- ✅ Return normalization handles domain variation automatically
- ✅ Entropy-advantage balance emerges naturally

### How Return Normalization Makes This Work

The actor loss is:
```python
actor_loss = -mean(sg(advantages) * logprobs + η * entropy)
```

**Advantages are normalized by percentile-based scaling**:
```python
S = max(1.0, percentile_95(returns) - percentile_5(returns))
advantages = (lambda_returns - values) / S
```

This keeps advantages in roughly [-1, 1] range regardless of reward scale.

**Evolution Over Training**:

| Phase | Return Distribution | S | Advantages | Entropy Effect |
|-------|---------------------|---|------------|----------------|
| Early | Highly variable | Large | Small (~0.01) | Entropy dominates → explore |
| Mid | Concentrating | Medium | Medium (~0.1) | Balanced |
| Late | Concentrated | Small | Large (~1.0) | Advantages dominate → exploit |

**Key Insight**: The balance shifts automatically. Don't micromanage it!

### Expected Entropy Ranges (4-dim Continuous)

| Phase | Entropy | Interpretation |
|-------|---------|----------------|
| Early | +3 to +8 | High exploration, policy uncertain |
| Mid | +1 to +3 | Learning, reducing uncertainty |
| Converged | +0.5 to +1.5 | Confident but still stochastic |
| **Negative** | **BUG!** | Fix logStd bounds immediately |

### Monitoring Entropy

```python
# Log these every gradient step
metrics["behavior/entropy_mean"] = entropies.mean()
metrics["behavior/entropy_min"] = entropies.min()
metrics["behavior/entropy_max"] = entropies.max()
metrics["actor/min_std"] = torch.exp(logStd).min()

# Warning signs
if entropy_mean < 0:
    print("CRITICAL: Negative entropy! Check logStdMin bounds!")
if entropy_mean > 10 and gradient_step > 100000:
    print("WARNING: Entropy not decreasing. Check world model quality.")
```

---

## 9. Value Normalization (The Moments Class)

### Purpose

Normalize advantages to a consistent range so that:
1. Fixed entropy scale works across domains
2. Training is stable regardless of reward magnitude

### Implementation

```python
class Moments(nn.Module):
    def __init__(self, decay=0.99, percentile_low=0.05, percentile_high=0.95):
        self.decay = decay
        self.low = 0.0   # 5th percentile (EMA)
        self.high = 0.0  # 95th percentile (EMA)

    def forward(self, x):
        # Compute current batch percentiles
        low = torch.quantile(x, 0.05)
        high = torch.quantile(x, 0.95)

        # Update EMA
        self.low = self.decay * self.low + (1 - self.decay) * low
        self.high = self.decay * self.high + (1 - self.decay) * high

        # Compute scale (with floor)
        S = max(1.0, self.high - self.low)

        return self.low, S
```

### Usage in Actor Training

```python
# During behavior training
_, S = value_moments(lambda_returns)
advantages = (lambda_returns - values) / S
```

### The Floor (`S >= 1.0`)

**Critical for sparse rewards!**

Early in training, all returns might be ~0 (no goals yet). Without the floor:
- S → 0
- advantages = something / 0 → explosion

With floor S >= 1.0:
- Returns of 0 → advantages of 0 (safe)
- Entropy bonus provides exploration signal until real rewards appear

### Diagnostic: Return Range S

```python
metrics["diagnostics/return_range_S"] = S
metrics["diagnostics/return_range_at_floor"] = (S <= 1.01).float()
```

**Warning Signs**:
- S always at 1.0 for >100k steps: Reward signal not reaching agent (check DreamSmooth, world model)
- S exploding (>100): Unusual reward variance (check environment)

---

## 10. Two-Hot Symlog Encoding

### The Problem with Normal Distributions

**Sparse rewards are multi-modal**:
- 99% of timesteps: reward = 0
- 1% of timesteps: reward = ±10

A Normal distribution would predict mean ≈ 0.1, which is wrong for both cases.

### Symlog Transform

Compresses large values while preserving sign:
```python
symlog(x) = sign(x) * ln(|x| + 1)
symexp(x) = sign(x) * (exp(|x|) - 1)  # Inverse
```

**Properties**:
- symlog(0) = 0
- symlog(±10) ≈ ±2.4
- symlog(±1000) ≈ ±6.9

### Two-Hot Encoding

Instead of predicting a scalar, predict a **categorical distribution over bins**:

```python
class TwoHotSymlog(nn.Module):
    def __init__(self, bins=255, min_val=-20, max_val=20):
        self.bins = bins
        self.bin_centers = linspace(-20, 20, 255)  # In symlog space
```

**Encoding Process**:
1. Transform target: `y = symlog(reward)` → e.g., symlog(10) = 2.4
2. Find adjacent bins: bin_127 ≈ 0, bin_140 ≈ 2.5
3. Split probability: [0.1 to bin_139, 0.9 to bin_140]
4. Train with cross-entropy loss

**Decoding Process**:
1. Softmax over bins → probabilities
2. Weighted sum of bin centers → expected value in symlog space
3. Transform back: `symexp(expected_value)` → predicted reward

### Why This Works for Sparse Rewards

The network can learn to output:
- High probability on bin_127 (≈0) for most states
- High probability on bin_140 (≈+10) when near goal
- High probability on bin_115 (≈-10) when danger

This captures the **multi-modal structure** that Normal distributions miss.

---

## 11. DreamSmooth: Handling Sparse Rewards

### The Problem

Even with Two-Hot, **baseline DreamerV3 fails on sparse rewards**.

**Why?** The reward model minimizes prediction error. If 99% of rewards are 0:
- Predicting 0 everywhere minimizes error
- Goal events are ignored as "noise"

### DreamSmooth Solution

**Temporally smooth rewards before training the world model**:

```python
def smooth_rewards(rewards, alpha=0.5):
    """EMA smoothing: spread reward signal backwards in time."""
    smoothed = torch.zeros_like(rewards)
    smoothed[-1] = rewards[-1]

    for t in reversed(range(len(rewards) - 1)):
        smoothed[t] = rewards[t] + alpha * smoothed[t + 1]

    return smoothed
```

**Before** (sparse):
```
rewards = [0, 0, 0, 0, 0, 0, 0, 0, 10, 0]
```

**After** (smoothed with α=0.5):
```
smoothed = [0.039, 0.078, 0.156, 0.312, 0.625, 1.25, 2.5, 5.0, 10.0, 0]
```

### Why This Helps

1. **Reward model sees signal earlier**: Instead of sudden +10, sees gradual increase
2. **Easier to predict**: Smooth functions are easier to learn than step functions
3. **Credit assignment hint**: Smoothing implicitly tells model "states before goal are valuable"

### DreamSmooth Parameters

```yaml
useDreamSmooth: true     # Enable (essential for hockey)
dreamsmoothAlpha: 0.5    # Smoothing factor
```

**Alpha Tradeoffs**:
- α=0.1: Minimal smoothing, preserves sparsity
- α=0.5: Good balance (recommended)
- α=0.9: Heavy smoothing, might blur credit assignment

### Impact on Hockey

From research:
> "DreamSmooth successfully predicts most of the (smoothed) sparse rewards. Vanilla DreamerV3's reward model misses most of the sparse rewards."

**Performance improvement: +60-80% on sparse reward tasks.**

---

## 12. The Complete Training Loop

### High-Level Algorithm

```python
def train_dreamerv3():
    # Initialize
    world_model = WorldModel()
    actor = Actor()
    critic = Critic()
    buffer = ReplayBuffer(capacity=250000)

    # Warmup: collect random episodes
    for _ in range(warmup_episodes):  # 100 episodes
        episode = collect_episode(random_policy)
        buffer.add(episode)

    # Main loop
    for gradient_step in range(total_gradient_steps):

        # === TRAINING (every step) ===
        for _ in range(replay_ratio):  # 32 gradient updates per env step

            # Sample batch of sequences
            batch = buffer.sample(batch_size=32, seq_len=32)

            # Train world model
            world_model_loss, latent_states = train_world_model(batch)

            # Train actor-critic in imagination
            actor_loss, critic_loss = train_behavior(latent_states)

        # === ENVIRONMENT INTERACTION (every replay_ratio steps) ===
        if gradient_step % replay_ratio == 0:
            episode = collect_episode(actor)
            buffer.add(episode)

            # Evaluation periodically
            if episode_count % eval_interval == 0:
                evaluate(actor)
```

### World Model Training Step

```python
def train_world_model(batch):
    obs, actions, rewards, dones = batch

    # Apply DreamSmooth to rewards
    if use_dreamsmooth:
        rewards = smooth_rewards(rewards, alpha=0.5)

    # Encode observations
    embeds = encoder(obs)

    # Run RSSM to get latent states
    h, z_posterior = rssm.observe_sequence(embeds, actions)
    full_states = concat(h, z_posterior)

    # Compute losses
    recon_loss = decoder.loss(full_states, obs)
    reward_loss = reward_head.loss(full_states, rewards)
    continue_loss = continue_head.loss(full_states, ~dones)

    # KL loss (with free nats)
    z_prior = prior_net(h)
    kl_loss = kl_divergence(z_posterior, z_prior, free_nats=1.0)

    # Total loss
    loss = recon_loss + reward_loss + continue_loss + kl_loss

    # Update
    world_model_optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(world_model.parameters(), max_norm=100)
    world_model_optimizer.step()

    return loss, full_states.detach()
```

### Behavior Training Step

```python
def train_behavior(initial_states):
    # === IMAGINATION ===
    # Rollout policy in latent space for H steps
    states, actions, rewards, continues = imagine(
        initial_states, horizon=15
    )

    # === CRITIC TRAINING ===
    # Compute lambda returns
    with torch.no_grad():
        values = slow_critic(states)  # Use slow critic for targets!
    lambda_returns = compute_lambda_returns(rewards, values, continues, lambda_=0.95)

    # Critic loss
    critic_values = critic(states[:, :-1])
    critic_loss = twohot.loss(critic_values, lambda_returns)

    # Update critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Update slow critic (EMA)
    update_slow_critic(decay=0.98)

    # === ACTOR TRAINING ===
    # Re-imagine with gradients through actor
    states, actions, logprobs, entropies = imagine_with_gradients(
        initial_states, horizon=15
    )

    # Normalize advantages
    _, S = value_moments(lambda_returns)
    advantages = (lambda_returns - values[:, :-1]) / S

    # Actor loss
    actor_loss = -mean(
        advantages.detach() * logprobs +
        entropy_scale * entropies
    )

    # Update actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss, critic_loss
```

---

## 13. Hyperparameters Explained

### Critical Parameters (Don't Touch Without Reason)

| Parameter | Value | Why |
|-----------|-------|-----|
| `entropyScale` | 3e-4 | DreamerV3 paper constant, domain-invariant |
| `discount` | 0.997 | γ^250 ≈ 0.47, good for 250-step episodes |
| `lambda_` | 0.95 | Bias-variance balance, don't increase |
| `replay_ratio` | 32 | Sample efficiency, don't decrease below 16 |
| `lr_actor` | 1e-4 | Must be ≤ lr_critic |
| `lr_critic` | 1e-4 | Standard |

### Architecture Parameters (Task-Dependent)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `recurrentSize` | 256 | GRU hidden, sufficient for 18-dim obs |
| `latentLength` | 16 | Categorical variables |
| `latentClasses` | 16 | Classes per variable (256 total stochastic dim) |
| `imaginationHorizon` | 15 | Steps to imagine, balance accuracy vs credit assignment |
| `batchSize` | 32 | Sequences per batch |
| `batchLength` | 32 | Timesteps per sequence |

### Training Schedule Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `warmup_episodes` | 100 | Collect before training (need ~40-60 goal examples) |
| `buffer_capacity` | 250000 | ~1000 episodes, balance freshness vs diversity |
| `free_nats` | 1.0 | KL threshold, prevents posterior collapse |

### Learning Rates

| Component | LR | Notes |
|-----------|-----|-------|
| World Model | 3e-4 | Foundation, can be higher |
| Actor | 1e-4 | Must be ≤ critic |
| Critic | 1e-4 | Standard |

### The Actor-Critic Hierarchy

**NEVER invert this relationship**:
```
lr_actor ≤ lr_critic
```

Why? Actor learns from critic's value estimates. If actor learns faster, it overfits to stale values.

---

## 14. Common Pitfalls & Debugging

### Pitfall 1: Negative Entropy

**Symptom**: `behavior/entropy_mean` goes negative.

**Cause**: `logStdMin` too low (we had -2, giving σ_min=0.135).

**Fix**: Set `logStdMin = -0.5` (σ_min=0.606).

### Pitfall 2: Bad CLI Overrides

**Symptom**: Training converges 3-5× slower than expected.

**Cause**: Overriding good defaults with bad values:
```bash
# BAD:
python train.py --replay_ratio 4 --lr_actor 0.0005

# GOOD:
python train.py  # Use defaults!
```

**Fix**: Trust the defaults. Only override with research-backed reasoning.

### Pitfall 3: Return Range S Stuck at Floor

**Symptom**: `diagnostics/return_range_S` always = 1.0.

**Cause**: No reward signal reaching the value function.

**Debug Steps**:
1. Check `world/reward_loss` - is reward model learning?
2. Check DreamSmooth is enabled
3. Check environment is giving rewards at all

### Pitfall 4: Entropy Not Decreasing

**Symptom**: Entropy stays at +10 for >200k steps.

**Cause**: World model not learning → advantages meaningless → entropy dominates.

**Debug Steps**:
1. Check `world/recon_loss` - should decrease
2. Check `world/kl_loss` - should stabilize around free_nats
3. Check warmup was sufficient (100 episodes)

### Pitfall 5: Policy Divergence

**Symptom**: Actor loss explodes, actions saturate at ±1.

**Cause**: Actor LR too high relative to critic.

**Fix**: Ensure `lr_actor ≤ lr_critic`. Try reducing to 5e-5.

### Debugging Checklist

```
□ Entropy positive? (Check logStdMin bounds)
□ Return range S growing? (Check DreamSmooth, world model)
□ Reconstruction loss decreasing? (World model learning)
□ KL loss stabilizing? (Prior matching posterior)
□ Advantages reasonable? (Not exploding, not all zero)
□ Win rate increasing? (Overall progress)
```

---

## 15. Our Hockey-Specific Adaptations

### Environment Characteristics

| Property | Value | Implication |
|----------|-------|-------------|
| Observation dim | 18 | Simple MLP encoder sufficient |
| Action dim | 4 | Continuous, bounded [-1, 1] |
| Episode length | 250 steps | discount=0.997 appropriate |
| Reward structure | Sparse ±10 | DreamSmooth essential |
| Keep-mode | Puck holding | Temporal credit assignment critical |

### Our Final Configuration

```yaml
# Critical (don't change)
entropyScale: 0.0003
discount: 0.997
lambda_: 0.95
replay_ratio: 32
lr_actor: 0.0001
lr_critic: 0.0001

# Architecture (tuned for hockey)
recurrentSize: 256
latentLength: 16
latentClasses: 16
imaginationHorizon: 15

# Training schedule
warmup_episodes: 100
buffer_capacity: 250000

# Sparse reward handling (essential)
useDreamSmooth: true
dreamsmoothAlpha: 0.5

# Actor bounds (fixed bug)
logStdMin: -0.5  # σ_min = 0.606, ensures positive entropy
logStdMax: 2.0   # σ_max = 7.39
```

### Auxiliary Tasks (Optional)

We implemented auxiliary prediction heads to help world model learn goal-relevant features:

1. **Goal Prediction**: "Will goal happen in next 15 steps?" (binary)
2. **Distance Head**: "How far is puck from opponent goal?" (regression)
3. **Shot Quality**: "How good is current scoring opportunity?" (regression)

These help the latent space encode scoring-relevant features without corrupting the reward signal.

### Expected Performance

| Hardware | Time to 70% Win Rate |
|----------|---------------------|
| RTX 4080 | 4-5 hours |
| RTX 2080 Ti (optimized config) | 16-20 hours |

### Recommended Training Command

```bash
cd 02-SRC/DreamerV3

# Use all good defaults, just enable DreamSmooth
python train_hockey.py \
    --opponent weak \
    --seed 42 \
    --use_dreamsmooth

# Monitor in another terminal
tensorboard --logdir results/
# Or check wandb.ai
```

---

## Summary

DreamerV3 is a powerful world-model-based RL algorithm that excels at:
- Sample efficiency (learns from limited data)
- Long-horizon credit assignment (connects distant rewards to actions)
- Handling sparse rewards (with DreamSmooth)

**Key Insights from Our Work**:

1. **Entropy scale is fixed (3e-4)** - return normalization handles domain variation
2. **Negative entropy is catastrophic** - ensure σ_min > 0.242
3. **Actor LR must be ≤ Critic LR** - never invert the hierarchy
4. **DreamSmooth is essential** for sparse rewards (+60-80% performance)
5. **Trust the defaults** - only override with research-backed reasoning
6. **Monitor entropy, return range S, and world model losses** for debugging

**The DreamerV3 Philosophy**:
> Learn the world. Imagine the future. Plan intelligently.

---

**Document Version**: 1.0
**Last Updated**: January 2026
