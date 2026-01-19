# Understanding DreamerV3: From Intuition to Implementation

This document explains how DreamerV3 works, first through intuition, then through our concrete implementation.

---

## Part 1: The Intuition

### 1.1 The Fundamental Problem

Consider how you learn to play a new video game. You don't just memorize "when I see screen X, press button Y." Instead, you build a **mental model** of how the game works: "enemies move this way," "my character jumps this high," "hitting that button shoots." Then you **imagine** scenarios: "if I jump here and shoot there, I could clear this level."

Traditional reinforcement learning (like TD3) skips this mental model entirely. It learns a direct mapping: observation → action. This works, but it's sample-inefficient because every lesson requires real experience.

DreamerV3 takes the human approach: **learn how the world works, then practice in your imagination.**

### 1.2 Model-Free vs. Model-Based: The Core Difference

**TD3 (Model-Free):**
```
Real World → Collect Experience → Update Policy Directly
     ↑                                    │
     └────────────────────────────────────┘
                 (repeat forever)
```

TD3 treats the environment as a black box. It collects (state, action, reward, next_state) tuples, then uses them to train a policy. Every learning signal requires a real environment interaction. If you want to learn that "jumping off cliffs is bad," you must actually jump off many cliffs.

**DreamerV3 (Model-Based):**
```
Real World → Collect Experience → Train World Model
                                        │
                                        ↓
                            ┌─────────────────────┐
                            │   IMAGINATION       │
                            │                     │
                            │  World Model        │
                            │       ↓             │
                            │  Predict Future     │
                            │       ↓             │
                            │  Train Policy       │
                            └─────────────────────┘
```

DreamerV3 first learns a **world model** that predicts what happens next. Then it trains its policy entirely by **imagining** thousands of futures using this model. One real experience can generate hundreds of imagined training examples.

### 1.3 The World Model: Learning to Dream

The world model answers: "If I'm in state S and take action A, what happens next?"

But here's the challenge: real observations (like game pixels or sensor readings) are **high-dimensional and noisy**. A hockey observation has 18 numbers. Predicting exactly how all 18 will change after each action is hard and unnecessary.

DreamerV3 solves this with **latent states** – compressed representations that capture only what matters:

```
Real Observation (18 dims)
        │
        ↓
    [ENCODER]
        │
        ↓
Latent State (much smaller, cleaner)
        │
        ↓
    [DECODER]
        │
        ↓
Reconstructed Observation
```

The encoder compresses observations into a latent space. The decoder proves the latent state is good by reconstructing the original observation. If reconstruction works, the latent state captured the important information.

### 1.4 The RSSM: Memory + Uncertainty

The core of DreamerV3's world model is the **Recurrent State-Space Model (RSSM)**. It maintains two types of state:

**Deterministic State (h):** Like memory. It accumulates information over time through a recurrent neural network (GRU). This captures "what has happened so far."

**Stochastic State (z):** Captures uncertainty. The world is inherently unpredictable – the puck might bounce unexpectedly, the opponent might do something surprising. The stochastic state represents this as a probability distribution, not a single point.

Together, they form the complete latent state:
```
[h, z] = latent state
   │
   ├── h: "I remember the puck was moving left"
   └── z: "The opponent might be at position X±uncertainty"
```

### 1.5 Prior vs. Posterior: Two Ways to Predict

The RSSM learns two prediction modes:

**Prior (Imagination Mode):**
"Given only my memory (h) and the action I took, where do I think I am?"

This is what we use during imagination. We don't have real observations – we're dreaming. The prior must predict the next state from dynamics alone.

**Posterior (Reality Mode):**
"Given my memory (h) AND the actual observation I just saw, where am I?"

This is what we use during real experience. We can look at the actual observation to get a better estimate of our state.

The training objective is to make the prior match the posterior. When they match, the world model has learned accurate dynamics – it can predict the future without needing to see it.

```
                ┌─────────────────┐
                │  Prior (dream)  │──────┐
                │  p(z | h)       │      │
                └─────────────────┘      │
                                         │  KL Divergence
                ┌─────────────────┐      │  (should be small)
                │ Posterior (real)│──────┘
                │ q(z | h, obs)   │
                └─────────────────┘
```

### 1.6 Learning in Imagination

Once the world model is trained, the magic happens:

1. **Start from a real state** (collected during actual gameplay)
2. **Imagine forward** using the prior:
   - Sample action from policy
   - Predict next state using prior
   - Predict reward from the predicted state
   - Repeat for N steps (the "imagination horizon")
3. **Train the policy** on these imagined trajectories

This is incredibly efficient. One real trajectory of 100 steps can generate thousands of imagined trajectories starting from different points. The policy gets massive amounts of training data without additional real experience.

### 1.7 Why This Works for Sparse Rewards

TD3 struggles with sparse rewards (like hockey where you only get +1/-1 at the end). The reward signal is too rare – the agent doesn't know which actions contributed to winning.

DreamerV3 handles this through **imagination horizon**:

```
Real trajectory: [...no reward...no reward...no reward...WIN!]

Imagination (50-step horizon):
  Start: step 150 → Imagine 50 steps → See win at step 200 → Credit actions!
  Start: step 140 → Imagine 50 steps → See win at step 190 → Credit actions!
  Start: step 130 → Imagine 50 steps → See win at step 180 → Credit actions!
```

By imagining long into the future, DreamerV3 can "see" the eventual reward and propagate credit backward through its imagined trajectory. The world model provides the missing information about what leads to what.

### 1.8 The Actor-Critic in Latent Space

DreamerV3 uses actor-critic, but entirely in the latent space:

**Actor (Policy):** Maps latent state [h, z] → action distribution

**Critic (Value):** Maps latent state [h, z] → expected future reward

Neither ever sees the raw observation during training. They operate purely on the compressed latent representations. This is more efficient and more robust – they learn from the essential features, not the noise.

### 1.9 Summary: The Three Learning Loops

DreamerV3 has three interleaved learning processes:

```
┌──────────────────────────────────────────────────────────────┐
│ LOOP 1: Collect Real Experience                              │
│   - Act in real environment                                  │
│   - Store (obs, action, reward, done) in replay buffer       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ LOOP 2: Train World Model                                    │
│   - Sample sequences from buffer                             │
│   - Encode observations → latent states                      │
│   - Train to reconstruct observations                        │
│   - Train to predict rewards                                 │
│   - Train prior to match posterior (KL loss)                 │
└──────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ LOOP 3: Train Actor-Critic in Imagination                    │
│   - Get starting states from real sequences                  │
│   - Imagine forward using policy + world model               │
│   - Compute value targets (TD(λ))                            │
│   - Update critic to predict these targets                   │
│   - Update actor to maximize predicted value                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Part 2: Our Implementation

Now let's see how these concepts map to our code.

### 2.1 File Structure Overview

```
DreamerV3/
├── agents/
│   └── hockey_dreamer.py    # Main agent (orchestrates everything)
├── models/
│   ├── sequence_model.py    # RSSM (the dreaming core)
│   ├── world_model.py       # Complete world model
│   └── behavior.py          # Actor-Critic
├── utils/
│   ├── math_ops.py          # symlog, lambda_returns
│   ├── distributions.py     # TanhNormal, GaussianDist
│   └── buffer.py            # Episode replay buffer
└── train_hockey.py          # Training loop
```

### 2.2 The SequenceModel (RSSM) – `models/sequence_model.py`

This is the **dreaming core** – implements the RSSM from Section 1.4.

**State Representation:**
```python
state = {
    'h': tensor,    # Deterministic state (GRU hidden), shape: (batch, 256)
    'z': tensor,    # Stochastic state (sampled), shape: (batch, 64)
    'mean': tensor, # Mean of z distribution
    'std': tensor,  # Std of z distribution
}
```

**Key Components:**

```python
# Process (previous_z, action) into GRU input
self.input_net = build_mlp(latent_size + action_dim, hidden_size, ...)

# Recurrent core - accumulates history into h
self.gru = nn.GRUCell(hidden_size, recurrent_size)

# Prior: predict z from h only (for imagination)
self.prior_net = build_mlp(recurrent_size, latent_size * 2, ...)

# Posterior: infer z from h AND observation (for training)
self.posterior_net = build_mlp(recurrent_size + embed_dim, latent_size * 2, ...)
```

**The Two Step Functions:**

`prior_step()` – Used during imagination (no observation available):
```python
def prior_step(self, state, action):
    # Update deterministic state
    x = input_net(concat(state['z'], action))
    h_next = self.gru(x, state['h'])

    # Predict z using ONLY h (no observation)
    mean, std = self.prior_net(h_next)
    z_next = mean + std * noise  # Sample from predicted distribution

    return {'h': h_next, 'z': z_next, 'mean': mean, 'std': std}
```

`posterior_step()` – Used during training (observation available):
```python
def posterior_step(self, state, action, embed):
    # Update deterministic state (same as prior)
    h_next = self.gru(...)

    # Prior prediction (for KL loss)
    prior_mean, prior_std = self.prior_net(h_next)

    # Posterior inference (using observation)
    post_mean, post_std = self.posterior_net(concat(h_next, embed))
    z_post = post_mean + post_std * noise

    return posterior_state, prior_state  # Both, for KL computation
```

**imagine_sequence()** – The dreaming function:
```python
def imagine_sequence(self, policy, start_state, horizon):
    """Imagine H steps into the future."""
    state = start_state
    for _ in range(horizon):
        features = concat(state['h'], state['z'])
        action = policy(features).sample()
        state = self.prior_step(state, action)  # No observation!
        # Collect states and actions
    return states, actions
```

This is where learning happens without real experience. The policy proposes actions, the prior predicts what happens, and we unroll this for `horizon` steps.

### 2.3 The WorldModel – `models/world_model.py`

The world model wraps the RSSM and adds prediction heads.

**Components:**
```python
# Encoder: raw observation → latent embedding
self.encoder = build_mlp(obs_dim, embed_dim, ...)

# RSSM dynamics
self.dynamics = SequenceModel(...)

# Decoder: latent state → predicted observation
self.obs_decoder = build_mlp(feature_dim, obs_dim, ...)

# Prediction heads
self.reward_head = build_mlp(feature_dim, 1, ...)    # Predict reward
self.continue_head = build_mlp(feature_dim, 1, ...)  # Predict if episode continues
```

**The Encoding Flow (Reality):**
```
Raw Observation (18 dims)
        │
        ↓ symlog (compress large values)
        ↓ encoder MLP
        │
Embedding (256 dims)
        │
        ↓ posterior_step (with GRU + posterior_net)
        │
Latent State [h, z]
```

**compute_loss()** – Train the world model:
```python
def compute_loss(self, batch):
    # 1. Encode observations
    embeds = self.encode(batch['obs'])

    # 2. Run through RSSM to get posterior and prior states
    posteriors, priors = self.dynamics.observe_sequence(embeds, actions, is_first)

    # 3. Reconstruction loss - can we decode back to observation?
    features = concat(posteriors['h'], posteriors['z'])
    obs_pred = self.obs_decoder(features)
    recon_loss = MSE(obs_pred, obs_target)

    # 4. Reward prediction loss
    reward_pred = self.reward_head(features)
    reward_loss = MSE(reward_pred, actual_rewards)

    # 5. Continue prediction loss
    continue_pred = self.continue_head(features)
    continue_loss = BCE(continue_pred, actual_continues)

    # 6. KL loss - make prior match posterior
    kl_loss = self.dynamics.kl_loss(posteriors, priors)

    return recon_loss + reward_loss + continue_loss + kl_loss
```

**imagine()** – Generate training data for actor-critic:
```python
def imagine(self, policy, start_state, horizon):
    # Use RSSM to imagine forward
    states, actions = self.dynamics.imagine_sequence(policy, start_state, horizon)

    # Predict rewards and continues for imagined states
    features = concat(states['h'], states['z'])
    rewards = self.reward_head(features)
    continues = self.continue_head(features)

    return states, actions, rewards, continues
```

### 2.4 The Behavior Model – `models/behavior.py`

The actor-critic that learns entirely in imagination.

**Policy (Actor):**
```python
class Policy(nn.Module):
    def forward(self, features):
        # features = [h, z] from RSSM
        out = self.net(features)
        mean, std = split(out)
        return TanhNormal(mean, std)  # Bounded actions in [-1, 1]
```

**ValueNetwork (Critic):**
```python
class ValueNetwork(nn.Module):
    def forward(self, features):
        # features = [h, z] from RSSM
        return self.net(features)  # Scalar value estimate
```

**train_step()** – Learning from imagination:
```python
def train_step(self, states, actions, rewards, continues):
    # states, actions, rewards, continues all come from imagination!

    features = concat(states['h'], states['z'])

    # 1. Compute TD(λ) return targets
    values = self.value_target(features)
    targets = lambda_returns(rewards, values, continues, bootstrap)

    # 2. Critic loss - predict the targets
    values_pred = self.value(features)
    critic_loss = MSE(values_pred, targets)

    # 3. Actor loss - maximize advantage-weighted log probability
    advantages = targets - values_pred
    action_dists = self.policy(features)
    log_probs = action_dists.log_prob(actions)
    actor_loss = -(log_probs * advantages).mean()

    # 4. Entropy bonus - encourage exploration
    entropy_loss = -entropy_scale * action_dists.entropy().mean()

    return actor_loss + entropy_loss, critic_loss
```

### 2.5 The HockeyDreamer Agent – `agents/hockey_dreamer.py`

Orchestrates everything into a clean interface.

**Acting (in real environment):**
```python
def act(self, obs, deterministic=False):
    # 1. Encode observation
    embed = self.world_model.encode(obs)

    # 2. Update internal state using POSTERIOR (we have real observation)
    self._state = self.world_model.dynamics.posterior_step(
        self._state, self._last_action, embed
    )

    # 3. Get action from policy
    features = concat(self._state['h'], self._state['z'])
    action = self.behavior.act(features, deterministic)

    return action
```

**Training (the three loops from Section 1.9):**
```python
def train_step(self, batch):
    # LOOP 2: Train World Model
    world_loss = self.world_model.compute_loss(batch)
    world_loss.backward()
    self.world_opt.step()

    # LOOP 3: Train Actor-Critic in Imagination
    # Get starting states from real batch
    start_states = self.world_model.get_start_states(batch)

    # Imagine forward
    states, actions, rewards, continues = self.world_model.imagine(
        policy=self.behavior.policy,
        start_state=start_states,
        horizon=self.horizon,  # 15 steps into future
    )

    # Train critic
    _, critic_loss, _ = self.behavior.train_step(states, actions, rewards, continues)
    critic_loss.backward()
    self.critic_opt.step()

    # Train actor (re-imagine to get gradients through policy)
    states, actions, rewards, continues = self.world_model.imagine(...)
    actor_loss, _, _ = self.behavior.train_step(...)
    actor_loss.backward()
    self.actor_opt.step()
```

### 2.6 The Training Loop – `train_hockey.py`

Puts it all together:

```python
while total_steps < max_steps:
    # LOOP 1: Collect real experience
    episode_reward, episode_length, info = run_episode(env, agent, buffer)

    # LOOPS 2 & 3: Train (world model + actor-critic in imagination)
    if len(buffer) >= min_buffer_size:
        batch = buffer.sample(batch_size, sequence_length)
        metrics = agent.train_step(batch)

    # Periodically evaluate
    if total_steps % eval_interval == 0:
        eval_metrics = evaluate(eval_env, agent)
```

### 2.7 Key Implementation Details

**Symlog Transformation** (`utils/math_ops.py`):
```python
def symlog(x):
    return sign(x) * log(|x| + 1)
```
Compresses large values while preserving sign. Allows the network to handle rewards/observations across many orders of magnitude.

**Lambda Returns** (`utils/math_ops.py`):
```python
def lambda_returns(rewards, values, continues, bootstrap, gamma, lambda_):
    """TD(λ) for value targets."""
    # G_t = r_t + γ * [(1-λ)*V_{t+1} + λ*G_{t+1}]
```
Balances between one-step TD (λ=0) and Monte Carlo (λ=1). We use λ=0.95 for good bias-variance tradeoff.

**TanhNormal Distribution** (`utils/distributions.py`):
```python
class TanhNormal:
    """Squashed Gaussian for bounded actions."""
    def sample(self):
        gaussian_sample = self.mean + self.std * noise
        return tanh(gaussian_sample)  # Bounded to [-1, 1]
```

**Episode Buffer** (`utils/buffer.py`):
Stores complete episodes and samples contiguous sequences. This preserves temporal structure needed for training the RSSM.

### 2.8 The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           REAL EXPERIENCE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Hockey Env → Observation (18 dims)                                     │
│                    │                                                    │
│                    ↓                                                    │
│              ┌─────────┐                                                │
│              │ Encoder │ (symlog → MLP)                                 │
│              └────┬────┘                                                │
│                   │                                                     │
│                   ↓                                                     │
│         Embedding (256 dims)                                            │
│                   │                                                     │
│                   ↓                                                     │
│           ┌──────────────┐                                              │
│           │   RSSM       │                                              │
│           │ (posterior)  │ ← uses real observation                      │
│           └──────┬───────┘                                              │
│                  │                                                      │
│                  ↓                                                      │
│        Latent State [h, z]                                              │
│                  │                                                      │
│                  ↓                                                      │
│           ┌──────────┐                                                  │
│           │  Policy  │                                                  │
│           └────┬─────┘                                                  │
│                │                                                        │
│                ↓                                                        │
│         Action (4 dims) → Hockey Env                                    │
│                                                                         │
│  (obs, action, reward, done) → Replay Buffer                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         WORLD MODEL TRAINING                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Replay Buffer → Sample sequence (B, T, dims)                          │
│                        │                                                │
│                        ↓                                                │
│                 ┌──────────────┐                                        │
│                 │   Encoder    │                                        │
│                 └──────┬───────┘                                        │
│                        │                                                │
│                        ↓                                                │
│                ┌───────────────┐                                        │
│                │     RSSM      │                                        │
│                │  (posterior   │                                        │
│                │   + prior)    │                                        │
│                └───────┬───────┘                                        │
│                        │                                                │
│              ┌─────────┴─────────┐                                      │
│              │                   │                                      │
│              ↓                   ↓                                      │
│        ┌──────────┐       ┌───────────┐                                │
│        │ Decoder  │       │  Reward   │                                │
│        │          │       │   Head    │                                │
│        └────┬─────┘       └─────┬─────┘                                │
│             │                   │                                      │
│             ↓                   ↓                                      │
│        Recon Loss          Reward Loss                                 │
│                                                                         │
│  + KL Loss (prior ↔ posterior) + Continue Loss                         │
│                        │                                                │
│                        ↓                                                │
│                   Total Loss → Backprop → Update World Model            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     ACTOR-CRITIC TRAINING (IMAGINATION)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Real Sequence → get_start_states() → Starting [h, z]                  │
│                                            │                            │
│                                            ↓                            │
│                  ┌─────────────────────────────────────────┐            │
│                  │           IMAGINATION LOOP              │            │
│                  │                                         │            │
│                  │  for t in range(horizon):               │            │
│                  │      features = [h_t, z_t]              │            │
│                  │      action = policy(features).sample() │            │
│                  │      [h_{t+1}, z_{t+1}] = RSSM.prior()  │  ← No obs! │
│                  │      reward_t = reward_head(features)   │            │
│                  │      continue_t = continue_head(...)    │            │
│                  │                                         │            │
│                  └────────────────┬────────────────────────┘            │
│                                   │                                     │
│                                   ↓                                     │
│            Imagined: states, actions, rewards, continues                │
│                                   │                                     │
│                    ┌──────────────┴──────────────┐                      │
│                    │                             │                      │
│                    ↓                             ↓                      │
│             ┌────────────┐               ┌────────────┐                 │
│             │   Critic   │               │   Actor    │                 │
│             │  Training  │               │  Training  │                 │
│             └──────┬─────┘               └──────┬─────┘                 │
│                    │                            │                       │
│                    ↓                            ↓                       │
│           TD(λ) targets              Policy gradient                    │
│           MSE(V, targets)            -(log_prob × advantage)           │
│                    │                            │                       │
│                    ↓                            ↓                       │
│              Critic Loss                  Actor Loss                    │
│                    │                            │                       │
│                    └──────────┬─────────────────┘                       │
│                               ↓                                         │
│                         Backprop → Update Actor & Critic                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Connecting It All

### Why This Matters for Hockey

**The Challenge:** Hockey has sparse rewards. You play ~250 steps, then get +1 (win), -1 (lose), or 0 (draw). TD3 struggles because there's no learning signal during the episode.

**DreamerV3's Solution:**
1. The world model learns "physics" – how the puck moves, how the opponent reacts
2. When imagining 15 steps from step 235, the model can "see" the impending goal
3. This predicted reward provides gradient signal throughout the episode
4. The actor learns: "at step 235, positioning here leads to predicted win at step 250"

### The Key Insight

TD3 asks: "What action maximizes future reward?" and tries to learn this directly from sparse data.

DreamerV3 asks: "What will happen if I do X?" (world model) and then "Given what will happen, what should I do?" (actor-critic in imagination).

By separating "understanding the world" from "choosing actions," DreamerV3 can learn both more efficiently. The world model provides rich supervision (reconstruction, reward prediction, dynamics) even when rewards are sparse. The actor-critic then leverages this understanding to make better decisions.

---

## Summary Table: TD3 vs. DreamerV3

| Aspect | TD3 | DreamerV3 |
|--------|-----|-----------|
| **Learning paradigm** | Model-free | Model-based |
| **World model** | None | Full (RSSM + heads) |
| **Policy training** | On real transitions | In imagination |
| **State representation** | Raw observations | Learned latent [h, z] |
| **Sparse rewards** | Struggles | Handles via long horizon |
| **Sample efficiency** | Lower | Higher |
| **Computational cost** | Lower per step | Higher per step |
| **Memory** | Simple replay buffer | Episode buffer + models |
| **Key components** | Actor, Critic | World Model, Actor, Critic |

---

*AI Usage Declaration: This document was developed with assistance from Claude Code.*
