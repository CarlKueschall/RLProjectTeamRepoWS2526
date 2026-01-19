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
Latent State (compressed, cleaner)
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
   ├── h: "I remember the puck was moving left" (deterministic, 256-dim)
   └── z: "The game state is likely X, Y, or Z" (stochastic, 256-dim categorical)
```

**DreamerV3's Key Innovation: Categorical Latents**

Unlike DreamerV1/V2 which used Gaussian stochastic states, DreamerV3 uses **categorical distributions**:
- Instead of `z ~ N(mean, std)` (continuous)
- We have `z ~ Categorical(logits)` (discrete)
- Specifically: 16 categorical variables, each with 16 classes
- Total stochastic state = 16 × 16 = 256 one-hot values flattened

Why categorical?
1. **Better representation capacity** - discrete choices can represent multimodal distributions
2. **More stable gradients** - straight-through estimator works well
3. **Uniform mixing** - can inject 1% uniform distribution to prevent latent collapse

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

**Free Nats Threshold**: KL divergence below a threshold (e.g., 1.0 nats) is not penalized. This prevents the posterior from collapsing to exactly match a weak prior, preserving useful information in the stochastic state.

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

Imagination (15-step horizon):
  Start: step 235 → Imagine 15 steps → See win at step 250 → Credit actions!
  Start: step 230 → Imagine 15 steps → See potential win → Credit actions!
  Start: step 225 → Imagine 15 steps → Moving toward goal → Credit actions!
```

By imagining into the future, DreamerV3 can "see" the eventual reward and propagate credit backward through its imagined trajectory. The world model provides the missing information about what leads to what.

### 1.8 The Actor-Critic in Latent Space

DreamerV3 uses actor-critic, but entirely in the latent space:

**Actor (Policy):** Maps latent state [h, z] → action distribution (TanhNormal)

**Critic (Value):** Maps latent state [h, z] → expected future reward distribution (Normal)

Neither ever sees the raw observation during training. They operate purely on the compressed latent representations. This is more efficient and more robust – they learn from the essential features, not the noise.

**Value Normalization**: DreamerV3 normalizes advantages using percentile-based scaling (5th to 95th percentile). This makes the actor gradient scale-invariant, enabling the same hyperparameters to work across different reward scales.

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

Now let's see how these concepts map to our code. Our implementation is based on NaturalDreamer, simplified for low-dimensional observations like hockey's 18-dim state vector.

### 2.1 File Structure Overview

```
DreamerV3/
├── dreamer.py              # Main agent class (Dreamer)
├── networks.py             # All neural network components (incl. auxiliary task heads)
├── buffer.py               # Replay buffer for sequence sampling
├── utils.py                # Helper functions (lambda returns, moments, TwoHotSymlog)
├── train_hockey.py         # Training loop
└── configs/
    └── hockey.yml          # Configuration file
```

### 2.2 The Networks – `networks.py`

All neural network components are defined here, built from simple MLPs.

**RecurrentModel (GRU-based dynamics):**
```python
class RecurrentModel(nn.Module):
    """GRU-based recurrent model for RSSM dynamics."""

    def forward(self, recurrentState, latentState, action):
        # Combine latent state and action
        x = torch.cat((latentState, action), -1)
        x = self.activation(self.linear(x))
        # Update recurrent state with GRU
        return self.recurrent(x, recurrentState)  # h_next
```

**PriorNet & PosteriorNet (Categorical distributions):**
```python
class PriorNet(nn.Module):
    """Prior network: predicts latent state from recurrent state only."""

    def forward(self, h):
        rawLogits = self.network(h)
        # Reshape to (batch, latentLength, latentClasses)
        probabilities = rawLogits.view(-1, 16, 16).softmax(-1)

        # Mix with 1% uniform distribution to prevent collapse
        uniform = torch.ones_like(probabilities) / 16
        finalProbabilities = 0.99 * probabilities + 0.01 * uniform

        # Sample with straight-through gradient
        logits = probs_to_logits(finalProbabilities)
        sample = OneHotCategoricalStraightThrough(logits=logits).rsample()
        return sample.view(-1, 256), logits  # z and logits
```

The posterior network is identical but takes `concat(h, embed)` as input – it has access to the encoded observation.

**Actor (TanhNormal for bounded actions):**
```python
class Actor(nn.Module):
    """Actor network: outputs actions with tanh squashing."""

    def forward(self, fullState, training=False):
        mean, logStd = self.network(fullState).chunk(2, dim=-1)
        # Bound log_std to [-5, 2]
        logStd = -5 + 7/2 * (torch.tanh(logStd) + 1)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        action = torch.tanh(sample) * actionScale + actionBias  # Bounded

        if training:
            # Jacobian correction for tanh
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(actionScale * (1 - tanh(sample)^2) + 1e-6)
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action
```

**Critic & Reward Model (Normal distributions):**
```python
class Critic(nn.Module):
    """Critic network: outputs Normal distribution for value."""

    def forward(self, fullState):
        mean, logStd = self.network(fullState).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
```

Both output distributions rather than point estimates, enabling probabilistic predictions.

### 2.3 The Dreamer Agent – `dreamer.py`

The main agent class orchestrating all components.

**Initialization:**
```python
class Dreamer:
    def __init__(self, observationSize, actionSize, actionLow, actionHigh, device, config):
        # State dimensions
        self.recurrentSize = 256                    # h dimension
        self.latentSize = 16 * 16 = 256             # z dimension (16 vars × 16 classes)
        self.fullStateSize = 256 + 256 = 512        # [h, z] concatenated

        # World model components
        self.encoder = EncoderMLP(18, 256, ...)     # obs → embedding
        self.decoder = DecoderMLP(512, 18, ...)     # fullState → obs
        self.recurrentModel = RecurrentModel(...)    # GRU dynamics
        self.priorNet = PriorNet(...)               # h → z (imagination)
        self.posteriorNet = PosteriorNet(...)       # [h, embed] → z (reality)
        self.rewardPredictor = RewardModel(...)     # fullState → reward
        self.continuePredictor = ContinueModel(...) # fullState → continue prob

        # Behavior model
        self.actor = Actor(512, 4, ...)             # fullState → action
        self.critic = Critic(512, ...)              # fullState → value
```

**World Model Training (`worldModelTraining`):**
```python
def worldModelTraining(self, data):
    """Train world model on a batch of sequences."""
    # data.observations: (batchSize=32, batchLength=32, obsSize=18)

    # Encode all observations
    encodedObs = self.encoder(data.observations)  # (32, 32, 256)

    # Initialize states
    h = torch.zeros(32, 256)  # deterministic
    z = torch.zeros(32, 256)  # stochastic

    # Process sequence step by step
    for t in range(1, 32):
        # Step recurrent model
        h = self.recurrentModel(h, z, data.actions[:, t-1])

        # Get prior (from h only) and posterior (from h + observation)
        _, priorLogits = self.priorNet(h)
        z, posteriorLogits = self.posteriorNet(concat(h, encodedObs[:, t]))

        # Collect states and logits

    # === Losses ===
    fullStates = concat(recurrentStates, posteriors)  # (32, 31, 512)

    # Reconstruction loss
    decodedObs = self.decoder(fullStates)
    reconLoss = -Normal(decodedObs, 1).log_prob(data.observations[:, 1:]).mean()

    # Reward prediction loss
    rewardDist = self.rewardPredictor(fullStates)
    rewardLoss = -rewardDist.log_prob(data.rewards[:, 1:]).mean()

    # KL loss (with free nats threshold)
    priorDist = OneHotCategorical(logits=priorsLogits)
    posteriorDist = OneHotCategorical(logits=posteriorsLogits)

    # Prior loss: train prior to match posterior
    priorLoss = kl_divergence(posteriorDist.detach(), priorDist)
    # Posterior loss: train posterior to match prior
    posteriorLoss = kl_divergence(posteriorDist, priorDist.detach())

    # Free nats: only penalize KL above threshold
    priorLoss = max(priorLoss, freeNats=1.0) * betaPrior
    posteriorLoss = max(posteriorLoss, freeNats) * betaPosterior

    # Continue prediction loss
    continueDist = self.continuePredictor(fullStates)
    continueLoss = -continueDist.log_prob(1 - data.dones[:, 1:]).mean()

    totalLoss = reconLoss + rewardLoss + priorLoss + posteriorLoss + continueLoss

    return fullStates.detach(), metrics
```

**Behavior Training (`behaviorTraining`):**
```python
def behaviorTraining(self, fullState):
    """Train actor and critic entirely in imagination."""
    # fullState: (B*T, 512) starting states from world model training

    h, z = fullState.split([256, 256], dim=-1)

    # Imagine trajectories (no observations!)
    for _ in range(imaginationHorizon=15):
        # Get action from actor
        action, logprob, entropy = self.actor(fullState.detach(), training=True)

        # Step world model using PRIOR (no observation available)
        h = self.recurrentModel(h, z, action)
        z, _ = self.priorNet(h)  # Prior only!

        fullState = concat(h, z)
        # Collect states, logprobs, entropies

    # Get predictions for imagined trajectory
    predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
    values = self.critic(fullStates).mean
    continues = self.continuePredictor(fullStates).mean  # or fixed discount

    # Compute lambda returns (TD(λ) targets)
    lambdaValues = computeLambdaValues(predictedRewards, values, continues, lambda_=0.95)

    # Normalize advantages using percentiles
    _, inverseScale = self.valueMoments(lambdaValues)  # 5th-95th percentile range
    advantages = (lambdaValues - values[:, :-1]) / inverseScale

    # Actor loss: maximize advantage-weighted log probability + entropy
    actorLoss = -mean(advantages.detach() * logprobs + entropyScale * entropies)

    # Critic loss: predict lambda returns
    valueDist = self.critic(fullStates[:, :-1].detach())
    criticLoss = -mean(valueDist.log_prob(lambdaValues.detach()))

    return metrics
```

**Acting in Real Environment:**
```python
def act(self, observation, h=None, z=None):
    """Select action for a single observation."""
    if h is None:
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 256)

    # Encode observation
    obs_t = torch.from_numpy(observation).float().unsqueeze(0)
    encodedObs = self.encoder(obs_t)

    # Update recurrent state (using dummy action for first step)
    h = self.recurrentModel(h, z, action_dummy)

    # Get POSTERIOR (we have observation in reality)
    z, _ = self.posteriorNet(concat(h, encodedObs))

    # Get action from actor
    fullState = concat(h, z)
    action = self.actor(fullState, training=False)

    return action.numpy(), h, z
```

### 2.4 Lambda Returns – `utils.py`

TD(λ) return computation for value targets:

```python
def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    """
    Compute TD(λ) returns.

    G_t = r_t + γ_t * [(1-λ) * V_{t+1} + λ * G_{t+1}]

    where γ_t = discount * continue_prob_t
    """
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]  # Final value estimate

    for i in reversed(range(rewards.shape[-1])):
        returns[:, i] = rewards[:, i] + continues[:, i] * (
            (1 - lambda_) * values[:, i] + lambda_ * bootstrap
        )
        bootstrap = returns[:, i]

    return returns
```

Lambda interpolates between:
- λ=0: One-step TD (high bias, low variance)
- λ=1: Monte Carlo (low bias, high variance)
- λ=0.95: Good balance for most tasks

### 2.5 Value Normalization – `utils.py`

Percentile-based normalization for stable actor training:

```python
class Moments(nn.Module):
    """Exponential moving average of percentiles for return normalization."""

    def __init__(self, device, decay=0.99, percentileLow=0.05, percentileHigh=0.95):
        self.low = torch.zeros(())   # 5th percentile
        self.high = torch.zeros(())  # 95th percentile

    def forward(self, x):
        # Update EMA of percentiles
        low = torch.quantile(x, 0.05)
        high = torch.quantile(x, 0.95)
        self.low = 0.99 * self.low + 0.01 * low
        self.high = 0.99 * self.high + 0.01 * high

        # Scale is the range (minimum 1.0)
        inverseScale = max(1.0, self.high - self.low)
        return self.low, inverseScale
```

This makes actor gradients independent of reward scale.

### 2.6 Replay Buffer – `buffer.py`

Stores transitions and samples contiguous sequences:

```python
class ReplayBuffer:
    def __init__(self, observationSize, actionSize, config, device):
        self.capacity = 100000
        self.observations = np.empty((capacity, 18), dtype=np.float32)
        self.actions = np.empty((capacity, 4), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

    def add(self, observation, action, reward, nextObservation, done):
        """Add single transition (FIFO rotation)."""
        self.observations[self.bufferIndex] = observation
        # ... store all fields
        self.bufferIndex = (self.bufferIndex + 1) % self.capacity

    def sample(self, batchSize, sequenceSize):
        """Sample batch of contiguous sequences."""
        # Sample random starting indices
        sampleIndex = np.random.randint(0, maxIdx, batchSize)
        # Get sequences of length 32
        sequenceOffset = np.arange(sequenceSize)
        sampleIndex = (sampleIndex.reshape(-1, 1) + sequenceOffset) % self.capacity

        # Return as batch object
        return Batch(observations, actions, rewards, dones)
```

### 2.7 The Training Loop – `train_hockey.py`

Orchestrates the three learning loops:

```python
# Main training loop
for gradient_step in range(total_gradient_steps):

    # === LOOP 1: Collect Real Experience ===
    if should_collect_experience():
        obs, _ = env.reset()
        h, z = None, None

        while not done:
            action, h, z = dreamer.act(obs, h, z)
            next_obs, reward, done, info = env.step(action)

            # Add PBRS if enabled
            if use_pbrs:
                reward += pbrs_shaper.shape(obs, next_obs, done)

            dreamer.buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

    # === LOOPS 2 & 3: Train World Model and Behavior ===
    if len(dreamer.buffer) >= min_buffer_size:
        # Sample sequence batch
        batch = dreamer.buffer.sample(batchSize=32, sequenceSize=32)

        # Train world model (returns starting states)
        fullStates, world_metrics = dreamer.worldModelTraining(batch)

        # Train actor-critic in imagination
        behavior_metrics = dreamer.behaviorTraining(fullStates)

        dreamer.totalGradientSteps += 1

    # Periodic evaluation
    if gradient_step % eval_interval == 0:
        win_rate = evaluate(env, dreamer, num_episodes=10)
```

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
│              │ Encoder │ (MLP: 18 → 256)                                │
│              └────┬────┘                                                │
│                   │                                                     │
│                   ↓                                                     │
│         Embedding (256 dims)                                            │
│                   │                                                     │
│                   ↓                                                     │
│           ┌──────────────┐                                              │
│           │   RSSM       │                                              │
│           │ (POSTERIOR)  │ ← uses real observation                      │
│           └──────┬───────┘                                              │
│                  │                                                      │
│                  ↓                                                      │
│        Latent State [h=256, z=256]                                      │
│                  │                                                      │
│                  ↓                                                      │
│           ┌──────────┐                                                  │
│           │  Actor   │                                                  │
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
│  Replay Buffer → Sample sequences (32 × 32 timesteps)                  │
│                        │                                                │
│                        ↓                                                │
│                 ┌──────────────┐                                        │
│                 │   Encoder    │                                        │
│                 └──────┬───────┘                                        │
│                        │                                                │
│                        ↓                                                │
│                ┌───────────────┐                                        │
│                │     RSSM      │                                        │
│                │ (process seq) │                                        │
│                │               │                                        │
│                │ t=1: h₁ = GRU(h₀, z₀, a₀)                             │
│                │      z₁_prior ~ PriorNet(h₁)                          │
│                │      z₁_post ~ PosteriorNet(h₁, embed₁)               │
│                │ ...repeat for t=2..31                                  │
│                └───────┬───────┘                                        │
│                        │                                                │
│              ┌─────────┴─────────┐                                      │
│              │                   │                                      │
│              ↓                   ↓                                      │
│    ┌───────────────┐     ┌───────────────┐                             │
│    │    Decoder    │     │ Reward Head   │                             │
│    │   (→ obs)     │     │   (→ R)       │                             │
│    └───────┬───────┘     └───────┬───────┘                             │
│            │                     │                                      │
│            ↓                     ↓                                      │
│       Recon Loss            Reward Loss                                 │
│                                                                         │
│  + KL Loss (prior ↔ posterior) with free nats                          │
│  + Continue Loss (Bernoulli)                                            │
│                        │                                                │
│                        ↓                                                │
│                   Total Loss → Backprop → Update World Model            │
│                                                                         │
│  Output: fullStates (32 × 31 × 512) detached for behavior training     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     ACTOR-CRITIC TRAINING (IMAGINATION)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  fullStates from world model → Starting [h, z] (flattened)             │
│                                            │                            │
│                                            ↓                            │
│                  ┌─────────────────────────────────────────┐            │
│                  │           IMAGINATION LOOP              │            │
│                  │                                         │            │
│                  │  for t in range(horizon=15):            │            │
│                  │      fullState_t = [h_t, z_t]           │            │
│                  │      action, logprob, entropy = Actor() │            │
│                  │                                         │            │
│                  │      h_{t+1} = RecurrentModel(h, z, a)  │            │
│                  │      z_{t+1} ~ PRIOR(h_{t+1})           │ ← No obs!  │
│                  │                                         │            │
│                  │      (using prior, not posterior!)      │            │
│                  │                                         │            │
│                  └────────────────┬────────────────────────┘            │
│                                   │                                     │
│                                   ↓                                     │
│       Imagined trajectory: fullStates, logprobs, entropies             │
│                                   │                                     │
│                    ┌──────────────┴──────────────┐                      │
│                    │                             │                      │
│                    ↓                             ↓                      │
│           ┌────────────────┐           ┌────────────────┐               │
│           │ Reward Head    │           │ Continue Head  │               │
│           │ (predict R)    │           │ (predict γ)    │               │
│           └───────┬────────┘           └───────┬────────┘               │
│                   │                            │                        │
│                   └──────────┬─────────────────┘                        │
│                              ↓                                          │
│                   ┌───────────────────┐                                 │
│                   │  Lambda Returns   │                                 │
│                   │  TD(λ) targets    │                                 │
│                   └─────────┬─────────┘                                 │
│                             │                                           │
│                             ↓                                           │
│               ┌─────────────┴─────────────┐                             │
│               │                           │                             │
│               ↓                           ↓                             │
│        ┌────────────┐              ┌────────────┐                       │
│        │   Critic   │              │   Actor    │                       │
│        │  Training  │              │  Training  │                       │
│        │            │              │            │                       │
│        │ V(s) → λ   │              │ -logπ × A  │                       │
│        │ returns    │              │ + entropy  │                       │
│        └──────┬─────┘              └──────┬─────┘                       │
│               │                           │                             │
│               ↓                           ↓                             │
│          Critic Loss               Actor Loss                           │
│          -log_prob(λ)              -advantage×logπ - η×entropy          │
│               │                           │                             │
│               └──────────┬────────────────┘                             │
│                          ↓                                              │
│                    Backprop → Update Actor & Critic                     │
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

### Key Hyperparameters and Their Effects

| Parameter | Value | Effect |
|-----------|-------|--------|
| `recurrentSize` | 256 | GRU hidden state size (temporal memory) |
| `latentLength × latentClasses` | 16 × 16 | Stochastic state capacity (256-dim) |
| `imaginationHorizon` | 15 | How far to look ahead in imagination |
| `lambda_` | 0.95 | TD(λ) bias-variance tradeoff |
| `discount` | 0.997 | Future reward importance |
| `entropyScale` | 0.003 | Exploration vs. exploitation |
| `freeNats` | 1.0 | KL threshold to prevent collapse |
| `replayRatio` | 32 | Gradient steps per environment step |

### Common Issues and Solutions

1. **Entropy Collapse (policy becomes deterministic)**
   - Symptom: `behavior/entropy` goes to 0 or negative
   - Solution: Increase `entropyScale` (try 3e-3 or higher)

2. **World Model Not Learning**
   - Symptom: `world/recon_loss` stays high
   - Solution: Check encoder/decoder architecture, learning rate

3. **Slow Training**
   - Symptom: Many seconds per gradient step
   - Solution: Reduce `imaginationHorizon` (try 10), reduce `replayRatio` (try 8)

4. **Memory Issues**
   - Symptom: OOM on GPU/MPS
   - Solution: Reduce `batchSize` or `batchLength`

---

## Part 4: Two-Hot Symlog Encoding – Fixing Sparse Reward Prediction

This section explains the Two-Hot Symlog encoding, a critical DreamerV3 innovation that enables accurate prediction of sparse rewards.

### 4.1 The Problem: Normal Distributions Can't Handle Sparse Rewards

In hockey, rewards are **multi-modal**:
- 99% of timesteps: reward = 0 (nothing happened)
- ~1% of timesteps: reward = ±10 (goal scored or conceded)

Our original implementation used **Normal distributions** to predict rewards:

```python
# Old approach: predict mean and std, return Normal distribution
mean, std = reward_network(latent_state)
reward_dist = Normal(mean, std)
loss = -reward_dist.log_prob(actual_reward).mean()
```

**Why this fails catastrophically:**

A Normal distribution is unimodal – it has a single peak. When trained on data that's mostly 0 with occasional ±10:

```
Actual reward distribution:      What Normal learns:

     ↑  ↑                              ╱╲
     │  │                             ╱  ╲
     │  │                            ╱    ╲
─────┼──┼────────────              ═╱══════╲═══
   -10  0  +10                      0 (mean)

Multi-modal                      Unimodal (compromises)
```

The Normal distribution **regresses to the mean**. It learns to predict ~0 for everything because that minimizes average error. It cannot represent "this state usually gives 0, but might give +10."

**Empirical evidence from our training:**
```
sparse_vs_nonsparse_error_ratio: 100-300x
sparse_pred_error: ~10 (error equals the reward magnitude!)
sparse_pred_mean: ~0 (predicts 0 even for goals)
imagination/reward_significant_frac: 0.0 (imagination NEVER predicts goals)
```

The world model was blind to sparse rewards. Even when goals occurred, the reward predictor said "probably 0."

### 4.2 The Solution: Discretize with Two-Hot Encoding

Instead of predicting a continuous value, DreamerV3 **discretizes** the prediction into bins and predicts a **categorical distribution**:

```
Instead of:  network → (mean, std) → Normal distribution
We use:      network → 255 logits → Categorical distribution over bins
```

The bins span the reward range in **symlog space** (more on this below):

```
Bin index:    0    63   127   191   254
              │     │     │     │     │
Symlog val: -20   -10    0    +10   +20
              │     │     │     │     │
Real val:  -485M  -22k   0   +22k  +485M
```

**Why categorical works for multi-modal distributions:**

A categorical distribution can assign probability mass to **multiple bins simultaneously**:

```
Predicted distribution for "might score soon" state:

Probability
    ↑
0.6 │           ████
0.4 │           ████
0.2 │    ██     ████     ██
    └────────────────────────→ bins
        -10      0       +10

"Probably 0, but 20% chance of ±10"
```

This is impossible with a unimodal Normal distribution but trivial with a categorical.

### 4.3 The "Two-Hot" Encoding

Standard one-hot encoding puts all probability mass on a single bin. But what if the target value falls **between** bins?

**Two-hot encoding** spreads probability between the two adjacent bins:

```
Target reward: 7.5
Bin 190 center: 7.0
Bin 191 center: 8.0

Target distribution (two-hot):
bin 190: 50% (closer to 7.0)
bin 191: 50% (closer to 8.0)
all other bins: 0%
```

More precisely, if target value falls between bins k and k+1:

```
α = (target - bin_k_center) / (bin_{k+1}_center - bin_k_center)
P(bin k) = 1 - α
P(bin k+1) = α
```

**Training loss:** Cross-entropy between predicted logits and two-hot target

```python
# Two-hot cross-entropy loss
log_probs = F.log_softmax(logits, dim=-1)
loss = -((1 - alpha) * log_probs[k] + alpha * log_probs[k+1])
```

This gives **smooth gradients** even when the target falls between bins.

### 4.4 Symlog Transform: Handling Large Value Ranges

Raw rewards range from -10 to +10, but **value estimates** can be much larger (cumulative discounted rewards over an episode). We need bins to cover a wide range without wasting resolution on unlikely values.

**Symlog** (symmetric logarithm) compresses large values while preserving behavior near zero:

```
symlog(x) = sign(x) * ln(|x| + 1)

Examples:
  symlog(0)     = 0
  symlog(1)     = 0.69
  symlog(10)    = 2.40   ← sparse rewards
  symlog(100)   = 4.62
  symlog(1000)  = 6.91
  symlog(-10)   = -2.40
```

Properties:
- **symlog(0) = 0** – zero stays zero
- **Symmetric** – negative values mirror positive
- **Gradient near 0 ≈ 1** – well-behaved for small values
- **Compresses large values** – 1000 maps to only 6.91

The inverse, **symexp**, converts back:

```
symexp(x) = sign(x) * (exp(|x|) - 1)
```

### 4.5 The Complete Two-Hot Symlog Pipeline

**Encoding a target value (for training):**

```
                    symlog              find bins           compute weights
Target (10.0) ──────────────→ 2.40 ──────────────→ k=190, k+1=191 ──────────→ P(190)=0.3, P(191)=0.7
```

**Decoding a prediction (for inference):**

```
                    softmax              weighted sum         symexp
255 logits ──────────────→ probabilities ──────────────→ ~2.40 ──────────────→ ~10.0
```

**Our implementation in `utils.py`:**

```python
class TwoHotSymlog(nn.Module):
    def __init__(self, bins=255, min_val=-20.0, max_val=20.0):
        # 255 bins from -20 to +20 in symlog space
        self.bin_centers = torch.linspace(min_val, max_val, bins)
        self.step = (max_val - min_val) / (bins - 1)  # ~0.157

    def loss(self, logits, target):
        """Two-hot cross-entropy loss."""
        # Transform target to symlog space
        y = symlog(target)
        y = torch.clamp(y, -20, 20)

        # Find bin indices
        continuous_idx = (y - self.min_val) / self.step
        k = continuous_idx.long().clamp(0, 253)  # Lower bin
        k_plus_1 = k + 1                          # Upper bin

        # Compute two-hot weights
        alpha = (continuous_idx - k.float()).clamp(0, 1)

        # Cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -((1 - alpha) * log_probs.gather(-1, k)
                 + alpha * log_probs.gather(-1, k_plus_1))
        return loss

    def decode(self, logits):
        """Decode logits to scalar value."""
        probs = F.softmax(logits, dim=-1)
        y_hat = (probs * self.bin_centers).sum(dim=-1)  # Expected value in symlog space
        return symexp(y_hat)  # Convert back to original scale
```

### 4.6 Where Two-Hot Is Applied

In DreamerV3, two-hot symlog is used for:

1. **Reward Prediction** (world model):
   ```python
   # networks.py
   class RewardModel(nn.Module):
       def forward(self, latent_state):
           return self.network(latent_state)  # → (batch, 255) logits

   # dreamer.py - worldModelTraining
   reward_logits = self.rewardPredictor(full_states)  # (B, T, 255)
   reward_loss = self.twoHot.loss(reward_logits, actual_rewards).mean()
   ```

2. **Value Prediction** (critic):
   ```python
   # networks.py
   class Critic(nn.Module):
       def forward(self, latent_state):
           return self.network(latent_state)  # → (batch, 255) logits

   # dreamer.py - behaviorTraining
   value_logits = self.critic(full_states)  # (B, H, 255)
   values = self.twoHot.decode(value_logits)  # → scalar values
   critic_loss = self.twoHot.loss(value_logits, lambda_returns).mean()
   ```

3. **Observation Preprocessing** (symlog only, not two-hot):
   ```python
   # dreamer.py - worldModelTraining
   obs_symlog = symlog(data.observations)  # Compress velocity spikes
   encoded_obs = self.encoder(obs_symlog)
   ```

### 4.7 Why This Fixes Our Sparse Reward Problem

**Before Two-Hot (Normal distribution):**

```
State near goal → Reward predictor → Normal(mean=0.1, std=0.5)
                                     ↓
                                 Predicts: ~0.1
                                 Actual: +10 or 0
                                 Error on +10: 9.9
```

The model learned to hedge by predicting near zero, which minimized average error but gave zero gradient signal for sparse events.

**After Two-Hot (Categorical distribution):**

```
State near goal → Reward predictor → 255 logits
                                     ↓
                                 softmax → [0, ..., 0.7 at bin_0, ..., 0.3 at bin_+10, ...]
                                 Decoded: weighted average ≈ 3.0

If +10 occurs: loss increases for bin_+10, model learns!
If 0 occurs: loss increases for bin_0, model learns!
```

The model can represent "probably 0, but maybe +10" and gets **gradient signal regardless of which outcome occurs**.

**Expected metrics after fix:**

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| `sparse_vs_nonsparse_error_ratio` | 100-300x | <10x |
| `sparse_pred_error` | ~10 | <3 |
| `sparse_sign_accuracy` | ~0.5 | >0.8 |
| `imagination/reward_significant_frac` | 0.0 | >0.01 |

### 4.8 Critic Initialization: Why Zeros Matter

The critic output layer is initialized to zeros:

```python
class Critic(nn.Module):
    def __init__(self, ...):
        # ... build network ...
        # Initialize output layer to zeros
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()
```

**Why?** Zero logits → uniform distribution over bins → expected value ≈ 0.

This means the critic starts by predicting "value ≈ 0" for all states, which is a reasonable prior. The actor then explores without strong biases from an untrained critic.

### 4.9 Summary: The Complete Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-HOT SYMLOG ENCODING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENCODING (Training):                                                       │
│                                                                             │
│  Target value (e.g., +10)                                                   │
│         │                                                                   │
│         ↓  symlog                                                           │
│  Compressed value (e.g., 2.40)                                              │
│         │                                                                   │
│         ↓  find adjacent bins                                               │
│  Bin k=190, k+1=191                                                         │
│         │                                                                   │
│         ↓  compute weights (α = 0.7)                                        │
│  Two-hot: P(190)=0.3, P(191)=0.7                                            │
│         │                                                                   │
│         ↓  cross-entropy with network logits                                │
│  Loss = -(0.3 × log_prob[190] + 0.7 × log_prob[191])                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DECODING (Inference):                                                      │
│                                                                             │
│  Network output: 255 logits                                                 │
│         │                                                                   │
│         ↓  softmax                                                          │
│  Probabilities: [p₀, p₁, ..., p₂₅₄]                                         │
│         │                                                                   │
│         ↓  weighted sum of bin centers                                      │
│  Expected symlog value: Σ pᵢ × bin_centerᵢ ≈ 2.40                           │
│         │                                                                   │
│         ↓  symexp                                                           │
│  Decoded value: ≈ 10.0                                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHY IT WORKS:                                                              │
│                                                                             │
│  • Categorical distributions can be multi-modal                             │
│  • Each bin gets its own gradient signal                                    │
│  • Sparse events (+10) update bins around +10                               │
│  • Common events (0) update bins around 0                                   │
│  • No "regression to mean" problem                                          │
│                                                                             │
│  • Symlog compresses large values into finite range                         │
│  • 255 bins from -20 to +20 in symlog space                                 │
│  • Covers rewards from -485M to +485M in original scale                     │
│  • But most resolution where it matters (near 0, ±10)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Auxiliary Tasks – Replacing PBRS for World Model Learning

This section explains auxiliary tasks, a DreamerV3-native approach to improve world model representations without corrupting the reward signal.

### 5.1 The Problem with PBRS in DreamerV3

**Potential-Based Reward Shaping (PBRS)** works well for model-free algorithms like TD3:
- Adds dense reward signal to guide exploration
- Mathematically proven to not change the optimal policy

However, PBRS is problematic for world-model based algorithms like DreamerV3:

```
The fundamental issue:
┌─────────────────────────────────────────────────────────────────────────────┐
│ DreamerV3 trains its actor-critic ENTIRELY in imagination                   │
│                                                                             │
│ Real Environment → Collects data → Trains World Model → Predicts rewards   │
│                                                                             │
│ If PBRS corrupts rewards:                                                   │
│   1. World model learns to predict PBRS-corrupted rewards                   │
│   2. Policy optimizes for PBRS-corrupted rewards IN IMAGINATION             │
│   3. Policy doesn't learn optimal behavior for TRUE sparse rewards          │
│                                                                             │
│ Unlike TD3 which eventually sees true rewards and adjusts,                  │
│ DreamerV3's policy NEVER sees true rewards during training!                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why PBRS fails for DreamerV3:**

1. **Reward prediction learns wrong targets**: The reward head predicts PBRS-shaped rewards, not true game outcomes
2. **No grounding**: Policy optimizes for a surrogate signal that doesn't represent actual winning/losing
3. **Imagination mismatch**: When evaluating without PBRS, behavior is misaligned

### 5.2 The Solution: Auxiliary Tasks

Instead of corrupting the reward signal, we add **auxiliary prediction tasks** that help the world model learn goal-relevant representations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUXILIARY TASKS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  The key insight: We want the world model to understand goal-relevant       │
│  features, but WITHOUT changing the reward signal.                          │
│                                                                             │
│  Solution: Add prediction heads that learn these features as EXTRA TASKS    │
│                                                                             │
│  World Model now predicts:                                                  │
│    1. Observations (reconstruction) ← existing                              │
│    2. Rewards (sparse, true values) ← existing, UNCHANGED                   │
│    3. Continue probability ← existing                                       │
│    4. Goal Prediction ← NEW auxiliary task                                  │
│    5. Puck-Goal Distance ← NEW auxiliary task                               │
│    6. Shot Quality ← NEW auxiliary task                                     │
│                                                                             │
│  The auxiliary tasks improve representations WITHOUT touching rewards!      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Our Three Auxiliary Tasks

**1. Goal Prediction (Binary Classification)**

```python
# networks.py
class GoalPredictionHead(nn.Module):
    """Predicts: Will a goal happen in the next K steps?"""

    def forward(self, full_state):
        logits = self.network(full_state)  # → (batch, 1)
        return logits  # Binary: 0 = no goal, 1 = goal incoming
```

Labels are computed from actual rewards:
- `y = 1` if any `|reward| > 1` in next `goal_horizon` steps (default: 15)
- `y = 0` otherwise

This teaches the world model to recognize "about to score" situations.

**2. Puck-Goal Distance (Regression)**

```python
# networks.py
class DistanceHead(nn.Module):
    """Predicts distance from puck to opponent goal."""

    def forward(self, full_state):
        return self.network(full_state)  # → (batch, 1) normalized distance
```

Computed from observations:
```python
puck_pos = obs[..., 12:14]  # Puck x, y
opponent_goal = [-1.0, 0.0]  # Left goal position
distance = torch.norm(puck_pos - opponent_goal, dim=-1)
```

This teaches spatial awareness of offensive position.

**3. Shot Quality (Regression)**

```python
# networks.py
class ShotQualityHead(nn.Module):
    """Predicts combined shooting opportunity quality."""

    def forward(self, full_state):
        return self.network(full_state)  # → (batch, 1) quality score
```

Combines position and momentum:
```python
# Shot quality = closeness to goal + positive x-velocity (toward goal)
shot_quality = (1 - normalized_distance) * 0.5 + positive_vel_component * 0.5
```

This teaches the model to value attacking positions with good momentum.

### 5.4 Integration into World Model Training

```python
# dreamer.py - worldModelTraining()

def worldModelTraining(self, data):
    # ... existing world model training ...

    # Compute standard losses (UNCHANGED)
    recon_loss = self.decoder.loss(full_states, data.observations)
    reward_loss = self.twoHot.loss(reward_logits, data.rewards)  # TRUE rewards!
    continue_loss = -continue_dist.log_prob(1 - data.dones).mean()
    kl_loss = self.compute_kl_loss(prior_logits, posterior_logits)

    # === AUXILIARY TASKS ===
    if self.use_auxiliary_tasks:
        # 1. Goal Prediction
        goal_labels = self.compute_goal_labels(data.rewards, horizon=15)
        goal_logits = self.goal_head(full_states)
        goal_loss = F.binary_cross_entropy_with_logits(goal_logits, goal_labels)

        # 2. Distance Prediction
        true_distances = self.compute_puck_goal_distance(data.observations)
        pred_distances = self.distance_head(full_states)
        distance_loss = F.mse_loss(pred_distances, true_distances)

        # 3. Shot Quality
        true_quality = self.compute_shot_quality(data.observations)
        pred_quality = self.shot_quality_head(full_states)
        quality_loss = F.mse_loss(pred_quality, true_quality)

        aux_loss = self.aux_scale * (goal_loss + distance_loss + quality_loss)
    else:
        aux_loss = 0

    total_loss = recon_loss + reward_loss + continue_loss + kl_loss + aux_loss
```

**Critical difference from PBRS:**
- Auxiliary losses train the ENCODER and LATENT SPACE
- Reward prediction still trains on TRUE sparse rewards
- Policy optimizes for TRUE rewards in imagination

### 5.5 Why Auxiliary Tasks Work

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 HOW AUXILIARY TASKS IMPROVE LEARNING                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WITHOUT auxiliary tasks:                                                   │
│    Latent space optimized for: reconstruction + reward + continue           │
│    Problem: Reconstruction doesn't emphasize goal-relevant features         │
│             Sparse rewards provide weak signal                              │
│                                                                             │
│  WITH auxiliary tasks:                                                      │
│    Latent space optimized for: reconstruction + reward + continue           │
│                                + goal prediction                            │
│                                + distance awareness                         │
│                                + shot quality                               │
│                                                                             │
│    Result: Latent space encodes "am I about to score?" information          │
│            Reward predictor has BETTER FEATURES to work with                │
│            Two-Hot Symlog can now predict sparse rewards accurately         │
│            Policy gets useful gradients in imagination                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The key insight:** Auxiliary tasks don't change WHAT the model predicts for rewards. They change HOW WELL the latent space represents goal-relevant information, which makes reward prediction more accurate.

### 5.6 Inverse Frequency Weighting for Sparse Events

Even with Two-Hot Symlog, sparse reward events (goals) are outnumbered by non-events ~100:1. We apply inverse frequency weighting:

```python
# dreamer.py - worldModelTraining()

# Count sparse vs non-sparse samples
sparse_mask = (rewards.abs() > 1.0).float()
n_sparse = sparse_mask.sum()
n_nonsparse = (1 - sparse_mask).sum()

# Compute inverse frequency weights (capped at 100x)
if n_sparse > 0 and n_nonsparse > 0:
    sparse_weight = min(100.0, n_nonsparse / n_sparse)
    nonsparse_weight = 1.0
else:
    sparse_weight = 1.0
    nonsparse_weight = 1.0

# Weight the reward loss
weights = sparse_mask * sparse_weight + (1 - sparse_mask) * nonsparse_weight
weighted_reward_loss = (reward_loss_per_sample * weights).mean()
```

This ensures goals contribute meaningfully to the gradient despite their rarity.

### 5.7 Hyperparameters for Auxiliary Tasks

| Parameter | Default | Description |
|-----------|---------|-------------|
| `useAuxiliaryTasks` | True | Enable auxiliary task training |
| `auxTaskScale` | 1.0 | Weight for auxiliary losses relative to world model losses |
| `goalPredictionHorizon` | 15 | How many steps ahead to predict goal occurrence |
| `auxHiddenSize` | 128 | Hidden layer size for auxiliary prediction heads |

**Tuning guidance:**

1. **`auxTaskScale`**: Start with 1.0. If auxiliary losses dominate (check W&B), reduce to 0.1-0.5
2. **`goalPredictionHorizon`**: Match your `imagination_horizon`. Default 15 works well
3. **`auxHiddenSize`**: 128 is sufficient. Larger values don't help much

### 5.8 Critical: Entropy Scale with Auxiliary Tasks

**This is the most important hyperparameter when using auxiliary tasks:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTROPY COLLAPSE = AUXILIARY TASK FAILURE                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  If entropy_scale is too low (< 0.001):                                     │
│    1. Policy becomes deterministic early                                    │
│    2. Agent stops exploring diverse states                                  │
│    3. Auxiliary tasks see repetitive/boring data                            │
│    4. Goal prediction never sees "about to score" states                    │
│    5. Auxiliary tasks learn nothing useful                                  │
│    6. Latent space doesn't improve                                          │
│                                                                             │
│  RECOMMENDED: entropy_scale >= 0.003                                        │
│                                                                             │
│  Monitor: behavior/entropy should stay POSITIVE (> 0)                       │
│           If it goes negative, policy has collapsed                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.9 Recommended Configuration

For optimal auxiliary task training:

```bash
python train_hockey.py \
    --opponent weak \
    --seed 42 \
    --gradient_steps 1000000 \
    --batch_size 32 \
    --batch_length 32 \
    --imagination_horizon 15 \
    --recurrent_size 256 \
    --latent_length 16 \
    --latent_classes 16 \
    --lr_world 0.0003 \
    --lr_actor 0.00008 \
    --lr_critic 0.0001 \
    --entropy_scale 0.003 \
    --gradient_clip 100 \
    --free_nats 1.0 \
    --discount 0.997
```

**Key changes from PBRS-based training:**
- Remove all `--use_pbrs`, `--pbrs_scale`, `--pbrs_w_*` arguments (they no longer exist)
- Increase `--entropy_scale` to 0.003 (was often 0.0005 with PBRS)
- Increase `--lr_world` to 0.0003 (faster auxiliary task learning)
- Increase `--gradient_clip` to 100 (DreamerV3 default)

### 5.10 Summary: PBRS vs Auxiliary Tasks

| Aspect | PBRS | Auxiliary Tasks |
|--------|------|-----------------|
| **Modifies rewards** | Yes (corrupts signal) | No (preserves true rewards) |
| **Policy training** | Optimizes shaped rewards | Optimizes true rewards |
| **World model** | Learns wrong reward targets | Learns true rewards + extra features |
| **Evaluation** | Behavior may differ without PBRS | Consistent behavior always |
| **Implementation** | Modifies reward before storage | Adds extra prediction heads |
| **Hyperparameters** | pbrs_scale, weights | aux_scale, entropy_scale |
| **DreamerV3 compatible** | No (breaks imagination training) | Yes (native approach) |

---

## Summary Table: TD3 vs. DreamerV3

| Aspect | TD3 | DreamerV3 |
|--------|-----|-----------|
| **Learning paradigm** | Model-free | Model-based |
| **World model** | None | Full (RSSM + heads) |
| **Policy training** | On real transitions | In imagination |
| **State representation** | Raw observations | Learned latent [h, z] |
| **Stochastic state** | N/A | Categorical (16×16) |
| **Sparse rewards** | Struggles | Handles via long horizon |
| **Sample efficiency** | Lower | Higher |
| **Computational cost** | Lower per step | Higher per step |
| **Memory** | Simple replay buffer | Sequence buffer + models |
| **Key components** | Actor, 2× Critic | World Model, Actor, Critic |
| **Reward shaping** | Often needed (PBRS) | Not needed (uses auxiliary tasks) |

---

*AI Usage Declaration: This document was developed with assistance from Claude Code.*
