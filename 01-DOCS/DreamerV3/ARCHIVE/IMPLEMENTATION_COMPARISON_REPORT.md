# DreamerV3 Implementation Comparison Report

## Executive Summary

**NaturalDreamer**: 6 files, ~700 lines total, works.
**Our Implementation**: 20+ files, ~2500+ lines total, doesn't work.

The fundamental problem: **We over-engineered everything.**

---

## File Count Comparison

| Component | NaturalDreamer | Our Implementation |
|-----------|----------------|-------------------|
| Main training | `main.py` (60 lines) | `train_hockey.py` (400+ lines) |
| Agent | `dreamer.py` (260 lines) | `hockey_dreamer.py` + `world_model.py` + `behavior.py` (800+ lines) |
| Networks | `networks.py` (160 lines) | `sequence_model.py` + parts of others (450+ lines) |
| Utils | `utils.py` (180 lines) | `math_ops.py` + `distributions.py` (450+ lines) |
| Buffer | `buffer.py` (55 lines) | `buffer.py` (similar) |
| **TOTAL** | **~700 lines** | **~2500+ lines** |

---

## Critical Differences

### 1. Reward/Value Prediction

**NaturalDreamer (SIMPLE)**:
```python
class RewardModel(nn.Module):
    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
```
- Outputs Normal distribution
- 2 outputs: mean and log_std
- Loss: `-distribution.log_prob(target).mean()`

**Our Implementation (COMPLEX)**:
```python
class TwoHotDistLayer(nn.Module):
    # 255 bins from -20 to +20 in symlog space
    # Custom two-hot encoding
    # Complex log_prob calculation with interpolation
    # Symlog/symexp transformations
```
- Outputs categorical distribution over 255 bins
- Custom two-hot encoding for targets
- ~200 lines of code for this alone

**Impact**: Two-hot encoding is theoretically elegant for handling multi-modal value distributions, but for hockey with rewards in [-1, 0, +1], **a simple Normal distribution is sufficient**. The complexity of Two-Hot is hurting us without any benefit.

---

### 2. KL Divergence Computation

**NaturalDreamer (SIMPLE)**:
```python
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough

priorDistribution       = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits), 1)
posteriorDistribution   = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits), 1)
priorDistributionSG     = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()), 1)
posteriorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach()), 1)

priorLoss     = kl_divergence(posteriorDistributionSG, priorDistribution)
posteriorLoss = kl_divergence(posteriorDistribution, priorDistributionSG)
```
- Uses PyTorch's built-in `kl_divergence()` function
- Uses PyTorch's `OneHotCategoricalStraightThrough` (discrete sampling with straight-through gradients)
- 6 lines of code

**Our Implementation (COMPLEX)**:
```python
class CategoricalDist:
    def kl_divergence(self, other):
        kl = (self.probs * (torch.log(self.probs + 1e-8) - torch.log(other.probs + 1e-8))).sum(dim=-1)
        return kl.sum(dim=-1)
```
- Custom CategoricalDist class (~90 lines)
- Manual KL computation
- Manual Gumbel-softmax sampling
- Manual unimix handling

**Impact**: We reimplemented what PyTorch already provides. Potential bugs, harder to understand.

---

### 3. Lambda Returns

**NaturalDreamer (7 LINES)**:
```python
def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
        bootstrap = returns[:, i]
    return returns
```

**Our Implementation (55 LINES)**:
```python
def lambda_returns(rewards, values, continues, bootstrap, gamma, lambda_):
    # Handle dimension variations (10 lines)
    if rewards.dim() == 3: rewards = rewards.squeeze(-1)
    # ... more dimension handling ...

    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else bootstrap
        td_target = rewards[t] + gamma * continues[t] * next_value
        returns[t] = (1 - lambda_) * td_target + lambda_ * (
            rewards[t] + gamma * continues[t] * next_return
        )
        next_return = returns[t]
    return returns
```

**Impact**: Our version is more "correct" mathematically, but NaturalDreamer's simpler formula **works just fine** and is easier to understand.

---

### 4. MLP Construction

**NaturalDreamer**:
```python
def sequentialModel1D(inputSize, hiddenSizes, outputSize, activation="Tanh"):
    layers = []
    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(getattr(nn, activation)())
    layers.append(nn.Linear(currentInputSize, outputSize))
    return nn.Sequential(*layers)
```
- Linear → Activation → Linear → Activation → Linear
- Simple, standard MLP

**Our Implementation**:
```python
def build_mlp(input_dim, output_dim, hidden_dims, activation=nn.SiLU, norm=True):
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if norm:
            layers.append(RMSNorm(hidden_dim))  # <-- Extra complexity
        layers.append(activation())
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
```
- Linear → RMSNorm → Activation → ...
- RMSNorm adds parameters and computation
- Paper uses LayerNorm only for specific components, not everywhere

**Impact**: RMSNorm in every MLP is unnecessary overhead. It can actually hurt training by normalizing gradients in unhelpful ways.

---

### 5. Actor Design

**NaturalDreamer**:
```python
class Actor(nn.Module):
    def forward(self, x, training=False):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        std = torch.exp(logStd)
        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh * self.actionScale + self.actionBias

        if training:
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(self.actionScale * (1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action
```
- Single forward pass returns everything needed
- Action scaling built in
- Training mode returns (action, logprob, entropy) together

**Our Implementation**:
```python
class Policy(nn.Module):
    def forward(self, features):
        # Returns TanhNormal distribution
        return TanhNormal(mean, std)

# Usage in behavior.py:
action_dists = self.policy(features)
log_probs = action_dists.log_prob(actions)
entropy = action_dists.entropy()
```
- Returns distribution object
- Caller must extract everything separately
- Additional TanhNormal class (~85 lines)

**Impact**: Our abstraction creates more code paths, more places for bugs.

---

### 6. World Model Training Loop

**NaturalDreamer**:
```python
def worldModelTraining(self, data):
    encodedObs = self.encoder(data.observations.view(-1, *shape)).view(B, T, -1)
    h = torch.zeros(B, recurrentSize)
    z = torch.zeros(B, latentSize)

    for t in range(1, T):
        h = self.recurrentModel(h, z, data.actions[:, t-1])
        _, priorLogits = self.priorNet(h)
        z, postLogits = self.posteriorNet(torch.cat((h, encodedObs[:, t]), -1))
        # Collect states...

    # Compute losses directly
    reconLoss = -Independent(Normal(decoderMeans, 1), 3).log_prob(data.obs[:, 1:]).mean()
    rewardLoss = -self.rewardPredictor(fullStates).log_prob(data.rewards[:, 1:]).mean()
    klLoss = ...

    loss = reconLoss + rewardLoss + klLoss
```
- Direct loop over timesteps
- All logic in one method
- ~60 lines

**Our Implementation**:
```python
def compute_loss(self, batch):
    posteriors, priors = self.observe(obs, actions, is_first)  # Calls observe_sequence
    features = self.get_features(posteriors)

    # Reconstruction loss
    obs_pred = self.decode_obs(features_flat)
    recon_loss = F.mse_loss(obs_pred, obs_target)

    # Reward loss with Two-Hot
    reward_dist = self.reward_head(features_flat)
    reward_log_probs = reward_dist.log_prob(rewards_flat)
    reward_weights = torch.where(...)  # Complex weighting
    reward_loss = -(reward_log_probs * reward_weights).mean()

    # KL loss
    kl_loss, kl_value = self.dynamics.kl_loss(posteriors, priors, ...)
```
- Logic spread across multiple methods
- observe() → observe_sequence() → posterior_step() chain
- ~150 lines across multiple files

**Impact**: Harder to trace execution flow, harder to debug.

---

### 7. Behavior Training

**NaturalDreamer**:
```python
def behaviorTraining(self, fullState):
    h, z = torch.split(fullState, (recurrentSize, latentSize), -1)

    for _ in range(imaginationHorizon):
        action, logprob, entropy = self.actor(fullState.detach(), training=True)
        h = self.recurrentModel(h, z, action)
        z, _ = self.priorNet(h)  # Prior only - no observation!
        fullState = torch.cat((h, z), -1)
        # Collect...

    rewards = self.rewardPredictor(fullStates[:, :-1]).mean
    values = self.critic(fullStates).mean
    lambdaValues = computeLambdaValues(rewards, values, continues, lambda_)

    _, inverseScale = self.valueMoments(lambdaValues)
    advantages = (lambdaValues - values[:, :-1]) / inverseScale

    actorLoss = -torch.mean(advantages.detach() * logprobs + entropyScale * entropies)
    criticLoss = -torch.mean(self.critic(fullStates[:, :-1]).log_prob(lambdaValues.detach()))
```
- Single method, ~50 lines
- Clear flow: imagine → compute returns → compute losses → backprop

**Our Implementation**:
- `imagine()` in world_model.py
- `train_critic()` in behavior.py
- `train_actor()` in behavior.py
- `compute_value_targets()` in behavior.py
- Coordinated from `train_step()` in hockey_dreamer.py
- ~250 lines across files

---

## What We Should Fix

### Immediate Simplifications

1. **Replace TwoHotDist with Normal distribution**
   - For rewards: `Normal(mean, std)` with 2-output MLP
   - For values: Same `Normal(mean, std)`
   - Remove symlog transforms (not needed for [-1, +1] rewards)

2. **Use PyTorch's categorical distribution**
   - Replace `CategoricalDist` with `OneHotCategoricalStraightThrough`
   - Use `kl_divergence()` from torch.distributions
   - Keep unimix as probability mixing before creating distribution

3. **Simplify lambda returns**
   - Use NaturalDreamer's 7-line version
   - It's mathematically equivalent for our use case

4. **Remove RMSNorm from MLPs**
   - Simple Linear → Activation → Linear
   - Only use normalization where paper specifically recommends

5. **Combine actor outputs**
   - Actor forward returns (action, logprob, entropy) in training mode
   - Simpler interface, less code

### Structural Changes

6. **Merge files**
   - Combine `sequence_model.py` + `world_model.py` → single `world_model.py`
   - Keep `behavior.py` separate but simpler
   - Networks can stay in their own file

7. **Inline the training loops**
   - WMT: Direct `for t in range(1, T):` loop
   - BT: Direct `for _ in range(horizon):` loop
   - Remove `observe_sequence()` and `imagine_sequence()` abstractions

8. **Remove terminal reward weighting**
   - With Normal distribution, model naturally handles sparse rewards
   - Two-Hot was the problem, not the solution

---

## Key Lessons from NaturalDreamer

1. **Use PyTorch distributions directly** - they work, they're tested, they're fast

2. **Simple > Elegant** - A 7-line lambda return beats a 55-line "proper" implementation

3. **Everything in one place** - Training logic visible in one method is easier to debug

4. **Normal distribution works** - Two-hot encoding is for extreme reward ranges; hockey doesn't need it

5. **No custom distributions** - `OneHotCategoricalStraightThrough`, `Normal`, `Independent` cover everything

6. **Actor returns everything** - (action, logprob, entropy) together simplifies downstream code

7. **Direct loops** - `for t in range(T):` is clearer than abstracted sequence processing

---

## Recommended Action

**Option A: Refactor our implementation**
- Time: 2-3 hours
- Risk: Introduce bugs while changing working(ish) code
- Benefit: Keep our file structure

**Option B: Adapt NaturalDreamer for hockey** (RECOMMENDED)
- Time: 1-2 hours
- Risk: Low - NaturalDreamer works
- Benefit: Start from known-good code, add hockey-specific pieces

The recommended path is **Option B**:
1. Copy NaturalDreamer structure
2. Replace Conv encoder/decoder with MLP for 18-dim observations
3. Add PBRS reward shaping in environment interaction
4. Add self-play/opponent logic
5. Connect to hockey environment

This gives us a working baseline that we understand, rather than debugging our over-engineered mess.

---

## Appendix: Code Examples to Adopt

### Simple Lambda Returns
```python
def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
        bootstrap = returns[:, i]
    return returns
```

### Simple Value Normalization
```python
class Moments(nn.Module):
    def __init__(self, device, decay=0.99, min_=1, percentileLow=0.05, percentileHigh=0.95):
        super().__init__()
        self.register_buffer("low", torch.zeros(()))
        self.register_buffer("high", torch.zeros(()))

    def forward(self, x):
        low = torch.quantile(x.detach(), self._percentileLow)
        high = torch.quantile(x.detach(), self._percentileHigh)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        inverseScale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverseScale.detach()
```

### Simple Reward Predictor
```python
class RewardModel(nn.Module):
    def __init__(self, inputSize, hiddenSize=256, numLayers=2):
        self.network = sequentialModel1D(inputSize, [hiddenSize]*numLayers, 2)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
```

### Simple KL Loss
```python
priorDist = Independent(OneHotCategoricalStraightThrough(logits=priorLogits), 1)
postDist = Independent(OneHotCategoricalStraightThrough(logits=postLogits), 1)
priorDistSG = Independent(OneHotCategoricalStraightThrough(logits=priorLogits.detach()), 1)
postDistSG = Independent(OneHotCategoricalStraightThrough(logits=postLogits.detach()), 1)

klDyn = kl_divergence(postDistSG, priorDist)
klRep = kl_divergence(postDist, priorDistSG)

klLoss = betaPrior * torch.maximum(klDyn, freeNats) + betaPosterior * torch.maximum(klRep, freeNats)
```

---

## Conclusion

**We made DreamerV3 harder than it needs to be.**

NaturalDreamer proves that:
- 700 lines can implement DreamerV3
- PyTorch distributions are sufficient
- Simple Normal distributions work for rewards/values
- Clear loops beat abstracted sequences

The path forward is to embrace simplicity, not fight it.
