# Auxiliary Rewards in Imagination: Implementation Plan

## Executive Summary

**Problem:** The actor receives tiny gradients because imagination produces near-zero rewards, leading to advantages ≈ 0 and no policy learning.

**Solution:** Use the already-trained auxiliary task heads (goalPredictor, puckGoalDistPredictor, shotQualityPredictor) to provide dense reward signals during imagination, giving the actor gradient signal even when the true reward predictor outputs zero.

---

## 1. Current Architecture Analysis

### 1.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WORLD MODEL COMPONENTS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  encoder           : obs (18-dim) → embedding (256-dim)                      │
│  recurrentModel    : (h, z, action) → h' (GRU)                               │
│  priorNet          : h → z (categorical latent, no observation)              │
│  posteriorNet      : (h, embedding) → z (categorical latent, with obs)       │
│  decoder           : fullState → obs reconstruction                          │
│  rewardPredictor   : fullState → reward logits (255 bins, two-hot)           │
│  continuePredictor : fullState → continue probability                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                         AUXILIARY TASK HEADS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  goalPredictor         : fullState → P(goal in next K steps)                 │
│  puckGoalDistPredictor : fullState → distance to opponent goal               │
│  shotQualityPredictor  : fullState → shot quality score [0, 1]               │
├─────────────────────────────────────────────────────────────────────────────┤
│                          BEHAVIOR COMPONENTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  actor  : fullState → action (TanhNormal distribution)                       │
│  critic : fullState → value logits (255 bins, two-hot)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 State Representation

```python
fullState = concat(h, z)  # Shape: (batch, 512)
           ↑    ↑
           │    └── z: stochastic latent (16 × 16 = 256 dims, categorical one-hot)
           └─────── h: deterministic recurrent state (256 dims, GRU hidden)
```

---

## 2. Current Training Flow

### 2.1 World Model Training (Real Data)

```
INPUT: Batch of real sequences from replay buffer
       data.observations: (B, T, 18)
       data.actions:      (B, T, 4)
       data.rewards:      (B, T, 1)
       data.dones:        (B, T, 1)

PROCESS:
┌────────────────────────────────────────────────────────────────────────────┐
│ FOR t = 1 to T-1:                                                          │
│   1. h = recurrentModel(h, z, action[t-1])     ← Uses PREVIOUS action      │
│   2. z_prior, _ = priorNet(h)                  ← Prediction without obs    │
│   3. z_posterior, _ = posteriorNet(h, enc(obs[t]))  ← Inference WITH obs   │
│   4. z = z_posterior                           ← Use posterior for training│
│   5. fullState = concat(h, z)                                              │
│   6. Collect: fullStates[t-1] = fullState                                  │
└────────────────────────────────────────────────────────────────────────────┘

LOSSES (all use fullStates from posterior):
  reconstruction_loss = -log_prob(decoder(fullState), obs)
  reward_loss         = two_hot_loss(rewardPredictor(fullState), reward)
  kl_loss             = KL(posterior || prior) + KL(prior || posterior)
  continue_loss       = -log_prob(continuePredictor(fullState), 1-done)

  # AUXILIARY LOSSES (trained on REAL targets):
  goal_loss           = BCE(goalPredictor(fullState), actual_goal_in_K_steps)
  distance_loss       = MSE(distPredictor(fullState), actual_puck_goal_dist)
  shot_quality_loss   = MSE(qualityPredictor(fullState), actual_shot_quality)

GRADIENT FLOW:
  worldModelLoss.backward() updates:
    ✓ encoder, decoder, recurrentModel, priorNet, posteriorNet
    ✓ rewardPredictor, continuePredictor
    ✓ goalPredictor, puckGoalDistPredictor, shotQualityPredictor  ← AUX HEADS

OUTPUT: fullStates.detach()  ← Detached for behavior training
```

### 2.2 Behavior Training (Imagination) - CURRENT

```
INPUT: fullState from world model training (detached)
       Shape: (B*(T-1), 512) = (~992 starting states)

PROCESS:
┌────────────────────────────────────────────────────────────────────────────┐
│ FOR i = 0 to horizon-1:                                                    │
│   1. action, logprob, entropy = actor(fullState.detach())                  │
│   2. h = recurrentModel(h, z, action)      ← Step dynamics                 │
│   3. z, _ = priorNet(h)                    ← Use PRIOR (no observation!)   │
│   4. fullState = concat(h, z)                                              │
│   5. Collect: fullStates[i], logprobs[i], entropies[i]                     │
└────────────────────────────────────────────────────────────────────────────┘

PREDICTIONS (from imagined fullStates):
  predictedRewards = twoHot.decode(rewardPredictor(fullStates[:, :-1]))
  values           = twoHot.decode(critic(fullStates))
  continues        = continuePredictor(fullStates).mean

LAMBDA RETURNS:
  λ_returns[t] = r[t] + γ * ((1-λ)*V[t+1] + λ*bootstrap)

  WHERE: r[t] = predictedRewards[t]  ← ONLY from rewardPredictor!

  PROBLEM: predictedRewards ≈ 0 because random policy doesn't reach goal states
           → λ_returns ≈ values
           → advantages ≈ 0
           → No actor gradient!

ACTOR LOSS:
  advantages = (λ_returns - values[:-1]) / scale
  actor_loss = -mean(advantages.detach() * logprobs + entropy_scale * entropy)

GRADIENT FLOW (actor_loss.backward()):
  ✓ actor           ← Updated via logprobs gradient
  ✗ critic          ← advantages are detached
  ✗ world model     ← fullState.detach() at actor input
  ✗ aux heads       ← NOT USED AT ALL!
```

---

## 3. Proposed Solution: Auxiliary Rewards in Imagination

### 3.1 Core Idea

Add dense reward signal from auxiliary predictors during imagination:

```python
# Current (sparse, near-zero):
predictedRewards = rewardPredictor(fullStates)  # ≈ 0 for random policy

# Proposed (dense + sparse):
baseRewards = rewardPredictor(fullStates)       # Sparse true rewards
auxRewards  = compute_aux_rewards(fullStates)   # Dense shaping rewards
predictedRewards = baseRewards + auxRewards     # Combined signal
```

### 3.2 Auxiliary Reward Design

```python
def compute_aux_rewards(fullStates, detach_aux_heads=True):
    """
    Compute dense auxiliary rewards from aux task predictions.

    Args:
        fullStates: Imagined latent states (B, H, fullStateSize)
        detach_aux_heads: If True, don't backprop through aux heads

    Returns:
        auxRewards: (B, H-1) dense reward signal
    """
    B, H, _ = fullStates.shape
    flat_states = fullStates[:, :-1].reshape(-1, fullStateSize)  # (B*(H-1), 512)

    # Get aux predictions
    if detach_aux_heads:
        # Use aux heads as fixed feature extractors (no gradient to world model)
        with torch.no_grad():
            goalProb = torch.sigmoid(goalPredictor(flat_states))
            puckDist = puckGoalDistPredictor(flat_states)
            shotQual = torch.sigmoid(shotQualityPredictor(flat_states))
    else:
        # Allow gradient flow (NOT RECOMMENDED - see section 4.2)
        goalProb = torch.sigmoid(goalPredictor(flat_states))
        puckDist = puckGoalDistPredictor(flat_states)
        shotQual = torch.sigmoid(shotQualityPredictor(flat_states))

    # Reshape
    goalProb = goalProb.view(B, H-1)  # (B, H-1)
    puckDist = puckDist.view(B, H-1)  # (B, H-1)
    shotQual = shotQual.view(B, H-1)  # (B, H-1)

    # === Reward Shaping ===
    # Goal probability: Higher probability → positive reward
    goalReward = auxGoalScale * goalProb  # e.g., 0.1 * [0, 1]

    # Distance: Closer to goal → positive reward (IMPROVEMENT reward)
    # Use difference between consecutive steps for potential-based shaping
    # distReward[t] = -scale * (dist[t+1] - dist[t])  # Negative if getting closer
    # But we only have dist at current timestep, so use inverse distance
    distReward = auxDistScale * (1.0 / (puckDist + 0.5) - 0.4)  # Centered around 0

    # Shot quality: Higher quality → positive reward
    shotReward = auxShotScale * shotQual  # e.g., 0.05 * [0, 1]

    auxRewards = goalReward + distReward + shotReward
    return auxRewards
```

### 3.3 Potential-Based Reward Shaping (PBRS) Consideration

**Important:** Simple reward shaping can change the optimal policy. PBRS is the only shaping that preserves optimality:

```
shaped_reward = r(s,a,s') + γ * Φ(s') - Φ(s)
```

For our auxiliary rewards:
- `goalProb` is NOT PBRS (it's a direct bonus)
- `shotQual` is NOT PBRS (it's a direct bonus)
- `distReward` CAN be PBRS if we compute: `γ * dist[t+1] - dist[t]`

**Decision:** Since we're in imagination (not real environment), and the goal is to provide learning signal to escape the "random policy trap", we accept non-PBRS shaping with small coefficients. The true reward (±10 for goals) will dominate once the policy starts scoring.

---

## 4. Gradient Flow Analysis

### 4.1 Proposed Gradient Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMAGINATION FORWARD PASS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Starting fullState (from world model, DETACHED)                             │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐                                                             │
│  │   ACTOR     │ ← Has gradient (we want to update this!)                    │
│  └─────────────┘                                                             │
│         │                                                                    │
│         ▼ action, logprob, entropy                                           │
│         │                                                                    │
│  ┌──────┴──────┐                                                             │
│  │ WORLD MODEL │ (recurrentModel + priorNet)                                 │
│  │  (frozen)   │ ← NO gradient (fullState.detach() at actor input)           │
│  └──────┬──────┘                                                             │
│         │                                                                    │
│         ▼ fullState (imagined)                                               │
│         │                                                                    │
│    ┌────┴────┬────────────────┬────────────────┐                             │
│    │         │                │                │                             │
│    ▼         ▼                ▼                ▼                             │
│ ┌──────┐ ┌──────┐      ┌───────────┐   ┌───────────┐                         │
│ │REWARD│ │CRITIC│      │ AUX HEADS │   │ AUX HEADS │                         │
│ │PRED  │ │      │      │(detached) │   │(gradient) │                         │
│ └──┬───┘ └──┬───┘      └─────┬─────┘   └─────┬─────┘                         │
│    │        │                │               │                               │
│    ▼        ▼                ▼               ▼                               │
│  baseR    values         auxRewards      auxRewards                          │
│    │        │           (no grad)       (WITH grad)                          │
│    │        │                │               │                               │
│    └────┬───┘                │               │                               │
│         │                    │               │                               │
│         ▼                    │               │                               │
│    predictedRewards ◄────────┘               │                               │
│    = baseR + auxR                            │                               │
│         │                                    │                               │
│         ▼                                    │                               │
│    λ_returns = f(rewards, values, γ)         │                               │
│         │                                    │                               │
│         ▼                                    │                               │
│    advantages = (λ_returns - values) / scale │                               │
│         │                                    │                               │
│         ▼                                    │                               │
│    actor_loss = -mean(adv.detach() * logprobs + ent_scale * entropy)         │
│         │                                    │                               │
│         │                                    │                               │
└─────────┼────────────────────────────────────┼───────────────────────────────┘
          │                                    │
          ▼                                    ▼
    GRADIENT TO ACTOR                   GRADIENT TO ACTOR
    (via logprobs)                      AND WORLD MODEL!
                                        (NOT WANTED!)
```

### 4.2 Why Detach Auxiliary Heads?

**If we DON'T detach aux heads:**
```
actor_loss.backward() would update:
  ✓ actor         ← Good, intended
  ✓ aux heads     ← BAD! These should only be trained on real data
  ✓ world model   ← BAD! via aux heads → fullStates → recurrentModel/priorNet
```

**If we DO detach aux heads (RECOMMENDED):**
```
actor_loss.backward() updates:
  ✓ actor         ← Good, intended
  ✗ aux heads     ← Frozen, only trained in worldModelTraining
  ✗ world model   ← Frozen, only trained in worldModelTraining
```

### 4.3 Detailed Gradient Proof

Let's trace gradients mathematically:

```
actor_loss = -mean(advantages.detach() * logprobs + entropy_scale * entropy)

∂(actor_loss)/∂(actor_params) = -mean(adv.detach() * ∂logprobs/∂actor)
                                 - entropy_scale * ∂entropy/∂actor

Since:
  - logprobs = actor.log_prob(action)
  - entropy = actor.entropy()
  - Both depend only on actor parameters

The gradient flows ONLY to actor, NOT through advantages.
```

**Key insight:** `advantages.detach()` prevents gradient flow through:
- λ_returns → predictedRewards → rewardPredictor/auxHeads
- λ_returns → values → critic
- λ_returns → continues → continuePredictor

**Therefore:** Even if auxRewards has gradient, it's blocked by `advantages.detach()`.

### 4.4 Wait - Is Detaching Even Necessary Then?

Let's check more carefully:

```python
# Line 534 in dreamer.py:
actorLoss = -torch.mean(advantages.detach() * logprobs + self.config.entropyScale * entropies)
```

Since `advantages.detach()` is used, gradients don't flow back through advantages regardless of whether aux heads are detached.

**HOWEVER**, the issue is memory and computational efficiency:
- If aux heads aren't detached, PyTorch builds a computation graph through them
- This wastes memory even though gradients are blocked later
- Using `torch.no_grad()` or `.detach()` on aux head outputs is cleaner

**Recommendation:** Use `torch.no_grad()` when computing aux rewards for clarity and efficiency.

---

## 5. Complete Implementation Plan

### 5.1 Changes to `dreamer.py`

#### 5.1.1 Add Configuration Options

```python
# In __init__, after self.useAuxiliaryTasks:
self.useAuxRewardsInImagination = getattr(config, 'useAuxRewardsInImagination', False)
self.auxGoalRewardScale = getattr(config, 'auxGoalRewardScale', 0.1)
self.auxDistRewardScale = getattr(config, 'auxDistRewardScale', 0.05)
self.auxShotRewardScale = getattr(config, 'auxShotRewardScale', 0.05)
```

#### 5.1.2 Add Helper Method

```python
def _compute_aux_rewards_imagination(self, fullStates):
    """
    Compute dense auxiliary rewards from imagined states.

    Args:
        fullStates: (B, H, fullStateSize) imagined latent states

    Returns:
        auxRewards: (B, H-1) auxiliary reward signal
    """
    B, H, _ = fullStates.shape

    # Flatten for batch processing (exclude last state - no reward after terminal)
    flat_states = fullStates[:, :-1].reshape(-1, self.fullStateSize)

    # Get predictions WITHOUT gradient (aux heads are frozen in imagination)
    with torch.no_grad():
        # Goal probability: P(goal in next K steps)
        goalLogits = self.goalPredictor(flat_states)
        goalProb = torch.sigmoid(goalLogits)  # (B*(H-1),)

        # Puck-goal distance (predicted by aux head)
        puckDist = self.puckGoalDistPredictor(flat_states)  # (B*(H-1),)

        # Shot quality score
        shotLogits = self.shotQualityPredictor(flat_states)
        shotQual = torch.sigmoid(shotLogits)  # (B*(H-1),)

    # Reshape to (B, H-1)
    goalProb = goalProb.view(B, H-1)
    puckDist = puckDist.view(B, H-1)
    shotQual = shotQual.view(B, H-1)

    # === Compute Reward Components ===

    # 1. Goal probability reward: encourage states where goals are likely
    #    Range: [0, auxGoalRewardScale]
    goalReward = self.auxGoalRewardScale * goalProb

    # 2. Distance reward: encourage getting puck closer to goal
    #    Convert distance to reward: closer = higher reward
    #    puckDist typically in range [0, 5], so 1/(dist+0.5) in range [0.18, 2.0]
    #    Center around 0 by subtracting baseline
    distReward = self.auxDistRewardScale * (1.0 / (puckDist.clamp(min=0.1) + 0.5) - 0.5)

    # 3. Shot quality reward: encourage good scoring positions
    #    Range: [0, auxShotRewardScale]
    shotReward = self.auxShotRewardScale * shotQual

    # Combine
    auxRewards = goalReward + distReward + shotReward

    return auxRewards
```

#### 5.1.3 Modify behaviorTraining

```python
def behaviorTraining(self, fullState):
    # ... existing imagination loop (lines 483-504) ...

    # Stack imagined trajectories
    fullStates = torch.stack(fullStates, dim=1)    # (B, horizon, fullStateSize)
    logprobs = torch.stack(logprobs[1:], dim=1)    # (B, horizon-1)
    entropies = torch.stack(entropies[1:], dim=1)  # (B, horizon-1)

    # Get base reward predictions
    rewardLogits = self.rewardPredictor(fullStates[:, :-1].reshape(-1, self.fullStateSize))
    rewardLogits = rewardLogits.view(fullStates.shape[0], -1, self.twoHotBins)
    baseRewards = self.twoHot.decode(rewardLogits)  # (B, H-1)

    # === NEW: Add auxiliary rewards if enabled ===
    if self.useAuxiliaryTasks and self.useAuxRewardsInImagination:
        auxRewards = self._compute_aux_rewards_imagination(fullStates)
        predictedRewards = baseRewards + auxRewards
    else:
        predictedRewards = baseRewards
        auxRewards = None

    # ... rest of behavior training unchanged ...

    # === NEW: Add aux reward metrics ===
    if auxRewards is not None:
        metrics["imagination/aux_reward_mean"] = auxRewards.mean().item()
        metrics["imagination/aux_reward_std"] = auxRewards.std().item()
        metrics["imagination/aux_reward_min"] = auxRewards.min().item()
        metrics["imagination/aux_reward_max"] = auxRewards.max().item()
        metrics["imagination/base_reward_mean"] = baseRewards.mean().item()
        metrics["imagination/total_reward_mean"] = predictedRewards.mean().item()
```

### 5.2 Changes to `train_hockey.py`

```python
# Add CLI arguments
parser.add_argument("--use_aux_rewards_imagination", action="store_true",
                    help="Use auxiliary task predictions as dense rewards in imagination")
parser.add_argument("--aux_goal_reward_scale", type=float, default=0.1,
                    help="Scale for goal probability reward in imagination")
parser.add_argument("--aux_dist_reward_scale", type=float, default=0.05,
                    help="Scale for distance reward in imagination")
parser.add_argument("--aux_shot_reward_scale", type=float, default=0.05,
                    help="Scale for shot quality reward in imagination")

# In config override section
if args.use_aux_rewards_imagination:
    config.dreamer.useAuxRewardsInImagination = True
if args.aux_goal_reward_scale is not None:
    config.dreamer.auxGoalRewardScale = args.aux_goal_reward_scale
# ... etc for other scales
```

---

## 6. Risk Analysis and Mitigations

### 6.1 Risk: Auxiliary Rewards Dominate True Rewards

**Problem:** If aux rewards are too large, agent optimizes for aux tasks instead of scoring goals.

**Mitigation:**
- Use small scales (0.05-0.1) so aux rewards are ~0.1-0.2 vs true rewards ±10
- True rewards are 50-100x larger, so they dominate when goals are scored
- Monitor `imagination/aux_reward_mean` vs `imagination/base_reward_mean`

### 6.2 Risk: Aux Heads Produce Bad Predictions in Imagination

**Problem:** Aux heads trained on real data may extrapolate poorly to imagined states.

**Mitigation:**
- Aux heads share latent representation with world model, so should generalize
- Use `torch.no_grad()` so bad predictions don't corrupt world model
- Monitor `aux/goal_accuracy` and `aux/puck_goal_dist_error` during training

### 6.3 Risk: Changed Optimal Policy

**Problem:** Non-PBRS reward shaping can change optimal policy.

**Mitigation:**
- Accept this for early training to escape "random policy trap"
- Use small scales so effect diminishes as true rewards become frequent
- Can decay aux reward scales over training if needed

### 6.4 Risk: Gradient Leakage

**Problem:** Accidentally updating world model through aux heads.

**Mitigation:**
- Use `torch.no_grad()` context (belt)
- `advantages.detach()` already blocks gradients (suspenders)
- Verify with `torch.autograd.grad()` check in tests

---

## 7. Verification Plan

### 7.1 Gradient Flow Verification

```python
def test_no_gradient_leakage():
    """Verify aux rewards don't leak gradients to world model."""
    # Setup
    agent = Dreamer(...)
    agent.useAuxRewardsInImagination = True

    # Get some states
    data = agent.buffer.sample(32, 32)
    fullStates, _ = agent.worldModelTraining(data)

    # Store world model params before
    wm_params_before = [p.clone() for p in agent.worldModelParameters]

    # Run behavior training
    agent.behaviorTraining(fullStates)

    # Check world model params unchanged
    for p_before, p_after in zip(wm_params_before, agent.worldModelParameters):
        assert torch.allclose(p_before, p_after), "World model was updated!"

    print("✓ No gradient leakage to world model")
```

### 7.2 Expected Metric Changes

| Metric | Without Aux Rewards | With Aux Rewards | Reason |
|--------|---------------------|------------------|--------|
| `imagination/reward_mean` | ≈ 0 | 0.05 - 0.2 | Aux rewards add signal |
| `advantages_abs_mean` | 0.08 - 0.69 | 0.5 - 2.0 | More varied returns |
| `behavior/entropy_mean` | 13.67 (max) | < 12, decreasing | Policy learns to exploit |
| `stats/win_rate` | 0.3 - 0.5, flat | Increasing | Actual improvement |

---

## 8. Implementation Order

1. **Add `_compute_aux_rewards_imagination()` method** to `Dreamer` class
2. **Add config attributes** in `__init__`
3. **Modify `behaviorTraining()`** to use aux rewards
4. **Add CLI arguments** to `train_hockey.py`
5. **Add metrics logging** for aux rewards
6. **Test gradient flow** with verification script
7. **Run training** with `--use_aux_rewards_imagination`

---

## 9. Recommended Hyperparameters

For initial testing:

```bash
python train_hockey.py \
    --opponent weak \
    --use_aux_rewards_imagination \
    --aux_goal_reward_scale 0.1 \
    --aux_dist_reward_scale 0.05 \
    --aux_shot_reward_scale 0.05 \
    --no_advantage_normalization \
    --entropy_scale 0.003
```

The scales are chosen so:
- Max aux reward per step: ~0.1 + 0.1 + 0.05 = 0.25
- Over 15-step horizon: ~3.75 cumulative
- Still much less than true goal reward (±10)

---

## 10. Summary

**What we're adding:**
- Dense reward signal from already-trained auxiliary heads
- Provides gradient to actor even when true rewards are zero
- Breaks the "random policy → no rewards → no learning" cycle

**What we're NOT changing:**
- World model training (aux heads still trained on real data)
- True reward prediction (still used, aux is additive)
- Critic targets (λ-returns now include aux rewards)

**Safety guarantees:**
- `torch.no_grad()` prevents aux head updates during imagination
- `advantages.detach()` prevents gradient flow through returns
- Small scales ensure true rewards dominate when goals are scored
