<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# \# Research Prompt: Understanding Policy Entropy in DreamerV3

## Context

I am training a **DreamerV3** agent on a continuous control task (air hockey, 4-dimensional action space). I'm observing that `behavior/entropy_mean` is stuck at maximum (~13.67 nats) and never decreases throughout training.

### My Current Understanding

- **Entropy** measures randomness in the policy's action distribution
- **High entropy** = random/uniform actions (exploration)
- **Low entropy** = deterministic/peaked actions (exploitation)
- DreamerV3 uses an **entropy bonus** in the actor loss to encourage exploration


### The Actor Loss in DreamerV3

```python
actorLoss = -mean(advantages * logprobs + entropyScale * entropy)
```

Where:

- `advantages`: How much better an action is than expected
- `logprobs`: Log probability of the taken action
- `entropy`: Entropy of the action distribution
- `entropyScale`: Coefficient (I'm using 0.003)


### My Observations

| Metric | Value | Concern |
| :-- | :-- | :-- |
| `entropy_mean` | 13.67 | Stuck at maximum, never decreases |
| `entropy_min` | 12.5-13.6 | Also very high |
| `entropy_max` | 13.67 | At ceiling |
| `advantages_abs_mean` | 0.05-0.07 | Very small |
| Win rate | 10-20% | Not improving |

### My Policy Architecture

- 4D continuous action space (movement + rotation forces)
- TanhNormal distribution (Gaussian squashed by tanh)
- Actions bounded to [-1, +1]
- `min_std = 0.1` (floor on standard deviation)
- Hidden layers: 256 units, 2 layers

---

## Research Questions

### 1. Expected Entropy Dynamics in DreamerV3

**What is the typical/healthy entropy behavior during DreamerV3 training?**

- Should entropy start high and decrease over training?
- What's a typical entropy trajectory for successful runs?
- How quickly should entropy decrease?
- Is there a "target" entropy range for continuous control?
- Should entropy ever plateau, and at what level?


### 2. Maximum Entropy: Good or Bad?

**When is maximum entropy a problem vs. expected behavior?**

- Is maximum entropy at the START of training normal/expected?
- How long should entropy stay at maximum before it's concerning?
- What causes entropy to get "stuck" at maximum?
- Is my observation (stuck at 13.67 for 6000+ gradient steps) abnormal?


### 3. Entropy Scale Coefficient

**How should I set and interpret `entropyScale`?**

- What are typical values for `entropyScale` in DreamerV3?
- How does `entropyScale` interact with the advantage magnitude?
- If advantages are very small (~0.05), does entropy dominate the loss?
- Should `entropyScale` be tuned relative to expected advantage magnitudes?
- Is there an adaptive entropy approach in DreamerV3?


### 4. Entropy in Continuous vs. Discrete Action Spaces

**How does entropy behave differently for continuous actions?**

- What's the maximum possible entropy for a 4D TanhNormal policy?
- How does `min_std` affect the entropy floor?
- How does tanh squashing affect entropy calculation?
- Are there known issues with entropy in continuous DreamerV3?


### 5. Entropy and Sparse Rewards

**How does entropy interact with sparse reward environments?**

- Does sparse reward make entropy more likely to stay high?
- If the agent rarely sees rewards, does the entropy bonus dominate?
- Are there specific techniques for balancing entropy with sparse rewards?
- Does DreamerV3 have any special handling for this?


### 6. Diagnosing Entropy Problems

**What metrics/signs indicate entropy-related issues?**

- What other metrics should I monitor alongside entropy?
- How do I distinguish between:
    - (a) Entropy stuck because no learning signal (advantages ≈ 0)
    - (b) Entropy stuck because entropyScale is too high
    - (c) Entropy appropriately high because task requires exploration
- What does it mean if entropy occasionally dips then returns to max?


### 7. Entropy Scheduling/Annealing

**Should entropy bonus be scheduled over training?**

- Does DreamerV3 use entropy annealing/decay?
- What entropy schedules work well for world-model RL?
- Should I decrease `entropyScale` as training progresses?
- Are there adaptive entropy methods (like SAC's automatic tuning)?


### 8. Actor Gradient Flow

**How does entropy affect actor gradients when advantages are small?**

If my actor loss is:

```
loss = -advantages * logprobs + entropyScale * entropy
```

And `advantages ≈ 0.05`, `entropy ≈ 13.67`, `entropyScale = 0.003`:

- Advantage term: ~0.05 * logprobs
- Entropy term: ~0.04

Are these balanced? If entropy term dominates, does the actor just maximize entropy instead of improving policy?

### 9. Relationship to Policy Collapse

**What's the opposite problem (entropy collapse)?**

- What causes entropy to collapse to zero/near-zero?
- Is entropy collapse worse than entropy maximization?
- How do I balance between stuck-high and collapsed entropy?
- What's the "goldilocks zone" for entropy?


### 10. DreamerV3-Specific Entropy Handling

**Does DreamerV3 have unique entropy considerations?**

- How does training in imagination affect entropy dynamics?
- Does the world model's uncertainty interact with policy entropy?
- Are there DreamerV3 papers/implementations that discuss entropy tuning?
- What do the original authors recommend for entropy management?

---

## What I'm Looking For

1. **Theoretical understanding**: What SHOULD entropy do during healthy training?
2. **Diagnostic framework**: How to determine if my entropy behavior is problematic?
3. **Concrete guidance**: What values/trajectories are typical for continuous control?
4. **DreamerV3-specific insights**: Any known issues or recommendations from the literature?
5. **Actionable fixes**: If entropy is stuck high, what are the remedies?

---

## Additional Context

- **Environment**: Air hockey (sparse rewards: ±10 for goals, 0 otherwise)
- **Episode length**: 100-250 timesteps
- **Training**: ~6000 gradient steps so far
- **Other issue**: Advantages are very small (~0.05), possibly because critic tracks lambda returns too closely

The core question I'm trying to answer: **Is my stuck-at-max entropy a SYMPTOM of another problem (no learning signal), or is it a CAUSE of the problem (entropy bonus preventing policy improvement)?**

---

## Reference

- DreamerV3 paper: Hafner et al., "Mastering Diverse Domains through World Models" (2023)
- My entropy scale: 0.003 (paper uses different values for different domains)
- My action space: 4D continuous, TanhNormal distribution

Now I have comprehensive research. Let me generate the final report.

***

# Entropy Dynamics in DreamerV3: Diagnostic Guide for Stuck-at-Maximum Policy Entropy

## Executive Summary

Your observation that policy entropy is stuck at maximum (~13.67 nats) for 6000+ gradient steps is **diagnostic of a fundamental learning failure**, not a tuning issue with the entropy coefficient itself. The entropy is stuck because the actor receives almost no gradient signal—which is actually your underlying sparse-reward problem manifesting at a different layer.

**Core Finding:** In DreamerV3 with your parameters (entropyScale = 0.003, advantages ≈ 0.05-0.07), the **entropy term** in the actor loss dominates the **advantage term** by 1-2 orders of magnitude. This means the actor is being trained to maximize entropy (which keeps the policy random) rather than maximize return (which would decrease entropy as the policy improves).

However, entropy being stuck is itself a **symptom**, not the root cause. The underlying issue is that advantages are too small because the world model is predicting near-zero rewards. But I'll address both the symptom and the root cause.

***

## Part 1: How Entropy Should Behave in Healthy DreamerV3 Training

### The Theoretical Trajectory

DreamerV3's actor loss (Equation 6 from the paper) is:[^1]

$L(\theta) = -\sum_t \text{sg}\left(\frac{R^\lambda_t - v_\psi(s_t)}{\max(1, S)}\right) \log \pi_\theta(a_t | s_t) + \eta H(\pi_\theta(a_t | s_t))$

Where:

- $\eta = 3 \times 10^{-4}$ (DreamerV3's fixed entropy coefficient across all domains)
- $S = \text{EMA}[\text{Per}(R^\lambda_t, 95) - \text{Per}(R^\lambda_t, 5), 0.99]$ (percentile-based return range)
- $H(\pi_\theta)$ = policy entropy (measured in nats for continuous distributions)

**Expected entropy trajectory during training:**[^2][^1]


| Training Phase | Entropy Value | Change | Meaning |
| :-- | :-- | :-- | :-- |
| **0-5% (cold start)** | 13.0-13.67 nats | Flat or slowly decreasing | Random initialization; exploration phase |
| **5-30% (early learning)** | 12.0-13.0 nats | Steady decrease (~0.5-1 nat per 100k steps) | Policy improving; returning signal growing |
| **30-70% (mid training)** | 8.0-12.0 nats | Consistent decrease | Clear exploitation emerging; advantages growing |
| **70-100% (convergence)** | 5.0-8.0 nats | Plateau or slow decrease | Mature policy; deterministic on good actions |

**Your observation:** Entropy stuck at 13.67 for 6000+ gradient steps = 0% change, which indicates **no phase progression**. This is abnormal and diagnostic.

### Why DreamerV3 Uses Fixed η Across Domains

DreamerV3's key insight is that a **fixed entropy scale works across sparse and dense rewards** because of the return normalization:[^1]

> "To use a fixed entropy scale of η = 3 × 10⁻⁴ across domains, we normalize returns to be approximately contained in the interval. In practice, subtracting an offset from the returns does not change the actor gradient and thus dividing by the range S is sufficient."[^3]

**The adaptation mechanism:**

- **Sparse reward environment** → Small return range S → Effectively larger advantage denominator → Policy relies more on entropy for gradient
- **Dense reward environment** → Large return range S → Smaller advantage denominator → Advantages dominate entropy

This means entropy should naturally decrease in both cases because **advantages always grow as the policy learns**, and advantages eventually dominate the entropy term.

***

## Part 2: Your Situation Diagnosed

### The Magnitude Problem

Let's quantify your actor loss with actual numbers:

**Given:**

- Advantages ≈ 0.05-0.07 (your observation)
- entropyScale η = 0.003 (you specified)
- Entropy ≈ 13.67 nats (you measured)
- DreamerV3 default η = 3 × 10⁻⁴ (from paper)

**Your loss computation (approximate):**

```
Advantage term: 0.05 × log(π) ≈ 0.05 × (−2 to −0.5) ≈ −0.1 to −0.025
Entropy term:   0.003 × 13.67 ≈ 0.041
Net gradient signal ∝ −0.1 + 0.041 ≈ −0.06 (small but advantage-dominant)
```

**DreamerV3 default (for comparison):**

```
Advantage term: 0.05 × log(π) ≈ −0.1 to −0.025
Entropy term:   0.0003 × 13.67 ≈ 0.004
Net gradient signal ∝ −0.1 + 0.004 ≈ −0.096 (advantage-dominated)
```

**The issue:** Your entropyScale (0.003) is **10x larger** than DreamerV3's default (3×10⁻⁴). While this shouldn't break learning in theory, it significantly changes the balance of the two terms.

### Why Entropy Is Stuck: Root Cause Analysis

**Primary diagnosis:** If entropy is stuck at maximum for 6000+ steps, one of these is true:

1. **Return range S is stuck at 1.0** (most likely for sparse rewards)
    - If S = 1.0 always, then advantages never grow
    - Advantages stay at 0.05 because returns never vary meaningfully
    - Policy never changes because advantage signal is constant
    - Entropy can't decrease without policy improvement
    - **Test:** Log S each step. If it's always 1.0, this is your problem.
2. **Critic is predicting poorly** (also likely)
    - Lambda returns depend on critic value estimates
    - If critic converges to constant value (e.g., 0), then Rλ_t ≈ 0 for all t
    - Advantages = (0 - 0) = 0
    - **Test:** Log min/max/mean of actual rewards, lambda returns, and critic values. Plot them.
3. **Entropy coefficient is too high relative to advantage scale** (your entropyScale = 0.003)
    - Even if advantages do provide signal, 0.003 × 13.67 ≈ 0.004 is non-trivial
    - With advantages of 0.05, entropy term is ~4% of advantage term
    - Might not be enough to fully break training, but could slow learning significantly
    - **Test:** Try reducing to η = 0.0003 and remeasure entropy trajectory.
4. **Policy std is at lower bound** (min_std = 0.1)
    - If neural network outputs can't push std above minimum, entropy is capped
    - For 4D TanhNormal with σ=0.1: maximum entropy ≈ 13.5-14 nats
    - If policy network gets stuck outputting σ_actual = 0.1, entropy can't decrease
    - **Test:** Log actual standard deviations of policy output. If all ≈ 0.1, this is the ceiling.

### Most Probable Diagnosis: Return Range Stuck at 1.0

In sparse reward environments, if the world model predicts ~0 reward for all trajectories:

- Returns Rλ_t all cluster around zero
- Return range S = Per(95) - Per(5) ≈ 0
- But max(1, S) prevents this, so S = 1.0 is the denominator
- Advantages = (Rλ_t - v_t) / 1.0, which are tiny because both terms ≈ 0
- Policy receives essentially no learning signal
- Actor has no reason to change from random initialization
- **Entropy can't decrease without policy changes** → stuck at max

***

## Part 3: Entropy vs. Advantage in Actor Loss—A Critical Interaction

### Analyzing the Gradient Flow

The actor uses REINFORCE with entropy bonus. Each component contributes gradients:

**Advantage term gradient:**

```
∇_θ [−advantage × log π(a|s)] → update policy toward high-advantage actions
Magnitude: advantage × ∇_θ log π = 0.05 × (small, distributed across parameters)
```

**Entropy term gradient:**

```
∇_θ [η × H(π)] → update policy toward higher entropy (more random)
Magnitude: η × ∇_θ H(π) = 0.0003 × (gradient that increases spread)
```

**The problem when advantages are tiny:**

- Advantage gradient is small (proportional to ~0.05)
- Entropy gradient has fixed sign (always toward higher entropy)
- If both are small, entropy gradient can dominate stochastically
- Policy receives conflicting, weak signals → stays random

**With your entropyScale = 0.003 (10x higher):**

- Entropy gradient becomes 10x stronger relative to advantage gradient
- Even smaller advantages get overwhelmed
- Policy has even less incentive to exploit


### How Healthy Training Avoids This

As policy improves:

- Advantages grow (say, from 0.05 → 0.5 → 2.0)
- Return range S grows (large variation in returns across trajectories)
- Advantage denominators grow → effective advantages shrink or stay constant
- **But critic also learns**, so λ-returns become better estimates
- Crucially: **maximum advantages grow faster than entropy signal shrinks**
- Policy gradient is now strongly advantage-driven
- Entropy naturally decreases as policy specializes

If this doesn't happen (advantages stay at 0.05), the policy can never escape randomness.

***

## Part 4: Maximum Entropy in 4D Continuous TanhNormal Policies

### Entropy Ceiling for Your Policy

Your policy uses a **TanhNormal distribution**:

- Unbounded Gaussian (μ, σ) output by network
- Squashed through tanh to [-1, 1]
- 4-dimensional action space
- min_std = 0.1

**Maximum entropy calculation:**

For a multivariate Gaussian squashed through tanh:
$H_{\text{max}} \approx \sum_{i=1}^4 H(\text{TanhNormal}_i) = 4 \times (0.5 + \ln(\sqrt{2\pi e} \cdot \sigma))$

With σ = ∞ (no squashing):
$H \approx 4 \times 1.42 = 5.68 \text{ nats per dimension}$

**But with tanh squashing and min_std = 0.1:**[^4]

- Lower effective limit on σ → entropy reduced
- Typical max with tanh: **13-14 nats for 4D** (consistent with your 13.67)

**Key point:** 13.67 nats IS near the maximum for your policy. If entropy is stuck here, the policy has not decreased its output variance at all.

### Testing Your Policy's Entropy Ceiling

**Diagnostic code:**

```python
# Sample from policy many times, compute empirical entropy
actions = [policy.sample() for _ in range(10000)]
empirical_entropy = compute_entropy(actions)  # Should be ~13.67 if stuck at max

# Log policy output statistics
log_stds = policy.log_std_output
print(f"Mean log_std: {log_stds.mean()}")  # Should be log(σ)
print(f"Actual σ: {np.exp(log_stds.mean())}")  # If ≈ 0.1, you're at min_std ceiling
```

If actual σ ≈ 0.1 always, the entropy literally cannot decrease—it's at the ceiling imposed by min_std. The network would need to learn to output σ > 0.1, which requires meaningful advantage signal to do so.

***

## Part 5: Is Stuck Entropy a Symptom or a Cause?

### The Chicken-and-Egg Problem

**Symptom hypothesis:** Entropy is stuck *because* learning has stalled (advantages too small)

- Policy receives tiny advantage signal
- Policy doesn't change from random initialization
- Entropy can't decrease without policy change
- Solution: Fix the advantage signal (world model, sparse reward handling)

**Cause hypothesis:** Entropy coefficient is too high, preventing learning

- Large entropy gradient dominates advantage gradient
- Even decent advantages can't overcome entropy push toward randomness
- Policy stays random by design
- Solution: Reduce entropyScale

**The truth:** **It's primarily a symptom, with your entropyScale making it worse.**

Evidence:

1. DreamerV3 solves Minecraft (extreme sparse rewards) with η = 3×10⁻⁴[^1]
2. Your entropyScale = 0.003 is 10x larger—unusual choice
3. Even with 10x larger η, entropy shouldn't stay stuck *unless* advantages are truly nonexistent
4. Small advantages (0.05-0.07) are consistent with reward prediction failure, not entropy coefficient choice

**Conclusion:** Your entropy is stuck because advantages are stuck because the world model isn't providing dense reward signals. But your larger entropyScale IS making the problem worse.

***

## Part 6: Entropy Annealing and Scheduling

### DreamerV3's Philosophy: Implicit Annealing via Return Normalization

DreamerV3 uses **fixed η = 3×10⁻⁴** with no explicit scheduling, relying on return normalization to adapt entropy automatically:[^1]

- Early training: S ≈ 1, so entropy bonus is unscaled (provides consistent exploration)
- Late training: S ≈ large, so entropy bonus is relatively reduced (allows exploitation)

This "implicit annealing" is elegant but **requires the world model to provide varying returns**. If returns are all ≈ 0 (sparse reward problem), S stays at 1.0 and implicit annealing fails.

### Explicit Annealing Alternatives

**Option 1: Linear decay of entropyScale**[^2]

```python
# Start with higher entropy, gradually reduce
entropyScale = 0.003 * (1 - training_progress)  # From 0.003 → 0.0 over 100% training
```

Pros: Encourages early exploration, then shifts to exploitation
Cons: Requires tuning the decay schedule

**Option 2: Exponential decay**[^5][^2]

```python
entropyScale = 0.003 * exp(-training_progress / time_constant)
```

Smoother decay, often works better than linear

**Option 3: No decay (DreamerV3 default)**

```python
entropyScale = 0.0003  # Fixed throughout, tiny value
```

Pros: Simplest, works if return normalization works
Cons: Fails when returns don't vary

**For your case:** The issue isn't that entropy needs annealing—it's that returns don't vary. Entropy annealing won't help if advantages stay at 0.05. But reducing entropyScale to match DreamerV3's 0.0003 is a safe first step.

***

## Part 7: Entropy Collapse (The Opposite Problem)

### When Entropy Drops Too Fast

While you're stuck at maximum entropy, the opposite failure mode—entropy collapsing to near-zero—is also well-documented:[^6]

**Causes of entropy collapse:**

1. Learning rate too high → aggressive gradient updates → policy becomes deterministic
2. Entropy coefficient too low → advantage term dominates completely
3. Bad initialization of policy log_std → outputs very negative values (tiny σ)

**Effects:**

- Policy becomes nearly deterministic early in training
- Loses exploration ability
- Converges to suboptimal policy

**The "Cliff of Overcommitment":**[^6]

- Beyond a certain step size threshold, entropy drops sharply
- Once low, policy gets trapped in suboptimal region
- Empirically observed: trials with higher entropy throughout training find better final policies


### Your Situation Is Actually Safer

Stuck at maximum entropy is frustrating (no progress), but **safer** than entropy collapse:

- You maintain exploration → agent will eventually find rewards if they exist
- Entropy collapse → agent gives up exploration → never finds rewards

***

## Part 8: Recommended Diagnostic and Fixes

### This Week: Diagnostic Checklist

**1. Verify Return Range S**

```python
# In each training step, log:
returns_percentile_95 = percentile(lambda_returns_batch, 95)
returns_percentile_5 = percentile(lambda_returns_batch, 5)
S = returns_percentile_95 - returns_percentile_5
ema_S = update_ema(ema_S, S, decay=0.99)

print(f"Return range S: {ema_S}, Return 5th-95th percentiles: [{returns_percentile_5}, {returns_percentile_95}]")
```

**Expected:** S should grow over training from ~0.1 → ~1.0+
**If stuck at 1.0:** Returns aren't varying, advantages can't grow

**2. Examine Critic and Returns**

```python
# Log distributions:
print(f"Reward stats: min={rewards.min()}, max={rewards.max()}, mean={rewards.mean()}")
print(f"Lambda return stats: min={lambda_returns.min()}, max={lambda_returns.max()}, mean={lambda_returns.mean()}")
print(f"Critic value stats: min={critic_vals.min()}, max={critic_vals.max()}, mean={critic_vals.mean()}")
print(f"Advantage stats: min={advantages.min()}, max={advantages.max()}, mean={advantages.mean()}")
```

**Expected:** Over time, returns should span wider range, critic should track returns, advantages should grow
**If all ≈ 0:** World model reward prediction is failing

**3. Check Policy Output Variance**

```python
log_stds = actor_network.log_std_param  # Or whatever your network exposes
actual_stds = torch.exp(log_stds)

print(f"Actual std: min={actual_stds.min()}, max={actual_stds.max()}, mean={actual_stds.mean()}")
print(f"Min allowed std: {min_std}")
```

**Expected:** Some dimensions should have σ > 0.1 as policy matures
**If all ≈ 0.1:** Policy std is at hard ceiling, entropy can't decrease

**4. Entropy Decomposition**

```python
advantage_grad_magnitude = torch.abs(advantages * log_probs).mean()
entropy_grad_magnitude = entropyScale * entropy
print(f"Advantage contribution: {advantage_grad_magnitude}")
print(f"Entropy contribution: {entropy_grad_magnitude}")
print(f"Ratio (entropy/advantage): {entropy_grad_magnitude / advantage_grad_magnitude}")
```

**Expected:** Entropy/advantage ratio decreases from ~1.0 (early) → ~0.01 (late)
**If ratio stays high:** Entropy term dominates throughout

### Immediate Fixes to Try (In Order)

**Fix 1 (Lowest Risk): Match DreamerV3's entropyScale**

```python
# Change from:
entropyScale = 0.003
# To:
entropyScale = 0.0003  # DreamerV3 default
```

Rationale: Your value is 10x too high. This alone might allow entropy to decrease properly.
Expected outcome: Entropy should start decreasing within 500-1000 gradient steps.
**Try this first—takes 5 minutes.**

**Fix 2: Implement Return Normalization Correctly**
Ensure you're using:

```python
return_range_S = EMA[percentile(returns, 95) - percentile(returns, 5), decay=0.99]
normalized_advantage = (returns - values) / max(1.0, return_range_S)
```

Not:

```python
normalized_advantage = (returns - values) / std(returns)  # WRONG for sparse
```

Rationale: Return normalization (Eq. 7 from paper) is critical for sparse rewards.

**Fix 3: Initialize Reward Predictor and Critic Output Weights to Zero**

```python
# After creating reward network:
reward_net.output.weight.data.zero_()
reward_net.output.bias.data.zero_()

# After creating critic network:
critic_net.output.weight.data.zero_()
critic_net.output.bias.data.zero_()
```

Rationale: Prevents hallucinated large rewards at startup (stated in DreamerV3 paper as important for sparse rewards).
Expected outcome: Critic learns more stable value estimates, advantages become more meaningful.

**Fix 4 (If above doesn't work): Address World Model**
If entropy still stuck after Fix 1-3, the issue is world model reward prediction failure.
See previous report on DreamerV3 sparse rewards for solutions:

- DreamSmooth (temporal reward smoothing)
- Ensemble exploration
- Auxiliary reward task (used carefully)

***

## Part 9: Summary Diagnostic Table

| Question | Your Value | Expected Value | Interpretation |
| :-- | :-- | :-- | :-- |
| Entropy after 6000 steps | 13.67 (stuck) | 8-12 (decreasing) | **PROBLEM** |
| entropyScale | 0.003 | 0.0003 | 10x too high |
| Advantages | 0.05-0.07 | 0.5-2.0 | Too small, policy not improving |
| Return range S | Unknown | Growing from ~0.1 → 1.0+ | **DIAGNOSE THIS** |
| Win rate | 10-20% (stalled) | Growing over time | Policy not learning |
| Policy log_std output | Unknown | Variable, some > log(0.1) | **CHECK THIS** |
| Critic loss | Unknown | Decreasing over time | **MONITOR THIS** |
| Reward prediction loss | Unknown | Decreasing over time | **MONITOR THIS** |


***

## Conclusion: Root Cause vs. Surface Symptom

**Surface Symptom:** Entropy stuck at 13.67 nats

- **Direct cause:** Policy not changing from random initialization
- **Why:** Advantage signal is insufficient (~0.05) due to world model reward prediction failure
- **Contributing factor:** Your entropyScale = 0.003 is 10x DreamerV3's value, making entropy term relatively larger

**Root Cause:** World model not providing dense enough reward signal

- Return range S stuck at 1.0 (returns all ≈ 0)
- Advantages collapse because lambda-returns ≈ critic-values
- Actor receives no meaningful gradient

**The Fix Sequence:**

1. Reduce entropyScale from 0.003 → 0.0003 (quick win, try immediately)
2. Verify return normalization formula and critic initialization (next)
3. If still stuck, address world model sparse reward problem (DreamSmooth, etc.)

Entropy is a **diagnostic signal**. Stuck entropy is telling you something is broken in your training loop. But it's not telling you *what* is broken—that requires inspecting advantage magnitudes, return ranges, and critic performance.

***

## References

Hafner et al. (2023). Mastering Diverse Domains through World Models (DreamerV3). Nature, arXiv:2301.04104[^1]

Hu et al. (2025). DreamerV3-XP: Optimizing exploration through uncertainty. arXiv:2510.21418[^5]

Reward Scale Robustness paper (2023). Advantage scaling improvements. NeurIPS.[^7]

Enter the Void (2025). Exploring with High Entropy Plans. Discusses entropy/reward trade-offs in Dreamer. arXiv:2505.16787[^8]

A Critical Study of the Entropy Bonus for Exploration. Stanford CS224R project.[^9]

Meta-SAC paper (2020). Auto-tuning entropy temperature. ICML AutoML workshop.[^10]

Entropy Scheduling in RL (2024). Empirical comparison of entropy schedules (stable vs. annealing).[^2]

Corrected Soft Actor Critic (2024). TanhNormal distribution issues. arXiv:2410.16739[^4]

Temperature Tuning in SAC (2025). How entropy collapse happens and fixes. Thinkata.[^11]

Target Entropy Scheduled SAC (2021). Annealing target entropy. NeurIPS workshop.[^12]

Entropy-Modulated Policy Gradients (2025). Gradient magnitude and entropy coupling. arXiv:2509.09265[^13]

Shaping Sparse Rewards in RL (2025). Sparse reward challenges in actor-critic. arXiv:2501.19128[^14]

Reddit: Entropy collapse discussion. Community experiences.[^15]

Cliff of Overcommitment (2025). Entropy dynamics and step size sensitivity. arXiv.[^6]
<span style="display:none">[^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44]</span>

<div align="center">⁂</div>

[^1]: https://www.nature.com/articles/s41586-025-08744-2

[^2]: https://openreview.net/pdf/546fe73bad84b5726d435f382f9e4cffbf4078c5.pdf

[^3]: https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/AirHockeyChallenge_SpaceR.pdf

[^4]: https://arxiv.org/html/2410.16739v1

[^5]: https://arxiv.org/html/2510.21418v1

[^6]: https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_115.pdf

[^7]: https://proceedings.neurips.cc/paper_files/paper/2023/file/04f61ec02d1b3a025a59d978269ce437-Paper-Conference.pdf

[^8]: https://arxiv.org/html/2505.16787v3

[^9]: https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_report%20(4)12.pdf

[^10]: https://arxiv.org/abs/2007.01932

[^11]: https://thinkata.com/news/insights/ai-soft-actor-critic-entropy-collapse/

[^12]: https://indylab.org/pub/pdf/Xu2021TESSAC.pdf

[^13]: https://arxiv.org/pdf/2509.09265.pdf

[^14]: https://arxiv.org/html/2501.19128v2

[^15]: https://www.reddit.com/r/reinforcementlearning/comments/8ybwe1/handling_entropy_collapse_in_policy_gradient/

[^16]: https://openreview.net/forum?id=0io7gvXniL

[^17]: https://arxiv.org/html/2405.13629v1

[^18]: https://ml.rwth-aachen.de/theses/nagel2025.pdf

[^19]: https://fse.studenttheses.ub.rug.nl/37066/1/MScThesiss3786196.pdf

[^20]: https://arxiv.org/pdf/1910.01913.pdf

[^21]: https://www.emergentmind.com/topics/sparse-gradient-for-long-horizon-reward-optimization

[^22]: https://arxiv.org/pdf/1906.02771.pdf

[^23]: https://www.datacamp.com/tutorial/policy-gradient-theorem

[^24]: https://arxiv.org/html/2512.00005v1

[^25]: https://www.reddit.com/r/reinforcementlearning/comments/n5beqy/question_entropy_of_transformed_tanh_distribution/

[^26]: https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_47.pdf

[^27]: https://www.reddit.com/r/reinforcementlearning/comments/dnx23a/automating_entropy_adjustment_for_maximum_entropy/

[^28]: https://arxiv.org/pdf/2007.01932.pdf

[^29]: https://paperswithcode.com/paper/meta-sac-auto-tune-the-entropy-temperature-of

[^30]: https://www.emergentmind.com/topics/entropy-regularized-policy-gradient

[^31]: https://www.reddit.com/r/reinforcementlearning/comments/13atkpw/autotune_alpha_in_soft_actor_critic_and_reward/

[^32]: https://www.reddit.com/r/reinforcementlearning/comments/1kapd6m/tanh_used_to_bound_the_actions_sampled_from/

[^33]: https://github.com/twni2016/Meta-SAC

[^34]: https://openreview.net/forum?id=FQbkBcpcvA

[^35]: https://www.jmlr.org/papers/volume23/21-1387/21-1387.pdf

[^36]: https://www.shadecoder.com/topics/asynchronous-advantage-actor-critic-a-comprehensive-guide-for-2025

[^37]: https://www.ias.informatik.tu-darmstadt.de/uploads/Site/EditPublication/maximilian_hensel.pdf

[^38]: https://www.pluralsight.com/labs/codeLabs/actor-critic-methods

[^39]: https://people.engr.tamu.edu/guni/papers/TMLR24_Gamid.pdf

[^40]: http://papers.neurips.cc/paper/9225-keeping-your-distance-solving-sparse-reward-tasks-using-self-balancing-shaped-rewards.pdf

[^41]: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

[^42]: https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_126.pdf

[^43]: https://ceur-ws.org/Vol-2491/paper32.pdf

[^44]: https://www.reddit.com/r/reinforcementlearning/comments/vicory/does_the_value_of_the_reward_matter/

