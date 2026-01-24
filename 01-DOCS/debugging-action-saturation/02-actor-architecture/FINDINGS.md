# Actor Architecture: Why Actions Saturate at +/-1

## Actor Network Design

**File:** `02-SRC/DreamerV3/networks.py`, lines 173-212

```python
class Actor(nn.Module):
    # MLP outputs: 4 means + 4 log_stds = 8 values
    # Distribution: Normal(mean, std)
    # Squashing: tanh(sample) -> action in [-1, 1]
```

## Action Sampling Pipeline

```
MLP output → split into [mean, logStd]
    → logStd bounded to [-0.5, 2.0] via tanh scaling
    → std = exp(logStd)  →  std ∈ [0.606, 7.39]
    → sample ~ Normal(mean, std)
    → action = tanh(sample) * actionScale + actionBias
```

## The Saturation Problem

### Current std bounds:
- `logStdMin = -0.5` → `std_min = 0.606`
- `logStdMax = 2.0` → `std_max = 7.389`

### Entropy at 13 nats means std is near maximum (7.39):
- With std=7.39, samples from Normal(mean, 7.39) are huge
- `tanh(x)` saturates to ±1 when |x| > 2
- With std=7.39, most samples have |x| >> 2
- Therefore `tanh(sample) ≈ ±1` for almost all samples
- **abs_mean ≈ 0.98 is a DIRECT CONSEQUENCE of high std**

### The math:
```
P(|x| > 2) where x ~ N(0, 7.39) ≈ P(|z| > 0.27) ≈ 0.79
P(|x| > 3) where x ~ N(0, 7.39) ≈ P(|z| > 0.41) ≈ 0.68
```

With std=7.39, ~79% of samples produce |tanh(x)| > 0.96. This explains abs_mean=0.98.

## Why std Stays at Maximum

The actor loss:
```python
actorLoss = -mean(advantages.detach() * logprobs + entropy_scale * entropies)
```

With:
- advantages ≈ 0.08 (very small, normalized by return range)
- logprobs ≈ -37 (the logged metric shows logprobs_mean ≈ 37, meaning actual logprobs are large negative)
- entropy_scale = 1e-5
- entropies ≈ 13

The gradient signal to reduce std:
- From advantages: `∂L/∂std` via `advantages * ∂logprob/∂std` → weak because advantages are tiny
- From entropy: `∂L/∂std` via `entropy_scale * ∂entropy/∂std` → 1e-5 * (positive) = pushes std UP

**Net effect**: The advantage signal is too weak to push std down. The tiny entropy bonus (while negligible in magnitude) has the correct sign to push std UP. Result: std stays at maximum.

## The Tanh Gradient Death Problem

Even if advantages were larger, there's a secondary issue:

```
∂action/∂mean = ∂tanh(sample)/∂sample * ∂sample/∂mean
              = (1 - tanh²(sample)) * 1
              ≈ 0 when |sample| >> 2 (saturated region)
```

When std=7.39, samples are almost always in the saturated region of tanh. This means:
- Gradients w.r.t. the mean are near zero
- The network CAN'T learn to change the mean
- Even with stronger advantages, the actor is trapped

## The Vicious Cycle

```
High std → saturated tanh → abs_mean=0.98
    → dead gradients w.r.t. mean (tanh gradient ≈ 0)
    → weak actor learning
    → std stays high (no signal to reduce it)
    → repeat
```

## What Should Happen Instead

In a healthy DreamerV3 agent:
1. Early training: high std → exploration → find rewards
2. Advantages become meaningful → push actor toward good actions
3. std decreases → actions become more precise
4. tanh operates in linear region → gradients flow → further refinement

**But our agent is stuck at step 1** because sparse rewards mean advantages stay too small to ever start the collapse.

## Key Difference: entropy_mean is NOT the post-squash entropy

The logged `entropy_mean ≈ 13` is the **pre-squash Gaussian entropy**:
```
entropy = 0.5 * ln(2*pi*e*std²) per dimension
         = 0.5 * ln(2*pi*e*7.39²) ≈ 3.4 per dim
         × 4 dims ≈ 13.6 nats total
```

This matches the observed 13.6757 max entropy exactly.
The TRUE post-squash entropy (in action space) would be much lower since everything maps to ±1.

## Effective Action Distribution

With std=7.39 and any mean:
- Action dim ≈ Bernoulli(-1, +1) with slight bias toward mean's sign
- NOT a continuous policy - it's effectively a binary bang-bang controller
- This explains why puck control is impossible

## Proposed Fixes

1. **Reduce max std**: `logStdMax = 0.5` → std_max = 1.65, keeping tanh in useful range
2. **Add mean regularization**: Penalize |mean| to keep outputs in tanh linear region
3. **Use beta distribution**: Actions naturally in [0,1] without squashing issues
4. **Increase advantage magnitude**: Reduce normalization aggressiveness
5. **Scheduled std reduction**: Anneal logStdMax from 2.0 → 0.5 over training
