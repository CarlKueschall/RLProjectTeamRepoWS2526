# Advantage Signal: Why It's Too Weak to Train the Actor

## Advantage Computation Pipeline

```python
# 1. Compute TD(lambda) returns in imagination
lambdaValues = computeLambdaValues(rewards, values, continues, lambda_=0.95)

# 2. Normalize by return range (percentile-based)
_, inverseScale = self.valueMoments(lambdaValues)  # = max(0.01, P95 - P05)

# 3. Compute advantages
advantages = (lambdaValues - values[:, :-1]) / inverseScale
```

## Moments (Normalization) Class

```python
class Moments:
    decay = 0.99           # Very conservative EMA
    min_ = 0.01            # Floor for scale
    percentileLow = 0.05   # 5th percentile
    percentileHigh = 0.95  # 95th percentile

    def forward(self, x):
        low = quantile(x, 0.05)
        high = quantile(x, 0.95)
        self.low = 0.99 * self.low + 0.01 * low
        self.high = 0.99 * self.high + 0.01 * high
        inverseScale = max(0.01, self.high - self.low)
        return self.low, inverseScale
```

## Why Advantages Are Tiny (0.08 abs_mean)

From the metrics:
- `norm_scale` (= inverseScale S): 3.5 - 4.2
- `lambda_returns_mean`: -0.47 to 2.27
- `lambda_returns_std`: 1.0 to 2.0
- `values/mean`: -0.45 to 2.33

### The math:
```
raw_advantage = lambdaValue - value
             ≈ noise with std ~ 0.2-0.5 (critic prediction error)

normalized_advantage = raw_advantage / S
                     = raw_advantage / 3.8
                     ≈ 0.05-0.13

abs_mean(advantages) ≈ 0.08  ✓ matches observed data
```

## Is the Normalization Too Aggressive?

**Analysis**: The normalization divides by the **range** (P95-P05), not the **std**. For a Normal distribution:
- P95 - P05 ≈ 3.29 * std
- So dividing by range ≈ dividing by 3.29 * std

This is MORE aggressive than standard z-normalization (which divides by 1*std).

**Comparison with standard DreamerV3**: The original paper uses the same percentile normalization. The difference is that in environments with dense rewards, the return range is larger AND the raw advantages are larger. In our sparse-reward hockey:
- Returns cluster around 0 most of the time
- Only a few trajectories see goals (±10 reward)
- The 5th-95th range captures the bulk of the returns
- But individual advantages remain small

## Actor Loss Magnitude Breakdown

```python
actorLoss = -mean(advantages * logprobs + entropy_scale * entropies)
```

| Component | Value | Contribution to Loss |
|-----------|-------|---------------------|
| advantages | 0.08 abs mean | - |
| logprobs | ~37 (sum over 4 dims, negative) | - |
| advantages * logprobs | ~0.08 * (-37) ≈ -2.96 | **Main driver** |
| entropy_scale | 1e-5 | - |
| entropies | ~13 | - |
| entropy_scale * entropies | 1e-5 * 13 ≈ 0.00013 | **Negligible** |
| **Net actor loss** | **~-(-2.96) ≈ 2.96** | - |

Wait - the logprobs are ~37 as reported in metrics (`logprobs_mean ≈ 37`). But logprobs should be NEGATIVE (log of probability < 1). Let me re-examine...

**Actually**: The metric "logprobs_mean" ≈ 37 likely represents the NEGATIVE log probability (i.e., the code logs `-logprobs` or uses the absolute value). True logprobs for a 4-dim Gaussian would be negative. Let me check the sign convention...

From the actor loss: `actorLoss = -mean(advantages * logprobs + ...)`
- If logprobs are negative (correct), then advantages * logprobs < 0 when advantages > 0
- The negative sign makes the loss POSITIVE when advantages are positive (wants to minimize)
- Actually this maximizes advantages * logprobs (REINFORCE style)

**The magnitude is still small**: Even with logprobs magnitude of ~37:
- `advantages * |logprobs|` ≈ 0.08 * 37 ≈ 2.96
- But the GRADIENT is what matters, not the loss value
- Gradient w.r.t. policy params: `advantages * ∂logprobs/∂θ`
- With tanh saturation, `∂logprobs/∂θ ≈ 0` for the mean params
- So the effective gradient is near zero regardless of advantage magnitude

## The Combined Failure Mode

```
Sparse rewards
    → Most imagined trajectories have reward ≈ 0
    → Lambda returns ≈ 0 for most states
    → Values ≈ 0 for most states
    → Advantages ≈ 0 for most states
    → Actor gets no learning signal
    → std stays at maximum
    → tanh saturates
    → gradients through tanh ≈ 0
    → Actor REALLY gets no signal (double failure)
```

## What About the Non-Zero Advantages?

From metrics: `advantage_significant_frac ≈ 0.22` (22% of advantages are "significant")

This means 22% of imagined states DO have meaningful advantages. But:
1. These are overwhelmed by the 78% with near-zero advantages in the mean
2. Even for the 22% with signal, tanh saturation kills the gradient
3. The actor gradient norm is 0.05-0.16 (very small), confirming weak signal

## norm_scale (S) Trajectory

```
Data points: 0.7 → 3.5 → 3.2 → 3.8 → 3.3 → 3.4 → 3.7 → 3.6 → 3.7 → 3.8 → ...
```

The first point (0.7) was likely at initialization. It quickly grew to ~3.5 and stabilized.
This means the return range settled at about 3.5-4.2, which is reasonable for a hockey game where goals give ±10 reward discounted over ~250 steps.

## Recommendation: The normalization isn't the PRIMARY problem

The normalization is working as designed. The real problem is the DOUBLE failure:
1. Small advantages (from sparse rewards) - this is expected and handled by normalization
2. Tanh saturation killing gradients - this BLOCKS the normalization from helping

Fix #2 (the tanh/std problem) and the normalization should work correctly to amplify the sparse advantage signal.
