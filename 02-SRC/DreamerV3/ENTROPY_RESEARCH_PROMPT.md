# DreamerV3 Entropy & Exploration Research Prompt

## Context

We are implementing DreamerV3 for a hockey game environment (18-dim observations, 4-dim continuous actions, sparse rewards ±10 for goals). We're confused about how entropy and exploration/exploitation should be managed throughout training. This prompt documents our current implementation and asks specific questions we need answered.

---

## Our Current Implementation

### Actor Network (TanhNormal Policy)

```python
class Actor(nn.Module):
    def forward(self, x, training=False):
        # Log std bounded to [-2, 2] -> std in [0.135, 7.39]
        logStdMin, logStdMax = -2, 2
        mean, logStd = self.network(x).chunk(2, dim=-1)
        logStd = logStdMin + (logStdMax - logStdMin) / 2 * (torch.tanh(logStd) + 1)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh * self.actionScale + self.actionBias  # Scale to [-1, 1]

        if training:
            logprobs = distribution.log_prob(sample)
            # Jacobian correction for tanh squashing
            logprobs -= torch.log(self.actionScale * (1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()  # Gaussian entropy: 0.5*ln(2*pi*e*σ²) per dim
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action
```

**Key details:**
- TanhNormal distribution (Gaussian with tanh squashing)
- `logStd` bounded to `[-2, 2]`:
  - `std_min = e^(-2) ≈ 0.135`
  - `std_max = e^(2) ≈ 7.39`
- Previously used `[-5, 2]` but `std_min=0.0067` caused "negative entropy"
- Gaussian entropy per dimension = `0.5 * ln(2πe*σ²)`
  - At std=0.135: entropy/dim ≈ -0.84, total (4-dim) ≈ **-3.4**
  - At std=1.0: entropy/dim ≈ +1.42, total (4-dim) ≈ **+5.7**
  - At std=7.39: entropy/dim ≈ +3.4, total (4-dim) ≈ **+13.6**

### Actor Loss Function

```python
actorLoss = -torch.mean(advantages.detach() * logprobs + entropyScale * entropies)
```

**Details:**
- Entropy added as bonus (higher entropy = lower loss = encouraged)
- `entropyScale = 0.0003` (what we believe is DreamerV3 paper default)
- Some of our experiments used `0.003` (10x larger) and `0.001`
- Advantages are normalized via percentile-based scaling

### Value Normalization (Moments class)

```python
class Moments(nn.Module):
    def forward(self, x):
        low = torch.quantile(x, 0.05)   # 5th percentile
        high = torch.quantile(x, 0.95)  # 95th percentile
        self.low = 0.99 * self.low + 0.01 * low    # EMA update
        self.high = 0.99 * self.high + 0.01 * high
        inverseScale = max(0.01, self.high - self.low)
        return self.low, inverseScale

# In behavior training:
_, inverseScale = self.valueMoments(lambdaValues)
advantages = (lambdaValues - values[:, :-1]) / inverseScale
```

This normalizes advantages to roughly [-1, 1] range based on return distribution.

### Diagnostics We Log

```python
metrics = {
    # Entropy stats
    "behavior/entropy_mean": entropies.mean(),
    "behavior/entropy_std": entropies.std(),
    "behavior/entropy_min": entropies.min(),
    "behavior/entropy_max": entropies.max(),

    # Advantage stats
    "behavior/advantages_mean": advantages.mean(),
    "behavior/advantages_std": advantages.std(),
    "behavior/advantages_abs_mean": advantages.abs().mean(),

    # Loss component balance
    "diagnostics/advantage_contribution": (advantages.abs() * logprobs.abs()).mean(),
    "diagnostics/entropy_contribution": (entropyScale * entropies).mean(),
    "diagnostics/entropy_advantage_ratio": entropy_contribution / advantage_contribution,
}
```

### Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `entropyScale` | 0.0003 | Constant, no annealing |
| `logStdMin` | -2 | std_min ≈ 0.135 |
| `logStdMax` | 2 | std_max ≈ 7.39 |
| `imaginationHorizon` | 15 | Steps of imagined rollout |
| Action dimensions | 4 | Continuous, bounded [-1, 1] |
| Reward structure | Sparse | ±10 for goals, 0 otherwise |

---

## What We're Observing (Symptoms)

### Confusing Entropy Behavior
- Entropy starts high (~+4 to +10) and gradually decreases during training
- Sometimes entropy goes **negative** (around -1 to -3) late in training
- We don't know if negative entropy is a bug, expected, or indicates a problem
- Different `entropyScale` values (0.0003 vs 0.003) lead to very different dynamics

### Balance Issues
- The `entropy_advantage_ratio` metric is confusing - we don't know target values
- We wrote a comment "should decrease from ~1.0 to ~0.01" but we're not sure this is correct
- When advantages are small (~0.05), does entropy dominate the actor loss?

### Uncertainty About "Good" Behavior
- Is our entropy range (-3 to +10) normal?
- Should entropy ever be negative in a healthy run?
- How should exploration decay over training in DreamerV3?

---

## Specific Research Questions

### Section 1: Entropy Scale

**Q1.1**: What is the **exact `entropyScale` value used in the official DreamerV3 implementation** (Danijar Hafner's code)? Is it truly 3e-4 or does it vary by domain?

**Q1.2**: Is `entropyScale` **supposed to be constant** throughout training, or does DreamerV3 use any form of entropy annealing/scheduling?

**Q1.3**: How does `entropyScale` interact with the **number of action dimensions**? Should it be scaled by action dimensionality (e.g., `entropyScale * actionDim`) or is it already normalized?

**Q1.4**: Does `entropyScale` need to be **adjusted for sparse reward environments** where advantages may be small for long periods?

---

### Section 2: Entropy Bounds and Negative Entropy

**Q2.1**: What are the **recommended bounds for logStd** in DreamerV3's actor? Our [-2, 2] gives std in [0.135, 7.39]. What does the paper/official code use?

**Q2.2**: **Is negative entropy expected/acceptable** in DreamerV3? For a bounded Gaussian:
- At std=0.135, entropy = 0.5*ln(2πe*0.018) ≈ -0.84 per dimension
- With 4 dims, total entropy ≈ -3.4
- Is this a problem that needs fixing, or expected behavior?

**Q2.3**: The DreamerV3 paper mentions "entropy regularization" - does this specifically refer to **Gaussian entropy** (which can be negative) or something else (like categorical entropy which is always non-negative)?

**Q2.4**: If negative entropy is problematic, what's the fix? Options might include:
- Shift entropy by a constant (e.g., `entropy - entropy_floor`)
- Use a different entropy formulation
- Clamp logStd to prevent very small std
- Use entropy relative to a reference distribution

---

### Section 3: Entropy-Advantage Balance

**Q3.1**: In the actor loss `L = -(advantages * logprobs + entropyScale * entropy)`, what's the **expected relative magnitude** of these two terms?
- Should they be roughly equal early in training?
- How should this ratio evolve?

**Q3.2**: Our `entropy_advantage_ratio` diagnostic: what values indicate:
- **Healthy balance**: entropy and advantage both contributing?
- **Entropy dominance**: policy just maximizing randomness?
- **Advantage dominance**: policy exploiting but maybe not exploring enough?

**Q3.3**: Does **percentile-based value normalization** (our Moments class) interact with entropy scaling? Since advantages are normalized to roughly [-1, 1], does this change optimal `entropyScale`?

**Q3.4**: In the official DreamerV3 implementation, how is the **balance between entropy and advantage terms** managed?

---

### Section 4: Exploration in Imagination

**Q4.1**: In DreamerV3, the actor is trained **entirely in imagination**. Does this change how exploration should work compared to model-free methods?

**Q4.2**: Does entropy compound/accumulate over the **imagination horizon**? With 15 steps of imagination, does the entropy bonus apply to each step equally?

**Q4.3**: Should exploration happen:
- In **imagination only** (actor training)?
- In the **real environment** (data collection)?
- **Both**, and how does DreamerV3 handle this?

**Q4.4**: Is there any **exploration mechanism in DreamerV3 beyond the entropy bonus**? (e.g., curiosity, noise injection, uncertainty-driven exploration)

---

### Section 5: DreamerV3 Paper & Official Implementation Details

**Q5.1**: What does the **DreamerV3 paper (Hafner et al. 2023)** specifically say about:
- Entropy coefficient naming and value
- Actor network parameterization (mean/std bounds)
- Exploration strategy
- Any entropy scheduling

**Q5.2**: In the **official DreamerV3 codebase**, what are the exact values for:
- `entropy_scale` or equivalent
- Policy standard deviation bounds
- Any entropy-related hyperparameters

**Q5.3**: Does DreamerV3 use **different entropy settings for different domains** (Atari vs DMC vs continuous control)? What category would our hockey task (continuous control, sparse rewards) fall into?

**Q5.4**: Does DreamerV3 use anything similar to **SAC's automatic temperature tuning** (learned α targeting specific entropy)? If not, why was this design choice made?

---

### Section 6: Practical Diagnostics

**Q6.1**: What **concrete entropy values** indicate healthy exploration for a 4-dim continuous action space?
- Early training: entropy should be ___?
- Mid training: entropy should be ___?
- Late training (converged): entropy should be ___?

**Q6.2**: What metrics besides raw entropy should we monitor to diagnose exploration issues?

**Q6.3**: How do we distinguish between:
- (a) Entropy stuck high because no learning signal (advantages ≈ 0)
- (b) Entropy stuck high because entropyScale is too high
- (c) Entropy appropriately high because task requires exploration
- (d) Entropy collapsed because exploiting prematurely

**Q6.4**: What are **red flags / warning signs** in entropy metrics that indicate something is wrong?

---

### Section 7: Comparison with Other Methods

**Q7.1**: How does DreamerV3's entropy handling differ from **SAC** (Soft Actor-Critic)?
- SAC uses automatic temperature tuning with target entropy = -dim(A)
- SAC's target for 4-dim would be -4 nats
- Does DreamerV3 aim for similar entropy levels?

**Q7.2**: How does DreamerV3's entropy handling differ from **PPO**?
- PPO often uses entropy coefficient + clipping
- Some PPO implementations anneal entropy coefficient

**Q7.3**: Is there any relationship between DreamerV3's `entropyScale` and SAC's target entropy concept? Can we compute an equivalent target entropy from `entropyScale`?

---

## Summary of Key Unknowns

1. **What's the correct `entropyScale`?** (0.0003 vs 0.003 vs domain-specific?)
2. **Should `entropyScale` decay over training?** (constant vs annealed?)
3. **Is negative entropy OK?** (expected vs bug?)
4. **What's the target entropy range?** (concrete numbers for 4-dim continuous)
5. **How should entropy_advantage_ratio evolve?** (1.0 → 0.01? something else?)
6. **Does DreamerV3 use automatic entropy tuning?** (like SAC's learned α?)

---

## Desired Output Format

For each question, please provide:
1. **Direct answer** with specific numbers where applicable
2. **Citation** to DreamerV3 paper or official code if available
3. **Explanation of the principle** so we understand WHY
4. **Practical recommendation** for our hockey task specifically

Thank you!
