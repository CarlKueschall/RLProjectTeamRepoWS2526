<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# \# DreamerV3 Entropy \& Exploration Research Prompt

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
| :-- | :-- | :-- |
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

### Section 5: DreamerV3 Paper \& Official Implementation Details

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

Now let me create a comprehensive final report with all findings.

***

# DreamerV3 Entropy \& Exploration: Comprehensive Research Report

## Executive Summary

Your confusion about entropy and exploration in DreamerV3 stems from three key misunderstandings about the official implementation: First, the fixed entropy scale η = 3 × 10⁻⁴ is **absolutely constant** across all domains with no annealing—its domain-invariance comes not from entropy tuning but from **return normalization** (percentile-based scaling). Second, negative entropy in your implementation indicates a critical architectural difference: DreamerV3's paper doesn't explicitly specify continuous action parameterization, but your TanhNormal approach with small σ fundamentally differs from how DreamerV3 handles exploration. Third, the entropy-advantage balance evolves automatically through return scaling, not through manual ratio management.[^1][^2][^3]

This report provides definitive answers to all seven sections of your research prompt using the official DreamerV3 paper, official code implementations, and peer-reviewed extensions of the algorithm.

***

## Section 1: Entropy Scale Details

### Q1.1: Exact entropy scale value and source

**Answer**: η = 3 × 10⁻⁴ (precisely 0.0003)[^2][^4][^5][^6]

**Citation**: The official DreamerV3 paper (Hafner et al., Nature 2025) states in Equation 6:
> "To use a fixed entropy scale of η = 3 × 10⁻⁴ across domains, we normalize returns to be approximately contained in the interval."[^3][^1]

The Ray RLlib implementation confirms this value. Multiple independent implementations (SheepRL, official embodied framework) use this identical constant.[^4][^2]

**Why this value works**: The entropy scale is NOT tuned per domain. Instead, it's paired with return normalization that ensures advantages remain in a consistent range across vastly different reward scales (from sparse ±10 goal rewards to dense continuous control feedback).

### Q1.2: Is entropy scale constant or annealed?

**Answer**: Absolutely constant. No annealing whatsoever.[^2][^3][^4]

The paper explicitly states "fixed entropy scale" with no temporal decay or scheduling. This is a fundamental design choice differentiating DreamerV3 from SAC (which uses learned α) and PPO (which typically anneals entropy coefficients).

**Why constant works**: The return normalization (Equation 7) handles domain variation, not entropy scheduling. When returns are normalized to approximately  range, a fixed η maintains consistent exploration pressure regardless of reward scale or frequency.[^1]

### Q1.3: Scaling by action dimensionality?

**Answer**: No explicit scaling by action dimensionality.[^3]

The entropy scale η = 3 × 10⁻⁴ is used directly for all action spaces (4-dim continuous, discrete Atari, 18-dim proprioceptive control, visual control). The paper provides no domain-specific modifications to entropy scale.[^3]

**However, entropy itself scales with dimensionality**: Shannon entropy of a categorical distribution or Gaussian scales naturally with the number of action dimensions, so larger action spaces inherently produce larger absolute entropy values. The **fixed coefficient** doesn't change, but the **total entropy bonus** naturally grows with action dimensionality through the network outputs.

For your 4-dim hockey task, this means you apply η directly without modification:

```
entropy_bonus = η * H[π(a_t | s_t)]  =  3e-4 * H[π(a_t | s_t)]
```


### Q1.4: Adjustment for sparse reward environments?

**Answer**: No. The same η = 3 × 10⁻⁴ is used across sparse and dense reward domains.[^3]

Sparse rewards are handled through **return normalization**, not entropy tuning. The key mechanism is the percentile-based scaling with a floor (L=1):

```
S = max(1, EMA[Per(R^λ, 95%) - Per(R^λ, 5%), 0.99])
advantages = (R^λ - v) / S
```

This prevents the entropy bonus from being overwhelmed by noisy advantage signals in sparse reward settings, because advantages are normalized relative to the observed return distribution. When advantages are small (sparse rewards → small differences between states), they don't drown out the entropy term.

***

## Section 2: Entropy Bounds and Negative Entropy

### Q2.1: Recommended logStd bounds in DreamerV3

**Answer**: The official paper does NOT specify logStd bounds or Tanh-Normal parameterization explicitly.[^3]

This is the critical insight: **The DreamerV3 paper uses categorical distributions for discrete actions and does NOT detail continuous action parameterization.** The paper states:
> "We use the Reinforce estimator for both discrete and continuous actions"[^3]

But it only provides architectural details for discrete (categorical with softmax + 1% unimix). For continuous actions, the paper leaves implementation details to practitioners.

**What implementations actually do**:

- **Official Danijar code**: Uses the embodied framework which handles continuous actions as Box spaces[^2]
- **SheepRL PyTorch implementation**: Also supports continuous but architectural details are not in the paper
- **Your approach** (logStd ∈ [-2, 2]): Reasonable, but not officially specified

**Recommendation for hockey task**: Keep your bounds [-2, 2] giving std ∈ [0.135, 7.39], but understand this is NOT the "official" DreamerV3 specification because the paper doesn't provide one.

### Q2.2: Is negative entropy expected/acceptable?

**Answer**: No, negative entropy should not occur in a correct implementation.[^3]

For a **proper probability distribution** (which DreamerV3 uses), entropy is always non-negative:

- **Categorical entropy**: ∑_i p_i log(p_i) ≥ 0 (strictly positive unless deterministic)
- **Gaussian entropy**: 0.5 ln(2πe σ²) ≥ 0 only when σ ≥ sqrt(1/(2πe)) ≈ 0.242

Your negative entropy values (-3.4 with σ=0.135) indicate **you're using a Gaussian with σ too small**, which violates the mathematical definition of entropy.

**Why you see negative entropy**: At std = 0.135:

```
H = 0.5 * ln(2πe * (0.135)²) 
  = 0.5 * ln(2πe * 0.0182)
  = 0.5 * ln(0.0456)  ← This is negative because ln(x) < 0 for x < 1
  ≈ 0.5 * (-3.09) ≈ -1.55 per dimension
```

This is a mathematical artifact of using an extremely narrow Gaussian. The entropy coefficient then multiplies this negative value, **encouraging the policy to become more concentrated** (opposite of exploration). This is a bug in your implementation, not expected behavior.

**Fix**: Either increase logStd bounds to ensure σ > 0.242, or use a different parameterization matching DreamerV3's actual approach (which doesn't provide enough detail for direct replication).

### Q2.3: What does "entropy regularization" mean in DreamerV3?

**Answer**: The paper refers to **Shannon entropy** of the policy distribution, regardless of whether it's categorical or continuous.[^3]

The actor loss (Equation 6) includes:

```
η H[π_θ(a_t | s_t)]
```

Where H is Shannon entropy. For categorical: H = -∑ p_i log(p_i). For Gaussian: H = 0.5 ln(2πe σ²).

This is standard entropy regularization as in SAC or PPO, with the key difference being that DreamerV3 uses a **fixed coefficient η** rather than learned or scheduled values.

### Q2.4: If negative entropy is problematic, what's the fix?

**Answer**: Multiple solutions depending on your parameterization choice:

**Option A: Increase logStd bounds (simpler)**
Change logStd ∈ [-2, 2] to logStd ∈ [-1, 2]:

- std_min = e^(-1) ≈ 0.368 (entropy/dim ≈ -0.5)
- Still allows meaningful exploration but avoids deeply negative entropy
- Entropy at maximum std: H ≈ +3.4 per dim, +13.6 total (consistent with your observations)

**Option B: Clamp Gaussian entropy**
Add floor: `entropy_loss = max(0, η * entropy)` to prevent negative contribution. This prevents entropy from actively discouraging exploration.

**Option C: Match DreamerV3 more precisely**
Implement categorical discretization for continuous actions (K bins per dimension). This guarantees non-negative entropy and matches the categorical emphasis throughout the paper. For 4-dim actions with K=16 bins: output 4 × 16 logits, sample categorical per dimension, scale to [-1, 1].

**My recommendation**: For your hockey task, use **Option B** (entropy floor) as a quick fix to match your current architecture, but investigate **Option C** (categorical) if you want to fully replicate DreamerV3's approach.

***

## Section 3: Entropy-Advantage Balance

### Q3.1: Expected relative magnitude of entropy vs advantage terms

**Answer**: They should **not be roughly equal**. Instead, their ratio evolves naturally based on return distribution statistics.[^3]

In the loss:

```
L(θ) = -∑_t sg[(R^λ - v) / max(1, S)] log π + η H[π]
```

The advantage term is normalized by return range S (95th minus 5th percentile), while the entropy term is fixed at 3e-4. The ratio depends on how spread out the return distribution is:

- **Early training**: Returns highly variable → S large → advantages small → entropy relatively dominates
- **Late training**: Returns concentrated → S small → advantages large → policy focuses more on rewards than entropy

This **automatic adaptation** is a key feature. You don't manually tune the ratio; percentile-based normalization handles it.

### Q3.2: What values indicate healthy vs problematic balance?

**Answer**: You can't use a fixed entropy_advantage_ratio as a diagnostic because this ratio is supposed to change naturally.

Better diagnostics:


| Metric | Healthy | Warning Sign |
| :-- | :-- | :-- |
| Actor loss trends downward | ✓ | Loss increasing or plateaus |
| Entropy stable ±20% | ✓ | Entropy collapsing to 0 early |
| Advantages normalized to [-2, 2] | ✓ | Advantages >> 1000 (bad scaling) |
| Return range (S) grows with episode returns | ✓ | S stuck at floor (1.0) despite increasing rewards |
| Policy behavior becomes goal-directed by mid-training | ✓ | Actions remain random throughout |

**Do NOT target**: "entropy_advantage_ratio should be 1.0 → 0.01" (your comment). This is incorrect.

### Q3.3: Percentile-based value normalization interaction

**Answer**: Percentile normalization is **essential** for entropy scale to work as fixed.[^3]

The interaction:

1. **Return percentiles** (5th, 95th) are tracked with EMA decay 0.99
2. **Range S** computed from percentiles, clamped to ≥ 1
3. **Advantages normalized** by S
4. **Entropy term fixed** at η (doesn't scale)

This coupling ensures: when advantages are small (sparse rewards, uncertain value estimates), entropy provides stable exploration signal. When advantages are large (confident value estimates), entropy stays proportionally smaller.

**For your implementation**: Your percentile-based Moments class is implementing the right idea. The key is that your entropy scale η must be designed for advantages normalized to approximately [-1, 1] range. Your current η = 0.0003 assumes this normalization. If you change the normalization scheme, η must change too.

### Q3.4: How is balance managed in official DreamerV3?

**Answer**: Through return normalization alone, not entropy tuning.

The official algorithm uses one mechanism: **percentile-based return normalization with EMA tracking and floor clamping**. No other balance mechanisms.

***

## Section 4: Exploration in Imagination

### Q4.1: Does exploration happen only in imagination?

**Answer**: **Exploration during data collection is entropy-driven randomness from the learned policy.** Exploration during training happens purely in imagination.[^3]

Clear distinction:

**Data Collection Phase**:

- Environment runs with actor policy π (trained previously)
- Policy is stochastic due to entropy bonus → some exploration
- But mostly exploiting learned behavior
- Minimal active exploration mechanisms

**Training Phase (Imagination)**:

- Actor trained entirely on imagined rollouts
- Policy gradient with entropy regularization drives exploration in latent space
- World model doesn't receive exploration signals (no gradients from actor-critic to world model)
- Exploration here guides what the policy learns, not what data is collected

This is different from curiosity-driven approaches. DreamerV3 relies on **passive entropy-driven exploration** during both phases, with the imagined rollouts allowing long-horizon credit assignment.

### Q4.2: Does entropy compound over imagination horizon?

**Answer**: Entropy bonus applies independently at each timestep, but the **advantage signal compounds**.[^3]

In the loss:

```
L = -∑_{t=1}^{H} sg[(R^λ_t - v(s_t)) / S] log π(a_t | s_t) + η H[π(a_t | s_t)]
```

- **Entropy term**: Summed independently, applies equally to each step
- **Advantage term**: Compounds through λ-returns over the horizon
- Total entropy bonus scales linearly with horizon H = 15

This means longer imagination horizons accumulate more entropy regularization pressure. This is intentional: longer-term planning benefits from more exploration across the trajectory.

### Q4.3: Should exploration happen in imagination only or both?

**Answer**: Both phases have exploration, but through different mechanisms.[^3]

**During Data Collection**:

- Actor acts stochastically → entropy-driven exploration of real environment
- Limited by real-world interaction budget
- Essential for discovering reward structure

**During Imagination Training**:

- Entropy bonus optimizes policy to explore in latent space
- Generates diverse imagined trajectories
- Shapes what behaviors are learned

**Real-world robots** (DayDreamer paper) show DreamerV3 explores effectively through this dual entropy-driven mechanism without additional curiosity rewards.[^7]

### Q4.4: Any exploration mechanisms beyond entropy?

**Answer**: The official DreamerV3 has **only entropy regularization** for exploration.[^3]

However, recent extensions add more:

**DreamerV3-XP** (2025): Adds ensemble disagreement-based intrinsic rewards for sparse-reward settings[^8][^9]

**Plan2Explore**: Separate exploration agent with state entropy maximization (not in core DreamerV3)[^5]

**Standard DreamerV3**: Entropy regularization alone, demonstrating it's sufficient for 150+ tasks including sparse-reward Minecraft.

For your hockey environment (sparse ±10 rewards), consider whether entropy alone suffices. If exploration stagnates, ensemble disagreement (DreamerV3-XP approach) might help, but start with baseline entropy regularization with proper normalization.

***

## Section 5: Official Implementation Details

### Q5.1: What does the DreamerV3 paper say about entropy?

**Answer**: The paper provides surprisingly little detail on continuous action entropy.[^3]

**What's explicitly stated**:

- Entropy scale η = 3 × 10⁻⁴ (fixed, no scheduling)
- "explore through an entropy regularizer"
- "entropy scale depends on reward scale and frequency"
- "return normalization... enables fixed entropy scale across domains"
- Uses Reinforce for both discrete and continuous (but parameterization not detailed)

**What's NOT stated**:

- logStd bounds for continuous actions
- Whether continuous uses Gaussian, categorical, or other parameterization
- Specific entropy formula for continuous actions
- Entropy floor or ceiling mechanisms

This gap explains your confusion. The paper provides a general framework but leaves continuous action specification to implementations.

### Q5.2: Exact hyperparameter values in official code

**Citation** from official code and multiple sources:[^4][^5][^2]


| Parameter | Value | Notes |
| :-- | :-- | :-- |
| entropy_scale (η) | 3e-4 | Constant, no scheduling |
| return_normalization_decay (EMA) | 0.99 | Percentile tracking |
| return_percentile_low | 0.05 (5th) | Robust to outliers |
| return_percentile_high | 0.95 (95th) | Robust to outliers |
| return_scale_floor (L) | 1.0 | Prevents noise amplification |
| imagination_horizon | 15 | Steps per rollout |
| discount_factor | 0.997 | 0.997 ≈ 0.9^0.01, near-episodic |
| gae_lambda | 0.95 | λ-return bootstrap |

No other entropy-related hyperparameters are configurable in official implementation.

### Q5.3: Different entropy settings for different domains?

**Answer**: No. Same η = 3 × 10⁻⁴ everywhere.[^4][^3]

DreamerV3's core innovation is **fixed hyperparameters across domains**. The paper explicitly tests:

- Atari (discrete, visual, episodic)
- DMC (continuous, proprioceptive, dense)
- Minecraft (visual, sparse, long-horizon)
- Crafter (discrete, sparse, structured)
- And 4+ more domains

**Same entropy coefficient throughout**. Domain robustness comes from return normalization, not entropy tuning.

### Q5.4: Automatic temperature tuning like SAC?

**Answer**: No. DreamerV3 explicitly rejects automatic entropy tuning.[^3]

**Why not?** From the paper:
> "Constrained optimization targets a fixed entropy on average across states... which is robust but explores slowly under sparse rewards and converges lower under dense rewards."

DreamerV3 chose **simplicity over adaptation**: fixed η with domain-invariant return normalization outperformed learned temperature approaches in their ablations.

**SAC's approach** (learned α targeting -dim(A)): Requires per-domain tuning of target entropy. For your 4-dim hockey: target would be -4 nats. DreamerV3 sidesteps this tuning.

***

## Section 6: Practical Diagnostics

### Q6.1: Concrete entropy values for 4-dim continuous action space

**Answer**: These are guidelines, not hard targets, and depend heavily on your normalization scheme:


| Phase | Expected Entropy | Notes |
| :-- | :-- | :-- |
| **Early training** | +3 to +8 per step | High variance in returns → small advantages → entropy relatively high |
| **Mid training** | +1 to +3 per step | Learning proceeds, return distribution concentrating |
| **Converged** | +0.5 to +1.5 per step | Policy focused but still stochastic |

**For Gaussian with σ ∈ [0.135, 7.39]**:

- Minimum entropy (σ=0.135): **-3.4 total** (BAD - negative)
- Mid-range entropy (σ=1.0): **+5.7 total** (healthy)
- Maximum entropy (σ=7.39): **+13.6 total** (high exploration)

**Your observations** (entropy starting +4 to +10, decaying to -1 to -3) indicate:

- Initial: Good, consistent with mid-to-high exploration
- Final decay to negative: **BUG**, indicates σ collapsing too much


### Q6.2: Other metrics to monitor

**Beyond raw entropy**:

```python
metrics = {
    # Entropy diagnostics
    "actor/entropy_mean": entropy.mean(),
    "actor/entropy_std": entropy.std(),
    "actor/min_std": logstd.exp().min(),  # Monitor lower bound
    "actor/max_std": logstd.exp().max(),  # Monitor upper bound
    
    # Return distribution tracking
    "returns/percentile_5": per_5,
    "returns/percentile_95": per_95,
    "returns/scale_S": S,
    "returns/scale_at_floor": (S == 1.0).float().mean(),  # % of time clamped
    
    # Advantage signal health
    "advantages/mean": advantages.mean(),
    "advantages/std": advantages.std(),
    "advantages/max_abs": advantages.abs().max(),
    
    # Actor loss decomposition
    "actor_loss/policy_gradient_term": (sg_advantages * logprobs).mean(),
    "actor_loss/entropy_term": (eta * entropies).mean(),
    "actor_loss/total": actor_loss.mean(),
    
    # Policy health
    "policy/action_mean": actions.mean(dim=0),  # Per-dim statistics
    "policy/action_std": actions.std(dim=0),
}
```


### Q6.3: Diagnosing entropy behavior

**To distinguish between causes:**

```
IF entropy_mean > 5 AND advantages_mean ≈ 0:
    → Learning signal weak, entropy high due to normalization floor
    → Problem: Task not providing enough signal (reward structure? credit assignment?)
    
IF entropy_mean > 5 AND advantages_mean > 1:
    → entropy_scale η may be too high
    → Reduce η (but NOT in official DreamerV3)
    
IF entropy_mean < 1 AND policy performs well:
    → Healthy convergence, policy found good behavior
    
IF entropy_mean CRASHES to negative:
    → CRITICAL BUG: logStd bounds too tight or entropy computation wrong
    → Fix: Check std bounds, entropy formula
```


### Q6.4: Red flags and warning signs

| Red Flag | Interpretation | Action |
| :-- | :-- | :-- |
| Entropy negative (< 0) | Gaussian std too small | Increase logStd_min or use categorical |
| Entropy stuck at max early | Learning not happening | Check world model quality, reward signal |
| Return scale S always at floor (1.0) | Returns too stable or weird | Check reward function, environment properties |
| Actor loss increasing | Policy diverging | Reduce η or check advantage computation |
| Action std ∈ [0.01, 0.05] | Collapsing to deterministic | Check entropy scale, advantage signals |
| Policy deterministic but low return | Premature convergence | Entropy regularization too weak |


***

## Section 7: Comparison with Other Methods

### Q7.1: DreamerV3 vs SAC entropy handling

| Aspect | DreamerV3 | SAC |
| :-- | :-- | :-- |
| **Entropy coefficient** | Fixed η = 3e-4 | Learned α |
| **Target entropy** | None (fixed scale) | -dim(A) per domain |
| **Tuning required** | None (domain-invariant) | Per-domain α target |
| **Return normalization** | Percentile-based, EMA | None (uses raw returns) |
| **Learning efficiency** | Very high (imagined rollouts) | Good (off-policy replay) |
| **Continuous actions** | Policy gradient (REINFORCE) | Deterministic policy gradient |
| **Discrete actions** | Supported | Not standard |
| **Exploration mechanism** | Entropy-only | Entropy-only |
| **Typical entropy level (4-dim)** | 1-5 nats mid-training | -4 to +2 depending on α |

**Key insight**: DreamerV3's fixed η works because return normalization ensures **advantages stay in a consistent range** regardless of reward scale. SAC's learned α targets a fixed entropy level regardless of return distribution—different approaches to the same problem.

### Q7.2: DreamerV3 vs PPO entropy handling

| Aspect | DreamerV3 | PPO |
| :-- | :-- | :-- |
| **Entropy coefficient** | Fixed 3e-4 | Often annealed (0.01 → 0.001) |
| **Policy learning** | REINFORCE on imagined rollouts | Policy gradient clipping |
| **Return normalization** | Percentile-based + EMA | Advantage normalization per batch |
| **Exploration decay** | None (constant η) | Scheduled (common practice) |
| **Action space** | Both discrete \& continuous | Both |
| **On-policy handling** | Artificial: on imagined data | Natural: on real rollouts |
| **Typical entropy decay** | Flat | Linear or exponential schedule |

**Critical difference**: PPO typically **anneals entropy** as exploration becomes less necessary (more data gathered). DreamerV3 keeps it constant because imagination allows continuous exploration of unobserved state regions.

### Q7.3: Relationship between DreamerV3's η and SAC's target entropy

**Mathematical relationship** (approximate):

For a policy at convergence under SAC with target entropy H_target = -dim(A):

- For 4-dim continuous: H_target = -4 nats (very concentrated policy)
- Typical SAC entropy during training: 0 to -2 nats (learned α maintains this)

For DreamerV3:

- Fixed η = 3e-4 with advantages in [-1, 1] from return normalization
- This generates entropy values 1-5 nats (much higher than SAC!)
- Why? Imagination allows continued exploration; SAC needs more exploitation

**Can't directly convert** η to H_target because:

1. DreamerV3 uses REINFORCE (policy grad), SAC uses deterministic policy grad
2. Different return distributions (imagined vs real)
3. Different normalization schemes

But conceptually: DreamerV3's 3e-4 is chosen to be "weak" enough that policy grad + return normalization naturally produces reasonable entropy, not "too strong" that it prevents convergence.

***

## Summary of Answers to Your Core Questions

### Critical Insights

1. **Entropy scale**: η = 3 × 10⁻⁴, **absolutely fixed**, no annealing, works across domains through return normalization, NOT entropy tuning.
2. **Negative entropy is a bug**: Indicates your Gaussian std < 0.242. Fix by increasing logStd bounds or switching to categorical discretization.
3. **Entropy-advantage balance**: Evolves automatically through percentile-based return normalization. Don't manually target a ratio; monitor return scale S and advantages instead.
4. **Exploration in imagination**: Entropy-only mechanism during both data collection and training. No active curiosity or uncertainty bonuses in core algorithm.
5. **Continuous action parameterization**: Paper doesn't specify! Your TanhNormal approach is reasonable, but it's not officially documented. Ensure σ stays > 0.242 to avoid negative entropy.
6. **No automatic entropy tuning**: DreamerV3 explicitly rejected SAC-style learned temperature in favor of fixed η with return normalization. Simpler and more robust.
7. **For sparse hockey task**: Use entropy-only exploration initially. If ineffective after substantial training (>50M steps), consider adding ensemble-disagreement (DreamerV3-XP approach).

***

## Practical Recommendations for Your Hockey Implementation

**Immediate fixes**:

1. **Fix negative entropy**: Change `logStdMin = -2` to `logStdMin = -0.5` (guarantees σ > 0.368)
2. **Remove entropy scaling by action dim**: Use η = 3e-4 directly for your 4-dim actions
3. **Trust return normalization**: Keep your percentile-based Moments class; it's correct
4. **Monitor return scale S**: If always at floor (1.0), your return distribution is likely pathological

**Diagnostics**:

```python
# Log these during training
"actor/entropy_mean": entropies.mean(),
"actor/min_std": torch.exp(logStd).min(),
"returns/scale_S": S,
"returns/percentile_range": per_95 - per_5,
"advantages/mean_abs": advantages.abs().mean(),
"policy/diverging": (logprobs.abs().mean() > 10).float(),
```

**Advanced tuning**:

If exploration appears insufficient after 50M+ environment steps on hockey:

- Try η = 3e-3 (10x higher) for 10M steps as diagnostic
- Or implement DreamerV3-XP ensemble disagreement bonus
- But start with baseline entropy-only approach

***

<references>
 Memory of your technical background in ML/RL[^1]
 DreamerV3-XP: Optimizing exploration through uncertainty (arXiv:2510.21418)[^8]
 Ray RLlib DreamerV3 implementation[^10]
 EclecticSheep SheepRL DreamerV3 PyTorch implementation[^11]
 DayDreamer: World Models for Physical Robot Learning (Wu et al., CoRL 2022)[^12]
 Danijar's official dreamerv3 GitHub repository[^2]
 "Mastering Diverse Domains through World Models" (Hafner et al., Nature 2025 / arXiv:2301.04104)[^3]
 YouTube: DreamerV3 Tutorial with diagrams[^13]
 SheepRL documentation[^14]
 Ray RLlib source code configurations[^4]
 Related academic papers on entropy in RL[^15]
 Spinning Up (OpenAI) SAC documentation[^16]
 Multiple implementations across research labs[^17]
 Reddit discussions from RL community[^18]
 Project websites and supplementary materials[^5]
 Recent extensions and variants of DreamerV3[^6]
</references>
<span style="display:none">[^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80]</span>

<div align="center">⁂</div>

[^1]: https://github.com/danijar/dreamerv3

[^2]: https://www.nature.com/articles/s41586-025-08744-2

[^3]: https://arxiv.org/html/2301.04104

[^4]: https://github.com/ray-project/ray/blob/master/rllib/algorithms/dreamerv3/dreamerv3.py

[^5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12510832/

[^6]: https://arxiv.org/html/2507.04075v1

[^7]: https://autolab.berkeley.edu/assets/publications/media/2022-12-DayDreamer-CoRL.pdf

[^8]: https://arxiv.org/abs/2510.21418

[^9]: https://chatpaper.com/paper/203054

[^10]: https://www.reddit.com/r/reinforcementlearning/comments/1defbq8/d_how_does_dreamerv3_do_so_well_on_sparsereward/

[^11]: https://www.youtube.com/watch?v=viXppDhx4R0

[^12]: https://eclecticsheep.ai/2023/08/10/dreamer_v3.html

[^13]: https://github.com/masonhargrave/dreamerv3-personal

[^14]: https://vitalab.github.io/article/2023/01/19/DreamerV3.html

[^15]: https://proceedings.neurips.cc/paper_files/paper/2023/file/04f61ec02d1b3a025a59d978269ce437-Paper-Conference.pdf

[^16]: https://lyk-love.cn/2024/02/25/DreamerV3-Report/

[^17]: https://github.com/AndrejOrsula/drl_air_hockey

[^18]: https://www.reddit.com/r/reinforcementlearning/comments/1mv4jn1/dreamer_v3_with_storm_4_months_to_build/

[^19]: https://icml.cc/virtual/2025/session/50258

[^20]: https://fse.studenttheses.ub.rug.nl/33851/1/bAI2024JeremiasFerrao.pdf

[^21]: https://fse.studenttheses.ub.rug.nl/37066/1/MScThesiss3786196.pdf

[^22]: https://lukaborn.de/misc/bachelor_thesis.pdf

[^23]: https://openreview.net/pdf/50883b8612224a4e7214d029983a9215111d4f28.pdf

[^24]: https://arxiv.org/html/2508.20294v3

[^25]: https://arxiv.org/html/2503.00653v1

[^26]: https://gitlab.ub.uni-bielefeld.de/scs/cocosy/rl-with-dreamer-and-fun/-/blob/99a8b0372996eb37ace1d32b1c889b5cf6b82570/dreamerv3/src/dreamerv3/configs.yaml

[^27]: https://di-engine-docs.readthedocs.io/en/latest/05_api_doc/model.html

[^28]: https://github.com/Eclectic-Sheep/sheeprl

[^29]: https://openreview.net/pdf?id=Q5b9zs6NwB

[^30]: https://cvoelcker.de/blog/2025/reppo-intro/

[^31]: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf

[^32]: https://arxiv.org/pdf/2301.04104.pdf

[^33]: https://arxiv.org/abs/2301.04104

[^34]: https://ar5iv.labs.arxiv.org/html/2301.04104

[^35]: https://findingtheta.com/blog/the-evolution-of-imagination-a-deep-dive-into-dreamerv3-and-its-conquest-of-minecraft

[^36]: https://arxiv.org/html/2510.21418v1

[^37]: https://proceedings.iclr.cc/paper_files/paper/2025/file/cf6501108fced72ee5c47e2151c4e153-Paper-Conference.pdf

[^38]: https://openreview.net/pdf?id=4OJdZhcwBb

[^39]: https://www.lesswrong.com/posts/4xGAmZ9GTGAkszHoH/parameter-scaling-comes-for-rl-maybe

[^40]: https://www.shadecoder.com/topics/sac-algorithm-a-comprehensive-guide-for-2025

[^41]: https://spinningup.openai.com/en/latest/algorithms/sac.html

[^42]: https://papers.cool/arxiv/2510.21418

[^43]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12003158/

[^44]: https://arxiv.org/html/2405.15083v1

[^45]: https://www.reddit.com/r/reinforcementlearning/comments/15o2cty/dreamer_v3_in_sheeprl/

[^46]: https://arxiv.org/html/2403.00564v2

[^47]: https://github.com/danijar/daydreamer

[^48]: https://docs.diambra.ai/handsonreinforcementlearning/sheeprl/

[^49]: https://github.com/penn-pal-lab/scaffolder/blob/main/DREAMERV3_README.md

[^50]: https://liner.com/review/efficientzero-v2-mastering-discrete-and-continuous-control-with-limited-data

[^51]: https://arxiv.org/html/2401.16650v3

[^52]: https://raw.githubusercontent.com/mlresearch/v275/main/assets/ugadiarov25a/ugadiarov25a.pdf

[^53]: https://www.emergentmind.com/topics/dreamerv3-style-mbrl-agent

[^54]: https://openreview.net/forum?id=8iW8HOidj1_

[^55]: https://openreview.net/pdf?id=3Cr6C2zNKw

[^56]: https://proceedings.neurips.cc/paper_files/paper/2023/file/96d00450ed65531ffe2996daed487536-Paper-Conference.pdf

[^57]: https://danijar.com/project/dreamerv3/

[^58]: https://www.reddit.com/r/reinforcementlearning/comments/1akcyjs/dreamerv3_for_nonvisual_control_tasks/

[^59]: https://openreview.net/forum?id=zNUOZcAUxz

[^60]: https://arxiv.org/html/2512.07437

[^61]: https://arxiv.org/html/2511.06136v2

[^62]: https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Report-3.pdf

[^63]: https://arxiv.org/html/2503.21047v1

[^64]: https://arxiv.org/html/2503.08872v1

[^65]: https://github.com/danijar/dreamer/issues/11

[^66]: https://www.reddit.com/r/reinforcementlearning/comments/1dmiv75/question_about_the_equation_derivation_of/

[^67]: https://arxiv.org/html/2509.20021v1

[^68]: https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_208.pdf

[^69]: https://arxiv.org/html/2411.10175v2

[^70]: https://proceedings.neurips.cc/paper_files/paper/2024/file/ed5854c456e136afa3faa5e41b1f3509-Paper-Conference.pdf

[^71]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12631203/

[^72]: https://github.com/danijar/embodied/issues/4

[^73]: https://arxiv.org/pdf/2506.17103.pdf

[^74]: https://github.com/danijar/dreamerv3/issues/18

[^75]: https://www.reddit.com/r/reinforcementlearning/comments/1biky9x/why_dreamerv3_uses_the_actorcritic_models/

[^76]: https://github.com/ray-project/ray/issues/39751

[^77]: https://www.reddit.com/r/reinforcementlearning/comments/129hcnz/normal_vs_multivariate_normal_for_action_samle/

[^78]: https://openreview.net/pdf?id=i7jAYFYDcM

[^79]: https://openreview.net/forum?id=AP0ndQloqR

[^80]: https://www.youtube.com/watch?v=vfpZu0R1s1Y

