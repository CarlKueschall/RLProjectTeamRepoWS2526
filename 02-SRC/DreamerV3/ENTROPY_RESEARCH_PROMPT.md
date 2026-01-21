# Research Prompt: Understanding Policy Entropy in DreamerV3

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

| Metric                  | Value     | Concern                           |
| ----------------------- | --------- | --------------------------------- |
| `entropy_mean`        | 13.67     | Stuck at maximum, never decreases |
| `entropy_min`         | 12.5-13.6 | Also very high                    |
| `entropy_max`         | 13.67     | At ceiling                        |
| `advantages_abs_mean` | 0.05-0.07 | Very small                        |
| Win rate                | 10-20%    | Not improving                     |

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
