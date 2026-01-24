# Root Cause Synthesis: The Complete Failure Chain

## The Single Root Cause

**The actor's std is stuck at maximum (7.39), causing tanh to saturate, which kills gradients and prevents any policy learning.**

This single issue cascades into every observed symptom:

```
std = 7.39 (max)
    ↓
samples ~ N(mean, 7.39) → |sample| >> 2 for ~80% of draws
    ↓
tanh(sample) ≈ ±1 for 80% of actions → abs_mean = 0.98
    ↓
Agent applies 5880N force every step → puck velocity >> 0.1
    ↓
Puck acquisition condition NEVER met → never holds puck
    ↓
Agent can only score by blind collision → sparse, accidental goals
    ↓
World model can't learn puck-holding dynamics (never observed)
    ↓
Imagination doesn't include puck-holding trajectories
    ↓
No advantage signal for controlled behavior
    ↓
Actor has no reason to reduce std → STUCK
```

## Why std Is Stuck at Maximum

The only force that can push std DOWN is the advantage signal:
```
∂actorLoss/∂std ∝ advantages * ∂logprob/∂std + entropy_scale * ∂entropy/∂std
```

- **Advantage term**: advantages ≈ 0.08, but `∂logprob/∂std` is complex and partly cancelled by tanh saturation
- **Entropy term**: entropy_scale = 1e-5, and `∂entropy/∂std > 0` (entropy increases with std), so this term ACTIVELY pushes std UP

The net gradient on std is either:
- Near zero (advantage signal too weak)
- Slightly positive (entropy bonus pushes std up)

Result: std never decreases from initialization.

## Why This Wasn't Caught Earlier

The previous run against weak bot achieved 85% win rate. How?

**Hypothesis**: The earlier training may have had different initial conditions or the weak bot is beatable with random bang-bang actions (just hitting the puck randomly can score against a weak opponent by luck). The 85% win rate may have been an artifact of the weak bot being exploitable by aggressive random play, not evidence of learned precision.

Evidence:
- Win rate against strong is only ~37% (random baseline would be ~20-30%)
- The agent likely scores by accident when the puck bounces off it at high speed toward the goal
- Against weak bot (which can't defend well), this works ~85% of the time
- Against strong bot (which holds puck and plays strategically), it works ~37%

## The Three Fixes Needed (In Priority Order)

### Fix 1: Reduce logStdMax (CRITICAL)

**Current**: `logStdMax = 2.0` → std_max = 7.39
**Proposed**: `logStdMax = 0.5` → std_max = 1.65

With std_max=1.65:
- Samples from N(0, 1.65): P(|x| > 2) ≈ 22% (vs 79% currently)
- Many more samples in tanh linear region
- Gradients flow properly
- Agent can output moderate actions (0.3-0.7 range)

**Risk**: May be too restrictive if we reduce too aggressively.
**Safe approach**: `logStdMax = 1.0` → std_max = 2.72 as intermediate step.

### Fix 2: Reduce logStdMin slightly

**Current**: `logStdMin = -0.5` → std_min = 0.606
**Keep as-is** or reduce to `logStdMin = -1.0` → std_min = 0.368

This is less critical but allows the policy to become more precise once it starts learning.

### Fix 3: Address the shoot dimension (dim3)

The agent needs to learn:
- action[3] < 0.5 → HOLD the puck
- action[3] > 0.5 → SHOOT the puck

With std=7.39, action[3] is always |1|, meaning it always tries to shoot.
With reduced std, the agent can learn to output action[3] ≈ 0 (hold) vs action[3] ≈ 1 (shoot).

## Expected Impact of Fix 1 (logStdMax reduction)

| Metric | Current (logStdMax=2) | Expected (logStdMax=0.5) |
|--------|----------------------|--------------------------|
| abs_mean | 0.98 | 0.4-0.7 |
| Entropy | 13.6 nats | 5-8 nats |
| Tanh gradient | ≈ 0 | 0.3-0.9 |
| Actor grad norm | 0.09 | 0.5-2.0 (estimate) |
| Puck acquisition | Never | Possible |
| Puck holding | Never | Possible |

## Implementation Plan

1. Change `logStdMax = 2` to `logStdMax = 0.5` in networks.py line ~195
2. Retrain from scratch (not fine-tune - current policy is garbage)
3. Train against weak opponent first (to establish basic skills)
4. Monitor: entropy should be 5-8 nats, abs_mean should be 0.4-0.7
5. Once stable, add self-play with mixed opponents

## Alternative/Additional Fixes (Consider After Fix 1)

- **Action mean regularization**: Add `0.01 * mean.pow(2).mean()` to actor loss
- **Pre-squash action penalty**: Penalize |sample| to keep in tanh linear region
- **Separate shoot head**: Use a binary Bernoulli for dimension 3 instead of continuous
- **Reward shaping for puck acquisition**: Small bonus when obs[16] > 0
- **Beta distribution**: Replace TanhNormal with Beta(alpha, beta) for naturally bounded [0,1] actions

## Verification Plan

After applying Fix 1, verify:
1. `actions/abs_mean` drops below 0.8 within first 100 episodes
2. `behavior/entropy_mean` drops to 5-8 nats range
3. `gradients/actor_norm` increases to > 0.3
4. `advantages_abs_mean` increases (better advantage signal reaches actor)
5. Eventually: agent acquires puck (obs[16] becomes non-zero in some episodes)
