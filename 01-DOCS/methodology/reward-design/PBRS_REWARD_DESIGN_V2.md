# PBRS Reward Design V2: Directional Attack Incentive

**Date:** 2026-01-17
**Status:** Implemented
**Problem Solved:** Agent shooting backward to exploit PBRS proximity rewards

---

## Executive Summary

This document describes the redesigned Potential-Based Reward Shaping (PBRS) system that adds a **directional attack component** to prevent the "shoot backward" exploit discovered in training runs.

### Key Changes
1. Added **φ_attack**: Rewards puck being close to opponent goal
2. Rebalanced weights: W_ATTACK=1.0, W_CHASE=0.5
3. Slower annealing: 15,000 episodes instead of 3,000
4. Minimum weight floor: 0.1 (never fully removes attack incentive)
5. Epsilon reset at annealing start for re-exploration

---

## Problem Analysis

### The Exploit

With the original PBRS design (φ_chase only), the agent discovered an exploit:

| Action | φ_chase Change | Net PBRS | Result |
|--------|----------------|----------|--------|
| Chase puck | ↑ (distance ↓) | **+** | Encouraged ✓ |
| Shoot forward | ↓ (puck goes away) | **-** | Discouraged ✗ |
| Shoot backward | ↓ then ↑ (can chase again) | **~0** | Neutral → exploited |

**Observed behavior:** Agent learned to chase puck, then shoot it backward to keep it close, maximizing PBRS without scoring goals.

**Evidence from run `256-TD3-Hockey`:**
- `shoot_action_avg` (overall): +0.5 to +0.7 (forward when chasing)
- `shoot_action_when_possess`: -0.5 to -0.8 (backward when possessing)
- Win rate: Peaked at 62%, collapsed to 18-28%

---

## Mathematical Design

### Two-Component Potential Function

```
φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack
```

**Components:**
- **φ_chase** = -dist(agent, puck) / MAX_DIST → Range: [-1, 0]
- **φ_attack** = -dist(puck, opponent_goal) / MAX_DIST → Range: [-1, 0]

**Constants:**
- MAX_DIST = √(9² + 5²) ≈ 10.3 (table diagonal)
- OPPONENT_GOAL = (4.5, 0)
- OWN_GOAL = (-4.5, 0)

### Weight Derivation

For a shot of distance D:

**Shoot FORWARD (toward opponent goal):**
```
Δφ_chase = -D/MAX (puck goes away from agent)
Δφ_attack = +D/MAX (puck goes toward opponent goal)
ΔF = W_CHASE × (-D/MAX) + W_ATTACK × (+D/MAX)
   = (D/MAX) × (W_ATTACK - W_CHASE)
```

**Shoot BACKWARD (toward own goal):**
```
Δφ_chase = -D/MAX (puck goes away from agent)
Δφ_attack = -D/MAX (puck goes away from opponent goal)
ΔF = -(D/MAX) × (W_ATTACK + W_CHASE)
```

**Requirements:**
1. Forward > 0: Requires `W_ATTACK > W_CHASE`
2. Backward < 0: Always true when weights > 0
3. Chase still meaningful: `W_CHASE` not too small

**Chosen weights:** `W_ATTACK = 1.0`, `W_CHASE = 0.5`

---

## Reward Matrix

### Complete Behavior Comparison

| Scenario | Old PBRS (chase only) | New PBRS (chase + attack) | Fixed? |
|----------|----------------------|---------------------------|--------|
| **Approach puck** | + | + | Same |
| **Shoot forward** | - (lose proximity) | **+** (attack > chase) | ✅ |
| **Shoot backward** | - | **--** (double penalty) | ✅ |
| **Hold puck still** | ~0 | ~0 | Same |
| **Hover at own goal** | ~0 | **-** (attack penalty) | ✅ |
| **Puck near opp goal** | any | **+** (attack bonus) | ✅ |

### Numerical Verification

**Forward shot (D = 5 units):**
```
Δφ_chase = -5/10.3 = -0.485
Δφ_attack = +5/10.3 = +0.485
ΔF = 0.5 × (-0.485) + 1.0 × (+0.485) = +0.242
ΔF_scaled = +0.242 × 100 × 0.02 = +0.48
```
✅ Positive reward for shooting forward

**Backward shot (D = 5 units):**
```
Δφ_chase = -5/10.3 = -0.485
Δφ_attack = -5/10.3 = -0.485
ΔF = 0.5 × (-0.485) + 1.0 × (-0.485) = -0.728
ΔF_scaled = -0.728 × 100 × 0.02 = -1.46
```
✅ Strong negative reward for shooting backward

**Margin:** +0.48 - (-1.46) = **+1.94 per shot**

---

## Scale Calculation

### Constraint: PBRS Must Not Eclipse Sparse Rewards

**Sparse rewards:** +10 (goal), -10 (concede)

**New potential range:**
```
φ_combined = 0.5 × [-1, 0] + 1.0 × [-1, 0] = [-1.5, 0]
φ_scaled = 100 × [-1.5, 0] = [-150, 0]
```

**Maximum single-step PBRS:**
```
F_max = γ × φ(s') - φ(s) = 0.99 × 0 - (-150) = 150
F_scaled_max = 150 × 0.02 = 3.0
```

**Safety check:** 3.0 < 10 (sparse reward) ✅

**Typical episode PBRS:** ~1.0-1.5 (empirically verified)

---

## Annealing Strategy

### Problem with Fast Annealing

Original 3,000-episode annealing caused collapse:
- Agent learned PBRS-optimal policy (shoot backward)
- When PBRS removed, Q-function couldn't adapt fast enough
- Epsilon already at minimum → no re-exploration

### New Slow Annealing

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `anneal_start` | 3000 | 5000 | Learn basic skills longer |
| `anneal_duration` | 3000 | 15000 | Very gradual fade |
| `minimum_weight` | 0.0 | 0.1 | Retain attack incentive |

**Timeline for 100k episodes:**
```
Episode 0-5000:      weight = 1.0 (full guidance)
Episode 5000-20000:  weight = 1.0 → 0.1 (gradual fade)
Episode 20000+:      weight = 0.1 (minimal guidance retained)
```

### Epsilon Reset

When annealing starts, reset epsilon to force re-exploration:
```python
if episode == pbrs_anneal_start:
    epsilon = 0.4  # Re-explore with new reward landscape
```

---

## Implementation Configuration

```python
# PBRS Component Weights
W_CHASE = 0.5           # Agent → Puck (supporting role)
W_ATTACK = 1.0          # Puck → Opponent Goal (primary)
STATIONARY_WEIGHT = 0.3 # Only applies to φ_chase

# Scaling
EPISODE_SCALE = 100.0
pbrs_scale = 0.02       # Episode PBRS ≈ 1.0-1.5

# Annealing
pbrs_anneal_start = 5000
pbrs_anneal_episodes = 15000
pbrs_min_weight = 0.1

# Exploration Reset
epsilon_reset_at_anneal = True
epsilon_reset_value = 0.4
```

---

## CLI Arguments

```bash
python train_hockey.py \
    --pbrs_scale 0.02 \
    --pbrs_anneal_start 5000 \
    --pbrs_anneal_episodes 15000 \
    --pbrs_min_weight 0.1 \
    --epsilon_reset_at_anneal \
    --epsilon_anneal_reset_value 0.4
```

---

## Expected Outcomes

### Before (Old PBRS)
- Agent learns to approach puck ✓
- Agent shoots backward to keep puck close ✗
- Win rate plateaus at 20-30% ✗

### After (New PBRS)
- Agent learns to approach puck ✓
- Agent shoots toward opponent goal ✓
- Win rate continues improving toward 60%+ ✓

### Metrics to Monitor
- `behavior/shoot_action_when_possess`: Should stay positive (> 0)
- `behavior/shoot_action_avg`: Should stay positive (> 0)
- `eval/weak/win_rate`: Should continue improving, not plateau
- `pbrs/annealing_weight`: Should decrease smoothly from 1.0 to 0.1

---

## References

- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations.
- Wiewiora, E. (2003). Potential-based shaping and Q-value initialization are equivalent.
- Run analysis: `256-TD3-Hockey-NORMAL-weak-lr0.0010-seed42` (collapse case study)
- Run analysis: `pbrs0.03-annealingTD3-Hockey-NORMAL-weak-lr0.0010-seed42` (backward shooting evidence)
