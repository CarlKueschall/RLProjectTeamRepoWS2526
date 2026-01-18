# PBRS Reward Design V3: Balanced Offense/Defense

**Date:** 2026-01-17
**Status:** Implemented
**Problem Solved:** Agent fails to chase puck in defensive situations (our half)

---

## Executive Summary

This document describes PBRS V3 which adds **defensive incentives** and makes φ_attack **asymmetric** to fix the agent's failure to chase pucks that go behind it (into our half).

### Key Changes from V2
1. **Asymmetric φ_attack**: Only active in opponent half (disabled in our half)
2. **NEW φ_defense**: Rewards being between puck and own goal
3. **Full φ_chase in our half**: Stationary puck reduction only applies in opponent half
4. **Epsilon decay starts AFTER warmup**: Keeps exploration high during buffer filling

---

## Problem Analysis

### The Observed Issue (V2)

From GIF analysis of run `best-performer`:
1. Agent pushes too eagerly toward middle/opponent side
2. When opponent shoots puck into our half, agent fails to chase it

### Root Cause

With V2 PBRS, when puck is in our half:

```
φ_attack = -dist_puck_to_opp_goal / MAX ≈ -0.73 (very negative!)
φ_chase = 0.3 × (-dist_to_puck / MAX) ≈ -0.09 (weak due to stationary reduction)

Total: 0.5 × (-0.09) + 1.0 × (-0.73) = -0.77
```

**Problem:** The agent is heavily penalized for states where puck is far from opponent goal, REGARDLESS of agent's actions. This creates:
1. A "penalty zone" in our half that the agent avoids
2. Weak chase incentive when puck is stationary in our half (only 15% effective)

### Quantified Incentives (V2)

| Situation | φ_chase Weight | φ_attack | Net Incentive |
|-----------|---------------|----------|---------------|
| Chase moving puck (opp half) | 0.5 × 1.0 = 0.5 | active | Strong |
| Chase stationary puck (opp half) | 0.5 × 0.3 = 0.15 | active | Weak |
| Chase moving puck (OUR half) | 0.5 × 1.0 = 0.5 | **-0.73** | Mixed |
| Chase stationary puck (OUR half) | 0.5 × 0.3 = **0.15** | **-0.73** | Very Weak! |

---

## V3 Solution

### Three-Component Potential Function

```
φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack + W_DEFENSE × φ_defense
     = 0.5 × chase + 0.7 × attack + 0.3 × defense
```

### Component Details

#### 1. φ_chase (Modified)
- **Always active** (both halves)
- Stationary weight (30%) **only applies in opponent half**
- In our half: **FULL weight** regardless of puck speed

```python
if puck_is_moving or puck_in_our_half:
    phi_chase = phi_chase_base  # Full magnitude
else:
    phi_chase = 0.3 * phi_chase_base  # Reduced in opp half only
```

#### 2. φ_attack (Asymmetric)
- **Only active when puck is in opponent half** (x > 0)
- When puck in our half: **φ_attack = 0** (no penalty!)

```python
if not puck_in_our_half:
    phi_attack = -dist_puck_to_opp_goal / MAX_DISTANCE
else:
    phi_attack = 0.0  # No penalty for defensive situations
```

#### 3. φ_defense (NEW)
- **Only active when puck is in our half** (x < 0)
- Rewards agent being between puck and own goal
- Range: [-1, +1] (can be positive when in good defensive position)

```python
if puck_in_our_half:
    defensive_ratio = (dist_puck_to_own_goal - dist_agent_to_own_goal) / dist_puck_to_own_goal
    phi_defense = clip(defensive_ratio, -1, 1)
else:
    phi_defense = 0.0
```

---

## Reward Matrix (V3)

| Situation | φ_chase | φ_attack | φ_defense | Net | Result |
|-----------|---------|----------|-----------|-----|--------|
| Chase puck (opponent half) | + | 0 | 0 | **+** | Encouraged |
| Shoot toward opponent goal | - | + | 0 | **+** | Encouraged |
| **Chase puck (OUR half)** | **+** | **0** | **+** | **++** | **STRONGLY encouraged!** |
| Good defensive position | ~ | 0 | + | **+** | Encouraged |
| Ignore puck in our half | 0 | 0 | - | **-** | Penalized |
| Camp near stationary puck | weak | 0 | 0 | **~** | Not rewarded |

---

## Quantified Incentives (V3)

| Situation | φ_chase Weight | φ_attack | φ_defense | Net |
|-----------|---------------|----------|-----------|-----|
| Chase moving puck (opp half) | 0.5 | 0.7 × active | 0 | Strong |
| Chase stationary puck (opp half) | 0.15 | 0.7 × active | 0 | Moderate |
| **Chase moving puck (OUR half)** | **0.5** | **0** | **0.3 × active** | **Strong!** |
| **Chase stationary puck (OUR half)** | **0.5** | **0** | **0.3 × active** | **Strong!** |

### Key Improvement

In V2, chasing a stationary puck in our half had 0.15 effective weight with -0.73 penalty.
In V3, it has 0.5 + 0.3 = 0.8 effective weight with 0 penalty!

---

## Additional Fix: Epsilon Decay

### Problem
Epsilon was decaying during warmup when no learning was happening, wasting exploration budget.

### Solution
Epsilon decay now only starts AFTER warmup completes:

```python
if i_episode >= args.warmup_episodes:
    agent.decay_epsilon()
```

---

## Configuration

### Weights
```python
W_CHASE = 0.5    # Agent → Puck (always active)
W_ATTACK = 0.7   # Puck → Opponent Goal (opponent half only)
W_DEFENSE = 0.3  # Defensive positioning (our half only)
```

### Thresholds
```python
PUCK_MOVING_THRESHOLD = 0.3
STATIONARY_PUCK_WEIGHT = 0.3  # Only applied in opponent half
CENTER_X = 0.0  # Half detection
```

---

## Expected Behavioral Changes

1. **Defensive Recovery**: Agent should now actively chase pucks shot into our half
2. **Defensive Positioning**: Agent should position between puck and own goal when under attack
3. **No Middle Camping**: Agent won't be penalized for being in defensive positions
4. **Maintained Offense**: φ_attack still encourages shooting toward opponent goal in their half

---

## Timeline

```
Episode 0-warmup:    epsilon = initial (no decay)
Episode warmup+:     epsilon starts decaying
Episode 0-5000:      PBRS weight = 1.0 (full guidance)
Episode 5000-20000:  PBRS weight = 1.0 → 0.1 (gradual fade)
Episode 20000+:      PBRS weight = 0.1 (minimal guidance retained)
```
