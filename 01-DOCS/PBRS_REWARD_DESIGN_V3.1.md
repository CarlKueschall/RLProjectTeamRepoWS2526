# PBRS Reward Design V3.1: Strong Chase, Simple Math

**Date:** 2026-01-17
**Status:** Implemented
**Philosophy:** Strong chase handles everything; simple math ensures correct shooting

---

## Executive Summary

V3.1 simplifies the reward shaping to just TWO components with a key mathematical constraint:

```
φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack
     = 1.0 × (-dist_to_puck/MAX) + 1.2 × (-dist_puck_to_goal/MAX)
```

**The Key Insight:** `W_ATTACK > W_CHASE` guarantees forward shooting is always net positive.

---

## Design Evolution

| Version | Components | Complexity | Issue |
|---------|------------|------------|-------|
| V1 | φ_chase only | Low | Agent shot backward to keep puck close |
| V2 | φ_chase + φ_attack | Medium | Agent avoided defensive positions (penalty) |
| V3 | φ_chase + φ_attack + φ_defense | High | Too complex, untested edge cases |
| **V3.1** | **φ_chase + φ_attack** | **Low** | **Simple math, strong chase** |

---

## The Core Insight

**Strong chase (W=1.0) handles EVERYTHING:**
1. **Defense**: Agent races toward puck in our half
2. **Interception**: Agent is always trying to reach the puck
3. **Ready position**: Agent lines up at barrier when puck is in opponent half

**The only constraint:** Forward shooting must be net positive.

---

## Shooting Math

When agent shoots puck distance D:

```
φ_chase change: -D/MAX (puck moves away from agent)
φ_attack change: +D/MAX (puck moves toward opponent goal) [forward shot]
                 -D/MAX (puck moves away from opponent goal) [backward shot]
```

**Forward shot:**
```
Δ = W_ATTACK × (+D/MAX) + W_CHASE × (-D/MAX)
  = D/MAX × (W_ATTACK - W_CHASE)
  = D/MAX × (1.2 - 1.0)
  = +0.2D/MAX ✓ (positive!)
```

**Backward shot:**
```
Δ = W_ATTACK × (-D/MAX) + W_CHASE × (-D/MAX)
  = D/MAX × (-W_ATTACK - W_CHASE)
  = D/MAX × (-1.2 - 1.0)
  = -2.2D/MAX ✗ (heavily penalized!)
```

---

## Reward Matrix

| Action | φ_chase Δ | φ_attack Δ | Net PBRS | Result |
|--------|-----------|------------|----------|--------|
| **Chase puck** | +1.0 | 0 | **+1.0** | STRONG encourage |
| **Shoot forward** | -1.0 | +1.2 | **+0.2** | Encouraged |
| **Shoot backward** | -1.0 | -1.2 | **-2.2** | Heavily penalized |
| **Puck in our half** | active | penalty | chase wins | Agent races to puck |

---

## Why Puck in Our Half Works

When puck is in our half:
- **φ_attack is very negative** (puck far from opponent goal) → creates urgency
- **φ_chase is active** (strong W=1.0) → drives agent toward puck
- **Combined effect**: Agent feels urgency AND has strong incentive to chase

The φ_attack "penalty" actually HELPS because:
1. It creates pressure to do something about the situation
2. The only way to improve φ_attack is to get the puck and shoot it forward
3. Strong φ_chase ensures agent will race to get the puck

---

## Simplifications from V3

| Removed | Reason |
|---------|--------|
| φ_defense component | Strong chase handles defensive positioning |
| Stationary puck weight | Always want full chase incentive |
| Asymmetric φ_attack | Penalty in our half actually creates good urgency |
| Conditional logic | Simpler = more robust |

---

## Configuration

```python
# Component weights
W_CHASE = 1.0    # Strong chase, always active
W_ATTACK = 1.2   # Slightly higher to ensure forward shooting is positive

# Scaling
EPISODE_SCALE = 100.0
pbrs_scale = 0.02  # External scaling in training

# No thresholds or conditional logic needed!
```

---

## Potential Range

```
φ_chase range: [-1, 0] (normalized by MAX_DISTANCE ≈ 10.3)
φ_attack range: [-1, 0]

Combined: W_CHASE × 1 + W_ATTACK × 1 = 2.2
Scaled: 2.2 × 100 = 220

Max PBRS per episode: 220 × 0.02 = 4.4
Sparse reward (win): 10

Ratio: 4.4 / 10 = 0.44 < 1 ✓ (PBRS guides but doesn't dominate)
```

---

## Additional Fix: Epsilon Decay

Epsilon decay now only starts AFTER warmup:

```python
if i_episode >= args.warmup_episodes:
    agent.decay_epsilon()
```

This keeps exploration high during buffer filling when no learning is happening.

---

## Expected Behavior

1. **Always chasing**: Agent constantly moves toward puck
2. **Good interception**: Agent positioned to catch incoming pucks
3. **Defensive recovery**: When puck goes behind agent, it races back
4. **Forward shooting**: Mathematical guarantee that forward shots are rewarded
5. **No camping**: Agent doesn't sit idle anywhere

---

## To Run

```bash
cd 02-SRC/TD3
sbatch train_hockey_pbrs_v3.sbatch
```
