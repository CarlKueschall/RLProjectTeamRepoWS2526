# PBRS Scaling Analysis: Why PBRS Dominates Sparse Rewards

## Executive Summary

**The Core Problem:** `pbrs_scale=0.02` is **~10x too high**, causing PBRS to dominate the sparse reward signal.

| Metric | Current | Target | Fix |
|--------|---------|--------|-----|
| Episode PBRS | ~3.5 | ~0.3 | Reduce pbrs_scale |
| PBRS/Sparse Ratio | 3.5x | 0.3x | 10x reduction needed |
| pbrs_scale | 0.02 | 0.002-0.003 | **Key change** |

---

## The Math Breakdown

### Step 1: Sparse Reward Structure

From the hockey environment:
- **Win:** +10 raw → +1.0 after `reward_scale=0.1`
- **Loss:** -10 to -18 raw → -1.0 to -1.8 after scaling
- **Tie:** 0

**Per-episode sparse: ±1.0 to ±1.8**

### Step 2: PBRS Accumulation

Current configuration:
- `EPISODE_SCALE = 100.0` (internal scaling)
- `pbrs_scale = 0.02` (external scaling)
- Average episode length: **155 steps**

Per-step PBRS calculation:
```
F(s,s') = γ × φ(s') - φ(s)
        ≈ potential_change × EPISODE_SCALE × pbrs_scale
        ≈ small_delta × 100 × 0.02
        ≈ 0.02 per step on average
```

**Episode total: 155 steps × 0.02/step ≈ 3.1**

Observed from runs: `pbrs/avg_per_episode = 3.5`

### Step 3: The Imbalance

| Signal | Magnitude | Role |
|--------|-----------|------|
| Sparse (win/loss) | ±1.0 | **Should dominate** |
| PBRS total | ~3.5 | Should guide, not dominate |

**Current Ratio: PBRS is 3.5x the sparse signal!**

This means the agent learns more from PBRS than from actual game outcomes.

---

## Why This Causes Problems

### 1. Cross-Court Exploitation
The cross-court component (φ_cross) adds up to +0.4 per high-velocity shot away from opponent. Over many steps:
- Agent learns that "moving puck around" generates more reward than "scoring"
- Results in possession hoarding (observed: 0.32-0.36)
- Agent reluctant to shoot (observed: shoot_when_possess = -0.41 to -0.51)

### 2. Early Learning Collapse
Initial strong performance → collapse pattern:
1. **Early:** Random exploration + PBRS guidance → learns to chase puck
2. **Mid:** Agent realizes PBRS >> sparse → optimizes for PBRS
3. **Late:** Agent hoards puck, avoids risky shots → high tie rate

### 3. The "Amazing Initial Performance" Trap
PBRS helps early learning (good!) but then:
- Agent overfits to PBRS signal
- Sparse win/loss becomes noise compared to PBRS
- Performance plateaus or degrades

---

## The Fix: Reduce pbrs_scale

### Target Ratio
PBRS should be **20-30% of sparse**, not 350%.

```
Target episode PBRS = 0.3 × sparse = 0.3 × 1.0 = 0.3
Current episode PBRS = 3.5
Reduction needed = 3.5 / 0.3 = 11.7x
```

### New pbrs_scale Calculation

```
new_scale = current_scale / 11.7
          = 0.02 / 11.7
          = 0.0017
```

**Recommendation: `pbrs_scale = 0.002` to `0.003`**

### Verification

With `pbrs_scale = 0.003`:
```
Episode PBRS ≈ 3.5 × (0.003 / 0.02) = 3.5 × 0.15 = 0.53
PBRS/Sparse Ratio = 0.53 / 1.0 = 0.53x ✓
```

---

## Why Clipping Doesn't Solve This

Current `pbrs_clip=1.0` means:
- Max per-step PBRS = 1.0
- Over 155 steps, max episode PBRS = **155.0**
- The clip is way too loose to help

Even with `pbrs_clip=0.1`:
- Max episode PBRS = 15.5
- Still 15x the sparse reward

**Clipping is for outlier protection, not scaling control.**

---

## Alternative Solutions Considered

### Option 1: Reduce EPISODE_SCALE (Not Recommended)
- Would require changing internal constants
- Less intuitive to tune
- Breaks existing documentation

### Option 2: Tighter Clipping (Partial Solution)
- `pbrs_clip=0.01` would limit episode PBRS to ~1.5
- But this clips normal transitions too aggressively
- Loses the smooth guidance signal

### Option 3: Annealing (Helps but Doesn't Fix Root Cause)
- Current annealing reduces PBRS over time
- But if PBRS starts at 3.5x sparse, even 50% reduction = 1.75x
- Agent may have already learned bad habits

### Option 4: Reduce pbrs_scale ✓ (Best Solution)
- Simple, direct control
- Maintains smooth gradients
- Preserves relative component weights
- Easy to tune

---

## Recommended Configuration

```bash
python train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 100000 \
    --seed 42 \
    \
    # CRITICAL FIX: Reduce PBRS scale by 10x
    --pbrs_scale 0.003 \
    \
    # Keep cross-court but at lower effective weight
    --pbrs_cross_weight 0.4 \
    \
    # Safety clip (now actually useful)
    --pbrs_clip 0.1 \
    \
    # Other settings
    --reward_scale 0.1 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.9995 \
    --lr_actor 0.001 \
    --lr_critic 0.001
```

### Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Episode PBRS | ~3.5 | ~0.5 |
| PBRS/Sparse Ratio | 3.5x | 0.5x |
| Sparse dominance | No | **Yes** |
| Early learning | Guided | Guided |
| Late learning | PBRS-biased | Sparse-driven |

---

## Summary

**The Problem:**
- `pbrs_scale=0.02` was chosen based on per-step intuition
- But PBRS accumulates over 155 steps
- Result: 3.5x sparse signal

**The Fix:**
- Reduce `pbrs_scale` from 0.02 to **0.002-0.003**
- This makes episode PBRS ~0.3-0.5 (30-50% of sparse)
- PBRS guides, sparse rewards dominate

**The Insight:**
- PBRS scaling must account for **episode length**
- Per-step PBRS should be ~0.002-0.003, not 0.02
- With 155 steps: 0.003 × 155 = 0.47 per episode ✓

---

## Action Items

1. **Immediate:** Change `--pbrs_scale 0.003` in next run
2. **Verify:** Check `pbrs/avg_per_episode` is now ~0.3-0.5
3. **Monitor:** `hacking/pbrs_to_sparse_ratio` should be <1.0
4. **Observe:** Win rate should improve as sparse signals dominate

