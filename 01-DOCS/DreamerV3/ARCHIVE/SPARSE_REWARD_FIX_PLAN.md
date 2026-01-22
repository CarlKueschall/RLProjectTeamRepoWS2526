# DreamerV3 Sparse Reward Fix Plan

**Date:** 2026-01-21
**Status:** Phase 1 & 3 COMPLETE - Ready for validation experiments
**Core Insight:** "The issue isn't how to add dense rewards—it's why DreamerV3's existing sparse reward machinery isn't working."

---

## Phase 1: Verify Robustness Stack (CRITICAL) ✅ COMPLETE

DreamerV3 has built-in mechanisms for sparse rewards. We verified ALL and fixed missing pieces.

### 1.1 Return Normalization ✅

**What it does:** Scales advantages using 5th-95th percentile range, preventing collapse when reward std ≈ 0.

**Audit result:**
- [x] `Moments` class uses percentile-based normalization (5th-95th percentile)
- [x] `min_` parameter = 0.01 (prevents division by zero)
- [x] Normalization is ALWAYS ON (not optional)

**File:** `utils.py:348-365`

### 1.2 Two-Hot Symlog Encoding ✅

**What it does:** Discretizes rewards/values into bins, handling multi-modal distributions (0 vs ±10).

**Audit result:**
- [x] `TwoHotSymlog` used for BOTH reward predictor AND critic
- [x] Bins: 255 bins, -20 to +20 in symlog space
- [x] `symlog` transform applied to observations before encoding

**Files:** `utils.py:221-345`, `networks.py:86-114`, `networks.py:206-237`

### 1.3 Zero Initialization ✅ FIXED

**What it does:** Output layers start at 0, preventing hallucinated rewards that delay real signal discovery.

**Audit result:**
- [x] Critic output layer: Already had zero initialization
- [x] Reward predictor output layer: **ADDED** zero initialization (was missing!)

**Fix applied:** Added zero initialization to `RewardModel.__init__()` in `networks.py:105-110`

### 1.4 EMA Critic Regularizer ✅ IMPLEMENTED

**What it does:** Stabilizes critic bootstrap by using exponential moving average of critic weights.

**Audit result:** WAS NOT IMPLEMENTED - this was a key missing piece!

**Fix applied:**
- Added `slowCritic` (EMA copy of critic) in `dreamer.py:75-81`
- Added `_updateSlowCritic()` method for EMA updates (`dreamer.py:147-151`)
- Use slow critic for lambda return bootstrap targets (`dreamer.py:525-530`)
- Update slow critic after each critic optimization step (`dreamer.py:570-571`)
- Added to checkpoint save/load (`dreamer.py:734`, `dreamer.py:769-773`)
- Config: `slowCriticDecay: 0.98` (default)

### 1.5 Free Bits / KL Balance ✅

**What it does:** Prevents dynamics/representation collapse, keeps info-rich latents.

**Audit result:**
- [x] `freeNats: 1.0` threshold applied to KL loss
- [x] `betaPrior: 1.0`, `betaPosterior: 0.1` balance the KL terms

**File:** `dreamer.py:243-247`

---

## Phase 2: Validate World Model Quality

Before adding any interventions, confirm the world model is learning correctly.

### 2.1 Metrics to Check

| Metric | Healthy Range | Concern If |
|--------|---------------|------------|
| `world/recon_loss` | Decreasing | Flat or increasing |
| `world/reward_loss` | Decreasing | Flat after initial drop |
| `world/kl_loss` | Stable around free_nats | Collapsing to 0 or exploding |
| `sparse_signal/sparse_sign_accuracy` | ~1.0 | < 0.8 (can't predict reward sign) |
| `world/latent_entropy` | > 0.5 | Near 0 (posterior collapse) |

### 2.2 Diagnostic Questions

- [ ] Is the reward predictor learning to predict sparse events? (Check `sparse_signal/sparse_pred_error`)
- [ ] Is the world model seeing enough sparse events? (Check `sparse_signal/event_rate_in_batch`)
- [ ] Are latent states diverse? (Check `world/latent_entropy`)

---

## Phase 3: Implement DreamSmooth ✅ COMPLETE

**Priority:** HIGH
**Theoretical soundness:** Proven, no distribution shift concerns
**Implementation effort:** Low

### 3.1 What DreamSmooth Does

Smooths rewards when sampling from replay buffer:
- Instead of sparse spikes (0, 0, 0, +10, 0, 0), the model sees (0.41, 0.82, 1.64, 3.28, 1.56, 0.62)
- Reward prediction becomes easier
- Denser signal emerges naturally without explicit shaping
- Respects episode boundaries (resets smoothing at done=1)

### 3.2 Implementation ✅

**Files modified:**
- `buffer.py`: Added `dreamsmooth_ema()` function and buffer integration
- `train_hockey.py`: Added CLI arguments
- `configs/hockey.yml`: Added configuration documentation

**Function:** `dreamsmooth_ema(rewards, dones, alpha=0.5)`
- Bidirectional EMA smoothing within episodes
- Respects episode boundaries (resets at done flags)
- Applied when sampling from buffer (not when storing)

### 3.3 CLI Arguments ✅

- `--use_dreamsmooth` (flag to enable)
- `--dreamsmooth_alpha` (EMA alpha, default: 0.5)

### 3.4 Usage

```bash
# Enable DreamSmooth for sparse reward training
python train_hockey.py --opponent weak --use_dreamsmooth --dreamsmooth_alpha 0.5
```

### 3.5 Key Insight

DreamSmooth changes what the world model LEARNS, not what the actor SEES in imagination. The world model learns to predict smoothed rewards, so imagination naturally produces denser signal.

---

## Phase 4: Disable/Remove Auxiliary Reward System

**Priority:** HIGH
**Reason:** Research indicates this approach has fundamental issues (distribution shift, value-actor mismatch)

### 4.1 What to Keep

- [ ] Auxiliary TASK HEADS (goalPredictor, distPredictor, shotQualityPredictor) - useful for world model representation learning
- [ ] Auxiliary LOSSES in world model training - helps encoder learn goal-relevant features

### 4.2 What to Remove/Disable

- [ ] Remove `useAuxRewardsInImagination` config option
- [ ] Remove `_compute_aux_rewards_imagination()` method (if implemented)
- [ ] Remove any code that adds aux rewards to `predictedRewards` in `behaviorTraining()`

### 4.3 CLI Cleanup

Remove these arguments (if added):
- `--use_aux_rewards_imagination`
- `--aux_goal_reward_scale`
- `--aux_dist_reward_scale`
- `--aux_shot_reward_scale`

---

## Phase 5: Run Validation Experiments

### 5.1 Experiment A: Robustness Stack Verification

```bash
python train_hockey.py \
    --opponent weak \
    --seed 42 \
    --gradient_steps 100000 \
    --entropy_scale 0.003 \
    --no_wandb
```

**Monitor:** All robustness metrics from Phase 2.1

### 5.2 Experiment B: DreamSmooth

```bash
python train_hockey.py \
    --opponent weak \
    --seed 42 \
    --gradient_steps 100000 \
    --use_dreamsmooth \
    --dreamsmooth_window 5 \
    --entropy_scale 0.003
```

**Compare to Experiment A:** Does `imagination/reward_mean` become non-zero? Does entropy decrease?

### 5.3 Success Criteria

| Metric | Target |
|--------|--------|
| `behavior/entropy_mean` | < 12 and decreasing |
| `imagination/reward_mean` | > 0.1 (with DreamSmooth) |
| `behavior/advantages_abs_mean` | > 0.5 |
| `stats/win_rate` | > 50% and increasing |

---

## Implementation Order

```
Week 1:
├── Day 1-2: Phase 1 - Audit robustness stack
│   ├── Check all 5 mechanisms
│   ├── Fix any missing pieces (especially EMA critic)
│   └── Re-enable advantage normalization
├── Day 3: Phase 2 - Validate world model
│   └── Run diagnostic experiment, check metrics
├── Day 4-5: Phase 3 - Implement DreamSmooth
│   ├── Add smoothing function
│   ├── Add CLI arguments
│   └── Test locally
└── Day 6-7: Phase 4 - Clean up aux rewards code
    └── Remove/disable imagination aux rewards

Week 2:
└── Phase 5 - Run validation experiments on cluster
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `utils.py` | Verify Moments normalization, keep min_=0.01 |
| `dreamer.py` | Remove aux rewards in imagination, verify robustness |
| `networks.py` | Verify zero initialization for reward predictor |
| `buffer.py` | Add DreamSmooth reward smoothing |
| `train_hockey.py` | Add DreamSmooth CLI args, remove aux reward args |

---

## Reference

- DreamSmooth paper: arXiv:2311.01450
- DreamerV3 paper: Nature 2025, arXiv:2301.04104
- Full research report: `02-SRC/DreamerV3/RESEARCH_PROMPT.md`

---

## Notes

The auxiliary task HEADS are still valuable for:
1. World model representation learning (multi-task loss)
2. Debugging/visualization (what does the model think is happening?)
3. Potential future use in goal-conditioned variants

They just shouldn't be used to shape rewards in imagination due to posterior/prior distribution shift.
