# W&B Run Analysis: best-cross-court-problematic

**Run ID:** xw6cfpjm
**Duration:** 41,253 seconds (~11.5 hours)
**Status:** Running (incomplete)
**Data Points:** 233 total, 66 shown (discretized)

---

## Executive Summary

**Configuration:** TD3 + PBRS V3.2 (with cross-court) | Weak opponent | LR=0.001 | Seed=42

**Overall Assessment:** ⚠️ **Problematic Run** - High PBRS volatility, poor weak opponent win rate, moderate strong opponent performance

| Metric | Value | Status |
|--------|-------|--------|
| **Win Rate (weak)** | 35.7% | ❌ Low |
| **Win Rate (strong)** | 33.6% | ⚠️ Moderate |
| **Cumulative Win Rate** | 16.3% | ❌ Poor |
| **PBRS/Sparse Ratio** | 1.70 | ⚠️ Elevated |
| **Possession Hoarding** | 0.363 | ⚠️ High |

---

## Performance Against Opponents

### vs Weak Opponent
- **Win Rate:** 35.7% (range: 2-54%)
  - **Decisive Wins:** 53.5% of wins
  - **Loss Rate:** 31.2% (15-63 losses)
  - **Tie Rate:** 33.2% (15-56 ties)
  - **Avg Reward:** -1.59 (range: -25.6 to +2.5)
  - **Cumulative:** 2-2862 wins throughout training

**Issue:** High tie rate (33.2%) despite being vs weak opponent

### vs Strong Opponent
- **Win Rate:** 33.6% (range: 7-51%)
  - **Decisive Wins:** 44.1% of wins
  - **Loss Rate:** 42.2% (27-62 losses)
  - **Tie Rate:** 24.1% (12-62 ties)
  - **Avg Reward:** -3.92 (range: -24.1 to +0.2)
  - **Cumulative:** 7-2862 wins

**Observation:** Slightly lower than weak; balanced loss/tie ratio

---

## Agent Behavior

### Puck Interaction
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Dist to Puck (avg)** | 1.96 | 3.91 | 2.69 |
| **Dist to Puck (min)** | 0.19 | 1.47 | 0.55 |
| **Possession Ratio** | 0.18% | 28.7% | 11.7% |
| **Time Near Puck** | 3.0 | 43.8 | 16.9 steps |

**Analysis:** Decent puck pursuit; variable possession patterns

### Shooting Behavior
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Shoot Action (avg)** | -0.64 | 0.62 | -0.03 |
| **Shoot When Possess** | -0.76 | 0.43 | **-0.41** |

**⚠️ Critical Issue:** Negative shoot-when-possess = **reluctant to shoot**

### Action Execution
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Action Magnitude (avg)** | 0.96 | 1.21 | 1.09 |
| **Lazy Actions** | 0.0% | 1.1% | 0.23% |
| **Distance Traveled** | 6.24 | 18.3 | 12.3 units |

**Good:** Active movement, very few lazy actions

---

## Reward Shaping Analysis

### PBRS Metrics
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **PBRS/Sparse Ratio** | 0.18 | **12.40** | 1.70 |
| **PBRS per Episode** | 3.30 | 3.78 | 3.53 |
| **Annealing Weight** | 1.0 | 1.0 | 1.0 |

**🚨 Problem:** Max ratio of 12.4 is extremely high (should be < 1.0)
- Indicates cross-court component is creating large reward spikes
- Cross-court bonus may be too aggressive (W_CROSS=0.4?)

### Reward Composition
| Type | Min | Max | Mean |
|------|-----|-----|------|
| **P1 Reward** | -19.4 | 7.98 | -4.96 |
| **P2 Reward** | -4.94 | 23.3 | 8.35 |
| **Sparse Ratio** | -130.5 | 46.8 | -0.95 |

**Issue:** Sparse_ratio of -130.5 suggests extreme outlier (likely full episode loss)

---

## Learning Dynamics

### Q-Values
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Q_avg** | -3.84 | 0.07 | -2.19 |
| **Q_std** | 0.12 | 2.42 | 1.01 |
| **Q_min** | -7.55 | -0.20 | -5.11 |
| **Q_max** | -1.97 | 10.13 | 2.20 |

**Observation:** Wide Q-value spread (Q_max up to 10.1); potential overestimation

### Gradients
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Actor Grad Norm** | 0.0 | 0.54 | 0.033 |
| **Critic Grad Norm** | 0.0 | 3.15 | 0.73 |

**Status:** Reasonable gradient flow; critic stronger than actor

### Losses
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Actor Loss** | 0.0 | 3.41 | 1.85 |
| **Critic Loss** | 0.0 | 0.12 | 0.031 |

**Observation:** Actor loss elevated; some zero-loss episodes

---

## Regularization & Safety

### VF Regularization
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Active Ratio** | 0.72 | 0.91 | 0.80 |
| **Violation Ratio** | 0.05 | 0.31 | **0.22** |
| **Q Advantage Mean** | 0.0049 | 0.0495 | 0.016 |
| **Reg Loss** | 0.0 | 0.0003 | 0.0002 |

**Analysis:**
- Active ratio ~80% = most states are non-lazy
- Violation ratio 22% = Q-function occasionally prefers heuristic (acceptable)
- Reg loss negligible = VF reg not strongly active

---

## Training Progression

### Exploration
- **Epsilon (start):** 1.0
- **Epsilon (current):** 0.46-1.0 (wide range)
- **Mean Epsilon:** 0.75
- **Decay Rate:** 0.9995/step

**Issue:** Epsilon still quite high (mean 0.75) = limited exploitation

### Episode Length
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Avg Episode Length** | 75.5 | 221.9 | 154.3 steps |
| **Training Speed** | 0.185 | 6.65 | 2.44 eps/sec |

**Good:** Reasonable episode length; acceptable training speed

---

## Hacking Detection

### Reward Exploitation
| Metric | Value | Assessment |
|--------|-------|-----------|
| **Possession Hoarding** | 0.36 | ⚠️ Moderate (should be <0.2) |
| **No-Shoot While Possess** | 9.57 | ⚠️ High (should be <5) |
| **PBRS/Sparse Ratio** | 1.70 | ⚠️ Elevated |

**Concerns:**
1. Agent holds puck too long without shooting
2. PBRS spikes create training instability
3. Cross-court bonus may be creating exploitable patterns

---

## Key Issues & Recommendations

### 🚨 Critical Issues
1. **PBRS Volatility:** Max ratio of 12.4 is far too high
   - **Fix:** Add `--pbrs_clip 1.0` to stabilize rewards
   - **Alternative:** Reduce `--pbrs_cross_weight 0.2` (currently 0.4)

2. **Puck Hoarding:** Agent reluctant to shoot (mean shoot-when-possess = -0.41)
   - **Cause:** Cross-court component may penalize immediate shooting
   - **Fix:** Verify cross-court math not incentivizing dribbling

3. **Low Win Rate vs Weak:** Only 35.7% despite being weak opponent
   - **Cause:** High tie rate (33%) suggests conservative play
   - **Fix:** May improve with PBRS clipping

### ⚠️ Secondary Issues
1. **High Epsilon:** Still exploring heavily at mean 0.75
   - Acceptable for ongoing training, but limits exploitation

2. **Q-Value Spread:** Q_max reaches 10.1 (possible overestimation)
   - Consider tighter `--q_clip` value

### ✅ What's Working
1. Active movement (low lazy ratio: 0.23%)
2. Reasonable gradient flow
3. VF regularization active but not intrusive (loss negligible)
4. Training speed acceptable

---

## Comparison Context

**Expected Metrics (Healthy Run):**
- Win rate vs weak: 45-55%
- Win rate vs strong: 35-45%
- PBRS/Sparse ratio: 0.3-0.8
- Possession hoarding: 0.15-0.25
- Epsilon (early): 0.9-1.0, (late): 0.3-0.5

**This Run:** Underperforming on win rate and PBRS stability

---

## Recommendations for Next Run

```bash
# Add PBRS clipping to prevent explosions
--pbrs_clip 1.0

# Reduce cross-court weight to avoid over-incentivizing
--pbrs_cross_weight 0.2

# Tighter Q-value clipping
--q_clip 20.0

# Slightly increase epsilon decay to exploit learned policy
--eps_decay 0.999
```

