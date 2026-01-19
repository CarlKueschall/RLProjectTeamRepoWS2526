# W&B Run Analysis: best-mimic-new-rws

**Run ID:** (not specified in data)
**Duration:** Estimated ~24-30 hours
**Status:** Complete / Stable
**Data Points:** 233 total, 66 shown (discretized)

---

## Executive Summary

**Configuration:** TD3 + PBRS V3.2 (with cross-court, likely with mitigations) | Weak opponent | LR=0.001 | Seed=42

**Overall Assessment:** ✅ **Better Performing Run** - More stable PBRS, improved weak opponent win rate, comparable strong opponent performance

| Metric | Value | Status |
|--------|-------|--------|
| **Win Rate (weak)** | 46.1% | ✅ Good |
| **Win Rate (strong)** | 27.2% | ⚠️ Lower |
| **Cumulative Win Rate** | 18.4% | ✅ Improved |
| **PBRS/Sparse Ratio** | 1.03 | ✅ Stable |
| **Possession Hoarding** | 0.318 | ✅ Better |

---

## Performance Against Opponents

### vs Weak Opponent
- **Win Rate:** 46.1% (range: 3-69%)
  - **Decisive Wins:** 69.9% of wins (excellent!)
  - **Loss Rate:** 19.8% (1-62 losses)
  - **Tie Rate:** 34.0% (15-66 ties)
  - **Avg Reward:** +0.54 (range: -25.1 to +4.9)
  - **Cumulative:** 4-1829 wins throughout training

**✅ Strength:** 46.1% win rate against weak is good; decisive wins at 69.9% show domination

### vs Strong Opponent
- **Win Rate:** 27.2% (range: 3-40%)
  - **Decisive Wins:** 39.75% of wins
  - **Loss Rate:** 40.5% (21-54 losses)
  - **Tie Rate:** 32.3% (20-57 ties)
  - **Avg Reward:** -4.49 (range: -24.8 to -2.2)
  - **Cumulative:** 3-1829 wins

**Observation:** Lower than weak opponent (expected); more conservative play

---

## Agent Behavior

### Puck Interaction
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Dist to Puck (avg)** | 1.90 | 3.91 | 2.59 |
| **Dist to Puck (min)** | 0.17 | 1.47 | 0.51 |
| **Possession Ratio** | 0.32% | 28.5% | 14.7% |
| **Time Near Puck** | 0.7 | 56.4 | 22.0 steps |

**Analysis:** Similar puck pursuit to problematic run; slightly higher possession but better control

### Shooting Behavior
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Shoot Action (avg)** | -0.83 | 0.15 | -0.31 |
| **Shoot When Possess** | -0.87 | 0.17 | **-0.51** |

**Analysis:** Still reluctant to shoot (mean -0.51), but slightly better than problematic run (-0.41)
- Shows improvement from mitigation efforts

### Action Execution
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Action Magnitude (avg)** | 0.96 | 1.33 | 1.13 |
| **Lazy Actions** | 0.0% | 1.1% | 0.21% |
| **Distance Traveled** | 5.98 | 20.4 | 12.3 units |

**Good:** Slightly more aggressive (higher magnitude); very few lazy actions

---

## Reward Shaping Analysis

### PBRS Metrics
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **PBRS/Sparse Ratio** | 0.18 | **6.62** | 1.03 |
| **PBRS per Episode** | 3.45 | 3.83 | 3.65 |
| **Annealing Weight** | 1.0 | 1.0 | 1.0 |

**✅ Improvement:** Max ratio reduced from 12.4 to 6.62 (better, but still high)
- Suggests partial mitigation applied
- Still occasional spikes but less severe

### Reward Composition
| Type | Min | Max | Mean |
|------|-----|-----|------|
| **P1 Reward** | -19.3 | 9.93 | -3.13 |
| **P2 Reward** | -7.24 | 23.3 | 6.60 |
| **Sparse Ratio** | -15.2 | 5.18 | +0.44 |

**✅ Improvement:** Sparse ratio much more stable (mean +0.44, max -15.2 vs -130.5)
- Indicates clipping or weight reduction applied

---

## Learning Dynamics

### Q-Values
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Q_avg** | -3.76 | 0.07 | -2.10 |
| **Q_std** | 0.12 | 2.45 | 1.00 |
| **Q_min** | -7.73 | -0.20 | -4.88 |
| **Q_max** | -2.11 | 9.75 | 1.91 |

**Improvement:** Q_max reduced from 10.1 to 9.75 (modest); Q_avg slightly better

### Gradients
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Actor Grad Norm** | 0.0 | 0.34 | 0.029 |
| **Critic Grad Norm** | 0.0 | 2.46 | 0.66 |

**Improvement:** Both reduced slightly; smoother learning

### Losses
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Actor Loss** | 0.0 | 3.57 | 2.02 |
| **Critic Loss** | 0.0 | 0.10 | 0.032 |

**Status:** Similar to problematic run; actor loss consistent

---

## Regularization & Safety

### VF Regularization
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Active Ratio** | 0.80 | 0.91 | **0.84** |
| **Violation Ratio** | 0.04 | 0.21 | **0.16** |
| **Q Advantage Mean** | 0.0048 | 0.0424 | 0.023 |
| **Reg Loss** | 0.0 | 0.0002 | 0.0001 |

**✅ Improvement:**
- Active ratio higher (0.84 vs 0.80) = better non-lazy detection
- Violation ratio lower (0.16 vs 0.22) = more stable critic
- Q advantage higher (0.023 vs 0.016) = better value separation

---

## Training Progression

### Exploration
- **Epsilon (start):** 1.0
- **Epsilon (current):** 0.16-1.0 (narrower range!)
- **Mean Epsilon:** 0.65 (lower than problematic run)
- **Decay Rate:** 0.9995/step

**✅ Improvement:** Lower mean epsilon indicates better exploitation phase

### Episode Length
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Avg Episode Length** | 70.9 | 222.1 | 155.4 steps |
| **Training Speed** | 0.191 | 7.17 | 2.50 eps/sec |

**Status:** Similar to problematic run; acceptable training speed

---

## Hacking Detection

### Reward Exploitation
| Metric | Value | Assessment |
|--------|-------|-----------|
| **Possession Hoarding** | 0.318 | ✅ Improved (down from 0.36) |
| **No-Shoot While Possess** | 9.59 | ⚠️ Similar (9.57 before) |
| **PBRS/Sparse Ratio** | 1.03 | ✅ Much better (down from 1.70) |

**Improvements:**
1. PBRS much more stable (1.03 vs 1.70)
2. Possession hoarding slightly reduced
3. Comparable no-shoot behavior (suggests different issue)

---

## Comparative Analysis: Cross-Court Problematic vs New RWS

### Performance Comparison

| Metric | Problematic | New RWS | Change |
|--------|------------|---------|--------|
| Win Rate (weak) | 35.7% | 46.1% | **+10.4%** ✅ |
| Win Rate (strong) | 33.6% | 27.2% | -6.4% |
| Cumulative Win Rate | 16.3% | 18.4% | +2.1% ✅ |
| PBRS/Sparse Ratio | 1.70 | 1.03 | **-0.67 (39% reduction)** ✅ |
| Possession Hoarding | 0.363 | 0.318 | **-0.045 (12% reduction)** ✅ |
| Q_avg | -2.19 | -2.10 | +0.09 ✅ |
| Epsilon (mean) | 0.75 | 0.65 | **-0.10 better** ✅ |

### Interpretation

**New RWS ("Mimic New RWS") is clearly better:**
- 10.4 percentage point improvement on weak opponent
- 39% reduction in PBRS volatility
- Lower epsilon = better exploitation
- Slightly lower possession hoarding

**Trade-off:** Slightly worse on strong opponent (-6.4%), but strong epsilon mean -0.10 in this scenario

---

## Likely Differences in Configuration

Based on comparative analysis, the "new RWS" (reward without spikes?) likely:

1. ✅ Uses `--pbrs_clip 1.0` or similar
   - Explains 39% reduction in max PBRS ratio

2. ✅ Possibly reduced `--pbrs_cross_weight` from 0.4 to 0.2
   - Explains lower hoarding and more stable rewards

3. ✅ Possibly adjusted `--pbrs_scale` downward
   - Could contribute to ratio stability

4. ✅ Better `--eps_decay` value
   - Explains lower mean epsilon (0.65 vs 0.75)

---

## Key Insights & Recommendations

### What Changed Successfully
1. **PBRS Stability:** 39% reduction in volatility
2. **Weak Opponent Performance:** +10.4% win rate
3. **Exploitation:** Better epsilon scheduling
4. **Regularization:** Tighter VF violation tracking

### Remaining Issues
1. **Shooting Reluctance:** Still mean -0.51 when possessing
   - Cross-court component may still be problematic
   - Needs defensive analysis (is agent intentionally avoiding)

2. **Strong Opponent Drop:** 6.4% lower win rate
   - May be due to reduced cross-court weight
   - Trade-off acceptable if weak performance is priority

3. **Possession Hoarding:** Still 0.318 (target <0.2)
   - Suggests agent still holding puck too long
   - May need dedicated anti-hoarding component

### Recommended Configuration for Production
```bash
python train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 100000 \
    --seed 42 \
    --reward_shaping \
    --pbrs_scale 0.02 \
    --pbrs_cross_weight 0.2 \
    --pbrs_clip 1.0 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.999 \
    --lr_actor 0.001 \
    --lr_critic 0.001
```

---

## Performance Grade

**Problematic Run:** C+ (35.7% win vs weak, high volatility)
**New RWS Run:** B+ (46.1% win vs weak, stable PBRS, good weak-opponent focus)

**Verdict:** New RWS represents meaningful progress. Further improvements would require:
1. Solving puck hoarding (add possession-time penalty)
2. Improving strong opponent play without sacrificing weak
3. Better cross-court incentive (may need different formula)

