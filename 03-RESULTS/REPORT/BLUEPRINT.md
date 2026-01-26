# DreamerV3 Hockey Report Blueprint

## Executive Summary

**Team:** Carl Kueschall + Serhat Alpay (2 people)
**Page Limit:** 8 pages (excluding references and appendix)
**Benchmark Performance:** 88.5% combined win rate (87% weak, 90% strong)
**Algorithm:** DreamerV3 (World-Model Based RL)

## Current Status

| Item | Status |
|------|--------|
| LaTeX skeleton | ✅ Complete (`INPUT/main.tex`) |
| Bibliography | ✅ Complete (`INPUT/main.bib`) |
| Carl's Methods section | ✅ Drafted |
| Carl's Experiments section | ✅ Skeleton with placeholders |
| Serhat's sections | 🔲 TODO (placeholders) |
| Placeholder figures | ✅ Generated |
| Ablation scripts | ✅ Ready (`run_ablations.sh`) |
| Ablation runs | 🔲 TODO |
| Real figures from ablations | 🔲 TODO |

---

## Project Requirements Checklist (from project.md)

- [x] Introduction with problem description (~0.5 page)
- [ ] Methods section with implementation details + math (~1+ page per person)
- [ ] Experimental evaluation with performance vs basic opponent (~1+ page per person)
- [ ] Final discussion comparing algorithms (~1 page)
- [ ] Each member's contribution clearly marked
- [ ] AI usage declaration

---

## Page Budget (8 Pages Total)

| Section | Pages | Owner |
|---------|-------|-------|
| Introduction | 0.5 | Shared |
| Methods - DreamerV3 (Carl) | 1.5 | Carl |
| Methods - [Serhat's Algorithm] | 1.5 | Serhat |
| Experiments - DreamerV3 | 1.5 | Carl |
| Experiments - [Serhat's] | 1.0 | Serhat |
| Discussion | 1.0 | Shared |
| AI Usage Declaration | 0.25 | Shared |
| **Total** | **~7.25** | - |

Buffer: ~0.75 pages for figures/tables that flow into text.

---

## Section-by-Section Content Plan

### 1. Introduction (0.5 page, SHARED)

**Content:**
- Hockey environment overview (18D state, 4D action, 250-step episodes)
- Challenge: sparse rewards (+10/-10 goals only), continuous control, adversarial setting
- Our approach: DreamerV3 world-model-based RL with 3-phase training curriculum
- Contributions summary:
  - **Carl:** DreamerV3 implementation with Two-Hot Symlog, DreamSmooth, auxiliary tasks, mixed opponents, self-play, 3-phase training curriculum
  - **Serhat:** [TBD - likely different algorithm or significant modification]
- Final result: 88.5% combined win rate (benchmark performance)

**Key citations needed:**
- Hockey environment [martius-lab]
- DreamerV3 paper [Hafner et al., 2024]
- NaturalDreamer codebase (starting point)

---

### 2. Methods - DreamerV3 Implementation (1.5 pages, CARL)

**2.1 Starting Point & Background (0.25 page)**
- Based on NaturalDreamer codebase: https://github.com/InexperiencedMe/NaturalDreamer
- DreamerV3 philosophy: Learn world model, train policy entirely in imagination
- Why world-model for hockey: Sample efficiency, handles sparse rewards through imagined rollouts

**2.2 World Model Architecture (0.4 page)**
- RSSM: Deterministic (GRU, 256-dim) + Stochastic (16×16 categorical = 256-dim)
- Encoder/Decoder: MLP for 18-dim observations
- Reward Model: **Two-Hot Symlog** (critical for sparse rewards)
- Continue Model: Bernoulli for episode termination
- Include architecture diagram (Figure 1)

**2.3 Key Modifications from Baseline (0.5 page)**

| Modification | Problem Solved | Implementation |
|--------------|---------------|----------------|
| **Two-Hot Symlog** | Sparse reward prediction | 255 bins, symlog space for rewards/values |
| **DreamSmooth** | Temporal credit assignment | Exponential smoothing α=0.5 on rewards |
| **Auxiliary Tasks** | Better latent representations | Goal prediction, distance, shot quality heads |
| **Mixed Opponents** | Prevent overfitting | 50% weak / 50% strong during training |
| **Self-Play + PFSP** | Diverse opponent strategies | Pool of 15 checkpoints, variance-mode selection |

**2.4 Behavior Learning (0.35 page)**
- Actor: Tanh-squashed Gaussian, min_std=0.1
- Critic: Two-Hot Symlog value prediction
- Slow Critic: EMA target (decay=0.98) for stable bootstrapping
- Lambda returns: TD(λ) with λ=0.95
- Entropy regularization: η=3e-4 (fixed, no annealing)
- Percentile-based value normalization (5th/95th percentile)

**Equations to include:**
- RSSM dynamics
- Two-Hot Symlog encoding
- Lambda return computation
- Actor loss with entropy
- DreamSmooth temporal smoothing

---

### 3. Methods - [Serhat's Algorithm] (1.5 pages, SERHAT)

**Options:**
1. Different algorithm entirely (SAC, TD-MPC, etc.)
2. Significant DreamerV3 modification (different auxiliary tasks, different architecture)
3. Alternative training approach

**Note:** Per project.md, each team member should have one algorithm or significant modification implemented.

---

### 4. Experiments - DreamerV3 (1.5 pages, CARL)

**4.1 Training Curriculum: 3-Phase Approach (0.3 page)**

| Phase | Duration | Configuration | Result |
|-------|----------|---------------|--------|
| Phase 1 | 30h | Mixed + Self-play (RR=32) | 0→72% win rate |
| Phase 2 | 8h | Mixed only (RR=16) | 72→85% win rate |
| Phase 3 | 16h | Fine-tuning (RR=4, lower LR) | **88.5% final** |

Include: Training curve figure showing all 3 phases

**4.2 Ablation Studies (0.6 page)**

**CRITICAL: We need to run these ablations!**

| Ablation | Hypothesis | Status |
|----------|------------|--------|
| DreamSmooth ON vs OFF | Essential for sparse rewards | Need to run |
| Two-Hot vs MSE rewards | Two-Hot handles goals better | Need to run |
| Auxiliary tasks ON vs OFF | Improves world model | Need to run |
| Self-play vs No self-play | Adds robustness | Can compare Phase 1 vs Phase 2 |
| Replay ratio 32 vs 16 vs 4 | Higher = faster early, lower = better fine-tune | Need to run |

**Recommended ablations to prioritize (pick 2-3):**
1. **DreamSmooth ablation** - Most impactful, easy to run
2. **Auxiliary tasks ablation** - Shows value of our additions
3. **Mixed opponents vs single opponent** - Shows training diversity benefit

**4.3 Benchmark Evaluation (0.3 page)**

| Metric | vs Weak | vs Strong | Combined |
|--------|---------|-----------|----------|
| Win Rate | 87% | 90% | **88.5%** |
| Tie Rate | ~5% | ~3% | ~4% |
| Loss Rate | ~8% | ~7% | ~7.5% |

Include: Performance bar chart, comparison table

**4.4 Training Analysis (0.3 page)**
- World model loss convergence
- Entropy evolution (should stay positive)
- Win rate progression
- Gradient step vs episode relationship
- Include: Key W&B metrics plots

---

### 5. Experiments - [Serhat's] (1.0 page, SERHAT)

Same structure as above for Serhat's algorithm.

---

### 6. Discussion (1.0 page, SHARED)

**6.1 Key Findings - DreamerV3 (Carl, 0.3 page)**
- Two-Hot Symlog critical for sparse reward handling
- DreamSmooth significantly accelerates learning
- 3-phase curriculum effective: bootstrap → stabilize → fine-tune
- Mixed opponents prevent overfitting to single strategy
- Self-play provides modest but meaningful improvements

**6.2 Key Findings - [Serhat's] (0.3 page)**

**6.3 Algorithm Comparison (0.2 page)**
- Compare DreamerV3 vs [Serhat's algorithm]
- Sample efficiency, final performance, training stability
- Recommendation for future work

**6.4 Limitations (0.1 page)**
- Evaluation limited to weak/strong bots (tournament will reveal true diversity)
- 54 hours total compute (significant but feasible)
- Ablations still being run

**6.5 Future Work (0.1 page)**
- Population-based training
- Model-based planning during evaluation
- Transfer to different game variants

---

### 7. AI Usage Declaration (0.25 page, SHARED)

**Carl Kueschall:**
- **IDE Autocomplete:** AI-powered code completion used throughout development (Cursor AI IDE)
- **Claude Code (claude.ai/code):** Used for:
  1. **Learning & Understanding:** Deep discussions about DreamerV3 internals, debugging complex RL concepts, understanding world model training dynamics
  2. **Mundane Tasks:** Writing plotting scripts, adding/removing W&B metrics, debugging tensor shape mismatches, creating test scripts
- **Philosophy:** Sequential, step-by-step development to ensure genuine understanding of the system, while leveraging AI to accelerate learning and handle routine tasks that don't enhance knowledge

**Serhat Alpay:**
- [TBD]

---

## Figures Plan

| Figure | Description | Section |
|--------|-------------|---------|
| Fig 1 | DreamerV3 Architecture Diagram | Methods |
| Fig 2 | RSSM Dynamics Visualization | Methods |
| Fig 3 | Two-Hot Symlog Encoding | Methods |
| Fig 4 | 3-Phase Training Curve | Experiments |
| Fig 5 | Ablation Results (2-3 subplots) | Experiments |
| Fig 6 | Benchmark Performance Bar Chart | Experiments |
| Fig 7 | W&B Key Metrics (entropy, loss, etc.) | Experiments |

---

## Tables Plan

| Table | Description | Section |
|-------|-------------|---------|
| Tab 1 | Key Hyperparameters | Methods |
| Tab 2 | Modifications from Baseline | Methods |
| Tab 3 | 3-Phase Training Summary | Experiments |
| Tab 4 | Ablation Results | Experiments |
| Tab 5 | Final Benchmark Results | Experiments |
| Tab 6 | Algorithm Comparison | Discussion |

---

## Ablation Study Recommendations

**Priority 1 (Must Have):**
1. **DreamSmooth ON vs OFF** - Train same config with `--use_dreamsmooth` vs without
   - Expected: DreamSmooth significantly faster convergence, better final performance
   - Time: ~10-15 hours each

2. **Auxiliary Tasks ON vs OFF** - Train with vs without goal/distance/quality heads
   - Expected: Auxiliary tasks improve world model representations
   - Time: ~10-15 hours each

**Priority 2 (Nice to Have):**
3. **Replay Ratio Comparison** - Compare RR=32 vs RR=16 vs RR=8
   - Expected: Higher RR faster early, but diminishing returns
   - Time: ~10 hours each

4. **Mixed vs Single Opponent** - Compare mixed (weak+strong) vs weak-only
   - Expected: Mixed training more robust
   - Time: ~15 hours each

**Note:** Given time constraints, recommend running Priority 1 ablations only.

---

## Timeline

| Task | Deadline | Owner |
|------|----------|-------|
| Finalize ablation list | Jan 27 | Both |
| Start ablation runs | Jan 27 | Carl |
| Draft Methods section | Jan 30 | Both |
| Draft Experiments section | Feb 2 | Both |
| Complete ablation runs | Feb 5 | Carl |
| Draft Discussion | Feb 7 | Both |
| Final polish | Feb 10 | Both |
| **Submission** | **Feb 27** | Both |

---

## Key References (BibTeX needed)

1. Hafner et al. (2024) - DreamerV3: Mastering Diverse Domains through World Models
2. Hafner et al. (2020) - Dream to Control (DreamerV1)
3. Wu et al. (2023) - DreamSmooth: Temporal Reward Smoothing
4. Martius Lab - Hockey Environment
5. NaturalDreamer codebase - Starting point
6. Silver et al. - AlphaStar (for self-play/PFSP context)

---

## Checklist Before Submission

- [ ] All ablation runs completed
- [ ] All figures generated and polished
- [ ] All tables filled with actual data
- [ ] Both team members' contributions clearly marked
- [ ] AI usage declaration complete
- [ ] Page count ≤ 8 (excluding references)
- [ ] All claims supported by evidence
- [ ] Proofreading complete
- [ ] BibTeX references working
- [ ] PDF compiles without errors
