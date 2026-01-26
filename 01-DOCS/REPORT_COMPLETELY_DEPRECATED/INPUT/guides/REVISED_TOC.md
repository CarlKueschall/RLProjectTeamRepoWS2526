# Revised Report Table of Contents

Based on available WandB experiments:

- 256 vs 512 vs 1024 Hidden Size
- RS vs No-RS
- Strong vs Weak Training Opponent
- Baseline-vs-Strong vs. Self-Play

---

## REPORT STRUCTURE (8 Pages Maximum)

```
┌─────────────────────────────────────────────────────────────┐
│ PAGE 1: Title Page + Table of Contents                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 1.5-2: Introduction (0.25 pages, SHARED)               │
│   • Problem statement: Hockey environment                   │
│   • Research goals: TD3 + SAC with self-play                │
│   • Contributions bullet points                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 2-2.5: Methods - TD3 (0.5 pages, CARL)                │
│   • TD3 algorithm overview                                  │
│   • Network architecture (18D→256×256→4D)                   │
│   • 4 Domain modifications:                                │
│     - Q-clipping [-25, 25]                                  │
│     - Anti-lazy learning regularization                    │
│     - Dual replay buffers                                   │
│     - PBRS (α=0.005)                                        │
│   [DONE - minimal prose, already written]                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 2.5-2.75: Methods - SAC (0.25 pages, SERHAT)          │
│   [To be completed by Serhat]                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 2.75-3.5: Experiments - TD3 (0.75 pages, CARL)        │
│                                                              │
│   • Training Curriculum (3 sentences)                       │
│     - Phase 1: Weak (0-10.25k)                             │
│     - Phase 2: Strong (10.25-27.5k)                        │
│     - Phase 3: Self-Play (27.5-97.5k)                      │
│                                                              │
│   • Ablation Studies                                        │
│                                                              │
│     Ablation 1: Network Architecture                        │
│     [TODO: Figure 1, Panel A]                               │
│     - Architecture scaling: 256 vs 512 vs 1024 hidden      │
│     - Training curves comparison                            │
│                                                              │
│     Ablation 2: Reward Shaping                              │
│     [TODO: Figure 1, Panel B]                               │
│     - RS (PBRS + strategic bonuses) vs No-RS (sparse only) │
│     - Training curves comparison                            │
│                                                              │
│     [TODO: Table 1]                                         │
│     - Ablation results summary:                            │
│       • Architecture: 256, 512, 1024 metrics                │
│       • Reward Shaping: No-RS, RS metrics                   │
│                                                              │
│   • Training Opponent Comparison                            │
│     - Strong vs Weak training impact                       │
│     [Optional figure if data available]                      │
│                                                              │
│   • Self-Play vs Baseline Comparison                        │
│     [TODO: Table 2]                                         │
│     - Baseline (Strong) vs Self-Play metrics                │
│                                                              │
│   • Final Evaluation (Table already in LaTeX)              │
│     - 100 games vs weak/strong at 27.5k and 97.5k          │
│     - Shows 88%→92% improvement                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 3.5-4: Experiments - SAC (0.5 pages, SERHAT)          │
│   [To be completed by Serhat]                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 4-4.5: Self-Play Training (0.5 pages, CARL)           │
│                                                              │
│   • Pool Management (bullet points)                          │
│     - 12-agent pool, 400-episode checkpoints                │
│     - 40% weak + 60% pool mix                              │
│                                                              │
│   • PFSP Curriculum (bullet points)                         │
│     - Weighting: w_i = wr_i(1-wr_i)                        │
│     - Favors ~50% win-rate opponents                       │
│                                                              │
│   • Anti-Forgetting Mechanisms (bullet points)             │
│     - Dual buffers (1/3 anchor, 2/3 pool)                    │
│     - Dynamic mixing (ratio → 0.70 if drop >10%)           │
│     - Performance gates (>80% win-rate)                    │
│     - Regression rollback                                   │
│                                                              │
│   • Results Summary (1 sentence)                           │
│     - 88%→92% improvement over 70k episodes                │
│                                                              │
│   • PFSP Explanation (text)                                 │
│     - Weighting function: $w_i = \text{wr}_i(1-\text{wr}_i)$ │
│     - Favors ~50% win-rate opponents                       │
│                                                              │
│   [Figure 4]                                                │
│   - Self-play vs baseline comparison (2 subplots)           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 4.5-5: Discussion (0.5 pages, SHARED)                 │
│                                                              │
│   • TD3 Key Findings (2-3 sentences)                        │
│     - Domain modifications essential                       │
│     - 92% vs weak, 100% vs strong                           │
│     - Self-play improves robustness                         │
│                                                              │
│   • SAC Key Findings (2-3 sentences, SERHAT)               │
│     [To be completed]                                       │
│                                                              │
│   • Algorithm Comparison (1-2 sentences)                    │
│     [Optional table if SAC data available]                 │
│                                                              │
│   • Future Work (1-2 sentences)                             │
│     - League play, model-based approaches, etc.            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 5-5.75: Appendix                                       │
│   • Reward Shaping Components Table                        │
│     - PBRS components and strategic bonuses                 │
│                                                              │
│ PAGE 5.75-6.5: References (BibTeX, ~0.75 pages)            │
│   [Auto-generated from main.bib]                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PAGE 5.75-8: FIGURES AND TABLES (2.25 pages)                │
│                                                              │
│   Figure 1: Ablation Studies (2-panel)                      │
│     - Architecture: 256 vs 512 vs 1024                      │
│     - Reward Shaping: RS vs No-RS                           │
│                                                              │
│   Table 1: Ablation Results Summary                         │
│     - Architecture + Reward Shaping metrics                 │
│                                                              │
│   Table 2: Self-Play vs Baseline Comparison                │
│     - Baseline vs Self-Play final metrics                   │
│                                                              │
│   Figure 2: Pool Evolution Timeline                        │
│     - Checkpoint composition over time                      │
│                                                              │
│   Figure 3: PFSP Weighting Curve                            │
│     - w = wr(1-wr) visualization                            │
│                                                              │
│   [Optional] Figure 4: Training Opponent Comparison         │
│     - Strong vs Weak training curves                        │
└─────────────────────────────────────────────────────────────┘
```

---

## KEY CHANGES FROM ORIGINAL GUIDES

### Removed:

- ❌ Learning rate ablation (not available, not wanted)
- ❌ Exploration decay ablation (not in available experiments)
- ❌ TD3 vs DDPG comparison (not in available experiments)

### Kept/Updated:

- ✅ Network architecture ablation (256 vs 512 vs 1024) - **AVAILABLE**
- ✅ Reward shaping ablation (RS vs No-RS) - **AVAILABLE**
- ✅ Training opponent comparison (Strong vs Weak) - **AVAILABLE**
- ✅ Self-play vs baseline comparison - **AVAILABLE**

---

## EXPERIMENTAL COVERAGE

### Ablation Studies (2):

**Ablation 1: Network Architecture Scaling**

- **Configurations**: 256 vs 512 vs 1024 hidden units
- **Metrics**: Training curves, convergence speed, final performance
- **Analysis**: Computational cost vs performance trade-off
- **Figure**: Panel A of Figure 1 (3 training curves)

**Ablation 2: Reward Shaping Impact**

- **Configurations**:
  - RS: PBRS (α=0.005) + strategic bonuses
  - No-RS: Sparse rewards only (goal ±1)
- **Metrics**: Training curves, convergence acceleration, final performance
- **Analysis**: How much does reward shaping help?
- **Figure**: Panel B of Figure 1 (2 training curves)

### Training Comparisons (2):

3. **Training Opponent**: Strong vs Weak opponent training

   - Generalization analysis
   - Performance against both opponent types
4. **Self-Play**: Baseline (Strong training) vs Self-Play

   - Robustness improvement (88%→92%)
   - Final evaluation metrics

---

## FIGURES NEEDED (Total: 4)

1. **Figure 1**: Architecture Ablation (3 subplots side-by-side)

   - **Panel A**: Cumulative win rate (256/512/1024)
   - **Panel B**: vs Strong eval win rate (256/512/1024)
   - **Panel C**: vs Weak tie rate (256/512/1024)
   - **Key finding**: 1024 hidden units show best performance across all metrics

2. **Figure 2**: RS vs No-RS Ablation (2 subplots side-by-side)

   - **Panel A**: vs Weak win rate (RS vs No-RS)
   - **Panel B**: vs Weak tie rate (RS vs No-RS)
   - **Key finding**: Reduced tie rate shows increased agent activity/aggression

3. **Figure 3**: Strong vs Weak Training Comparison (3 subplots side-by-side)

   - **Panel A**: vs Strong eval (strong-trained vs weak-trained)
   - **Panel B**: vs Weak eval (strong-trained vs weak-trained)
   - **Panel C**: Tie rate (strong-trained vs weak-trained)
   - **Key finding**: Training on strong improves both weak and strong performance (top-down effect)

4. **Figure 4**: Self-Play vs Baseline (2 subplots side-by-side)

   - **Panel A**: Cumulative win rate progression (baseline vs self-play)
   - **Panel B**: vs Strong eval win rate (baseline vs self-play)
   - **Key finding**: Modest improvements, evaluation limited by available opponents

---

## TABLES NEEDED (Total: 3)

1. **Table 1**: Final Evaluation (already in LaTeX, Experiments section)
   - 100 games vs weak/strong at 27.5k and 97.5k episodes
   - Win/Loss/Tie percentages and average rewards

2. **Table 2**: Self-Play vs Baseline Comparison (optional, if space allows)
   - **Columns**: Training Method | Win% vs Weak | Win% vs Strong | Episodes | Avg Reward
   - **Rows**: Baseline (Strong), Self-Play, Improvement
   - Shows 88%→92% improvement

3. **Table 3**: Reward Shaping Components (Appendix)
   - PBRS components (offensive progress, puck proximity, defense, cushion)
   - Strategic bonuses (puck touch, proximity, goal direction, etc.)
   - Scaling factors (α=0.005 for PBRS)

---

## WORD COUNT BUDGET

- **Introduction**: ~100-150 words (SHARED)
- **Methods - TD3**: ~180 words (DONE)
- **Experiments - TD3**: ~200 words (bullet points + captions)
- **Self-Play**: ~150 words (bullet points + 1 sentence)
- **Discussion**: ~250 words (SHARED)
- **TOTAL**: ~780 words MAX

**Everything else = FIGURES AND TABLES**

---

## DATA TO EXTRACT FROM WANDB

### From "256 vs 512 vs 1024 Hidden Size":

- [ ] Training curves for all 3 architectures
- [ ] Episodes to 80%/90% win rates
- [ ] Final win rates vs weak/strong
- [ ] Computational metrics (if available)

### From "RS vs No-RS":

- [ ] Training curves for both configurations:
  - RS: With PBRS (α=0.005) + strategic bonuses
  - No-RS: Sparse rewards only
- [ ] Convergence speed comparison:
  - Episodes to reach 80% win rate
  - Episodes to reach 90% win rate
- [ ] Final performance metrics:
  - Final win rate vs weak opponent
  - Final win rate vs strong opponent
  - Average reward comparison

### From "Strong vs Weak Training Opponent":

- [ ] Training curves for both configurations
- [ ] Final evaluation metrics vs both opponent types
- [ ] Generalization analysis

### From "Baseline-vs-Strong vs. Self-Play":

- [ ] Baseline (Strong) final metrics
- [ ] Self-play final metrics
- [ ] Improvement quantification
- [ ] Training progression during self-play

---

## SUCCESS CRITERIA

✓ All 4 available experiments are covered
✓ No references to learning rate ablation
✓ No references to exploration decay ablation
✓ No references to TD3 vs DDPG (unless data available)
✓ Figures match available WandB reports
✓ Word count stays under 800 words
✓ Professional presentation with clear figures/tables
