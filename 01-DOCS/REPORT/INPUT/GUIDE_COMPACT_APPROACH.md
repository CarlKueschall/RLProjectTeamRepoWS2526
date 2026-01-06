# Compact Report Guide: How to Fill the [TODO] Sections

## Overview

Your LaTeX template is now **compact and efficient**. Instead of writing long prose, you'll fill [TODO] sections with **figures and tables** that show your results. This guide tells you exactly what to put in each [TODO].

---

## Quick Reference: [TODO] Sections in LaTeX

The `report_main.tex` file has these [TODO] placeholders:

### In Methods (TD3 subsection):
- No TODOs - already describes all 4 modifications

### In Experiments (TD3 subsection):
- **[TODO: Include training curves and quantitative results]** - Add 2-3 figures
- **[TODO: Comparative metrics table]** - Add comparison table
- **[TODO: Pool evolution visualization, opponent strength progression]** - In self-play section

### In Self-Play:
- **[TODO: Pool evolution visualization, PFSP weight analysis]** - Add 2 figures

### In Discussion:
- **[TODO: TD3 vs SAC table comparing...]** - Add comparison table

---

## SECTION-BY-SECTION FILLING GUIDE

### SECTION 1: Introduction (Already ~75% complete)

**What's already written:**
- Problem statement (hockey environment)
- Your and Serhat's contributions listed

**What you might add (optional):**
- Enhance bullet points with 1-2 specific results: "achieves 92% win rate vs weak baseline"
- Done. Keep it concise.

**Word count:** ~100-150 words (already done)

---

### SECTION 2: Methods - TD3 (Already ~95% complete)

**What's written:**
```
TD3 (Twin Delayed DDPG) mitigates overestimation bias through twin critics,
delayed policy updates, and target smoothing. Our implementation uses actor
networks (18D→256×256→4D) and dual critic networks (22D→256×256×128→1D) with
Adam learning rate 3×10⁻⁴, batch size 256, γ=0.99, τ=0.005. OU noise decays
from ε=1.0 to 0.05 over training.

Domain-Specific Modifications: (1) Q-clipping: Hard clamping to [-25, 25]
prevents Q-value explosion. (2) Anti-lazy learning: Regularization penalizes
passive actions when puck-distant. (3) Dual replay buffers: Anchor (1/3, weak)
+ Pool (2/3, self-play) with dynamic mixing. (4) PBRS: Conservative reward
shaping (α=0.005) with potential functions.
```

**What you can add (optional):**
- Small **equation box** showing Q-clipping formula: `Q(s,a) = clamp(Q(s,a), -25, 25)`
- Small **equation box** showing VF-reg: `L_vf = λ_vf × max(0, Q_passive - Q_active)`
- Small **figure** showing potential function components (optional - not critical)

**Word count:** ~180 words (DONE - don't expand)

---

### SECTION 3: Experiments - TD3 (0.75 pages, mostly figures)

**Structure in LaTeX:**
```
TEXTBF: Training Curriculum: Phase 1 (weak, 0-10.25k), Phase 2 (strong, 10.25-27.5k),
        Phase 3 (self-play, 27.5-97.5k)

TEXTBF: Ablation Studies: Network architecture (256/512/1024), learning rates
        (1e-4/3e-4/1e-3), exploration decay rates, reward shaping impact.
        [TODO: Include training curves and quantitative results]

TEXTBF: TD3 vs DDPG: DDPG baseline comparison isolates twin critic and delayed
        update benefits. [TODO: Comparative metrics table]

TABLE 1: Final Evaluation (100 games each): Shows win rates for weak/strong at 27.5k
         and 97.5k episodes
```

**What to ADD in [TODO] sections:**

#### [TODO #1] "Include training curves and quantitative results"

**Create Figure A1: Training Curves Comparison (2-panel)**
- **Panel 1:** Network architecture scaling (256 vs 512 vs 1024)
  - X-axis: Episodes (0-27,500)
  - Y-axis: Win rate or smoothed reward
  - 3 lines (small/medium/large networks)
  - Include legend

- **Panel 2:** Learning rate sensitivity (1e-4 vs 3e-4 vs 1e-3)
  - X-axis: Episodes (0-27,500)
  - Y-axis: Win rate or smoothed reward
  - 3 lines (different learning rates)
  - Include legend

**Optional:** Add third panel for exploration decay rates

**Create Table A1: Ablation Results Summary**
```
| Hyperparameter | Config | Episodes to 80% | Episodes to 90% | Final Win % | Stability |
|---|---|---|---|---|---|
| **Network** | 256 | X | Y | Z% | [metric] |
| | 512 | X | Y | Z% | [metric] |
| | 1024 | X | Y | Z% | [metric] |
| **Learning Rate** | 1e-4 | X | Y | Z% | [metric] |
| | 3e-4 | X | Y | Z% | [metric] |
| | 1e-3 | X | Y | Z% | [metric] |
| **Exploration Decay** | Slow | X | Y | Z% | [metric] |
| | Baseline | X | Y | Z% | [metric] |
| | Fast | X | Y | Z% | [metric] |
```

#### [TODO #2] "Comparative metrics table"

**Create Table A2: TD3 vs DDPG Comparison**
```
| Algorithm | Win vs Weak | Win vs Strong | Episodes to 90% | Final Reward |
|---|---|---|---|---|
| DDPG | XX% | YY% | Z episodes | R |
| TD3 | XX% | YY% | Z episodes | R |
| Improvement | +X% | +Y% | -Z episodes | +R |
```

**If you didn't run DDPG:**
Leave as [?] and note: "DDPG comparison conducted with same hyperparameters [citation to implementation]"

---

### SECTION 4: Self-Play (0.5 pages)

**Structure (already written):**
```
Pool Management: 12-agent pool, checkpoints every 400 episodes, 40% weak + 60% pool
PFSP Curriculum: Weighting w_i = wr_i(1-wr_i), favors 50% win-rate opponents
Anti-Forgetting: Dual buffers (1/3 anchor, 2/3 pool), ratio → 0.70 when drop >10%
Performance Gates: >80% win-rate vs weak AND variance <0.3
Regression Rollback: Auto-recovery if drop >15% AND 2+ consecutive drops
Results: Self-play improved weak robustness 88%→92% over 70k episodes, 100% vs strong.
[TODO: Pool evolution visualization, PFSP weight analysis]
```

**What to ADD in [TODO] section:**

#### [TODO] "Pool evolution visualization, PFSP weight analysis"

**Create Figure A3: Pool Evolution Timeline**
- X-axis: Training episodes (27.5k to 97.5k)
- Y-axis: Checkpoint ID or opponent strength
- Show: Which agents in pool at each time, pool turnover
- Could be: Timeline graphic, Gantt chart, or heatmap

**Create Figure A4: PFSP Weighting**
- X-axis: Win rate (0% to 100%)
- Y-axis: Selection weight
- Curve: w = wr × (1 - wr)
- Show: Peak at 50%, drops at extremes

**Optional Figure A5: Win Rate vs Pool Over Time**
- X-axis: Episodes during self-play (27.5k-97.5k)
- Y-axis: Win rate against pool (rolling 100-game average)
- Shows: How agent performance in self-play progresses

---

### SECTION 5: Discussion (0.5 pages - SHARED)

**Already written:**
```
TD3 Key Findings: Domain modifications essential, TD3 achieves 92% vs weak and 100%
vs strong. Self-play improves 88%→92% over 70k episodes. PBRS scaling (α=0.005)
prevents Q-value explosion.

Algorithm Comparison: [TODO: TD3 vs SAC table]

Future Work: Investigate league-play, model-based approaches, policy distillation,
transfer learning.
```

**What to ADD in [TODO] section:**

#### [TODO] "TD3 vs SAC table comparing..."

**Create Table A6: Algorithm Comparison (Fill in with your data where available)**
```
| Property | TD3 | SAC | Better | Notes |
|---|---|---|---|---|
| Win Rate vs Weak | 92% | [?]% | [TD3/SAC/Tie] | [your insight] |
| Win Rate vs Strong | 100% | [?]% | [TD3/SAC/Tie] | [your insight] |
| Sample Efficiency | [data] | [data] | [TD3/SAC] | Episodes to 80% |
| Training Stability | [data] | [data] | [TD3/SAC] | Lower variance = better |
| Hyperparameter Sensitivity | Low | [data] | [TD3/SAC] | [your analysis] |
```

**If SAC not complete:** Just note "[To be completed by Serhat]"

---

## FIGURE CHECKLIST (TOTAL 6-8 FIGURES RECOMMENDED)

- [x] **Figure 1 (Methods):** Optional - Network architecture diagram OR equations only
- [ ] **Figure 2 (Experiments):** Architecture/LR/Exploration curves (2-3 subpanels)
- [ ] **Figure 3 (Experiments):** Ablation results table visualization (bar charts)
- [ ] **Figure 4 (Experiments):** TD3 vs DDPG comparison curves
- [ ] **Figure 5 (Self-Play):** Pool evolution timeline
- [ ] **Figure 6 (Self-Play):** PFSP weighting curve
- [ ] **Figure 7 (Results):** Win rate evolution during training
- [ ] **Figure 8 (Results):** Evaluation bar charts (weak vs strong)

---

## TABLE CHECKLIST (TOTAL 3-4 TABLES)

- [ ] **Table 1 (Experiments):** Final evaluation results (already in LaTeX)
- [ ] **Table 2 (Experiments):** Ablation studies summary
- [ ] **Table 3 (Experiments):** TD3 vs DDPG comparison
- [ ] **Table 4 (Discussion):** Algorithm comparison (optional)

---

## Data Extraction Checklist

Before creating figures, extract:

```
From WandB:
☐ Training curves (reward/win-rate over episodes)
☐ All ablation runs' metrics
☐ Final evaluation scores (100-game test sets)
☐ Any graphs already in WandB (can screenshot)

From your code:
☐ Architecture specs (exact layer sizes)
☐ Hyperparameter values used
☐ Self-play pool statistics
☐ PFSP weighting data if logged

From evaluations:
☐ Win/loss/tie counts vs weak opponent (27.5k and 97.5k)
☐ Win/loss/tie counts vs strong opponent (27.5k and 97.5k)
☐ Average rewards per game
☐ Reward std dev/variance

From DDPG (if you have it):
☐ DDPG training curves
☐ DDPG final performance
☐ DDPG vs TD3 convergence speed
```

---

## CRITICAL: Word Count Management

**Your prose budget (MAXIMUM):**
- Methods: 180 words (DONE)
- Experiments: 200 words (3 bullet sections + table captions)
- Self-Play: 150 words (bullet points + 1 result sentence)
- Discussion: 250 words (key findings + future work)
- **TOTAL: 780 words MAX**

**Everything else = FIGURES AND TABLES**

If you exceed 800 words, you've written too much. Cut ruthlessly.

---

## LaTeX Insertion Examples

**How to insert figures in [TODO] sections:**

```latex
[TODO: Include training curves and quantitative results]

% BECOMES:

\begin{figure}[h]\centering
\includegraphics[width=0.9\textwidth]{figures/architecture_ablation.pdf}
\caption{Training curves comparing network architectures (256, 512, 1024).}
\label{fig:arch-ablation}
\end{figure}

\begin{table}[h]\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Architecture} & \textbf{Ep to 80\%} & \textbf{Final Win\%} & ... \\
...
\end{tabular}
\caption{Architecture ablation results.}
\end{table}
```

---

## Timeline

**Week 1 - DATA FIRST:**
- Extract all metrics from WandB, checkpoints, evaluations
- Create 6-8 figures (the hard part)
- Create 3-4 result tables
- Total time: 2-3 days

**Week 2 - WRITING + INTEGRATION:**
- Write 780 words total prose (very quick - ~2 hours)
- Insert figures/tables into [TODO] sections
- Final formatting and layout
- Total time: 1-2 days

---

## Success Criteria

✓ Figures are clean, labeled, with legends
✓ Tables are easy to read with clear column headers
✓ All claims in prose are supported by figures/tables
✓ Word count < 800 words for your sections
✓ All [TODO] sections filled with data, not placeholders
✓ Layout looks professional (proper spacing, captions)

