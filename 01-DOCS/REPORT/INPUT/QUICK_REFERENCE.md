# Quick Reference Card: Filling Your Report [TODOs]

## The Three [TODO] Sections You Need to Fill

### [TODO #1] In Experiments - "Include training curves and quantitative results"
**Location:** Page 2-2.75, after "Ablation Studies" bullet

**What to create:**
- **Figure:** 2-3 panel plot showing training curves for architecture/LR/exploration ablations
- **Table:** Ablation results (convergence speed, final performance, stability)

**Size:** ~half page (2 figures or 1 figure + 1 table)

---

### [TODO #2] In Experiments - "Comparative metrics table"
**Location:** Page 2, after "TD3 vs DDPG" bullet

**What to create:**
- **Table:** TD3 vs DDPG side-by-side comparison
  - Columns: Algorithm, Win% vs Weak, Win% vs Strong, Episodes to 90%, Final Reward
  - Rows: DDPG, TD3, Improvement

**Size:** ~2 inches wide, ~1 inch tall table

---

### [TODO #3] In Self-Play - "Pool evolution visualization, PFSP weight analysis"
**Location:** Page 3.25-3.75

**What to create:**
- **Figure A:** Pool evolution timeline (which agents in pool over time)
- **Figure B:** PFSP weighting curve showing w = wr(1-wr) with peak at 50%

**Size:** ~0.75 page (2 small figures)

---

## Data You Need (Extract from WandB/Code)

**Ablation Runs (extract metrics):**
- [ ] Architecture 256: episodes to 80%, episodes to 90%, final win%
- [ ] Architecture 512: same metrics
- [ ] Architecture 1024: same metrics
- [ ] LR 1e-4: same metrics
- [ ] LR 3e-4: same metrics
- [ ] LR 1e-3: same metrics
- [ ] Exploration slow/baseline/fast: same metrics

**TD3 vs DDPG (from your implementations):**
- [ ] DDPG final win rate vs weak
- [ ] DDPG final win rate vs strong
- [ ] TD3 final win rate vs weak
- [ ] TD3 final win rate vs strong
- [ ] Episodes for DDPG to reach 90%
- [ ] Episodes for TD3 to reach 90%

**Self-Play:**
- [ ] Pool composition over time (which checkpoints in pool at each episode)
- [ ] Win rates against each opponent at different training stages
- [ ] Checkpoint strength estimates

---

## Writing Checklist

**Methods (0.5 page):** ✓ Already done, don't modify

**Experiments (0.75 page):**
- [ ] 3 sentences explaining training curriculum (already written)
- [ ] 1 sentence about ablations (already written)
- [ ] 1 sentence about TD3 vs DDPG (already written)
- [ ] Insert Figure 1 (training curves)
- [ ] Insert Table 1 (ablation results)
- [ ] Insert Table 2 (TD3 vs DDPG)
- [ ] Insert Table 3 (final evaluation - already in template)

**Self-Play (0.5 page):**
- [ ] Bullets about pool/PFSP/gates (already written)
- [ ] 1 sentence result summary (already written)
- [ ] Insert Figure 2 (pool evolution)
- [ ] Insert Figure 3 (PFSP weighting)

**Discussion (0.5 page, shared):**
- [ ] TD3 key findings (already written)
- [ ] Algorithm comparison (partially written)
- [ ] Future work (already written)

---

## Timeline (Compact Version)

**If you have 1 week:**
- Day 1: Extract all data from WandB (~2 hours)
- Day 2-3: Create all 6 figures (~4 hours)
- Day 3: Create all 4 tables (~1 hour)
- Day 4: Insert into LaTeX and format (~1 hour)
- Day 5-6: Write/edit prose (~2 hours total)
- Day 7: Final proofread

**If you have 2 weeks:**
- Days 1-3: Leisurely data extraction and figure creation
- Days 4-5: Polish figures, verify data accuracy
- Days 6-7: Write prose
- Days 8-14: Iterate, get feedback, refine

---

## Pro Tips

1. **Figures first, prose last.** Figures take longer, writing is fast.
2. **Use grid layout for multi-panel figures.** Saves space.
3. **Label everything.** X-axis, Y-axis, legend, title, caption.
4. **Error bars matter.** Show ±1 std dev or confidence intervals.
5. **Tables need headers.** Bold column names, use consistent units.
6. **Captions should be descriptive.** "Figure 1" is bad; "Figure 1: Training curves..." is good.
7. **Don't exceed 800 total words.** Be ruthless in cutting prose.
8. **Numbers matter more than words.** "88% → 92% improvement" beats "significant improvement"

---

## Figure/Table Creation Tools

**Python (recommended):**
```python
import matplotlib.pyplot as plt
import numpy as np

# Typical flow
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(episodes, small_net_rewards, label='256')
axes[0].plot(episodes, medium_net_rewards, label='512')
axes[0].plot(episodes, large_net_rewards, label='1024')
axes[0].set_xlabel('Episodes')
axes[0].set_ylabel('Win Rate (%)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('Architecture Ablation')

# Similar for axes[1]...

plt.tight_layout()
plt.savefig('ablation_curves.pdf', dpi=300, bbox_inches='tight')
```

**Excel/Numbers:**
- Create table in spreadsheet
- Format nicely (borders, bold headers)
- Export as PDF or image
- Paste into LaTeX

**Pandas (for tables):**
```python
import pandas as pd

df = pd.DataFrame({
    'Architecture': ['256', '512', '1024'],
    'Episodes to 80%': [5000, 4500, 4200],
    'Win Rate': ['88%', '90%', '91%']
})

# Export to LaTeX
print(df.to_latex(index=False))
```

---

## Common Mistakes to Avoid

❌ Writing 1.5-page Methods section (old guide)
✓ Keep Methods to 0.5 pages

❌ Using low-resolution figures
✓ Export figures at 300 DPI minimum

❌ Missing figure labels/captions
✓ Every figure needs title, axis labels, legend, caption

❌ Hiding important data in prose
✓ Show data in tables/figures instead

❌ Comparing ablations with different hyperparameters
✓ Keep everything else constant, vary one parameter at a time

❌ No error bars on curves
✓ Include confidence intervals or std dev bands

---

## File Organization

```
01-DOCS/REPORT/INPUT/
├── report_main.tex          # Main LaTeX file (edit [TODOs])
├── main.bib                 # Bibliography (auto-populated)
├── figures/                 # Create this folder
│   ├── arch_ablation.pdf
│   ├── lr_sensitivity.pdf
│   ├── self_play_pool.pdf
│   └── ...
└── tables/
    ├── ablation_results.txt
    ├── td3_vs_ddpg.txt
    └── ...
```

---

## Compilation

```bash
cd /Users/carlkueschall/workspace/RLProjectHockey/01-DOCS/REPORT/INPUT

pdflatex report_main.tex
bibtex report_main
pdflatex report_main.tex
pdflatex report_main.tex

# Output: report_main.pdf
```

---

## Questions?

- **"How do I insert a figure?"** → Look in `GUIDE_COMPACT_APPROACH.md` section "LaTeX Insertion Examples"
- **"What should my table look like?"** → Check same guide, sections with [TODO] explanations
- **"Is my prose too long?"** → Count words; should be <200 per section
- **"Should I expand Methods?"** → No. Keep it 0.5 pages. Use space for figures instead.

