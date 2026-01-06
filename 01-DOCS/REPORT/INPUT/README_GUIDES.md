# Report Guides: Quick Navigation

## Files in This Directory

You now have **5 guide documents** to help you write your report:

### 1. **GUIDE_COMPACT_APPROACH.md** ⭐ START HERE
**PURPOSE:** Explains how to fill the [TODO] sections in the LaTeX template.
**USE THIS FOR:** Understanding exactly what figures/tables to create and where they go.
**Content:**
- What each [TODO] means
- Exact figures/tables to create
- Data extraction checklist
- LaTeX insertion examples

**Timeline: 1-2 days of work**

---

### 2. **GUIDE_OVERALL_STRATEGY.md**
**PURPOSE:** High-level report structure and writing strategy.
**USE THIS FOR:** Understanding page layout and overall approach.
**Content:**
- Compact structure (4.25 pages content + 3-4 pages figures)
- Section-by-section overview
- Word count budgets
- Writing timeline

**Timeline: Quick reference only**

---

### 3. **GUIDE_TD3_METHOD_SECTION.md**
**PURPOSE:** Detailed breakdown of TD3 method section.
**STATUS:** Still valid but SUPERSEDED by compact approach.
**USE THIS FOR:** Deep understanding of TD3 concepts (for actual writing understanding).
**CAVEAT:** This was written for expanded report (1.5 pages). Your actual Methods is 0.5 pages. Use it for learning, not as word count guide.

---

### 4. **GUIDE_TD3_EXPERIMENTS_SECTION.md**
**PURPOSE:** Detailed breakdown of ablation studies and experiments.
**STATUS:** Still valid but condensed in actual report.
**USE THIS FOR:** Understanding what ablations to run/analyze, what metrics matter.
**CAVEAT:** This was written for 1.5-page Experiments section. Your actual section is 0.75 pages. Use for experimental design ideas, not word count.

---

### 5. **GUIDE_SELF_PLAY_SECTION.md**
**PURPOSE:** Detailed breakdown of self-play system.
**STATUS:** Still valid but condensed in actual report.
**USE THIS FOR:** Understanding self-play mechanisms, what to measure.
**CAVEAT:** This was written for 0.75-page section, which matches current template. This guide is most aligned with reality.

---

## WHICH GUIDE SHOULD I USE?

### For Writing Your Report:
**Primary resource:** `GUIDE_COMPACT_APPROACH.md` ← Start here, follow it step-by-step

### For Understanding Your Algorithms:
**Secondary resources:** `GUIDE_TD3_METHOD_SECTION.md` and `GUIDE_TD3_EXPERIMENTS_SECTION.md` ← Read for conceptual understanding

### For Overall Context:
**Reference:** `GUIDE_OVERALL_STRATEGY.md` ← Skim for structure understanding

---

## KEY CHANGES FROM ORIGINAL GUIDES

| Aspect | Original Guides | Compact Approach |
|--------|---|---|
| **Methods length** | 1.5 pages | 0.5 pages |
| **Focus** | Balance prose + figures | Prose minimal, figures dominant |
| **Experiments length** | 1.5 pages | 0.75 pages |
| **Ablation detail** | 700+ words per ablation | Bullet points + tables |
| **Self-Play length** | 0.75 pages | 0.5 pages |
| **Required figures** | 10-12 | 6-8 |
| **Word count per section** | ~400-500 | ~150-250 |
| **Timeline** | 3 weeks | 1-2 weeks |

---

## WORKING APPROACH (RECOMMENDED)

### Phase 1: Preparation (Day 1)
1. Read `GUIDE_COMPACT_APPROACH.md` thoroughly
2. Skim `GUIDE_OVERALL_STRATEGY.md` for context
3. Gather all data from WandB, checkpoints, evaluations

### Phase 2: Create Figures & Tables (Days 2-3)
1. Follow the **Figure Checklist** in `GUIDE_COMPACT_APPROACH.md`
2. Follow the **Table Checklist**
3. Save all figures as PDF files
4. Have tables in both Excel and LaTeX format

### Phase 3: Write Prose (Days 4-5)
1. Write 0.5p Methods section (use `GUIDE_TD3_METHOD_SECTION.md` for inspiration on what to say, but keep it SHORT)
2. Write 0.75p Experiments section (use `GUIDE_TD3_EXPERIMENTS_SECTION.md` for what results matter)
3. Write 0.5p Self-Play section (use `GUIDE_SELF_PLAY_SECTION.md` as detailed reference)
4. Write 0.5p Discussion (discuss key findings from your data)

### Phase 4: Integration (Day 6)
1. Insert all figures/tables into LaTeX [TODO] sections
2. Format properly with captions and labels
3. Check page count and layout
4. Final proofread

---

## TL;DR - Quick Start

**Just want to know what to do? Follow this:**

1. Open `GUIDE_COMPACT_APPROACH.md`
2. Look at the section that matches what you're working on
3. See what [TODO] is there
4. See what figure/table to create
5. Extract data from WandB
6. Create figure/table
7. Insert into LaTeX
8. Move to next section

**Don't overthink it.** Figures > Prose. Keep prose short.

---

## FAQ

**Q: Should I follow the detailed guides or the compact approach?**
A: Follow `GUIDE_COMPACT_APPROACH.md`. The detailed guides are reference material only.

**Q: How much should I write for Methods?**
A: ~180 words. It's already written in the LaTeX template. Don't expand it.

**Q: Can I make my Methods section longer?**
A: No. Keep it to 0.5 pages max. Use extra space for figures instead.

**Q: What if my ablations show negative results?**
A: That's fine. Show the data honestly. Discuss why a modification didn't help.

**Q: Do I need all 6-8 figures?**
A: Aim for at least 5-6. More figures = less prose required = better use of space.

**Q: Should I follow the old 1.5-page structure from the detailed guides?**
A: No. That was for a longer report. Your actual template is much more compact.

---

## Support

- LaTeX questions: Check `report_main.tex` for structure
- Data/figure questions: Check `GUIDE_COMPACT_APPROACH.md` section "Data Extraction Checklist"
- Conceptual questions: Check the detailed guides (`GUIDE_TD3_METHOD_SECTION.md`, etc.)

