# DreamerV3 Hockey Report Blueprint

## Executive Summary

**Author:** Carl Kueschall (single person)
**Page Limit:** 5 pages (excluding references)
**Presentation:** 3 min video
**Benchmark Performance:** 93.5% combined win rate (90% weak, 97% strong)
**Algorithm:** DreamerV3 (World-Model Based RL)

---

## Current Status

| Item | Status |
|------|--------|
| LaTeX report | ✅ Overhauled for single-person |
| Introduction + exmatriculation note | ✅ Complete |
| Methods section | ✅ Complete |
| Experiments section | ✅ Complete |
| Discussion | ✅ Complete |
| AI usage declaration | ✅ Complete |
| Bibliography | ✅ Complete (`INPUT/main.bib`) |
| Placeholder figures | Run `python generate_report_figures.py --placeholder` |
| Ablation runs | ✅ DreamSmooth + Two-Hot complete |
| Real figures from ablations | 🔲 TODO |

---

## Project Requirements Checklist (from project.md)

- [x] Introduction with problem description (~0.5 page)
- [x] Methods section with implementation details + math (min 1 page)
- [x] Experimental evaluation with performance vs basic opponent (min 1 page)
- [x] Final discussion (~1 page)
- [x] AI usage declaration
- [ ] Page count ≤ 5 (excluding references)
- [x] Integrate completed ablation results (DreamSmooth, Two-Hot)

---

## Page Budget (5 Pages Total)

| Section | Pages |
|---------|-------|
| Introduction | ~0.5 |
| Methods | ~1.25 |
| Experiments | ~1.25 |
| Discussion | ~0.75 |
| AI Usage Declaration | ~0.25 |
| **Total** | **~5** |

Appendix (hyperparameters, architecture) follows references and may count toward limit—verify with project rules.

---

## Presentation (3 min)

| Segment | Time | Content |
|---------|------|---------|
| Intro | ~20 s | Hockey problem, sparse rewards, why world-model RL |
| Approach | ~45 s | DreamerV3 in 2–3 sentences; Two-Hot Symlog, DreamSmooth, 2-phase curriculum |
| Results | ~60 s | 93.5% win rate, training curve, 1–2 ablations |
| Wrap-up | ~15 s | Takeaways, limitations, future work |

---

## Ablation Study Recommendations

**Priority 1 (Must Have):**
1. **DreamSmooth ON vs OFF** - `--use_dreamsmooth` vs without
2. **Two-Hot ON vs Gaussian** - Two-Hot Symlog vs Gaussian/MSE reward-value modeling

**Priority 2 (Nice to Have):**
3. Replay ratio comparison (RR=32 vs 16 vs 8)
4. Mixed vs single opponent

Integrate generated ablation tables/figures from `02-SRC/DreamerV3/ablation-analysis/output`.

---

## Checklist Before Submission

- [x] Ablation placeholders resolved (DreamSmooth + Two-Hot)
- [ ] Training curve figure exists (`INPUT/figures/training_curve_placeholder.png`)
- [ ] Page count ≤ 5 (excluding references)
- [ ] All claims supported by evidence
- [ ] Proofreading complete
- [ ] PDF compiles without errors
- [ ] 3-min video presentation recorded
- [ ] Code submitted and tournament client running
