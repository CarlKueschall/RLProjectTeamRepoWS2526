# Slide 7 — Ablations: What Helped, What Was Neutral

## On-Slide Text
- DreamerV3 core already handles sparsity strongly
- DreamSmooth: clear late-stage stability/performance gains
- Two-Hot vs Gaussian heads: mixed to neutral in this setup

## Visuals (pick 2–3 for readability)
- DreamSmooth combined win rate:
  - `/Users/carlkueschall/workspace/RLProjectHockey/03-RESULTS/REPORT/figures/experiments/dreamsmooth-combined-win-rate.png`
- DreamSmooth sparse pred error:
  - `/Users/carlkueschall/workspace/RLProjectHockey/03-RESULTS/REPORT/figures/experiments/dreamsmooth-sparse-pred-error.png`
- Two-Hot combined win rate:
  - `/Users/carlkueschall/workspace/RLProjectHockey/03-RESULTS/REPORT/figures/experiments/twohot-combined-win-rate.png`

## Speaker Notes (20s)
"Ablations showed an important nuance: DreamerV3 is already strong in this sparse environment, but DreamSmooth improved late-stage stability and sparse prediction quality. Two-Hot did not show a clear net gain in this single-seed setup, so we report it as mixed or neutral rather than over-claiming."

## Slide Build
1. Show DreamSmooth plots first.
2. Add Two-Hot plot with 'mixed evidence' tag.
