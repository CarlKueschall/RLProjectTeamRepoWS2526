# Slide 6 — Results: Robust Selection Instead of Single-Metric Picking

## On-Slide Text
- Late strong checkpoints: **556k, 612k, 634k**
- Fixed-opponent performance stays high in late phase
- Cross-play robustness metrics became the true discriminator:
  - `probe_mean_win_rate`
  - `probe_min_win_rate`
  - robust score / worst-case cross-play

## Primary Visual
![Benchmark progression](/Users/carlkueschall/workspace/RLProjectHockey/03-RESULTS/REPORT/figures/benchmark/final_benchmark_eval_progression_stitch.png)

## Speaker Notes (25s)
"In the final phase, many checkpoints looked similarly strong on fixed bots, so that metric stopped being enough. We therefore selected checkpoints using robust cross-play statistics. This led to a stable late shortlist around 556k, 612k, and 634k."

## Slide Build
1. Show progression figure.
2. Highlight shortlist checkpoints.
3. Emphasize robust metrics bullet.
