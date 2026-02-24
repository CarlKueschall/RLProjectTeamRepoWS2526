# SUMMIT: Performance Improvement Notes (Tournament Push)

## Current Situation

- Best checkpoint: `336k_selfplay` (`93.5%` combined on weak+strong benchmark).
- Main issue is robustness against arbitrary opponents, not fixed-bot score ceiling.
- Metrics show late-phase regression/cycling during self-play rather than clear undertraining.

## Key Findings from 336k Benchmark Metrics

1. World model is not the primary bottleneck.
- `world/loss`, `recon_loss`, `reward_loss`, sparse-signal diagnostics are stable.
- No obvious collapse in RSSM or reward prediction.

2. Late self-play behavior shows drift.
- `eval/combined_win_rate` peaked at `0.99`, ended at `0.88`.
- `eval/strong_win_rate` peaked at `1.00`, ended at `0.84`.
- `selfplay/winrate_newest_third` peaked near `0.80`, ended near `0.68`.
- `selfplay/winrate_vs_pool_overall` peaked near `0.868`, ended near `0.818`.

3. Likely curriculum/sampling bug in training loop.
- In `train_hockey.py`, mixed-opponent override runs after self-play selection.
- This can remap selected anchor opponents and skew true exposure.
- Observed mismatch supports this:
  - Self-play manager counters show near-balanced anchors.
  - Logged opponent counts show strong-heavy exposure.

## Interpretation

- Extending self-play unchanged is unlikely to reliably improve robustness.
- Most promising gains come from:
  - fixing opponent sampling logic,
  - reducing update aggressiveness in continuation,
  - improving opponent diversity/hardness scheduling.

## High-Value Action Plan (Prioritized)

1. Fix mixed/self-play interaction in `train_hockey.py`.
- Ensure mixed-opponent logic only applies before self-play activation (or when self-play inactive).
- Preserve self-play manager’s selected opponent distribution exactly.

2. Continue training from `best_selfplay_336k.pth` with conservative settings.
- Suggested:
  - `replay_ratio`: `16` (or `8`),
  - `lr_world`: `2e-4`,
  - `lr_actor`: `5e-5`,
  - `lr_critic`: `5e-5` to `7e-5`,
  - keep `DreamSmooth` on.

3. Increase pool refresh rate for diversity.
- Keep `self_play_pool_size=20`.
- Reduce `self_play_save_interval` from `500` to `250`.

4. Two-stage self-play schedule.
- Stage A (stabilize): `pfsp_mode=variance`, anchor ratio around `0.4`.
- Stage B (harden): short run with `pfsp_mode=hard`, anchor ratio around `0.25-0.30`.

5. Select final checkpoint by robustness composite, not single metric.
- Track and rank by:
  - fixed-bot combined win rate,
  - `winrate_newest_third`,
  - `pfsp_min_winrate`,
  - `winrate_vs_pool_overall`.

## Decision on “Should self-play run longer?”

- Yes, but only **after** fixing sampling logic and reducing update aggressiveness.
- Longer run without those adjustments likely continues late-cycle regression.

