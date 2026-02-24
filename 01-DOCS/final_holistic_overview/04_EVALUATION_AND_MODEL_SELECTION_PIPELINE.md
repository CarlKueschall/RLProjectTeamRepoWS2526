# Evaluation and Model Selection Pipeline

## Scope
This document explains how checkpoints are evaluated, ranked, and converted into training assets for self-play and league continuation.

Main files:
- `02-SRC/DreamerV3/test_hockey.py`
- `02-SRC/DreamerV3/scripts/eval_summit_matrix.py`
- `02-SRC/DreamerV3/scripts/eval_select_pool.py`
- `02-SRC/DreamerV3/scripts/prepare_league_pool.py`

## Core principle
Selection is robustness-driven, not single-score driven. The pipeline explicitly considers:
1. fixed-bot competence,
2. cross-checkpoint robustness,
3. worst-case and low-quantile behavior,
4. diversity for league composition.

## A) Matrix ranking (`eval_summit_matrix.py`)
## What it runs
For each candidate checkpoint:
1. fixed-vs-weak eval
2. fixed-vs-strong eval
3. candidate vs each opponent checkpoint eval

## Output artifacts
- `all_matches.csv`
- `candidate_summary.csv`
- `candidate_ranking.md`

## Robust score used in this script
For each candidate:
- `fixed_combined = mean(weak_wr, strong_wr)`
- `cp_mean = mean(winrate_vs_each_checkpoint)`
- `cp_worst = min(winrate_vs_each_checkpoint)`

Score:
- `robust_score = 0.60 * cp_worst + 0.25 * fixed_combined + 0.15 * cp_mean`

Interpretation:
- heavily penalizes brittle checkpoints (worst-case failure dominates).

## B) Fast large-set pool selection (`eval_select_pool.py`)
Designed for dozens of checkpoints.

## Stage 0: compatibility prefilter
- Attempts model-only load with `load_optimizers=False`.
- Incompatible checkpoints are skipped before expensive evaluation.

## Stage 1: screening
Per candidate:
1. fixed weak/strong eval.
2. sparse directed checkpoint-vs-checkpoint set:
- always near temporal neighbors
- plus random sampled opponents.

Stage-1 quality:
- `stage_quality = 0.55*cp_q20 + 0.25*cp_mean + 0.20*fixed_combined`

Where:
- `cp_q20` is 20th percentile of cp winrates.

## Stage 2: top-K refinement
- Evaluate top-K with denser cross-play.
- Optional bidirectional matches (`A->B` and `B->A`) to reduce side bias.

If both directions available, pair score uses:
- `s = 0.5 * (A_vs_B_winrate + B_vs_A_lossrate)`

Final quality:
- `final_quality = 0.60*cp_q20 + 0.25*cp_mean + 0.15*fixed_combined`

## C) Diversity-aware pool recommendation
Given final summaries + pairwise scores:
1. normalize `final_quality` to `[0,1]`.
2. build per-candidate pair-score vectors.
3. greedy selection up to pool size with blend:
- `blend = quality_weight * quality_norm + diversity_weight * min_cosine_distance_to_selected`

Outputs:
- `recommended_pool.csv`
- `recommended_pool.txt`
- `recommended_pool.md`

## D) League materialization (`prepare_league_pool.py`)
Purpose:
- convert recommended list into a clean training-ready directory.

Behavior:
1. reads `recommended_pool.csv`
2. resolves source checkpoints (with optional fallback source-dir)
3. materializes via `symlink`, `hardlink`, or `copy`
4. writes:
- `league_checkpoints.txt` (paths for probe loading)
- `league_weights.csv` (rank, blend score, quality fields)

## E) In-training robustness evaluation (`train_hockey.py`)
During periodic eval interval:
1. evaluate weak and strong bots.
2. evaluate all loaded probe checkpoints.
3. compute online robust score:
- `robust = 0.60*probe_min + 0.25*combined_fixed + 0.15*probe_mean`

This mirrors matrix philosophy and gives live checkpoint-selection signal during long runs.

## Practical selection policy
1. Use matrix ranking for shortlists and direct A/B validation.
2. Use pool-select for large candidate sets and league construction.
3. Use online robust score to choose continuation points before expensive re-eval.
4. Reconfirm final contenders with higher episode counts and bidirectional matches.

## Common pitfalls
1. Path mismatches across local/cluster contexts.
2. architecture-mismatched legacy checkpoints contaminating candidate folders.
3. ranking from too few episodes causing noisy false positives.
4. relying only on fixed-bot scores (high fixed score can still be weak in cp robustness).
