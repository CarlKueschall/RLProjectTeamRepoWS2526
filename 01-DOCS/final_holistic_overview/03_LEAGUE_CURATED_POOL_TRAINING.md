# League-Curated Pool Training (Weighted Bootstrap + PFSP)

## Scope
This document describes the strongest training regime in this project: continuation from a strong checkpoint while bootstrapping self-play with a curated league of historical checkpoints.

Relevant implementation:
- `02-SRC/DreamerV3/train_hockey.py`
- `02-SRC/DreamerV3/opponents/self_play.py`
- `02-SRC/DreamerV3/scripts/eval_select_pool.py`
- `02-SRC/DreamerV3/scripts/prepare_league_pool.py`

## Conceptual objective
Use an explicit checkpoint league to increase strategic diversity and stabilize robustness gains faster than plain self-play from a single trajectory.

## Pipeline
## 1) Candidate universe
- Collect many checkpoints from multiple runs/seeds.
- Keep only architecture-compatible checkpoints.

## 2) Quality/diversity selection
- Run `eval_select_pool.py` to produce `recommended_pool.csv` and `recommended_pool.txt`.
- This does two stages:
  - fast screening
  - dense top-K refinement
- Then greedy blend selection on quality + diversity.

## 3) Materialize league directory
- Use `prepare_league_pool.py` to create:
  - `league_checkpoints.txt`
  - `league_weights.csv`
  - linked/copied checkpoint files

## 4) Continue training from best checkpoint
- Resume from best current policy (example: `634k.pth`).
- Configure self-play bootstrap from curated league.

## Runtime behavior in this mode
## Startup load
1. Probe checkpoints can be loaded from `league_checkpoints.txt` for robustness eval.
2. Self-play manager reads `league_weights.csv` and resolves prior weights.
3. At self-play activation, bootstrap checkpoints are converted into internal opponent snapshots and inserted into pool.

## Sampling behavior after activation
1. Anchor branch probability = `self_play_weak_ratio`.
2. Self-play branch probability = `1 - self_play_weak_ratio`.
3. Inside self-play branch:
- PFSP weight from empirical matchup results.
- multiplied by static prior term `prior^(alpha)` if enabled.

This yields hybrid sampling:
- dynamic hardness/variance curriculum
- quality-biased prior on curated checkpoints

## Why this mode is stronger than naive self-play
1. Immediate diversity at activation (not waiting many save intervals).
2. Better coverage of strategic modes from earlier and alternative run trajectories.
3. Lower risk of local cyclic overfitting to a narrow recent-self history.
4. Faster ramp into meaningful adversarial training pressure.

## Key configuration levers
1. `--self_play_bootstrap_dir`
- league directory containing checkpoint files.

2. `--self_play_bootstrap_max`
- how many external checkpoints are injected at activation.

3. `--self_play_bootstrap_strategy`
- deterministic recency/oldest/uniform, or score-aware ranked/weighted.

4. `--self_play_bootstrap_weights_csv`
- quality table source.

5. `--self_play_bootstrap_weight_column`
- usually `blend_score` from pool selection script.

6. `--self_play_prior_alpha`
- strength of static quality prior in PFSP sampling.

## Monitoring signals that confirm it is working
1. Startup logs show successful probe loads and non-zero bootstrap added count.
2. `selfplay/pool_size` jumps beyond 1 immediately after activation.
3. `selfplay/bootstrap_added_count` equals intended bootstrap size.
4. `stats/opponent_recent_selfplay_frac` converges near target split.
5. Stratified self-play winrates move over time rather than collapsing to trivial extremes.

## Operational caveats
1. Absolute-path traps
- if league files reference absolute local paths, remote cluster runs fail.
- must materialize cluster-local paths (fixed with `prepare_league_pool.py` usage).

2. Missing checkpoint files in scratch
- if SLURM copies code but not checkpoint assets, bootstrap load fails.
- either rsync checkpoint directories to home and copy to scratch, or resolve from shared filesystem.

3. Architecture mismatch in old checkpoints
- older head formats (e.g., reward head shape mismatch) must be filtered out pre-selection.

## Expected outcome pattern
1. Early continuation can look noisy after injecting many opponents.
2. Mid training should show stronger pool-overall and newest-third robustness.
3. Best checkpoints are usually found by periodic matrix/league re-eval, not final step only.

## Practical decision rule
If league-curated continuation is healthy and resource-constrained:
1. keep run alive,
2. periodically harvest checkpoints,
3. rerun pool selection,
4. refresh curated pool,
5. continue from best robust checkpoint.
