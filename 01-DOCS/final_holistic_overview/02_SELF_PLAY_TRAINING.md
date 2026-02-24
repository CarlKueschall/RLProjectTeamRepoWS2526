# Self-Play Training (Pool + PFSP)

## Scope
This document describes DreamerV3 self-play as implemented in:
- `02-SRC/DreamerV3/train_hockey.py`
- `02-SRC/DreamerV3/opponents/self_play.py`
- `02-SRC/DreamerV3/opponents/pfsp.py`

Self-play is the transition from fixed-opponent learning to non-stationary curriculum learning against checkpointed versions of the same policy family.

## Why self-play is added
1. Fixed bots are mostly solved and stop providing gradient-rich variety.
2. Self-play creates an evolving curriculum tied to your own policy frontier.
3. PFSP concentrates training on opponents that are neither trivial nor impossible.

## Activation lifecycle
## 1) Activation condition
- Controlled by `--self_play_start <episode>`.
- In `train_hockey.py`, once current episode >= threshold:
  - `SelfPlayManager.activate(...)` is called.

## 2) Initial pool seeding
- Pool is seeded with a checkpoint of the current agent (`selfplay_seed_ep<episode>.pth`).
- This gives immediate valid opponent availability.

## 3) Optional bootstrap seeding at activation
- If configured, external checkpoints are converted and added:
  - `--self_play_bootstrap_dir`
  - `--self_play_bootstrap_glob`
  - `--self_play_bootstrap_max`
  - `--self_play_bootstrap_strategy`
- Supported strategies:
  - `uniform`, `recent`, `oldest`, `random`, `ranked`, `weighted`

## Opponent-type sampling logic per episode
Each episode, selection is split into anchor-vs-self-play:
1. Draw anchor route with probability `self_play_weak_ratio`.
2. If anchor route:
   - choose weak/strong anchors with balancing toward 50/50 realized split.
3. Else:
   - sample from self-play pool (PFSP or uniform).

Important interaction:
- Once self-play is active, mixed-opponent pre-phase logic is bypassed.
- This prevents accidental distortion of configured self-play ratios.

## PFSP implementation details
PFSP score function (`pfsp_weight`) supports:
1. `variance`
- Weight = `winrate * (1 - winrate)`
- Peaks at 0.5 win rate.

2. `hard`
- Weight = `(1 - winrate)^p`
- Emphasizes hardest opponents.

3. `uniform`
- Constant weight.

In `SelfPlayManager._pfsp_select()`:
- If opponent has >=5 recorded matches, PFSP weight is used.
- Otherwise fallback is a static prior term (optionally quality-biased via prior alpha).

## Static quality prior blend
With league/bootstrap weights loaded:
- Each opponent can carry a prior score.
- Sampling multiplier: `prior^(prior_alpha)`.
- `prior_alpha=0` disables prior influence.
- Positive alpha biases early sampling toward higher-quality curated opponents.

## Pool growth and eviction
1. Every `self_play_save_interval` episodes:
- current agent is snapshotted into pool.

2. FIFO pool cap:
- If size exceeds `self_play_pool_size`, oldest checkpoint is evicted.

3. Tracking maps are kept in sync on eviction:
- recent results deque
- games count
- insertion episode
- source mapping
- prior weights

## Result recording and stratified diagnostics
After self-play episode result:
- Winner encoded as `1` win, `-1` loss, `0` draw.
- Stored per-opponent for PFSP.
- Also logged into age-tercile buckets:
  - oldest third
  - middle third
  - newest third

This gives direct signal on forgetting vs adaptation:
- rising newest-third with collapsing oldest-third can indicate forgetting.

## Metrics emitted by `SelfPlayManager.get_stats()`
### Core state
- active flag, pool size, episodes since activation.

### Anchor/self-play realized mix
- anchor counts, self-play counts.
- actual anchor ratio and self-play ratio.
- actual weak ratio among anchors.

### PFSP health
- number tracked, mean/std/min/max winrates.

### Opponent age/info
- current opponent index and insertion episode.
- oldest and newest opponent episodes in pool.

### Stratified win-rate diagnostics
- `winrate_oldest_third`
- `winrate_middle_third`
- `winrate_newest_third`
- `winrate_vs_pool_overall`

### Bootstrap/prior diagnostics
- bootstrap attempted/added counts.
- prior min/mean/max.

## Practical interpretation of this phase
1. This is where robustness against non-fixed policies improves.
2. Performance noise is expected; trend and robustness metrics matter more than single-point eval.
3. Pool composition quality and sampling policy strongly influence final tournament transfer.

## Failure modes to watch
1. Pool too homogeneous
- high overall win vs pool but weak transfer.

2. Pool too hard too early
- learning stalls, critic destabilizes.

3. Poor anchor ratio
- too little anchor: fixed-bot performance regresses.
- too much anchor: self-play gains slow.

4. PFSP without enough data
- early PFSP winrates are noisy; static priors and bootstrap quality become important.
