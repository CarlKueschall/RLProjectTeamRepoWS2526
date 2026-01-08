# Self-Play Metrics Debugging & Fix

## Problem

Self-play metrics were not appearing in WandB logs, even when training with self-play enabled.

## Root Cause Analysis

The issue was in `train_hockey.py` at line 615:

```python
if args.self_play_start > 0:
    sp_stats = self_play_manager.get_stats()
    # ... add to log_metrics
```

**The Problem:** This condition only logs self-play metrics when `args.self_play_start > 0`. 

However, there are two scenarios where this fails:
1. **During initial training phase (before self-play starts)**: When `self_play_start=25000`, episodes 0-24999 won't log any self-play metrics, making it impossible to track the ramp-up to self-play activation.
2. **When using `--self_play_start 0`**: This was the actual problem in your run! With `self_play_start=0`, the condition evaluates to `False`, and NO self-play metrics are ever collected or logged.

## Solution

Changed the condition from:
```python
if args.self_play_start > 0:
```

To:
```python
if self_play_manager is not None:
```

This ensures that:
- Self-play metrics are logged **whenever self-play infrastructure is initialized**, regardless of whether self-play has activated yet
- The metrics are always available for tracking opponent mixing, buffer distribution, and curriculum progression
- Pre-self-play phases can be analyzed to understand the agent's behavior before self-play kicks in

## Additional Improvements

Also added explicit tracking of which opponent type was faced in each episode:
- `selfplay/episode_opponent_type_weak`: 1.0 if episode was vs weak opponent, 0.0 otherwise
- `selfplay/episode_opponent_type_strong`: 1.0 if episode was vs strong opponent, 0.0 otherwise
- `selfplay/episode_opponent_type_selfplay`: 1.0 if episode was vs self-play opponent, 0.0 otherwise

This provides a clear per-episode record of opponent mixing in WandB.

## What Gets Logged Now

### Always (from `self_play_manager.get_stats()`):
- `selfplay/active`: Whether self-play is currently active (0 or 1)
- `selfplay/pool_size`: Number of historical agents in the opponent pool
- `selfplay/weak_ratio_target`: Target ratio of weak vs strong episodes
- `selfplay/anchor_weak_episodes`: Count of weak opponent episodes in anchor buffer
- `selfplay/anchor_strong_episodes`: Count of strong opponent episodes in anchor buffer
- `selfplay/anchor_weak_ratio`: Ratio of weak to total anchor episodes
- `selfplay/anchor_strong_ratio`: Ratio of strong to total anchor episodes
- And many more PFSP, regression, and mixing metrics...

### Per-Episode:
- `selfplay/episode_opponent_type_weak/strong/selfplay`: Which opponent type was faced

## WandB Charts to Create

With these metrics now being logged, you can create useful charts:

1. **Opponent Mixing Over Time** (stacked area):
   - `selfplay/episode_opponent_type_weak`
   - `selfplay/episode_opponent_type_strong`
   - `selfplay/episode_opponent_type_selfplay`

2. **Anchor Buffer Balance**:
   - `selfplay/anchor_weak_ratio` (should oscillate around 0.5 if balancing works)

3. **Self-Play Ramp-Up**:
   - `selfplay/active` (shows when self-play activates)
   - `selfplay/pool_size` (shows opponent pool growth)

4. **PFSP Difficulty Curriculum**:
   - `selfplay/pfsp_avg_winrate` (should track around 50% for well-selected opponents)
   - `selfplay/pfsp_diversity_metric` (tracks opponent difficulty variety)

