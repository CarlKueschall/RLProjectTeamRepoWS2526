# Puck Touches Metric Bug - Investigation and Fix

## Issue Summary

The `behavior/puck_touches` metric was showing **0.0 throughout training** despite the agent winning games (eval results: W/W/T), which logically requires touching the puck.

## Root Cause

The puck_touches counter was **only tracked inside the strategic_shaper.compute() method**, which is guarded by:

```python
if args.reward_shaping and args.use_strategic_rewards:
    strategic_bonuses = strategic_shaper.compute(obs_next, info, dist_to_puck)
    # puck_touches incremented here
```

The focused training scripts use `--no_strategic_rewards` to disable strategic reward bonuses. This causes:
1. `args.use_strategic_rewards = False`
2. `strategic_shaper.compute()` is never called
3. `puck_touches` counter is never incremented
4. Metric shows 0.0 for all episodes

## Why the Agent Still Won

Despite the metric showing 0 touches, the agent won because:
- **Sparse goal rewards** (+10 for win, -10 for loss, 0 for tie) were the primary learning signal
- **PBRS bonus** (with `--pbrs_scale 0.5`) provided auxiliary guidance for being close to puck
- **Strategic rewards** (disabled with `--no_strategic_rewards`) were not needed for basic winning behavior
- The agent learns to score through the main RL objective, not through reward shaping bonuses

## The Fix

### Changes Made

**1. train_hockey.py (lines 580-587)**
- Added **always-on puck_touches tracking** before the strategic_rewards guard
- Increments counter whenever environment detects a touch OR when agent is within 0.3 of puck
- This ensures metric accuracy regardless of `--no_strategic_rewards` flag

```python
# ALWAYS track puck touches (decoupled from strategic_rewards flag)
# This ensures behavior metrics are accurate regardless of reward configuration
env_touch_reward = info.get('reward_touch_puck', 0.0)
if env_touch_reward > 0:
    strategic_shaper.puck_touches += 1
elif dist_to_puck < 0.3 and env_touch_reward == 0:
    # Backup: if env didn't detect, use distance threshold
    strategic_shaper.puck_touches += 1
```

**2. strategic_rewards.py (lines 59-61, 124-128)**
- **Removed puck_touches increments** from compute() method
- Prevents double-counting now that global tracking is in place
- Reward bonus for touching (`env_touch_reward * 0.3`) is still applied if strategic_rewards enabled

## Impact

| Scenario | Before | After |
|----------|--------|-------|
| `--use_strategic_rewards` | Correct (counted in compute) | Correct (counted globally) |
| `--no_strategic_rewards` | BROKEN (never incremented) | FIXED (tracked globally) |
| Double-counting risk | None with `--no_strategic_rewards` | Prevented by removing compute() increment |

## Verification

Next training run with seed 55 Stage 2 (or any run with `--no_strategic_rewards`) should show:
- `behavior/puck_touches` gradually increasing from 0
- Metric values correlating with observed agent behavior in videos
- No more all-zeros metric despite agent winning games

## Timeline

1. **Seed 55 Stage 1 (6400/30000 eps)**: Shows 0 puck_touches (metric bug)
2. **After this fix**: Next runs will have accurate puck_touches tracking
3. **All future runs**: Metric will be decoupled from strategic_rewards flag

## Technical Notes

- The puck_touches tracking now uses two methods:
  1. **Physics-based**: `info.get('reward_touch_puck', 0.0)` from environment
  2. **Fallback**: Distance threshold `dist_to_puck < 0.3` if env didn't register
- Works for all reward configurations: PBRS only, strategic only, both, or neither
- The reward bonus for touching (`bonuses['puck_touches']`) is still only applied if strategic_rewards enabled
