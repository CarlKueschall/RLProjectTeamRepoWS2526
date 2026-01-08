# Self-Play Activation Bug Fixes

## Summary

Found and fixed **3 critical bugs** preventing self-play from ever activating, even with high win rates and loose performance gates.

---

## Bug #1: Self-Play Management Gated by Parameter Value

### Location
`train_hockey.py`, line 357

### The Bug
```python
if args.self_play_start > 0:
    # ... all self-play management code ...
    opponent_type = self_play_manager.select_opponent(i_episode)
```

### Impact
When running with `--self_play_start 0` (immediate self-play), the entire self-play management block was **completely skipped**. The code never even attempted to check if self-play should activate.

### Root Cause
The code was checking the parameter value (`args.self_play_start > 0`) instead of checking whether self-play infrastructure was initialized (`self_play_manager is not None`).

### Fix
```python
if self_play_manager is not None:
    # ... all self-play management code ...
    opponent_type = self_play_manager.select_opponent(i_episode)
```

Now self-play management runs whenever self-play infrastructure exists, regardless of the `self_play_start` parameter.

---

## Bug #2: Hardcoded 500-Episode Requirement

### Location
`opponents/self_play.py`, lines 156-158

### The Bug
```python
def should_activate(self, episode, eval_vs_weak, rolling_variance):
    if not self.performance_gated:
        return episode >= self.start_episode
    
    # HARDCODED: Need at least 500 episodes of eval data
    episodes_needed = 500
    if episode < self.start_episode + episodes_needed:
        return False  # ← Always returns False for first 500 episodes!
```

### Impact
Even with performance gates set to trivial values (e.g., `--selfplay_gate_winrate 0.2`), self-play could never activate before episode 500+ if using performance gating.

This makes no sense when:
1. You load a checkpoint at episode 9,500
2. You want immediate self-play with `--self_play_start 0`
3. You set loose gates like `0.2` win rate

The agent would have to train an additional 500 episodes before self-play could even be considered.

### Root Cause
The 500-episode requirement was meant to ensure sufficient eval data exists, but it was hardcoded and didn't account for:
- Loaded checkpoints starting mid-training
- Explicit requests for immediate self-play
- Very loose performance gates indicating user wants quick activation

### Fix
```python
def should_activate(self, episode, eval_vs_weak, rolling_variance):
    if not self.performance_gated:
        return episode >= self.start_episode
    
    # Need at least one evaluation before we can check performance gates
    if eval_vs_weak is None:
        return False
    
    # Check win-rate gate
    if eval_vs_weak < self.gate_winrate:
        return False
    
    # Check variance gate (if rolling variance available)
    if rolling_variance is not None and rolling_variance > self.gate_variance:
        return False
    
    return True
```

Now activation happens immediately after the **first evaluation** (at `eval_interval`), as long as performance gates are met.

---

## Bug #3: Related Gating Issues

### Locations
- `train_hockey.py`, line 546: PFSP result tracking
- `train_hockey.py`, line 730: Dynamic anchor mixing updates
- `train_hockey.py`, line 854: Regression rollback checkpoint updates

### The Bug
These sections were also gated by `if args.self_play_start > 0`, preventing them from running with `--self_play_start 0`.

### Fix
Changed all to check `if self_play_manager is not None` for consistency.

---

## Expected Behavior After Fix

### With `--self_play_start 0 --performance_gated_selfplay --selfplay_gate_winrate 0.2`:

1. **Episode 1-249**: No eval yet, self-play can't activate
2. **Episode 250** (first eval): 
   - Evaluate vs weak opponent
   - If win rate ≥ 0.2 and variance gate passes → **SELF-PLAY ACTIVATES**
3. **Episode 251+**: Start playing against self-play pool
4. WandB shows:
   - `selfplay/active` switches from 0 to 1
   - `selfplay/pool_size` starts growing
   - `selfplay/episode_opponent_type_selfplay` becomes 1.0

### With `--self_play_start 25000` (original behavior):
1. **Episode 1-24999**: Train only vs strong opponent
2. **Episode 25000+**: Begin checking for self-play activation
3. Activation subject to performance gates (if enabled)

---

## Testing the Fix

To verify self-play now activates correctly:

1. Run with immediate self-play:
   ```
   python3 train_hockey.py \
     --self_play_start 0 \
     --performance_gated_selfplay \
     --selfplay_gate_winrate 0.2 \
     --selfplay_gate_variance 0.5 \
     --eval_interval 250 \
     ... other args ...
   ```

2. Check WandB at episode 250+:
   - `selfplay/active` should change from 0 → 1
   - `selfplay/pool_size` should start increasing from 0

3. Check opponent type metrics:
   - Before activation: `episode_opponent_type_strong` = 1.0, others = 0.0
   - After activation: `episode_opponent_type_selfplay` = varies (mixed with weak/strong)

---

## Why These Bugs Weren't Caught

1. Most prior runs used `--self_play_start > 0` (e.g., 25000), so Bug #1 never manifested
2. The 500-episode requirement was hidden as a magic constant without clear documentation
3. The gates were tested with episodes starting from scratch (episode 0), not loaded checkpoints
4. No integration test checked immediate self-play activation with loaded checkpoints

---

## Lessons Learned

- ✅ Always gate self-play logic on **infrastructure existence** (`is not None`), not **parameter values** 
- ✅ Don't use magic constants for temporal requirements without flexibility options
- ✅ When loading checkpoints, treat them as if they "started" at that episode for timing logic
- ✅ Test with realistic scenarios: loaded checkpoints + immediate self-play

