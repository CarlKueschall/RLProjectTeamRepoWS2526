# W&B Metrics Configuration Guide

## Overview

This document describes the comprehensive W&B metrics system for tracking TD3 self-play hockey training. The system provides detailed insights into opponent mixing, buffer distribution, PFSP curriculum progression, and performance across different opponent types.

---

## Evaluation Metrics (`eval/` namespace)

### Per-Opponent-Type Evaluation

The training system now evaluates performance against **three distinct opponent types** at regular intervals (configurable via `--eval_interval`):

#### **Weak Opponent** (`eval/weak/`)
- **Purpose**: Baseline performance metric; should maintain high win-rate to prevent catastrophic forgetting
- **Metrics**:
  - `win_rate`: Win rate including ties (W/(W+L+T))
  - `win_rate_decisive`: Win rate in decisive games only (W/(W+L))
  - `tie_rate`: Frequency of draws
  - `loss_rate`: Frequency of losses
  - `avg_reward`: Average episode reward against weak opponent
  - `wins`, `losses`, `ties`: Raw game counts

#### **Strong Opponent** (`eval/strong/`)
- **Purpose**: Primary training opponent during pre-self-play phase; indicates learning progress
- **Metrics**: Same as weak opponent
- **Interpretation**: Should improve monotonically until self-play activation

#### **Self-Play Opponent** (`eval/selfplay/`)
- **Purpose**: Mirror-match evaluation; tests generalization and robustness against similar-strength opponents
- **Active**: Only logged when `--self_play_start > 0` and self-play is active
- **Metrics**:
  - Same as above, plus:
  - `opponent_age`: Age of current self-play opponent in episodes (indicates curriculum progression)
- **Interpretation**: Lower variance in self-play evaluation indicates more consistent generalization

---

## Self-Play Metrics (`selfplay/` namespace)

### Self-Play Status
- `active`: 1.0 if self-play active, 0.0 if training against fixed opponents
- `pool_size`: Number of past checkpoints in self-play opponent pool

### Buffer Management

#### **Target Buffer Composition**
- `weak_ratio_target`: Target percentage of sampling from weak opponent buffer (e.g., 0.4 = 40%)

#### **Anchor Buffer Balance** (weak vs strong episodes)
Critical for preventing catastrophic forgetting during self-play:
- `anchor_weak_episodes`: Number of episodes in anchor buffer from weak opponent
- `anchor_strong_episodes`: Number of episodes in anchor buffer from strong opponent
- `anchor_weak_ratio`: Ratio of weak episodes in anchor (target: 0.5 for balance)
- `anchor_strong_ratio`: Ratio of strong episodes in anchor (target: 0.5 for balance)
- `anchor_balance_score`: 1.0 (perfect balance) → 0.0 (all one type)
  - Formula: `1.0 - |weak_ratio - 0.5| * 2.0`
  - Used to detect if anchor buffer is becoming skewed

### Dynamic Anchor Mixing

When `--dynamic_anchor_mixing` is enabled:
- `anchor_ratio_current`: Current sampling ratio from anchor buffer
  - May differ from `weak_ratio_target` if forgetting is detected
  - Increases automatically when performance vs weak opponent drops

### PFSP (Prioritized Fictitious Self-Play) Metrics

Available when `--use_pfsp` and self-play is active:

#### **Opponent Pool Statistics**
- `pfsp_num_opponents_tracked`: Number of opponents with sufficient game history (≥10 games)
- `pfsp_avg_winrate`: Average win-rate across all tracked opponents
- `pfsp_std_winrate`: Standard deviation of win-rates (curriculum diversity)
- `pfsp_min_winrate`: Weakest tracked opponent's win-rate
- `pfsp_max_winrate`: Strongest tracked opponent's win-rate
- `pfsp_median_winrate`: Median opponent strength
- `pfsp_diversity_metric`: Measure of opponent difficulty spread (higher = more diverse curriculum)

#### **Current Opponent Information**
- `opponent_pool_index`: Index of current opponent in pool (0 = oldest, N-1 = newest)
- `opponent_checkpoint_episode`: Episode at which current opponent was saved
- `opponent_age_episodes`: How many episodes have passed since opponent was saved (curriculum progression metric)

### Regression Rollback Protection

When `--regression_rollback` is enabled:
- `best_eval_vs_weak`: Highest win-rate against weak opponent achieved so far
- `consecutive_eval_drops`: Number of consecutive eval intervals with performance drops
  - Triggers rollback when ≥2 consecutive drops of >`--regression_threshold` magnitude
- `rollback_enabled`: 1.0 if regression rollback is active

---

## Training Behavior Metrics (`selfplay/episode_type_*`)

Track which opponent type is being used each episode:
- `episode_type_weak`: 1.0 if episode used weak opponent, 0.0 otherwise
- `episode_type_strong`: 1.0 if episode used strong opponent, 0.0 otherwise
- `episode_type_selfplay`: 1.0 if episode used self-play opponent, 0.0 otherwise

**Usage**: Can be summed over logging intervals to understand opponent distribution in training

---

## Behavior Visualization (GIFs)

### Per-Opponent-Type GIFs

The training system now records gameplay GIFs against all opponent types:

#### **Gameplay vs Weak** (`behavior/gameplay_vs_weak`)
- Shows baseline behavior and skill execution
- Should demonstrate consistent control and strategy

#### **Gameplay vs Strong** (`behavior/gameplay_vs_strong`)
- Shows ability to compete against stronger opponent
- Indicates whether agent adapts to opponent difficulty

#### **Gameplay vs Self-Play** (`behavior/gameplay_vs_selfplay`)
- Only recorded when self-play is active
- Shows behavior against similar-strength opponent
- Useful for detecting degenerate strategies

**Recording Frequency**: Controlled by `--gif_interval` and `--gif_episodes`

---

## Example W&B Dashboard Configuration

### Recommended Chart Layout

1. **Performance Trends** (line charts)
   - `eval/weak/win_rate_decisive` (should be high and stable)
   - `eval/strong/win_rate_decisive` (should improve until self-play)
   - `eval/selfplay/win_rate_decisive` (should converge around 0.5)

2. **Self-Play Buffer Distribution** (line charts)
   - `selfplay/anchor_weak_ratio` (target: 0.5)
   - `selfplay/anchor_strong_ratio` (target: 0.5)
   - `selfplay/anchor_balance_score` (target: 1.0)

3. **Curriculum Progression** (line chart)
   - `selfplay/opponent_age_episodes` (increases as curriculum progresses)
   - `selfplay/pfsp_diversity_metric` (variance of opponent difficulties)
   - `selfplay/pool_size` (number of opponents available)

4. **PFSP Statistics** (line charts)
   - `selfplay/pfsp_avg_winrate` (track average opponent strength)
   - `selfplay/pfsp_std_winrate` (curriculum diversity)
   - `selfplay/pfsp_median_winrate` (typical opponent strength)

5. **Safety Metrics** (line charts - important for monitoring)
   - `selfplay/best_eval_vs_weak` (should be ≥ 0.85 to activate self-play)
   - `selfplay/consecutive_eval_drops` (should stay 0; >0 indicates potential rollback)
   - `eval/weak/win_rate_decisive` (emergency check: should never drop critically)

6. **Training Mix** (stacked area chart)
   - `selfplay/episode_type_weak`
   - `selfplay/episode_type_strong`
   - `selfplay/episode_type_selfplay`
   - Shows opponent type distribution over time

---

## Metric Interpretation Guide

### Healthy Self-Play Training Indicators
✓ `anchor_balance_score` > 0.8 (anchor buffer well-balanced)
✓ `eval/weak/win_rate_decisive` ≥ 0.85 (no catastrophic forgetting)
✓ `eval/selfplay/win_rate_decisive` ≈ 0.50 ± 0.15 (competitive self-play)
✓ `pfsp_std_winrate` > 0.1 (good curriculum diversity)
✓ `consecutive_eval_drops` = 0 (no regression events)

### Warning Signs
⚠ `anchor_balance_score` < 0.6 (anchor buffer skewed; should increase `weak_ratio_target`)
⚠ `eval/weak/win_rate_decisive` < 0.80 (forgetting detected; check dynamic_anchor_mixing)
⚠ `eval/selfplay/win_rate_decisive` < 0.30 or > 0.70 (curriculum imbalanced)
⚠ `consecutive_eval_drops` ≥ 1 (approaching regression rollback threshold)
⚠ `pool_size` < `self_play_pool_size` / 2 (opponent diversity may be limited)

### Critical Issues
✗ `eval/weak/win_rate_decisive` < 0.70 (catastrophic forgetting; rollback likely)
✗ `consecutive_eval_drops` ≥ 2 (regression rollback triggered if enabled)
✗ `pfsp_std_winrate` = 0 (all opponents same strength; curriculum failure)

---

## Implementation Details

### Self-Play Manager (`opponents/self_play.py`)
- **`get_stats()` method**: Computes all self-play metrics using:
  - `opponent_winrates`: Dict mapping opponent paths to deque of game results
  - `anchor_weak_count`, `anchor_strong_count`: Episode counts from anchor buffer
  - `opponent_episodes`: Dict mapping opponent paths to save episodes
  - `best_eval_vs_weak`: Tracks best performance vs weak opponent
  - `consecutive_eval_drops`: Counts consecutive evaluation regressions

### Training Loop (`train_hockey.py`)
- **Evaluation Section**: Calls `evaluate_vs_opponent()` for each opponent type
- **GIF Recording**: Creates separate GIFs for weak, strong, and self-play opponents
- **Logging Section**: Aggregates metrics with `selfplay/` prefix and logs to W&B

### Evaluation Function (`evaluation/evaluator.py`)
- **`evaluate_vs_opponent()`**: Returns detailed win/loss/tie statistics
- Returns both overall and decisive-game-only win rates for nuanced performance tracking

---

## Configuration Examples

### Baseline Self-Play (Balanced)
```bash
--self_play_start 25000
--use_dual_buffers
--use_pfsp --pfsp_mode variance
--dynamic_anchor_mixing
--selfplay_gate_winrate 0.85
--self_play_weak_ratio 0.4
```
**W&B Expectations**:
- `anchor_balance_score` → 0.95-1.0 (excellent balance)
- `eval/weak/win_rate` → 0.85+ (strong baseline)
- `eval/selfplay/win_rate` → 0.45-0.55 (good competition)

### Aggressive Self-Play (Fast Convergence)
```bash
--self_play_start 10000
--use_dual_buffers
--dynamic_anchor_mixing
--selfplay_gate_winrate 0.75
--self_play_weak_ratio 0.2
--regression_rollback
```
**W&B Expectations**:
- More rapid progress vs self-play
- Risk of forgetting weak opponent skills (monitor `eval/weak/win_rate`)
- Regression rollback provides safety net

### Conservative Self-Play (Stability Focus)
```bash
--self_play_start 40000
--use_dual_buffers
--selfplay_gate_winrate 0.90
--self_play_weak_ratio 0.6
--regression_rollback
```
**W&B Expectations**:
- Slower but more stable training
- Excellent baseline maintenance
- May need longer total training time

---

## See Also
- `TOURNAMENT_KEEP_MODE.md`: Tournament environment configuration
- Main training script: `02-SRC/TD3/train_hockey.py`
- Self-play manager: `02-SRC/TD3/opponents/self_play.py`

