# RL Hockey Project - TD3 with V10 Strategic Quality Fixes

## Current Status

**Phase:** V10 Strategic Quality & Diversity - Ready for Self-Play Testing
**Approach:** TD3 + V10 strategic fixes (opponent-aware shooting, reward rebalancing, diversity incentives)
**Last Updated:** 2026-01-03 (V10 Strategic Quality Fixes Applied)
**Current Performance:** 88% decisive win rate vs weak opponent (V9 baseline)
**Expected After V10:** 90%+ win rate with improved shot quality and strategy diversity
**Tournament Date:** 24.02.2026 | Submission 27.02.2026 23:55

---

## Project Overview

Reinforcement Learning project to develop agents for a simulated hockey game using the gymnasium API. **Team of 2** implementing TD3 with self-play learning, competing in a tournament.

**Environment:** [hockey-env](https://github.com/martius-lab/hockey-env)
**W&B Dashboard:** https://wandb.ai/carlkueschalledu/rl-hockey

---

## Training Architecture (V8 Simplified)

### Direct Training in NORMAL Mode

```
┌─────────────────────────────────────────────────────────┐
│         TD3 WITH V8 STABILITY FIXES TRAINING            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Current Phase: Direct Training (Episodes 0+)          │
│  ┌─────────────────────────────────────────────────┐  │
│  │ • Train against WEAK opponent                  │  │
│  │ • Mode: NORMAL (standard tournament mode)      │  │
│  │ • Current: 92.3% win rate achieved             │  │
│  │ • Target: 95%+ win rate (expected ~ep 10000)   │  │
│  │ • Checkpoints: every 500 episodes              │  │
│  │ • Evaluation: every 1000 episodes (deterministic)│  │
│  └─────────────────────────────────────────────────┘  │
│                      │                                  │
│        ┌─────────────┴──────────────┐                  │
│        │                            │                  │
│        ▼                            ▼                  │
│   TD3 Agent                  Weak Opponent             │
│   ┌─────────────────────┐  ┌──────────────────┐      │
│   │ Actor Network       │  │ Built-in AI Bot  │      │
│   │ (400-400 hidden)    │  │ Hockey Environment│      │
│   │                     │  │ Level: weak      │      │
│   │ 2 Critic Nets       │  └──────────────────┘      │
│   │ (400-400-200)       │                             │
│   │                     │  Self-Play (Future)         │
│   │ Q-clip: ±25.0       │  ┌──────────────────┐      │
│   └─────────────────────┘  │ Pool: off for now│      │
│        ↓                    │ Activates ep8000+│      │
│   Replay Buffer (500k)      └──────────────────┘      │
│   • Diverse state distribution (random seeds)         │
│   • PBRS rewards (properly scaled 10x fix)           │
│   • Slow decay exploration (eps: 1.0 → 0.05)         │
│   • 4630+ episodes collected                         │
│                                                       │
└─────────────────────────────────────────────────────────┘
```

**Why V8 Changes:** Removed TRAIN_SHOOTING complexity and implemented critical bug fixes.

---

## V10 Strategic Quality & Diversity (2026-01-03)

### Problem Solved

Your observation was spot-on: **Dense reward imbalance (17:1) + blind shooting → wild shooting + infinite puck loops**

### Solutions Implemented

**Fix #1: Reward Rebalancing (50-70% reduction)**

- Reduced all dense reward multipliers to achieve 2:1 dense:sparse ratio
- Puck direction bonus cut 67% (was major wild-shooting driver)
- **Result:** Goals now ~10x more valuable than intermediate rewards

**Fix #2: Opponent-Aware Shooting**

- Calculates opponent position relative to shot corridor
- Penalizes blocked shots (-0.2), rewards clear shots (+0.15)
- **Result:** Agent learns to wait for openings instead of wild shooting

**Fix #3: Strategic Diversity Incentives**

- Tracks attack sides (left/right/center)
- Awards diversity bonus for varied attacks (+0.5 per unique side)
- **Result:** Agent attacks from multiple angles, forces opponent to move

**Fix #4: Opponent Forcing Metric**

- Rewards when agent makes opponent move significantly
- Implements tennis/table-tennis style play principle
- **Result:** Strategic pressure becomes valuable outcome

### Expected Impact

- ✅ Eliminate wild shooting behavior
- ✅ Improve shot quality (clear vs blocked)
- ✅ Increase attack diversity (left/right/center)
- ✅ Add strategic depth (forcing opponent movement)

---

## V9 Anti-Forgetting Improvements (2026-01-03, V9.1 Bug Fixes Applied)

### What's New in V9

**Priority 1: Catastrophic Forgetting Prevention**

1. **Dual Replay Buffers** - Separate buffers for anchor (vs weak) and pool (vs self-play) to prevent distributional interference
2. **Dynamic Anchor Mixing** - Automatically adjusts weak opponent ratio (40-70%) based on forgetting detection
3. **PFSP Opponent Selection** - Variance curriculum focuses training on ~50% win-rate opponents (most informative)

**Priority 2: Stability & Safety**
4. **Performance-Gated Self-Play** - Activates self-play only after decisive win-rate ≥85% + low variance (competence-based)
5. **Regression Rollback Guard** - Automatic checkpoint rollback if eval drops >15% for 2 consecutive evals
6. **PBRS Terminal Fix** - Already in V8 (forces Φ(terminal)=0 to prevent episodic bias)

### V9.1 Critical Bug Fixes (2026-01-03)

**Gate Check Logic Fix:**

- **Bug:** Performance gate checked `len(rolling_outcomes) >= 500` when `rolling_outcomes.maxlen=100` (impossible condition)
- **Fix:** Changed to episode-based check: `i_episode >= (args.self_play_start + 500)`
- **Impact:** Self-play gate now properly activates at episode 2000 (1500 + 500) when performance conditions are met
- **Metrics:** Gates on `eval_decisive ≥ 85%` AND `rolling_variance ≤ 0.12`

### Why V9 Matters

**Problem V8 Couldn't Solve:** Self-play causes 80%→60% regressions due to:

- Distribution shift (pool opponent != target opponent)
- Catastrophic forgetting of weak-bot strategy
- Poor opponent sampling (wasting time on trivial opponents)

**V9 Solution:** Research-backed anti-forgetting mechanisms from AlphaStar, OpenAI Five, and academic literature.

**Expected Outcome:** Stable 95%+ vs weak throughout self-play (tournament-ready).

---

## V8 Stability Improvements

### Why Direct Training is Working

1. **Fixed Q-Value Explosion:** Reduced reward multipliers 10x to match Q-clip bounds

   - Was: Rewards -30 to +272 (impossible to fit in Q-clip ±25)
   - Now: Rewards -15 to +30 (fits perfectly in Q-clip ±25)
2. **Fixed State Distribution Mismatch:** Random seeds instead of sequential

   - Was: Sequential seeds (46, 47, 48...) → correlated starting positions → overfitting
   - Now: Truly random seeds (reproducible via np.random.seed) → generalized learning
3. **Consistent Evaluation:** GIF recording now uses same seeding as metrics

   - Was: GIFs showed poor results despite 92.3% eval win rate
   - Now: GIFs accurately represent agent performance

### Self-Play (Future Option)

If needed after reaching 95%+ win rate:

- **Automatic Curriculum:** Pool naturally gets stronger over time
- **Robust Learning:** Agent learns against diverse versions of itself
- **No Manual Tuning:** Pool management is automatic
- **Proven Approach:** Used by AlphaGo, AlphaZero, modern game-playing agents

---

## Key Components

### TD3 Algorithm

- **Twin Critics:** Use `min(Q1, Q2)` to reduce overestimation bias
- **Delayed Updates:** Update actor every 2 policy update steps
- **Target Smoothing:** Add clipped noise to target actions
- **Q-Value Clipping:** Bound targets to ±25.0 (V8 fix: was ±100)
- **Gradient Clipping:** Norm-based clipping (max_norm=1.0) for stability

### State Distribution Management (V8 Fix)

- **Random Initialization:** Each episode uses truly random seed (not sequential)
- **Reproducibility:** Seeded via `np.random.seed(args.seed + i_episode)` for reproducibility
- **Generalization:** Prevents overfitting to specific starting positions
- **Evaluation Consistency:** Training distribution matches evaluation distribution

### Reward Shaping (PBRS) - V8 Scaled

Using potential-based reward shaping with **proper 10x scaling reduction**:

- Goal proximity: 0.233 multiplier (was 2.33)
- Puck touch bonuses: 0.8 multiplier (was 8.0)
- Directional incentives: 0.6 multiplier (was 6.0)

All implemented with potential function `Φ(s)` to preserve optimal policy.
**Expected total rewards:** -15 to +30 per episode (fits in Q-clip ±25)

### Exploration Strategy

- **Initial:** `eps=1.0` (fully random actions)
- **Decay:** `0.99985` per episode (slow, ~4000 episodes to reach min)
- **Minimum:** `eps_min=0.05` (always maintain exploration)
- **Training vs Eval:** rolling_win_rate includes noise, eval_win_rate is deterministic

---

## Training Command (V9 - Current)

```bash
cd 02-SRC/TD3

# V9: Advanced Self-Play with Anti-Forgetting Features
python train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 25000 \
    --seed 46 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.99985 \
    --warmup_episodes 500 \
    --batch_size 512 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    --buffer_size 500000 \
    --reward_shaping \
    --q_clip 25.0 \
    --q_warning_threshold 10.0 \
    --hidden_actor 400 400 \
    --hidden_critic 400 400 200 \
    --log_interval 100 \
    --save_interval 500 \
    --gif_interval 1000 \
    --gif_episodes 3 \
    --eval_interval 1000 \
    --self_play_start 1500 \
    --self_play_pool_size 10 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.4 \
    --use_dual_buffers \
    --use_pfsp \
    --pfsp_mode variance \
    --dynamic_anchor_mixing \
    --performance_gated_selfplay \
    --selfplay_gate_winrate 0.85 \
    --selfplay_gate_variance 0.12 \
    --regression_rollback \
    --regression_threshold 0.15
```

### V8 Command (Baseline - No V9 Features)

```bash
cd 02-SRC/TD3

python train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 25000 \
    --seed 46 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.99985 \
    --warmup_episodes 500 \
    --batch_size 512 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    --buffer_size 500000 \
    --reward_shaping \
    --q_clip 25.0 \
    --q_warning_threshold 10.0 \
    --hidden_actor 400 400 \
    --hidden_critic 400 400 200 \
    --log_interval 100 \
    --save_interval 500 \
    --gif_interval 1000 \
    --gif_episodes 3 \
    --eval_interval 1000
```

### Key Parameters (V9 Updated)

| Parameter                               | Value    | Purpose                                                                            | Notes                                                                                     |
| --------------------------------------- | -------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **V9 Anti-Forgetting Mechanisms** |          |                                                                                    |                                                                                           |
| `--use_dual_buffers`                  | True     | Split replay buffer into anchor (33%, weak) + pool (67%, self-play)                | Prevents catastrophic forgetting of weak-bot strategy                                     |
| `--use_pfsp`                          | True     | Enable PFSP (Prioritized Fictitious Self-Play) opponent selection                  | Focuses on ~50% win-rate opponents (most informative training)                            |
| `--pfsp_mode`                         | variance | PFSP selection metric:`winrate * (1 - winrate)` peaks at 50%                     | Alternative:`hard` focuses on difficult opponents                                       |
| `--dynamic_anchor_mixing`             | True     | Auto-adjust weak opponent ratio (range: 40-70%) based on forgetting detection      | Increases anchor time if eval drops, prevents distribution collapse                       |
| `--performance_gated_selfplay`        | True     | Enable competence-based self-play activation                                       | Only activates if BOTH conditions met (see gate thresholds)                               |
| `--regression_rollback`               | True     | Auto-rollback to best checkpoint if performance drops                              | Safety net against catastrophic forgetting                                                |
| **Self-Play Gate Thresholds**     |          |                                                                                    |                                                                                           |
| `--self_play_start`                   | 1500     | Episode threshold: earliest point where self-play logic starts checking conditions | NOT the activation episode - gate will check conditions starting here + 500 more episodes |
| `--selfplay_gate_winrate`             | 0.85     | Threshold: agent must reach ≥85% decisive win-rate vs weak opponent               | Customizable: lower = earlier activation, higher = more confident                         |
| `--selfplay_gate_variance`            | 0.12     | Threshold: rolling outcome variance must be ≤0.12 for stability                   | Customizable: lower = more stable training required, higher = faster activation           |
| **V8 Stability Fixes**            |          |                                                                                    |                                                                                           |
| `--mode`                              | NORMAL   | Standard tournament mode (250 steps per episode)                                   | Direct training (was TRAIN_SHOOTING complexity)                                           |
| `--q_clip`                            | 25.0     | Bound Q-values to prevent explosion                                                | Matches reward scale post-10x fix (was 100.0)                                             |
| `--reward_shaping`                    | True     | PBRS with proper 10x scaling + terminal fix (Φ(terminal)=0)                       | V8: fixed terminal state bias, annealed during self-play                                  |
| **Training Mechanics**            |          |                                                                                    |                                                                                           |
| `--self_play_pool_size`               | 10       | Max opponents to keep in pool (oldest removed)                                     | Customizable: more = broader coverage, fewer = recent focus                               |
| `--self_play_save_interval`           | 500      | Add new opponent to pool every N episodes                                          | Customizable: higher = more stable opponents, lower = faster curriculum                   |
| `--self_play_weak_ratio`              | 0.4      | Baseline anchor ratio (40% weak, 60% pool)                                         | With dynamic mixing, this becomes the starting point                                      |

---

## Self-Play Activation Logic (How Gate Works)

The **performance gate** controls when self-play activates. It's NOT automatic at a fixed episode - it depends on agent competence.

### Gate Activation Flow

```
Episode 0-1500:
└─ Direct training vs weak opponent
   (self-play logic inactive)

Episode 1500 onwards:
└─ Gate checks every 250 episodes if conditions are met:

   Condition 1: Episode Threshold
   └─ Must be ≥ 500 episodes past --self_play_start
   └─ With --self_play_start 1500 → checks start at episode 2000

   Condition 2: Eval Performance
   └─ eval/vs_weak_win_rate_decisive ≥ --selfplay_gate_winrate (default 0.85)
   └─ This is DECISIVE win-rate (excludes ties)

   Condition 3: Training Stability
   └─ rolling_variance ≤ --selfplay_gate_variance (default 0.12)
   └─ Measured over 100-episode window

   ALL THREE CONDITIONS MUST BE TRUE:
   └─ IF (episode ≥ 2000) AND (eval ≥ 85%) AND (variance ≤ 0.12)
   └─ THEN → 🎮 SELF-PLAY ACTIVATES

After Activation:
└─ Opponent pool starts growing
└─ Dual buffers route transitions: weak → anchor, pool → pool
└─ PFSP selects opponents with ~50% win-rate
└─ Dynamic anchor mixing auto-adjusts (40-70% weak)
└─ Regression rollback monitors for catastrophic forgetting
```

### Customizing Gate Behavior

| To Activate Self-Play... | Adjust                                          | Effect                            |
| ------------------------ | ----------------------------------------------- | --------------------------------- |
| **Earlier**        | Lower `--selfplay_gate_winrate` (e.g., 0.80)  | Activates with less competence    |
| **Earlier**        | Lower `--selfplay_gate_variance` (e.g., 0.08) | Requires more stable training     |
| **Later**          | Higher `--selfplay_gate_winrate` (e.g., 0.90) | Activates with more confidence    |
| **Later**          | Increase `--self_play_start` (e.g., 2000)     | Delay all checks by 500 episodes  |
| **Skip gate**      | Set `--performance_gated_selfplay` False      | Use episode-based activation only |

### What These Parameters DON'T Do

- `--self_play_start 1500` does NOT mean "activate at episode 1500" - it means "start checking at 1500+500=2000"
- `--selfplay_gate_winrate 0.85` is a THRESHOLD, not the actual win-rate - agent may exceed it
- `--selfplay_gate_variance 0.12` doesn't control pool size or mixing - it's just the gate threshold

---

## Metrics & Monitoring

### Real Progress (Trust These)

**Critical Distinction: Overall vs Decisive Win-Rate**

- **`eval/vs_weak_win_rate_decisive`** - Decisive win-rate (excludes ties) - USED BY GATE
- **`eval/vs_weak_win_rate`** - Overall win-rate (includes ties) - Reference only
- **`performance/rolling_win_rate`** - Recent win rate including exploration noise (100-episode window)
- **`performance/cumulative_win_rate`** - Total wins/(wins+losses+ties)

The gate checks **decisive** win-rate, not overall. If eval shows 46% overall but 88% decisive, that's normal (ties don't count).

### Training Health

- **`training/epsilon`** - Exploration decay (should decrease smoothly)
- **`values/Q_avg`** - Average Q-values (should stay < 75)
- **`behavior/puck_touches`** - Agent engagement metric
- **`behavior/action_magnitude_avg`** - Action intensity

### Self-Play Gate Monitoring (Before Activation)

Monitor these while waiting for self-play to activate (episodes 1500-2000+):

- **`self_play/gate_eval_decisive`** - Current decisive win-rate (need ≥0.85)
- **`self_play/gate_eval_overall`** - Overall win-rate (reference, includes ties)
- **`self_play/gate_rolling_variance`** - Training stability metric (need ≤0.12)
- **`self_play/gate_eval_ready`** - 1.0 if eval ≥ threshold, 0.0 otherwise
- **`self_play/gate_variance_ready`** - 1.0 if variance ≤ threshold, 0.0 otherwise
- **`self_play/gate_both_ready`** - 1.0 if BOTH conditions met, 0.0 otherwise
- **`self_play/gate_episodes_waiting`** - Episodes since `--self_play_start` (will check at 500+)

### Self-Play Active (After Activation)

- **`self_play/active`** - Boolean flag (1.0 = self-play activated)
- **`self_play/pool_size`** - Number of checkpoints in pool (grows from 0 to max)
- **`behavior/gameplay_gif_vs_target`** - 3 eps vs weak opponent (should stay ≥0.85)
- **`behavior/gameplay_gif_vs_selfplay`** - 3 eps vs pool opponent (variable, training difficulty)

### V9 Anti-Forgetting Metrics

- **`self_play/current_anchor_ratio`** - Dynamic weak opponent ratio (40-70%)
- **`self_play/peak_eval_vs_weak`** - Best eval performance so far
- **`self_play/drop_from_peak`** - Current drop from peak (triggers forgetting boost)
- **`pfsp/avg_winrate_vs_pool`** - Average win-rate across pool opponents
- **`pfsp/min_winrate_vs_pool`** - Minimum win-rate (forgetting proxy)
- **`regression/best_eval_vs_weak`** - Best checkpoint performance
- **`regression/consecutive_drops`** - Drop counter (rollback at 2)

---

## Expected Performance (V9)

| Phase                              | Opponent             | Decisive Win Rate            | Timing                             |
| ---------------------------------- | -------------------- | ---------------------------- | ---------------------------------- |
| **Phase 1: Direct Training** | Weak only            | 0% → 85%+                   | Episodes 0-1500+                   |
| **Phase 2: Gate Waiting**    | Weak only            | 85%+ required                | Episodes 1500-2000                 |
| **Phase 3: Gate Active**     | Weak only            | 85%+ required                | Episodes 2000+ (if conditions met) |
| **Phase 4: Self-Play**       | Weak + Pool          | 85%+ vs weak, 30-70% vs pool | Episodes 2000+ (after activation)  |
| **Evaluation**               | Weak (deterministic) | Should stay ≥85% throughout | Every 1000 eps                     |

### Success Criteria (V9)

**Phase 1 - Direct Training (Episodes 0-1500):**

- ✅ Agent engages with puck (action_magnitude > 0.05)
- ✅ Rolling win-rate increases smoothly
- ✅ Q-values stay below 20 (no explosion)
- ✅ Reach eval_vs_weak_win_rate_decisive ≥ 85% by episode ~1500

**Phase 2 - Gate Waiting (Episodes 1500-2000):**

- ✅ Continue training vs weak opponent
- ✅ Monitor gate metrics (eval_decisive, rolling_variance)
- ✅ Console logs show gate status every 250 episodes

**Phase 3 - Self-Play Activation (Episode ~2000):**

- 🎯 Gate conditions met: eval ≥85% AND variance ≤0.12
- 🎯 Self-play activates: `self_play/active` → 1.0
- 🎯 Pool starts growing: `self_play/pool_size` → 1, 2, 3, ...
- 🎯 Maintain eval_vs_weak_win_rate_decisive ≥85% (forgetting check)

**Phase 4 - Self-Play Scaling (Episodes 2000-10000+):**

- 🎯 Reach 90%+ decisive win-rate vs weak by episode ~10000
- 🎯 Opponent pool reaches 10 by episode ~5000
- 🎯 No catastrophic drops (regression_rollback monitoring)

**Tournament Target:**

- 🎯 Reach 90%+ decisive win-rate vs weak opponent
- 🎯 Stable self-play throughout training

---

## File Structure

```
02-SRC/TD3/
├── train_hockey.py          # Main training script with self-play
├── td3_agent.py             # TD3 agent implementation
├── feedforward.py           # Neural network modules
├── memory.py                # Replay buffer
├── README.md                # TD3-specific documentation
└── results/
    ├── checkpoints/         # Regular training checkpoints
    └── selfplay_checkpoints/# Self-play opponent pool
```

---

## GIF Recording

Training automatically records gameplay every `--gif_interval` episodes (default 1000):

**Before Self-Play (Episodes 0-7999):**

- 3 episodes vs weak opponent
- Shows basic learning progress

**During Self-Play (Episodes 8000+):**

- `behavior/gameplay_gif_vs_target` - Agent vs weak bot (real progress)
- `behavior/gameplay_gif_vs_selfplay` - Agent vs pool opponent (training difficulty)

Both GIFs logged to W&B with episode count and results.

---

## Troubleshooting

| Issue                                    | Solution                                                               |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| **Training hangs**                 | Check GPU/MPS (should NOT be CPU)                                      |
| **Self-play doesn't activate**     | Verify reaching episode 8000, check logs for "SELF-PLAY ACTIVATED"     |
| **Win rate stuck at 33%**          | "Lazy learning" - check reward_shaping enabled, action_magnitude > 0.1 |
| **GIF recording fails**            | Install imageio:`pip install imageio`                                |
| **Self-play opponents won't load** | Check `selfplay_checkpoints/` exists, verify pool_size > 0           |
| **Q-values exploding (>100)**      | Reduce learning rate or check reward scale                             |

---

## Previous Attempts

### Why We Removed Curriculum Learning

- ❌ Complex system (700+ lines)
- ❌ Required 10-stage manual progression
- ❌ Prone to failure during stage transitions
- ❌ Difficult to debug when things went wrong

### Why Self-Play is Better

- ✅ Simpler (100+ lines vs 700+)
- ✅ Automatic difficulty progression
- ✅ Robust to sudden difficulty changes
- ✅ Proven effective in modern RL

---

## References

- **TD3 Paper:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- **Self-Play Concept:** AlphaGo, AlphaZero (DeepMind)
- **PBRS:** [Policy Invariant Reward Shaping](https://www.cs.cmu.edu/~ggordon/papers/ng-etal-99.pdf)

---

## Timeline

| Milestone                   | Status         | Target     | Buffer  |
| --------------------------- | -------------- | ---------- | ------- |
| Phase 1 training (8000 eps) | 🔄 IN PROGRESS | 3-5 hours  | 45 days |
| Reach 40% vs weak           | ⏳ PENDING     | 5000 eps   | 45 days |
| Self-play activate          | ⏳ PENDING     | 8000 eps   | 42 days |
| Reach 55% eval win rate     | ⏳ PENDING     | 20000 eps  | 30 days |
| Generate report plots       | ⏳ PENDING     | Day 25     | 25 days |
| **Tournament**        | ⏳ PENDING     | 24.02.2026 | 0 days  |

---

**For detailed phase-by-phase progress:** See `PROGRESS.md`
**For system architecture & context:** See `CLAUDE.md`
**For TD3-specific documentation:** See `02-SRC/TD3/README.md`
