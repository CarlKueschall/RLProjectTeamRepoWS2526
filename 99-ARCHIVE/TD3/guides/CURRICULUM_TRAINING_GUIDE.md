# Curriculum Training Guide - 100k Episodes (Seed 53)

## Overview

This is a 3-stage curriculum training run with conservative hyperparameters:
- **Stage 1 (0-30k)**: Train against weak opponent to build foundations
- **Stage 2 (30k-50k)**: Switch to strong opponent for harder challenges
- **Stage 3 (50k-100k)**: Self-play automatically activates at 50k

**Seed**: 53
**Total Duration**: ~24-48 hours on A4000
**Expected Performance**: 75%+ vs strong by end

---

## Stage 1: Foundation Training (0-30k episodes)

**Objective**: Learn basic skills against weak opponent

**Command**:
```bash
cd TD3
python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 30000 \
    --seed 53 \
    --warmup_episodes 1000 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.999965 \
    --batch_size 512 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    --buffer_size 1000000 \
    --q_clip 25.0 \
    --q_clip_mode hard \
    --q_warning_threshold 10.0 \
    --hidden_actor 1024 1024 \
    --hidden_critic 1024 1024 200 \
    --reward_shaping \
    --tie_penalty -3.0 \
    --lr_decay \
    --lr_min_factor 0.1 \
    --episode_block_size 20 \
    --eval_interval 500 \
    --eval_episodes 50 \
    --log_interval 20 \
    --save_interval 500 \
    --gif_episodes 3 \
    --self_play_start 100000 \
    --self_play_pool_size 25 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.5 \
    --use_dual_buffers \
    --use_pfsp \
    --pfsp_mode variance \
    --dynamic_anchor_mixing \
    --performance_gated_selfplay \
    --selfplay_gate_winrate 0.90 \
    --regression_rollback \
    --regression_threshold 0.15
```

**What to Monitor**:
- Win rate vs weak should reach 60-75% by episode 20k
- Ties should decrease to <20%
- Watch W&B for smooth learning curves

**When to Stop**: After episode 30,000 completes

**Checkpoint Location**: `./results/checkpoints/checkpoint_episode_30000.pth`

---

## Stage 2: Difficulty Increase (30k-50k episodes)

**Objective**: Transfer skills to strong opponent

**Command** (run after Stage 1 completes):
```bash
cd TD3
python3 train_hockey.py \
    --mode NORMAL \
    --opponent strong \
    --max_episodes 50000 \
    --seed 53 \
    --warmup_episodes 1000 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.999965 \
    --batch_size 512 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    --buffer_size 1000000 \
    --q_clip 25.0 \
    --q_clip_mode hard \
    --q_warning_threshold 10.0 \
    --hidden_actor 1024 1024 \
    --hidden_critic 1024 1024 200 \
    --reward_shaping \
    --tie_penalty -3.0 \
    --lr_decay \
    --lr_min_factor 0.1 \
    --episode_block_size 20 \
    --eval_interval 500 \
    --eval_episodes 50 \
    --log_interval 20 \
    --save_interval 500 \
    --gif_episodes 3 \
    --self_play_start 100000 \
    --self_play_pool_size 25 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.5 \
    --use_dual_buffers \
    --use_pfsp \
    --pfsp_mode variance \
    --dynamic_anchor_mixing \
    --performance_gated_selfplay \
    --selfplay_gate_winrate 0.90 \
    --regression_rollback \
    --regression_threshold 0.15 \
    --resume ./results/checkpoints/checkpoint_episode_30000.pth
```

**Key Changes from Stage 1**:
- `--opponent weak` → `--opponent strong`
- `--max_episodes 30000` → `--max_episodes 50000`
- Added `--resume ./results/checkpoints/checkpoint_episode_30000.pth`

**What to Monitor**:
- Win rate will DROP initially (40-50%) - this is expected!
- Should gradually improve to 55-65% by episode 50k
- Loss/reward curves may spike briefly, then stabilize

**When to Stop**: After episode 50,000 completes

**Checkpoint Location**: `./results/checkpoints/checkpoint_episode_50000.pth`

---

## Stage 3: Self-Play Mastery (50k-100k episodes)

**Objective**: Polish against strong opponent + diverse self-play pool

**Command** (run after Stage 2 completes):
```bash
cd TD3
python3 train_hockey.py \
    --mode NORMAL \
    --opponent strong \
    --max_episodes 100000 \
    --seed 53 \
    --warmup_episodes 1000 \
    --eps 1.0 \
    --eps_min 0.05 \
    --eps_decay 0.999965 \
    --batch_size 512 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --grad_clip 1.0 \
    --buffer_size 1000000 \
    --q_clip 25.0 \
    --q_clip_mode hard \
    --q_warning_threshold 10.0 \
    --hidden_actor 1024 1024 \
    --hidden_critic 1024 1024 200 \
    --reward_shaping \
    --tie_penalty -3.0 \
    --lr_decay \
    --lr_min_factor 0.1 \
    --episode_block_size 20 \
    --eval_interval 500 \
    --eval_episodes 50 \
    --log_interval 20 \
    --save_interval 500 \
    --gif_episodes 3 \
    --self_play_start 50000 \
    --self_play_pool_size 25 \
    --self_play_save_interval 500 \
    --self_play_weak_ratio 0.5 \
    --use_dual_buffers \
    --use_pfsp \
    --pfsp_mode variance \
    --dynamic_anchor_mixing \
    --performance_gated_selfplay \
    --selfplay_gate_winrate 0.90 \
    --regression_rollback \
    --regression_threshold 0.15 \
    --resume ./results/checkpoints/checkpoint_episode_50000.pth
```

**Key Changes from Stage 2**:
- `--max_episodes 50000` → `--max_episodes 100000`
- `--self_play_start 100000` → `--self_play_start 50000` (activates immediately!)
- Updated `--resume` to episode 50000 checkpoint

**What to Monitor**:
- Self-play pool fills up (25 checkpoints saved every 500 episodes)
- Win rate vs self-play pool should stabilize around 50% (balanced opponents)
- Win rate vs strong opponent should reach 70-80%
- Regression rollback may activate if performance drops

**When to Stop**: After episode 100,000 completes (DONE!)

**Final Checkpoint**: `./results/checkpoints/checkpoint_episode_100000.pth`

---

## Quick Reference: Stage Transitions

| Stage | Episodes | Opponent | Self-Play | Command Changes |
|-------|----------|----------|-----------|-----------------|
| 1 | 0-30k | weak | disabled | Initial training |
| 2 | 30k-50k | strong | disabled | Change opponent, add --resume |
| 3 | 50k-100k | strong | ACTIVE | Change max_episodes, update --resume |

---

## Troubleshooting

**If training crashes between stages:**
1. Check last saved checkpoint: `ls -ltr ./results/checkpoints/`
2. Resume from latest checkpoint using `--resume <path>`
3. Adjust `--max_episodes` to desired end point

**If win rate is too low at Stage 2 transition (<50% vs weak):**
- Consider extending Stage 1 to 40k episodes
- Or lower learning rate for Stage 2 (--lr_actor 0.0002 --lr_critic 0.0002)

**If self-play causes performance collapse:**
- Regression rollback should auto-revert
- If persistent, lower `--self_play_weak_ratio` to 0.3 (more anchor opponent)

---

## Testing Final Checkpoint

After Stage 3 completes, test the final agent:

```bash
# Test vs weak
python test_hockey.py --checkpoint ./results/checkpoints/checkpoint_episode_100000.pth --opponent weak --episodes 100 --hidden_actor 1024 1024 --hidden_critic 1024 1024 200

# Test vs strong
python test_hockey.py --checkpoint ./results/checkpoints/checkpoint_episode_100000.pth --opponent strong --episodes 100 --hidden_actor 1024 1024 --hidden_critic 1024 1024 200
```

**Expected Results**:
- vs weak: 75-85% win rate
- vs strong: 70-80% win rate

---

## Notes

- All stages use the SAME seed (53) for reproducibility
- Buffer and replay memory persist across stage transitions
- Learning rate decay continues from previous stage (cosine annealing)
- W&B will show all stages as a continuous run (same seed)
