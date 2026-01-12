# Focused Training Guide - Winning-Oriented Approach

## 🎯 Philosophy Change

Based on analysis of previous runs, we identified a **fundamental misalignment** between reward structure and the objective of winning:

### Problems with Previous Approach:
1. **PBRS dominance**: Dense potential-based rewards (0.715/episode) were overshadowing sparse goal rewards (±10)
2. **Strategic reward noise**: Shot quality, diversity, and forcing bonuses added complexity without clear benefit
3. **Tie penalty misalignment**: -3.0 penalty encouraged risk-averse play (ties treated as 70% losses)
4. **Q-value saturation**: Clip at 25.0 was too low, preventing learning distinction between good and great wins
5. **Network overfitting**: 1024 hidden units too large for amount of training data

### New Focused Approach:
1. **Reduce PBRS magnitude by 50%** (`--pbrs_scale 0.5`): Let sparse goal rewards dominate decision-making
2. **Disable strategic rewards** (`--no_strategic_rewards`): Pure focus on scoring goals, not intermediate metrics
3. **Remove tie penalty** (`--tie_penalty 0.0`): Ties are neutral outcomes, encouraging offensive play
4. **Raise Q-clip to 35.0** (`--q_clip 35.0`): Prevent value saturation, allow learning quality of wins
5. **Use 400 hidden units** (`--hidden_actor 400 400`): Prevent overfitting, empirically better performance

---

## 📊 Two Training Strategies

### Strategy A: Aggressive Training (Recommended for Speed)
**File**: `train_hockey_aggressive_focused.sbatch`

- **Duration**: 25,000 episodes (~6-8 hours on A4000)
- **Opponent**: Strong (immediately)
- **Self-play**: Starts at episode 8,000
- **Seed**: 54

**Use case**:
- Fast iteration for testing hypothesis
- Direct challenge from start
- Good for experienced agents

**Expected performance**:
- Episode 8k: 30-40% vs strong
- Episode 15k: 45-55% vs strong
- Episode 25k: 55-65% vs strong

---

### Strategy B: Curriculum Training (Recommended for Final Performance)
**Files**:
- `train_hockey_curriculum_focused_stage1.sbatch` (0-30k)
- `train_hockey_curriculum_focused_stage2.sbatch` (30k-50k)
- `train_hockey_curriculum_focused_stage3.sbatch` (50k-100k)

- **Duration**: 100,000 episodes total (~24-36 hours on A4000)
- **Phase 1**: Weak opponent (0-30k)
- **Phase 2**: Strong opponent (30k-50k)
- **Phase 3**: Strong + self-play (50k-100k)
- **Seed**: 55

**Use case**:
- Maximum final performance
- Stable learning progression
- Good for final tournament agent

**Expected performance**:
- Episode 30k: 60-75% vs weak
- Episode 50k: 55-70% vs strong
- Episode 100k: 75-85% vs strong

---

## 🚀 How to Run

### Aggressive Training (Quick Test)

```bash
# Submit the job
cd ~/02-SRC
sbatch train_hockey_aggressive_focused.sbatch

# Monitor progress
watch -n 10 squeue -u stud432

# Check W&B dashboard for live metrics
# Look for tags: "scaled-pbrs", "no-strategic-rewards"
```

### Curriculum Training (Full Run)

```bash
cd ~/02-SRC

# Stage 1: Train against weak (0-30k episodes)
sbatch train_hockey_curriculum_focused_stage1.sbatch

# Wait for Stage 1 to complete (~8-10 hours)
# Check that checkpoint exists:
# ls ~/02-SRC/TD3/results/checkpoints/checkpoint_episode_30000.pth

# Stage 2: Train against strong (30k-50k episodes)
# Update CHECKPOINT_PATH in stage2.sbatch if needed
sbatch train_hockey_curriculum_focused_stage2.sbatch

# Wait for Stage 2 to complete (~6-8 hours)
# Check that checkpoint exists:
# ls ~/02-SRC/TD3/results/checkpoints/checkpoint_episode_50000.pth

# Stage 3: Self-play mastery (50k-100k episodes)
# Update CHECKPOINT_PATH in stage3.sbatch if needed
sbatch train_hockey_curriculum_focused_stage3.sbatch

# Wait for Stage 3 to complete (~15-20 hours)
# Final checkpoint:
# ls ~/02-SRC/TD3/results/checkpoints/checkpoint_episode_100000.pth
```

---

## 📈 What to Monitor

### Critical Metrics (W&B Dashboard)

1. **Win Rate vs Strong** (`eval/strong/win_rate`)
   - **Aggressive target**: 55-65% by episode 25k
   - **Curriculum target**: 75-85% by episode 100k

2. **Win Rate vs Weak** (`eval/weak/win_rate`)
   - **Curriculum target**: 60-75% by episode 30k

3. **Q-Value Headroom** (`values/Q_max`)
   - Should NOT hit 35.0 (the new clip)
   - If hitting clip, raise to 40.0

4. **Goals Scored vs Conceded Ratio** (`scoring/goals_scored` / `scoring/goals_conceded`)
   - Target: > 0.6 by 50% through training

5. **Tie Rate** (`eval/strong/tie_rate`, `eval/weak/tie_rate`)
   - Should decrease over time (more decisive play)
   - Target: < 15% by end of training

### Secondary Metrics

6. **Actor Loss Stability** (`losses/actor_loss`)
   - Should not oscillate wildly
   - Steady convergence = good

7. **PBRS Contribution** (`pbrs/avg_per_episode`)
   - With 0.5x scaling: ~0.0025-0.003 per episode
   - Should be much smaller than goal rewards

8. **Distance to Puck** (`behavior/dist_to_puck_avg`)
   - Should decrease (getting closer to puck)
   - Target: < 2.5 by 50% through training

---

## 🔍 Comparison to Previous Runs

| Metric | Old Approach (1024 hidden, strategic rewards, tie penalty) | New Approach (400 hidden, focused rewards) |
|--------|-------------------------------------------------------------|---------------------------------------------|
| **Win rate @ 30k** | 24% vs weak (!?) | Target: 60-75% vs weak |
| **Network size** | 1024-1024 (overfitting) | 400-400 (better generalization) |
| **PBRS magnitude** | 1.0x (0.715/episode) | 0.5x (0.36/episode) |
| **Strategic rewards** | Enabled (noisy) | Disabled (focused) |
| **Tie penalty** | -3.0 (risk-averse) | 0.0 (neutral) |
| **Q-clip** | 25.0 (saturated) | 35.0 (headroom) |
| **Focus** | Optimize puck proximity | Optimize goals scored |

---

## 🎓 Key Hyperparameter Changes

### Reward Configuration
```bash
--reward_shaping          # Keep PBRS enabled
--pbrs_scale 0.5          # NEW: Reduce PBRS magnitude by 50%
--no_strategic_rewards    # NEW: Disable shot quality, diversity, forcing bonuses
--tie_penalty 0.0         # NEW: Ties are neutral (was -3.0)
```

### Value Function
```bash
--q_clip 35.0             # NEW: Raised from 25.0 to prevent saturation
--q_clip_mode hard        # Hard clipping (unchanged)
```

### Network Architecture
```bash
--hidden_actor 400 400           # NEW: Reduced from 1024 1024
--hidden_critic 400 400 200      # NEW: Reduced from 1024 1024 200
```

### Everything Else
All other hyperparameters remain the same:
- LR: 3e-4 (actor & critic)
- Batch size: 512
- Gamma: 0.99
- Tau: 0.005
- Epsilon decay: 0.999965
- Buffer size: 1M
- Cosine LR decay enabled

---

## 🧪 Testing Final Checkpoint

After training completes, test the agent:

```bash
cd ~/02-SRC/TD3

# Test vs weak
python3 test_hockey.py \
    --checkpoint ./results/checkpoints/checkpoint_episode_100000.pth \
    --opponent weak \
    --episodes 100 \
    --hidden_actor 400 400 \
    --hidden_critic 400 400 200 \
    --verbose

# Test vs strong
python3 test_hockey.py \
    --checkpoint ./results/checkpoints/checkpoint_episode_100000.pth \
    --opponent strong \
    --episodes 100 \
    --hidden_actor 400 400 \
    --hidden_critic 400 400 200 \
    --verbose
```

**Success criteria**:
- vs weak: 75-85% win rate
- vs strong: 70-80% win rate

---

## 🔬 Experimental Validation

To verify the new approach is better, compare to the ongoing curriculum run (seed 53):

| Metric | Seed 53 (old approach) @ 8k episodes | Seed 54/55 (new approach) @ 8k episodes |
|--------|---------------------------------------|------------------------------------------|
| Win rate vs weak | 24% | Target: 40-50% |
| Win rate vs strong | 26% | Target: 30-40% |
| Tie rate | 18-40% | Target: < 20% |
| Q_max | Hitting 25.0 clip | Should be < 35.0 |

If seed 54/55 achieves these targets, the new focused approach is validated.

---

## 💡 Philosophy Summary

**Old mindset**: "Teach the agent to engage with the puck using dense rewards"
- Result: Agent optimizes puck proximity, not goals

**New mindset**: "Let the agent figure out HOW to score, just tell it WHAT to score"
- Result: Agent optimizes goals directly

**Key insight**: The agent learned to touch the puck 12x more (0.2 → 2.5 touches) but didn't learn to WIN. The dense rewards were misleading it.

With focused rewards:
- PBRS provides gentle guidance (0.5x magnitude)
- Sparse goal rewards provide the TRUE objective (±10)
- No noise from strategic bonuses
- No risk-aversion from tie penalties

**Expected outcome**: Better alignment between training objective and winning objective.

---

## 📝 Notes

- Both strategies use the SAME hyperparameters except opponent/self-play timing
- Seeds are different (54 aggressive, 55 curriculum) for fair comparison
- All advanced features remain enabled (PFSP, dual buffers, regression rollback)
- Episode blocking (20 episodes) stabilizes Q-learning during self-play
- LR decays from 3e-4 to 3e-5 over the full training duration

---

## 🚨 Troubleshooting

**If win rate is still low (<40% vs weak at 30k)**:
- Check if angle transformation bug is fixed in your code version
- Verify PBRS is actually scaled (check W&B config: `pbrs_scale: 0.5`)
- Check strategic rewards are disabled (W&B tag: `no-strategic-rewards`)

**If Q-values hit 35.0 clip**:
- Raise to 40.0 in next run
- This indicates rewards are larger than expected

**If tie rate remains high (>25%)**:
- Consider small positive tie reward (+1.0) to encourage offensive play
- Or slightly negative (-0.5) but much less than loss (-10)

**If training crashes mid-stage**:
- Resume from latest checkpoint using `--resume` flag
- Update CHECKPOINT_PATH in sbatch file
- Adjust `--max_episodes` to desired end point

---

**Good luck! Focus on winning, not intermediate metrics.**
