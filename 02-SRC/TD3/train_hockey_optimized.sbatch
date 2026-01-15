#!/bin/bash
#SBATCH --job-name=td3_stage1_gamma99
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=12:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=stud432@uni-tuebingen.de

###############################################################################
# STAGE 1: HIGH-GAMMA TRAINING FOR SPARSE REWARD CREDIT ASSIGNMENT
#
# ROOT CAUSE ANALYSIS (from successful vs failed run comparison):
#   SUCCESSFUL RUN (94.5% WR): gamma=0.99, eps=1.0, warmup=1000
#   FAILED RUNS (24-27% WR):  gamma=0.95, eps=0.3, warmup=500
#
# WHY FAILED RUNS COLLAPSED:
#   1. gamma=0.95 provides INSUFFICIENT credit assignment for sparse rewards
#      - Reward 100 steps away: 10 * 0.95^100 = 0.059 (essentially zero)
#      - Agent cannot learn which actions lead to winning
#   2. eps=0.3 means 70% greedy from episode 1 with RANDOM policy
#      - Buffer fills with losing experiences (80%+ losses)
#      - Agent learns passivity: "doing nothing is safest"
#   3. Q-values became NEGATIVE → pessimistic policy → passive play
#      - time_near_puck collapsed from 4-11 to 0.3-2 steps
#      - Agent actively AVOIDED the puck
#
# WHY THIS SCRIPT WILL WORK:
#   1. gamma=0.99 provides MEANINGFUL credit assignment
#      - Reward 100 steps away: 10 * 0.99^100 = 3.66 (useful signal!)
#      - Agent can learn cause-effect relationships
#   2. eps=1.0 ensures DIVERSE buffer from the start
#      - All transitions are exploratory initially
#      - Buffer contains balanced win/loss experiences
#   3. Slow eps decay (0.99995) reaches eps_min=0.15 around episode 30k
#      - Matches successful run's exploration schedule
#
# TWO-STAGE TRAINING APPROACH:
#   STAGE 1 (THIS SCRIPT): gamma=0.99, eps=1.0→0.15, 30k episodes
#     - Goal: Learn basic cause-effect, achieve >55% win rate
#     - Q-values should become POSITIVE
#   STAGE 2 (CONTINUE FROM CHECKPOINT): gamma=0.95, eps=0.15
#     - Goal: Refine tactical play with shorter credit assignment
#     - Target: 85%+ win rate
#
# EXPECTED METRICS AT END OF STAGE 1:
#   - Win rate: 55-65% vs weak opponent
#   - Q_avg: POSITIVE (0.2 to 1.0)
#   - puck_touches: 3-8 per episode (active play)
#   - time_near_puck: 4-8 steps (engaged with puck)
#
# ESTIMATED RUNTIME: 6-8 hours for 30k episodes
###############################################################################

# Copy code to scratch (faster I/O)
cp -R ~/02-SRC /scratch/$SLURM_JOB_ID/
cd /scratch/$SLURM_JOB_ID/02-SRC/TD3

# Set wandb API key
export WANDB_API_KEY="cc059bf17f0dffb378947bc37398b00eea0d1944"

# Run Stage 1 training with high gamma
singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
    --mode NORMAL \
    --opponent weak \
    --max_episodes 30000 \
    --seed 55 \
    \
    `# === EXPLORATION (CRITICAL: FULL EXPLORATION INITIALLY) ===` \
    `# eps=1.0 ensures diverse buffer, prevents greedy exploitation of random policy` \
    `# Slow decay reaches eps_min=0.15 around episode 30k (matching successful run)` \
    `# Calculation: 1.0 * 0.99995^30000 ≈ 0.22, continues decaying in stage 2` \
    --warmup_episodes 2000 \
    --eps 1.0 \
    --eps_min 0.15 \
    --eps_decay 0.99995 \
    \
    `# === PRIORITIZED EXPERIENCE REPLAY ===` \
    `# PER helps but is NOT a substitute for proper gamma/exploration` \
    `# Oversamples high-TD-error transitions (often rare wins)` \
    --use_per \
    --per_alpha 0.6 \
    --per_beta_start 0.4 \
    --per_beta_frames 100000 \
    \
    `# === NETWORK ARCHITECTURE ===` \
    --hidden_actor 256 256 \
    --hidden_critic 256 256 128 \
    \
    `# === TRAINING DYNAMICS ===` \
    `# gamma=0.99 is CRITICAL for sparse reward credit assignment` \
    `# With gamma=0.99: reward 100 steps away = 3.66 (useful signal)` \
    `# With gamma=0.95: reward 100 steps away = 0.059 (no signal)` \
    --batch_size 1024 \
    --buffer_size 1000000 \
    --train_freq 10 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --gamma 0.99 \
    --tau 0.005 \
    --grad_clip 1.0 \
    \
    `# === TD3 SPECIFIC ===` \
    --policy_freq 2 \
    --target_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    \
    `# === Q-VALUE STABILITY ===` \
    `# q_clip=10.0: Realistic Q-range for sparse ±10 rewards with gamma=0.99` \
    `# Soft tanh clipping: prevents Q-explosion while allowing smooth gradients` \
    `# Range: With gamma=0.99 and ~150-step episodes, Q-values naturally in [-10, +10]` \
    --q_clip 10.0 \
    --q_clip_mode soft \
    \
    `# === REWARD SHAPING ===` \
    `# PBRS at 0.5x guides exploration without creating local optimum` \
    `# reward_scale=1.0 keeps sparse rewards dominant (PBRS ratio ~2.3%)` \
    --reward_scale 1.0 \
    --reward_shaping \
    --pbrs_scale 0.5 \
    --pbrs_constant_weight \
    --no_strategic_rewards \
    --tie_penalty -1.5 \
    \
    `# === SELF-PLAY DISABLED FOR STAGE 1 ===` \
    `# Focus on mastering weak opponent first` \
    --disable_selfplay \
    --no_lr_decay \
    \
    `# === LOGGING ===` \
    --eval_interval 1000 \
    --eval_episodes 100 \
    --log_interval 50 \
    --save_interval 2000 \
    --gif_episodes 3

# Copy results back to home directory
cp -R /scratch/$SLURM_JOB_ID/02-SRC/TD3/results ~/02-SRC/TD3/

###############################################################################
# AFTER STAGE 1 COMPLETES:
#
# If win rate > 55% and Q_avg is POSITIVE, create Stage 2 script:
#
# singularity exec --nv ~/containers/hockey.sif python3 train_hockey.py \
#     --mode NORMAL \
#     --opponent weak \
#     --max_episodes 170000 \
#     --checkpoint results/checkpoints/TD3_Hockey_NORMAL_weak_30000_seed55.pth \
#     --gamma 0.95 \
#     --eps 0.15 \
#     --eps_min 0.05 \
#     --eps_decay 0.9999 \
#     --warmup_episodes 0 \
#     ... (rest of parameters same as above)
#
# Stage 2 can later progress to strong opponent and self-play.
###############################################################################
