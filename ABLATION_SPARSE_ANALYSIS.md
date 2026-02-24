# Sparse Reward Ablation Analysis

## Bottom line

Your intuition is partly right: DreamerV3 core already handles sparse hockey very well.
Two-Hot: no clear net gain here; if anything, slightly worse sustained policy quality in this ablation.
DreamSmooth: gives a real late-stage benefit (better sustained win/reward and cleaner sparse-signal prediction), even though early progress is similar.

## DreamSmooth ON vs OFF (from scratch)

Matched at same budget (288k steps):
win_rate_100 0.99 vs 0.88 (+11 pp), reward_mean 9.34 vs 7.76.

Late-window stability/performance:
win_rate_100 tail mean 0.974 vs 0.901, reward_mean tail 9.11 vs 7.86.

Sparse/world-model quality improved:
reward_pred_error_mean tail 0.0074 vs 0.0358 (~5x lower),
sparse_pred_error tail 0.0228 vs 0.1143 (~5x lower).

Caveat: DreamSmooth run is slower in wall-clock in this dataset (likely not solely algorithmic; run conditions differ).

## Two-Hot ON vs OFF (from scratch)

Overall policy quality: OFF is slightly better on sustained performance.
win_rate_100 tail 0.972 (OFF) vs 0.942 (ON),
reward_mean tail 9.05 (OFF) vs 8.55 (ON).

Sample efficiency: ON reaches 0.95 once earlier, but OFF is stronger/more consistent over most of training.

World/sparse metrics are mixed, but no convincing practical gain from ON in this run:
ON has higher entropy/return-range, but not better realized win/reward.

Net: Two-Hot looks neutral to slightly negative here.

## Scientifically defensible report claim

"In single-seed from-scratch ablations, DreamerV3's core architecture already learns strong policies under sparse rewards. DreamSmooth improves late-stage stability and final policy quality, while Two-Hot does not provide consistent gains in our setup."

## Important limitation

All four runs are marked crashed, and each condition is single-seed.
So this is strong directional evidence, not strict statistical significance.
If you want publication-grade certainty: run 3 seeds per condition with fixed stop steps.

## Why this likely happened

### Why DreamSmooth reduced prediction error

- DreamSmooth converts sparse, impulse-like reward targets into temporally smoother targets around informative events. That reduces target variance for the reward head and lowers gradient noise.
- In Dreamer training, better reward prediction directly improves imagined rollouts and lambda-return targets. Cleaner imagined rewards then stabilize critic targets and policy updates.
- The observed metrics fit this mechanism: much lower late reward prediction error and sparse prediction error, plus better late win/reward.
- The gain appears strongest in mid/late training because the replay buffer has enough event diversity; smoothing then acts more like variance reduction than bias.

### Why DreamerV3 is already strong in sparse hockey

- DreamerV3 does not rely only on immediate reward regression. It learns latent dynamics and trains actor-critic over imagined trajectories, which gives multi-step credit assignment even when rewards are rare.
- High replay ratio and sequence training extract more signal per environment step than purely reactive value methods.
- Lambda-returns, slow critic targets, and return normalization stabilize long-horizon learning once the world model is reasonably accurate.
- Result: even without extra tricks, the base algorithm can reach high fixed-bot performance in this domain.

### Why Two-Hot did not help much here

- Two-Hot is most useful when target distributions are broad/heavy-tailed and discretization improves optimization geometry. In this setup, practical reward/value scales seem already learnable with the baseline head.
- In your run, Two-Hot ON tends to increase entropy and return range without translating into better sustained wins. That indicates "wider" value dynamics, but not better control quality.
- Discretization can introduce bias or calibration mismatch if binning/symlog resolution is not the limiting factor. Then it may add complexity without improving policy learning.
- Given single-seed noise and crashes, the right conclusion is not "Two-Hot is bad", but "no consistent net gain was demonstrated here".

### Practical implication for final push

- Keep DreamSmooth ON for continuation training.
- Treat Two-Hot as non-critical for this project's final objective unless a short multi-seed rerun shows a repeatable benefit.
- Prioritize opponent-pool quality and robust cross-play evaluation, since those produced larger realized gains than this architectural toggle.
