# Synthesis Report: Why and How the Ablations Behave

## The Core Problem: Sparse Rewards

Hockey gives +10 for a goal, −10 for conceding, 0 otherwise. Episodes last up to 250 steps. The agent must connect actions at t=50 to a goal at t=200. Standard RL struggles because the gradient signal is diluted over many steps.

## Why DreamSmooth Works

**Mechanism:** DreamSmooth defines \(\tilde{r}_t = \alpha r_t + (1-\alpha)\tilde{r}_{t+1}\) with α=0.5. A goal at t=200 creates nonzero \(\tilde{r}\) for t=199, 198, … so earlier actions receive a gradient.

**Why OFF fails:** Without propagation, only the last step before a goal gets a strong signal. The world model and policy see mostly zeros. Learning is slow and brittle; the agent finds a good policy by chance, then drifts when the signal is too sparse to maintain it.

**Evidence:** DreamSmooth OFF peaks at 95% then drops to 88%. ON rises to 99% and stays there. ON also runs ~105k more steps—more stable training.

## Why Two-Hot Symlog Helps (Modestly)

**Mechanism:** MSE treats −10, 0, +10 as a regression target. The gradient is weak when predictions are far from these rare values. Two-Hot Symlog discretizes into bins and uses a two-hot encoding, so gradients flow even when the exact value is wrong.

**Why Gaussian still works:** In our hockey setup, rewards are not so rare that MSE completely fails. The world model sees enough goal events to learn. Gaussian reaches 97%.

**Why Two-Hot is still better:** 98% vs 97%, +0.38 mean reward. The gain is small but consistent. For sparser rewards or different domains, Two-Hot's advantage would likely grow.

## How They Interact

- **DreamSmooth + Two-Hot:** Both address sparse rewards. DreamSmooth handles time; Two-Hot handles reward magnitude. They complement each other.

## Final Recommendation

For DreamerV3 on hockey:

1. **Use DreamSmooth**—it is essential for stable, high performance.
2. **Use Two-Hot Symlog**—small but consistent gains, better robustness.

The 2-phase curriculum (mixed opponents → self-play) in the main report builds on these choices. DreamSmooth and Two-Hot enable learning from sparse goals; the curriculum then adds diversity via self-play.
