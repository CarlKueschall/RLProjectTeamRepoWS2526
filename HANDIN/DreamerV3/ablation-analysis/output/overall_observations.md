# Overall Observations: Ablation Studies

## Summary Table (All Runs)

| Run | Ablation | Max Steps | Final Win Rate | Final Mean Reward |
|-----|----------|-----------|----------------|-------------------|
| fbench-ABLATION-dreamsmooth-ON | DreamSmooth ON | 393k | **99%** | **9.45** |
| ABLATION-dreamsmooth-OFF | DreamSmooth OFF | 288k | 95% | 8.13 |
| ABLATION-twohot-ON | Two-Hot ON | 259k | **98%** | **9.33** |
| ABLATION-twohot-OFF-gaussian | Two-Hot OFF (Gaussian) | 249k | 97% | 8.95 |

## Cross-Ablation Insights

1. **DreamSmooth has the largest impact.** ON vs OFF: +4 pp win rate, +1.32 mean reward, and ON trains 37% longer. Without DreamSmooth, performance peaks then degrades.

2. **Two-Hot vs Gaussian is close.** Both reach 97–98%. Two-Hot has a small edge (+1 pp, +0.38 reward). Gaussian is viable but Two-Hot is preferred for sparse rewards.

3. **Best observed: DreamSmooth ON** (99%, 9.45).

## Recommended Configuration

For sparse-reward hockey:

- **DreamSmooth: ON** (essential)
- **Two-Hot Symlog: ON** (recommended)
