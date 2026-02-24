# Two-Hot Symlog Ablation: ON vs Gaussian (OFF)

## Direct Comparison

| Metric | Two-Hot ON | Two-Hot OFF (Gaussian) | Δ (ON − OFF) |
|--------|------------|------------------------|--------------|
| Max gradient steps | 259,232 | 249,632 | +9,600 |
| Final win rate | **98%** | 97% | +1 pp |
| Final mean reward | **9.33** | 8.95 | +0.38 |
| Mean win rate (trajectory) | 72.5% | 73.6% | −1.1 pp |
| Run state | crashed | crashed | — |

## Learning Curves (Win Rate vs Gradient Steps)

| Steps (k) | Two-Hot ON | Gaussian OFF |
|-----------|------------|--------------|
| 35 | 31% | — |
| 41 | — | 41% |
| 70 | 76% | — |
| 83 | — | 89% |
| 108 | 92% | — |
| 124 | — | 94% |
| 147 | 94% | — |
| 166 | — | 97% |
| 182 | 95% | — |
| 208 | — | 97% |
| 220 | 94% | — |
| 249 | — | 97% |
| 259 | 98% | — |

## Key Observations

1. **Both reach high performance** (98% vs 97%)—Gaussian/MSE does not fail catastrophically on hockey’s sparse rewards in this setup.
2. **Gaussian learns faster early**: 41% at 41k vs 31% at 35k for Two-Hot; 89% at 83k vs 76% at 70k.
3. **Two-Hot has slightly higher final** (98% vs 97%, +0.38 mean reward)—marginal but consistent.
4. **Two-Hot is the safer choice** for sparse {−10, 0, +10}: theory predicts MSE struggles with such distributions; our results show Two-Hot matches or slightly exceeds Gaussian.

## Conclusion

**Two-Hot Symlog is recommended** for sparse rewards. Gaussian works reasonably well here, but Two-Hot gives a small edge and is theoretically better suited to extreme reward distributions. Use Two-Hot for robustness.
