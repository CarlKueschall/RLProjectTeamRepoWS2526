# Slide 2 — Problem Setup and What "Success" Means

## On-Slide Text
- Environment: 2-player continuous-control hockey
- Observation: 18D, Action: 4D per player
- Reward: sparse goal signal (`+10` score, `-10` concede, mostly `0`)
- Success criterion in project context:
  - Strong fixed-opponent performance
  - Robust cross-play vs diverse checkpoints

## Visual
Optional minimal table:
| Item | Value |
|---|---|
| Observation | 18D |
| Action | 4D |
| Reward profile | Sparse |
| Evaluation | Fixed + cross-play |

## Speaker Notes (25s)
"The environment is adversarial continuous control with sparse terminal-like rewards. That makes credit assignment and stability hard. So we do not treat success as one mean win-rate number; we evaluate both fixed bots and cross-play robustness across diverse checkpoints."

## Slide Build
1. Show env facts.
2. Highlight sparse reward line in accent color.
3. Show success criteria as two-part bullet.
