# Slide 1 — DreamerV3 Hockey: From Sparse Rewards to Tournament-Ready Policy

## On-Slide Text
- **Project:** World-model RL agent for adversarial hockey
- **Core challenge:** Sparse rewards (`+10/-10/0`) and non-transitive opponent matchups
- **Core result:** Late performance improved via **curated league self-play**, not fixed-bot optimization

## Visual
Use a clean title slide with one compact pipeline icon row:
`Scratch Training -> Self-Play -> League Curation -> Final Checkpoint`

## Speaker Notes (20s)
"This project implements DreamerV3 for hockey, where rewards are extremely sparse and opponents create a moving target. The key finding is that after baseline competence, the biggest gains come from structured self-play with curated opponents, not from continuing to optimize against fixed bots."

## Slide Build
1. Title.
2. Challenge line.
3. Core result line.
