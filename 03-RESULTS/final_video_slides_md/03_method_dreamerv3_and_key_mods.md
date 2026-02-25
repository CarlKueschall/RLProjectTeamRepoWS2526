# Slide 3 — Method: DreamerV3 + Sparse-Reward-Specific Modifications

## On-Slide Text
- Base: DreamerV3 world model + imagination actor-critic
- Key modifications:
  - Two-Hot Symlog reward/value heads
  - DreamSmooth temporal reward smoothing
  - Sparse-event loss weighting for reward head
  - PFSP-driven opponent selection in self-play

## Visual
Use a 2-layer diagram:
- Layer 1: `World Model (RSSM, reward, continue)`
- Layer 2: `Imagination rollouts -> Actor/Critic updates`
Add side badges for the four modifications.

## Speaker Notes (25s)
"The core engine is DreamerV3: learn dynamics from real data, then optimize policy in imagined trajectories. For hockey sparsity, we added two-hot symlog targets, DreamSmooth for temporal credit assignment, event-aware reward loss weighting, and PFSP opponent sampling once self-play is active."

## Slide Build
1. Base DreamerV3 block.
2. Add sparse-reward modifications as side tags.
