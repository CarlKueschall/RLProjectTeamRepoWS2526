# Overfitting to Strong Bot

## Evidence

| Metric | Strong Bot | Weak Bot |
|--------|-----------|----------|
| Win Rate | 37% (improving) | 33% (declining from 85%) |
| Mean Reward | -2.04 | -2.96 |
| Losses | 2.67 avg | 4.0 avg |

## Why Overfitting Happens

1. **Training exclusively against one opponent**: The agent only sees the strong bot's patterns
2. **World model specializes**: Dynamics model learns to predict strong bot behavior
3. **Policy narrows**: Even with high entropy, the advantage signal (however weak) is specific to strong bot patterns
4. **No diversity in experience**: Buffer contains only strong-bot interactions

## Why Weak Bot Performance Degrades

The weak and strong bots have fundamentally different behaviors:
- **Weak bot**: Less aggressive, slower reactions, simpler strategy
- **Strong bot**: PD-controlled, strategic puck holding, precise timing

An agent optimized against strong-bot patterns may:
- Overreact to weak bot's slower movements
- Miss openings that only appear against weak opponents
- Adopt defensive postures that are unnecessary against weak

## This Is Secondary to the Action Saturation Problem

The overfitting is real but is NOT the root cause of poor performance. Even if we fixed the overfitting (mixed training), the agent would still:
- Output saturated actions (abs_mean=0.98)
- Never hold the puck
- Play as a random bang-bang controller

**Fix the action saturation first, THEN address overfitting with mixed/self-play training.**

## Recommended Fix (After Core Issues Resolved)

1. Use self-play with `--self_play_start 50 --self_play_pool_size 10`
2. Mix weak and strong opponents: `--self_play_weak_ratio 0.3`
3. Evaluate against BOTH opponents during training
4. Only fine-tune against strong after achieving 70%+ against weak
