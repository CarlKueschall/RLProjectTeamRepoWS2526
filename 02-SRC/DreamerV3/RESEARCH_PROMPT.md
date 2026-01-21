# Research Prompt: Sparse Rewards in Model-Based Reinforcement Learning

## Context and Problem Statement

I am training a **DreamerV3** agent to play air hockey. The environment has extremely sparse rewards:
- **+10** when our agent scores a goal
- **-10** when the opponent scores
- **0** for all other timesteps (~99% of the time)

Episodes last 100-250 timesteps, and goals occur in roughly 1-5% of episodes during early training with a random policy.

### The DreamerV3 Architecture

DreamerV3 is a model-based RL algorithm that:
1. **Learns a world model** from real experience (encoder, recurrent dynamics, reward/continue predictors)
2. **Trains actor-critic entirely in imagination** - rolls out the policy in the learned latent space without environment interaction
3. Uses **categorical latent states** and **two-hot symlog encoding** for rewards/values to handle multi-modal distributions

The key insight of DreamerV3 is that the actor never needs real environment rewards during training - it learns purely from imagined trajectories in the world model.

### The Problem We're Facing

Our agent is stuck in a failure mode:
- **Policy entropy is at maximum** (13.67 nats) and never decreases
- **Win rate plateaus at 30-45%** against a weak scripted opponent
- **Advantages are near-zero** (0.08-0.69 after normalization)

Root cause analysis revealed:

1. **In imagination**, the world model predicts near-zero rewards because:
   - Random policy produces random trajectories
   - Random trajectories rarely reach goal-scoring states
   - World model correctly predicts: "random state → no goal → reward ≈ 0"

2. **Lambda returns ≈ values** because:
   - λ-return = rewards + discounted future values
   - If rewards ≈ 0, then λ-return ≈ values
   - Advantages = λ-return - values ≈ 0

3. **Actor receives no learning signal**:
   - REINFORCE gradient ∝ advantage × ∇log π
   - If advantages ≈ 0 for all actions, no gradient direction
   - Policy stays at random initialization indefinitely

This is a **chicken-and-egg problem**: the actor needs reward signal to learn, but rewards only come from goal-scoring states, which require a non-random policy to reach.

### What We've Already Implemented

1. **Two-hot symlog encoding** for rewards and values (handles multi-modal ±10 vs 0)
2. **Inverse frequency weighting** for sparse reward events in world model training
3. **Auxiliary task heads** trained on real data:
   - `goalPredictor`: P(goal in next K steps)
   - `puckGoalDistPredictor`: Distance from puck to opponent goal
   - `shotQualityPredictor`: Combined position/velocity scoring opportunity metric
4. **Value normalization** using percentile-based moments (which we've now made optional)

### Our Proposed Solution

We're considering using the auxiliary task predictions as **dense reward signals during imagination**:

```python
# During imagination:
baseRewards = rewardPredictor(imagined_states)  # ≈ 0 for random policy
auxRewards = (
    0.1 * goalProb +           # Encourage goal-likely states
    0.05 * (1/(distance+0.5)) + # Encourage puck near goal
    0.05 * shotQuality          # Encourage good shooting positions
)
totalRewards = baseRewards + auxRewards
```

This would provide gradient signal even when true rewards are zero, hopefully bootstrapping the policy out of the random-policy trap.

---

## Research Questions

### Primary Questions

1. **Is auxiliary reward shaping in imagination a sound approach for DreamerV3 with sparse rewards?**
   - Are there theoretical concerns with adding shaped rewards only in imagination (not in the real environment)?
   - Does this violate any assumptions of the DreamerV3 algorithm?
   - How does this compare to reward shaping in the real environment?

2. **What are the established best practices for handling sparse rewards in model-based RL?**
   - How do papers like DreamerV3, Dreamer, PlaNet, MBPO handle sparse reward environments?
   - Are there specific techniques designed for model-based RL with sparse rewards?
   - What worked in Atari games with sparse rewards (e.g., Montezuma's Revenge)?

3. **Should we instead apply Potential-Based Reward Shaping (PBRS) in the real environment?**
   - If we shape rewards before they enter the replay buffer, the world model learns dense signals
   - This seems more principled than shaping only in imagination
   - But does it change what the world model learns in problematic ways?

### Secondary Questions

4. **What role does curiosity/intrinsic motivation play in model-based sparse reward RL?**
   - ICM (Intrinsic Curiosity Module), RND (Random Network Distillation)
   - Plan2Explore and other exploration-focused model-based methods
   - Are these complementary to or alternatives to reward shaping?

5. **How important is the quality of imagination starting states?**
   - Should we prioritize starting imagination from states that led to real goals?
   - Is there work on "goal-conditioned imagination" or similar?

6. **Are there hierarchical or goal-conditioned approaches suited for this?**
   - Breaking the task into subgoals (approach puck, control puck, shoot)
   - Goal-conditioned policies with hindsight relabeling
   - Feudal/hierarchical variants of world models

7. **What do practitioners report about DreamerV3 on sparse reward tasks?**
   - Are there known failure modes or required modifications?
   - Community wisdom, blog posts, or implementation notes?

### Specific Technical Questions

8. **In DreamerV3, should shaped rewards affect the critic target (λ-returns)?**
   - Option A: Only use shaped rewards for actor gradient (not critic)
   - Option B: Use shaped rewards for both actor and critic
   - What are the tradeoffs?

9. **How should auxiliary reward scales be set relative to true rewards?**
   - Our true rewards are ±10, proposed aux rewards sum to ~0.2 max
   - Is this ratio appropriate? Should aux rewards decay over training?

10. **Are there concerns about the auxiliary heads generalizing from posterior states (real) to prior states (imagined)?**
    - In world model training, we use posterior (with observation)
    - In imagination, we use prior (without observation)
    - Could this distribution shift cause aux heads to give bad predictions?

---

## Environment Details

- **Observation space**: 18-dimensional continuous (positions, velocities, angles of player, opponent, puck)
- **Action space**: 4-dimensional continuous (movement and rotation forces)
- **Episode length**: 100-250 timesteps (ends on goal or timeout)
- **Reward structure**: +10 (our goal), -10 (opponent goal), 0 (otherwise)
- **Opponent**: Scripted "weak" opponent (deterministic, predictable)
- **Goal frequency**: With random policy, ~1-5% of episodes end in any goal

---

## What I'm Looking For

1. **Literature review**: Key papers on sparse rewards in model-based RL, especially with world models
2. **Technique comparison**: Pros/cons of different approaches (PBRS, curiosity, hierarchical, goal-conditioned)
3. **DreamerV3-specific guidance**: Known issues, community solutions, author recommendations
4. **Practical recommendations**: What should we try first? What's most likely to work?
5. **Theoretical grounding**: Is our proposed approach sound, or are there fundamental issues?

---

## Summary of Our Situation

| Aspect | Current State | Desired State |
|--------|---------------|---------------|
| Policy entropy | 13.67 (maximum) | < 10, decreasing |
| Advantages | 0.08-0.69 (tiny) | 1.0+ (meaningful) |
| Win rate vs weak | 30-45% (plateau) | 70%+ (improving) |
| Imagination rewards | ≈ 0 | Dense signal toward goals |
| World model | Learns sparse rewards OK | Unchanged |
| Actor learning | Stuck (no gradient) | Active improvement |

We need to break the cycle where random policy → no goals in imagination → no learning → stays random.

---

## Constraints

- We want to stay within the DreamerV3 framework (not switch algorithms)
- Solution should be implementable with reasonable effort
- Prefer approaches with theoretical grounding over pure heuristics
- Need to work with our existing trained auxiliary heads if possible
