# Comprehensive Research Prompt: Optimal PBRS Reward Shaping for Air Hockey RL Agent

## Executive Summary

We are developing a TD3 (Twin Delayed DDPG) reinforcement learning agent to play air hockey. We use **Potential-Based Reward Shaping (PBRS)** to provide dense learning signals while maintaining theoretical guarantees about policy invariance. We need to design the **optimal potential function** and **reward balancing strategy** that will lead to the best possible air hockey performance while completely avoiding reward hacking.

---

## Our Current Setup

### Algorithm: TD3 (Twin Delayed DDPG)
- Continuous action space (4D: movement + shooting)
- Twin critics to reduce overestimation
- Delayed policy updates (every 2 critic updates)
- Target policy smoothing with clipped noise
- Gaussian exploration noise N(0, σ)

### Environment: Air Hockey
- **State space**: 18-dimensional per player
  - Player position (2D), angle (1D), velocity (2D), angular velocity (1D)
  - Opponent position (2D), angle (1D), velocity (2D), angular velocity (1D)
  - Puck position (2D), velocity (2D)
- **Action space**: 4-dimensional continuous [-1, 1]
  - 2D movement direction
  - 2D shooting/hitting direction
- **Sparse rewards**: +10 for scoring, -10 for being scored on, 0 otherwise
- **Episode length**: ~250 steps maximum
- **Discount factor**: γ = 0.99

### Current PBRS Implementation

We use the standard PBRS formulation:
```
F(s, s') = γ · φ(s') - φ(s)
```

Where φ(s) is our potential function. Currently we compute potential based on:
- Distance from player to puck
- Puck position relative to opponent's goal
- Puck velocity toward opponent's goal

The shaped reward is:
```
r_shaped = r_sparse + scale · F(s, s')
```

Where `scale` is a hyperparameter (currently 0.1).

---

## Critical Problems We're Experiencing

### 1. Agent Learns NOT to Shoot When Possessing
Our metrics show `shoot_action_when_possess` going from +0.1 to -0.5 over training. The agent learns to hold the puck rather than shoot. This suggests our potential function may inadvertently reward "being near the puck" more than "scoring goals."

### 2. Catastrophic Forgetting
Possession ratio peaks at 22% then collapses to 1-2%. The agent unlearns good behaviors.

### 3. Potential Reward Hacking
We suspect the agent may be optimizing the potential function directly rather than the true objective (winning).

### 4. Uncertainty About Optimal Balance
We don't know the mathematically correct way to scale PBRS relative to sparse rewards to guarantee:
- PBRS provides useful gradient signal
- Sparse rewards remain the dominant optimization target
- No reward hacking is possible

---

## Research Questions

### Part A: Optimal Air Hockey Strategy

1. **What is the theoretically optimal strategy for air hockey?**
   - Positioning relative to puck and goal
   - When to be aggressive vs defensive
   - Optimal shooting timing and angles
   - How should strategy change based on score differential and time remaining?

2. **What sub-goals decompose the air hockey objective?**
   - What intermediate achievements strongly correlate with winning?
   - What is the causal chain: action → sub-goal → scoring → winning?
   - Which sub-goals are necessary vs sufficient for winning?

3. **What behaviors should we explicitly encourage vs discourage?**
   - Should we reward puck possession? (Risk: agent hoards puck)
   - Should we reward being near puck? (Risk: agent shadows puck without engaging)
   - Should we reward puck velocity toward goal? (Risk: weak shots that don't score)

### Part B: Optimal PBRS Potential Function Design

4. **What is the optimal form for φ(s) in air hockey?**
   - Should it be additive (sum of components) or multiplicative?
   - What state features should φ depend on?
   - Should φ be time-dependent or stationary?
   - Should φ incorporate opponent state?

5. **How do we ensure φ(s) guides toward winning, not just sub-goals?**
   - The PBRS guarantee says optimal policy is unchanged, but learning dynamics ARE affected
   - How do we design φ so the learning path leads to the global optimum?
   - What potential functions lead to reward hacking despite PBRS guarantees?

6. **What are the failure modes of PBRS in competitive games?**
   - How can opponents exploit our potential function?
   - How does self-play interact with PBRS?
   - Should φ be different during self-play vs fixed opponent training?

7. **Should we use state-only potential φ(s) or state-action potential φ(s,a)?**
   - What are the theoretical differences?
   - Which is more appropriate for continuous action spaces?

### Part C: Mathematical Reward Balancing

8. **What is the mathematically correct scaling for PBRS?**
   - Given sparse rewards in range [-10, +10]
   - Given potential function with range [φ_min, φ_max]
   - What should the scale factor be to guarantee:
     - PBRS never dominates sparse rewards
     - PBRS provides sufficient gradient signal
     - No local optima are created by PBRS

9. **How should PBRS magnitude relate to the discount factor γ?**
   - PBRS telescopes to γ^T · φ(s_T) - φ(s_0) over an episode
   - How does this interact with episode length?
   - What constraints does this place on φ magnitude?

10. **Is there a principled way to compute the optimal scale factor?**
    - Based on reward statistics (mean, variance)
    - Based on Q-value magnitudes
    - Based on gradient magnitudes
    - Based on information-theoretic criteria

11. **How do we verify that reward hacking is impossible?**
    - What mathematical conditions must hold?
    - How do we test empirically?
    - What metrics indicate reward hacking is occurring?

### Part D: Implementation Best Practices

12. **Should we use potential shaping, reward shaping, or both?**
    - PBRS: F(s,s') = γφ(s') - φ(s)
    - Reward shaping: r' = r + F(s,a,s')
    - Look-ahead advice
    - Which is theoretically superior?

13. **How should PBRS interact with experience replay?**
    - Should we store shaped or unshaped rewards?
    - How does off-policy learning affect PBRS guarantees?
    - Are there corrections needed for importance sampling?

14. **How should PBRS interact with TD3 specifically?**
    - Twin critics and PBRS
    - Target networks and potential function stability
    - Delayed policy updates and shaped rewards

15. **Should the potential function be learned or hand-crafted?**
    - Pros/cons of learned potential functions
    - If learned, how to ensure it doesn't collapse to constant?
    - If hand-crafted, what domain knowledge is essential?

---

## Specific Technical Constraints

Please consider these constraints in your recommendations:

1. **Must maintain PBRS policy invariance guarantee** - We cannot use arbitrary reward shaping
2. **Must work with continuous action spaces** - Discrete methods don't apply
3. **Must be compatible with off-policy learning** - We use experience replay
4. **Must not require environment modification** - We can only modify the reward computation
5. **Must scale to self-play** - Strategy must work when opponent also improves
6. **Must be computationally efficient** - Potential function evaluated every step

---

## Requested Output Format

Please provide:

### 1. Theoretical Foundation
- Mathematical derivation of optimal PBRS for competitive games
- Proof or explanation of why certain potential functions avoid reward hacking
- Conditions under which PBRS guarantees hold in practice

### 2. Recommended Potential Function
- Explicit mathematical form of φ(s)
- Justification for each component
- Expected range of values
- How it guides toward winning (not just sub-goals)

### 3. Reward Scaling Formula
- Exact formula for computing scale factor
- Derivation showing why this prevents reward hacking
- How to adapt scaling during training (if needed)

### 4. Implementation Checklist
- Step-by-step implementation guide
- Common pitfalls to avoid
- Debugging/verification procedures

### 5. Empirical Validation Strategy
- Metrics to monitor for reward hacking
- Expected learning curves
- Red flags indicating problems

---

## Context: Why This Matters

We are participating in a reinforcement learning competition. Our agent must:
1. Beat a "weak" baseline opponent consistently (>90% win rate)
2. Beat a "strong" baseline opponent (>50% win rate)
3. Compete against other trained agents in a tournament

Current performance: ~10-15% win rate against weak opponent after 3000 episodes. This is unacceptable. We believe the reward shaping strategy is the bottleneck.

---

## References to Consider

Please incorporate insights from:
- Ng et al. (1999) - Original PBRS paper
- Wiewiora et al. (2003) - Potential-based shaping and Q-value initialization
- Devlin & Kudenko (2012) - Dynamic potential-based reward shaping
- Harutyunyan et al. (2015) - Expressing arbitrary reward functions as potential-based advice
- Any relevant work on reward shaping in competitive/adversarial settings
- Any relevant work on reward shaping in continuous control

---

## Summary of What We Need

1. **The optimal air hockey strategy** decomposed into rewardable sub-goals
2. **The optimal potential function φ(s)** that guides learning toward this strategy
3. **The mathematically precise scaling** that prevents reward hacking
4. **Implementation details** specific to TD3 with experience replay
5. **Verification methods** to confirm our approach is working

We want to get this RIGHT, with mathematical rigor, not trial-and-error tuning.
