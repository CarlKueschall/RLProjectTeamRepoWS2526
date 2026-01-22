
<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# \

# # Plan based on this research prompt

 This is an excellent research report. Let me distill the key insights and provide a clear
  recommendation.

  Key Insights from Research

1. Our Diagnosis is Correct

  The chicken-and-egg problem is well-documented and our root cause analysis is accurate.

2. Our Proposed Auxiliary Rewards Approach Has Significant IssuesConcern: Posterior/Prior Distribution ShiftDescription: Aux heads trained on posterior states (with observations), but used on priorstates (imagination). Research shows this creates "performance gaps"────────────────────────────────────────Concern: Policy Invariance ViolatedDescription: Shaping only in imagination (not real env) creates a biased estimator - actorsees shaped rewards but critic trains on true rewards────────────────────────────────────────Concern: Value-Actor MismatchDescription: Can lead to overestimated values and policy divergence
3. Better Alternatives Exist

  DreamSmooth (Recommended First):

- Smooth rewards in the replay buffer before world model training
- The reward predictor learns to predict smoothed (denser) rewards
- Theoretically clean - just preprocessing, no algorithmic changes
- Well-validated on sparse reward tasks (RoboDesk, ShadowHand, Crafter)

  Ensemble Disagreement (Second Choice):

- Train K ensemble members of forward dynamics
- Intrinsic reward = variance across predictions
- No auxiliary head distribution shift concerns
- Signal is grounded in model uncertainty, not hand-crafted metrics

4. We Should Verify DreamerV3's Robustness Stack

  The research emphasizes that DreamerV3 has built-in mechanisms for sparse rewards that we  should verify are working:

- Return normalization with 5th-95th percentile scaling
- Symexp twohot loss for both reward predictor and critic
- Zero initialization for output weights
- EMA critic regularizer

---

  My Recommendation: Implement DreamSmooth First

  Based on the research, here's what we should do:

  Phase 1: DreamSmooth Implementation (Priority)

# Before adding transitions to replay buffer, smooth rewards:

  def smooth_rewards(rewards, window_size=5):
    """Apply exponential moving average smoothing to sparse rewards."""
    smoothed = torch.zeros_like(rewards)
    for t in range(len(rewards)):
    start = max(0, t - window_size)
    end = min(len(rewards), t + window_size + 1)
    smoothed[t] = rewards[start:end].mean()
    return smoothed

  Why this is better than auxiliary rewards in imagination:

1. No distribution shift (smoothing happens on real data)
2. World model learns dense signal naturally
3. No value-actor mismatch
4. Theoretically sound - just preprocessing
5. Already validated on similar sparse reward tasks

  Phase 2: Only If DreamSmooth Is Insufficient

  If DreamSmooth doesn't work, then try auxiliary rewards BUT with these safeguards:

- Only use goalProb (categorical, less distribution-sensitive)
- Skip distance-based metrics (too sensitive to prior state distribution)
- Apply only to actor, NOT critic (critic should learn true rewards)
- Weight very low (0.01-0.05, not 0.1)
- Decay over training

  What NOT to Do

  ❌ Don't implement auxiliary rewards in imagination as the first solution
  ❌ Don't disable return normalization (we made it optional - should keep it ON for sparse
  rewards)
  ❌ Don't use distance-based aux rewards (distribution shift concerns)

---

  Action Plan

1. Immediate: Implement DreamSmooth in the replay buffer
2. Verify: Ensure all DreamerV3 robustness techniques are enabled
3. Monitor: Check if reward prediction loss improves and imagination produces more variedrewards
4. Only if needed: Fall back to carefully-constrained auxiliary rewards

  Should I implement DreamSmooth? It's a cleaner solution than what we planned, and the
  research strongly supports it as the first intervention to try.

# # Research Prompt: Sparse Rewards in Model-Based Reinforcement Learning

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

| Aspect              | Current State            | Desired State             |
| :------------------ | :----------------------- | :------------------------ |
| Policy entropy      | 13.67 (maximum)          | < 10, decreasing          |
| Advantages          | 0.08-0.69 (tiny)         | 1.0+ (meaningful)         |
| Win rate vs weak    | 30-45% (plateau)         | 70%+ (improving)          |
| Imagination rewards | ≈ 0                     | Dense signal toward goals |
| World model         | Learns sparse rewards OK | Unchanged                 |
| Actor learning      | Stuck (no gradient)      | Active improvement        |

We need to break the cycle where random policy → no goals in imagination → no learning → stays random.

---

## Constraints

- We want to stay within the DreamerV3 framework (not switch algorithms)
- Solution should be implementable with reasonable effort
- Prefer approaches with theoretical grounding over pure heuristics
- Need to work with our existing trained auxiliary heads if possible

Now I have sufficient research depth. Let me generate the comprehensive report.

---

# Sparse Rewards in Model-Based Reinforcement Learning: A Comprehensive Analysis for DreamerV3

## Executive Summary

Your situation represents a well-documented failure mode in sparse reward model-based RL where the agent becomes trapped in a random policy equilibrium. The root cause—that advantages collapse when reward predictions are near-zero during imagination—is correct and directly addressed in the DreamerV3 literature through specific robustness techniques. Rather than applying auxiliary rewards *only* in imagination, the evidence strongly suggests a multi-pronged approach combining world model improvements, explicit exploration mechanisms, and potential-based reward shaping *in the replay buffer* before imagination occurs.

**Key Finding:** Adding auxiliary rewards to imagination alone may work but introduces theoretical concerns around distribution shift (auxiliary heads trained on posterior states, applied to prior states) and creates learning signal contamination. The community has converged on cleaner solutions: temporal reward smoothing (DreamSmooth), ensemble exploration (DreamerV3-XP), and hierarchical subgoals.

---

## Part 1: Your Problem Diagnosis Is Correct

### The Chicken-and-Egg Cycle

Your analysis of why the policy is stuck is theoretically sound. Under sparse rewards, the actor receives approximately zero gradient signal because:[^2][^3]

1. **World model predicts zero rewards**: Random policy produces trajectories that rarely reach goal states, so the learned reward predictor correctly outputs ≈0[^5]
2. **Lambda returns collapse to values**: Since Rλ = r + γ·v and r ≈ 0, we get Rλ ≈ v[^6]
3. **Advantages vanish**: Advantage = Rλ - v ≈ 0, so ∇log π multiplied by ~0 produces no learning signal[^6]
4. **Entropy remains maximum**: Without meaningful gradients, the policy stays at its random initialization[^8]

This is not a bug in DreamerV3—it's a fundamental limitation of any actor-critic method when initial rewards are genuinely sparse and require exploration to discover. The air hockey environment amplifies this: goals occur in ~1-5% of early episodes with random policy, meaning the world model learns that "random → no goal → reward ≈ 0" with high confidence.[^1]

### Why DreamerV3 Succeeds on Minecraft Diamond Despite Sparse Rewards

DreamerV3 solves this by combining multiple mechanisms working together, not one silver bullet:

| Mechanism                            | Why It Matters for Sparse Rewards                                                | Relevance to Your Problem                                                         |
| :----------------------------------- | :------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Return normalization**[^6]   | Scales returns 5th-95th percentile; doesn't collapse under std dev ≈ 0          | Critical: prevents advantage collapse when rewards are truly sparse               |
| **Symexp twohot loss**[^6]     | Decouples gradient magnitude from target value via categorical bins              | Stabilizes reward prediction for ±10 signals in sea of zeros                     |
| **Zero initialization**[^6]    | Reward predictor\& critic outputs start at 0, not random large values            | Prevents hallucinated rewards that delay real signal discovery                    |
| **EMA critic regularizer**[^6] | Targets depend on critic's own predictions; EMA stabilizes this self-loop        | Prevents bootstrap divergence with sparse/delayed signal                          |
| **Reconstruction loss**[^3]    | Unsupervised prediction of observations; not dependent on sparse rewards         | Learns good representations; policy can generalize even with sparse reward signal |
| **Free bits + KL balance**[^6] | Prevents dynamics/representation losses from collapsing; keeps info-rich latents | Ensures world model retains predictive state info needed for planning             |

**Critical Insight:** The author's paper explicitly states:[^6]

> "Normalizing rewards or returns by their standard deviation can fail under sparse rewards where their standard deviation is near zero, which overly amplifies rewards and can cause instabilities... Return normalization with a denominator limit overcomes these challenges, exploring rapidly under sparse rewards and converging to high performance."

Your normalization is optional in DreamerV3, which is likely a mistake for your sparse air hockey domain. Ensure you're using the exact return normalization formula with the 5th-95th percentile scaling and EMA.

---

## Part 2: Should You Shape Rewards in Imagination Only?

### The Core Problem with Your Proposed Approach

Your proposal to use auxiliary rewards only during imagination:

```python
totalRewards = rewardPredictor(imagined_states) + 0.1*goalProb + 0.05*(1/(distance+0.5)) + 0.05*shotQuality
```

has **both theoretical and practical concerns**:

**Theoretical Issue 1: Posterior/Prior Distribution Shift**[^9]

Your auxiliary heads are trained on posterior states (with observations): `auxReward = f(h_t, z_t_posterior, o_t)`. During imagination, you apply them to prior states: `f(h_t, z_t_prior)` (no observation). This distribution mismatch is documented in recent literature:[^9]

- Posterior-dominated sampling (high reliance on posterior info) outperforms prior-dominated sampling
- Training-inference mismatch creates "performance gaps" in variational RL
- Even balanced sampling (50/50 posterior/prior) shows inferior performance to purely posterior approaches

Your auxiliary heads' weights were optimized for the posterior distribution. Applying them to prior states during imagination could cause them to "hallucinate" rewards in misleading ways, especially for distance-based metrics that depend on observability cues.

**Theoretical Issue 2: Policy Invariance Violated**[^11][^12]

When you add arbitrary auxiliary rewards *only to imagination* (not to the real environment or replay buffer), you're creating an asymmetry:

- Policy learns to maximize shaped rewards in imagination
- But those shaped rewards never appear in real trajectories or during critic training on replay data
- This is NOT equivalent to potential-based reward shaping (PBRS), which *preserves* policy optimality

With PBRS, if you shape rewards in the environment as F(s,a,s') = γ·φ(s') - φ(s), the optimal policy doesn't change—only the learning speed. But shaping only in imagination creates a **biased estimator** because:

- Actor sees shaped signal during training (encouraging certain behaviors)
- Critic is trained on real rewards, creating a value-actor mismatch
- This can lead to overestimated values and policy divergence[^13]

**Practical Issue 3: Auxiliary Head Generalization**

Your `puckGoalDistPredictor` was trained on a dataset where the agent was exploring (initially random, then gradually improving). As the agent's policy improves, it visits different state distributions. The auxiliary head's predictions on these new states may not transfer well, especially:

- Distance metrics in state regions the agent has never visited
- Reward predictors trained on early-game states being applied to late-game states
- Goal probability estimates trained on exploration trajectories applied to exploitation trajectories

---

## Part 3: What the Literature Actually Recommends

### Solution 1: DreamSmooth—Apply Temporal Smoothing Before Imagination

**What it does:**[^14][^4]

- Smooth rewards *in the replay buffer* before they're used to train the world model
- Instead of predicting exact reward at timestep t, predict the average reward over a window (e.g., [t, t+5])
- This "softens" the sparse reward signal, making it easier for the reward model to learn

**Why it works:**

- Sparse rewards create large prediction errors even if the model is slightly off (predicted reward at t+1 vs t causes big loss)
- Smoothing makes the reward prediction task easier, so the model stops omitting sparse rewards
- The actor benefits: instead of predicting "nothing", the reward model now predicts "small positive reward spreading across 5 timesteps"
- Denser signal emerges naturally without explicit shaping

**Evidence:**[^4][^5]

- Applied to RoboDesk, ShadowHand, Crafter—all sparse reward environments
- Improved both sample efficiency and final performance over vanilla DreamerV3
- Also works with TD-MPC (another model-based algorithm), suggesting broad applicability
- No theoretical concerns—just preprocessing the experience

**Implementation:** Before adding transitions to the replay buffer, apply Gaussian/exponential/EMA smoothing:

```python
# Pseudocode
for episode in env:
  for t, r_t in enumerate(episode.rewards):
    r_smoothed[t] = smooth_window(episode.rewards[max(0,t-k):t+k])
  # Add (s,a,r_smoothed,s') to replay buffer
```

### Solution 2: Intrinsic Motivation via Ensemble Disagreement (DreamerV3-XP / InDRiVE)

**What it does:**[^16][^17]

- Train K ensemble members of forward dynamics models (lightweight, just s_{t+1} prediction)
- Compute variance across ensemble predictions: Var_k(μ_k(s_t, a_t))
- Add intrinsic reward: r_int = Var[ŝ_{t+1}] or Var[r̂_t]
- Combine: r_total = r_ext + λ·r_int during imagination

**Why it works for sparse rewards:**

- Exploration isn't driven by reward frequency; it's driven by model uncertainty
- Agent explores regions where the world model disagrees, which tend to be underexplored
- Once the model improves, intrinsic reward naturally decreases (no need for explicit decay)
- Compatible with sparse extrinsic rewards; provides complementary signal

**Evidence:**[^17][^15]

- DreamerV3-XP showed 45% improvement in exploration efficiency on hard sparse tasks (Cup Catch, Reacher-hard)
- InDRiVE achieved zero-shot transfer in autonomous driving using purely ensemble disagreement
- Plan2Explore (predecessor) achieved state-of-the-art on unsupervised skill discovery

**Key advantage over auxiliary rewards:** No auxiliary heads needed; no posterior/prior mismatch; signal is grounded in model uncertainty, not hand-crafted metrics.

### Solution 3: Hierarchical/Goal-Conditioned Approach with PBRS

**What it does:**[^19]

- Decompose air hockey into subgoals: "Move to puck" → "Position for shot" → "Execute shot" → "Recover"
- Train goal-conditioned policy π(a|s,g) where g is the subgoal
- Use Hindsight Experience Replay (HER): retroactively relabel failures with achieved subgoals
- Shape rewards using potential function: F(s,a,g) = γ·φ(s',g) - φ(s,g)

**Why it works:**

- HER turns sparse-reward episodes into dense signal by treating achieved states as alternative goals
- Policy learns incrementally: "how to reach midfield" → "how to position from midfield" → etc.
- PBRS theoretically preserves optimality while accelerating learning[^10]
- Each subgoal gives positive feedback, breaking the zero-gradient cycle

**Implementation note:** This is more invasive (requires goal space and HER logic), so probably second choice unless hierarchical structure naturally fits your domain.

---

## Part 4: Specific Recommendations for Your Air Hockey Problem

### Recommended Approach (in order of priority)

**Phase 1: Stabilize the foundation (2-3 runs)**

1. Ensure you're using DreamerV3's full robustness stack:
   - Return normalization with percentile-based range (Eq. 7 in paper)[^6]
   - Symexp twohot loss for both reward predictor and critic[^6]
   - Zero initialization for output weights[^6]
   - EMA critic regularizer[^6]
2. Verify reward predictor is actually learning (check training curves: does reward loss decrease?)
3. Log imagination rollouts: what fraction of imagined trajectories reach a goal? (Should be higher than random over time)

**Phase 2: Improve reward signal density (1-2 runs)**
4. Implement **DreamSmooth**: temporal reward smoothing before replay buffer

- Window size k: try 3-5 timesteps
- Use exponential moving average or simple Gaussian window
- This is the lowest-risk intervention; it's preprocessing, not algorithmic change

**Phase 3: Add exploration (1-2 runs)**
5. If DreamSmooth + Phase 1 stabilization don't work, add **intrinsic reward via ensemble disagreement**:

- Train K=3-5 lightweight forward models (just predict next latent state)
- Compute reward variance as intrinsic signal
- Use dynamic weighting: λ_t = λ_0 · exp(-t/τ) to decay over training
- This gives exploration bonus without auxiliary head distribution shift concerns

**Phase 4: Only if still stuck (1-2 runs)**
6. Try **auxiliary rewards in imagination, but carefully**:

- Only use `goalProb` (categorical, less distribution-sensitive)
- Skip distance-based metrics (too sensitive to prior state distribution)
- Weight them very low (0.01-0.05 relative to true rewards)
- Apply only during actor training, NOT during critic training
- Monitor for evidence of policy divergence (critic values exploding)

### What NOT to Do

❌ **Do not use auxiliary rewards only in imagination without addressing distribution shift**. If you do:

- The posterior/prior mismatch will cause your auxiliary predictions to degrade
- The actor will learn to trust shaped signals that don't correspond to real value
- You'll likely see brief improvements followed by collapse

❌ **Do not disable your auxiliary heads if they were helping**. Instead:

- Keep them for analyzing agent behavior (is the policy attempting to score?)
- Use them for *evaluation*, not training
- Or use them as auxiliary losses on the world model (multi-task learning), not as reward shaping

❌ **Do not scale auxiliary rewards equal to true rewards**. The ratio 0.1 vs 0±10 is too aggressive—0.01-0.05 max.

---

## Part 5: Theoretical Grounding for Your Proposed Approach

### Is Auxiliary Reward Shaping in Imagination Sound in Principle?

**Cautiously yes, under specific conditions:**

1. **If the auxiliary rewards don't violate the value function assumptions.** The critic learns:

```
v(s) = E[∑_τ γ^τ r_τ | s]  # Under current actor
```

If imagination rewards include shaped signal but real trajectories don't, there's a fundamental mismatch. The critic trained on replay data (true rewards) has no basis for the value estimates the actor expects.
2. **If the auxiliary heads generalize well from posterior to prior.** This is your biggest risk. Option: fine-tune auxiliary heads *on prior states* (during imagination) to adapt to the distribution they'll encounter.
3. **If the shaped signal is small enough not to dominate.** Auxiliary ≈ 0.05 relative to ±10 true rewards is reasonable; it's a "nudge", not the signal.

### What Would Make This Approach Better?

Instead of auxiliary rewards, use auxiliary *losses* on the actor network during imagination training:

```python
# During actor training:
policy_loss = -log π(a|s) * (advantage)
aux_loss_goal = -goalProb(s)  # Encourage goal-likely states
aux_loss_dist = -(1/(distance+1)) # Encourage puck-near-goal

total_loss = policy_loss + 0.01*aux_loss_goal + 0.01*aux_loss_dist
```

This provides gradient signal *without* contaminating the reward signal or value function. The actor learns "when I take actions consistent with reaching the goal, my reward goes up, AND I reach goal-likely states." Both signals reinforce each other naturally.

---

## Part 6: Why Air Hockey Is Particularly Challenging

Air hockey is harder than Minecraft diamonds despite Minecraft's scale because:

1. **Temporal credit assignment is immediate**: In Minecraft, intermediate milestones give implicit credit (you can "tell" you're making progress—you have a pickaxe). In air hockey, until the ball enters the goal, you have no signal.
2. **Goal state is a narrow basin**: Scoring requires precise positioning and velocity. Random exploration almost never finds it naturally. Minecraft has many checkpoint-like states (crafting table → pickaxe → furnace) that provide stepping stones.
3. **Reward is truly antagonistic in episodes**: If opponent scores, that's -10. So the agent learns "avoid opponent scoring" before "score goals." This creates a local optimum: defensive play without scoring attempts.[^20]

### Specific Mitigations for Air Hockey

Beyond the general solutions above:

1. **Warm-start actor with behavioral cloning** from scripted opponent traces (if available). Even 1000 steps of imitation learning gives the actor a non-random starting distribution, making exploration more targeted.
2. **Use negative reward tolerance**: Instead of treating every opponent goal as catastrophic feedback, use:

```
r_shaped = 10*(our_goals) - 5*(opponent_goals) - 0.01  # Small per-step penalty
```

This keeps opponent-scoring as a signal without dominating. Apply PBRS: F(s,a,s') = γ·V(s') - V(s) where V estimates from unshaped rewards.
3. **Curriculum learning**: Start with opponent randomness 100%, gradually reduce to 50%, then scripted strategy. This gives the agent early wins to bootstrap exploration.
4. **Imagination starting states**: Prioritize starting imagination rollouts from states the agent has reached after real goals (draw from replay buffer, weighted by episodes with goals). This gives the actor better context.

---

## Part 7: Comparison Matrix of Approaches

|                                            | DreamSmooth | Ensemble Disagreement | PBRS + Hierarchical | Aux Rewards in Imagination |
| :----------------------------------------- | :---------- | :-------------------- | :------------------ | :------------------------- |
| **Ease of Implementation**           | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐              | ⭐⭐                | ⭐⭐⭐                     |
| **Theoretical Soundness**            | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐            | ⭐⭐⭐⭐⭐          | ⭐⭐⭐                     |
| **Prior Work Validation**            | ⭐⭐⭐⭐    | ⭐⭐⭐⭐              | ⭐⭐⭐⭐⭐          | ⭐⭐                       |
| **Speed of Results**                 | ⭐⭐⭐      | ⭐⭐⭐                | ⭐                  | ⭐⭐                       |
| **Robustness to Domain**             | ⭐⭐⭐⭐    | ⭐⭐⭐⭐              | ⭐⭐⭐              | ⭐⭐                       |
| **Distribution Shift Risk**          | None        | None                  | None                | **High**             |
| **Generalization Beyond Air Hockey** | High        | High                  | Medium              | Low                        |

---

## Part 8: When Your Proposed Approach *Might* Work

Your auxiliary reward shaping could succeed if:

1. **You carefully validate auxiliary head generalization.** Before using them, measure their accuracy on prior-sampled states vs. posterior-sampled states. If accuracy drops >20%, don't use them without adaptation.
2. **You apply auxiliary rewards only to actor, not critic.** The critic should learn from true rewards only. This minimizes value-actor mismatch.
3. **You decay auxiliary rewards over time.** Start with 0.1 weight, decay to 0.01 by 50% training completion. This bootstrap the actor early without long-term bias.
4. **You combine with other interventions.** Auxiliary rewards alone won't fix zero advantages. Use them as a supplement to return normalization, zero initialization, and DreamSmooth.
5. **You validate on simpler domains first.** Test on a synthetic sparse-reward task (e.g., reach-goal-2D) before deploying on air hockey.

---

## Conclusion: Recommended Sequence

1. **This week**: Run comprehensive ablations to confirm all of DreamerV3's robustness techniques are enabled and working (, Figure 6a).[^6]
2. **Next week**: Implement DreamSmooth (, ). This is a safe, well-validated intervention with no theoretical risks. Measure impact on reward prediction loss and imagination rollout success rate.[^14]
3. **Week 3**: If still insufficient, add ensemble disagreement intrinsic reward (). This provides exploration without auxiliary head risks.[^15]
4. **Week 4**: Only then, if needed, add auxiliary rewards in imagination—but with the safeguards outlined in Part 4, Phase 4.

The core issue isn't "how do we add dense rewards?" It's "why isn't DreamerV3's existing machinery for sparse rewards working as expected?" Start by confirming all robustness techniques are engaged, validate your world model (is it actually learning?), and then apply proven solutions in order of theoretical soundness.

Your air hockey problem is solvable. The research literature shows multiple paths forward. The key is matching the intervention to the root cause without introducing new problems through distribution mismatch or value function corruption.

---

## References

Orsuka et al. (2024). Learning to Play Air Hockey with Model-Based Deep Reinforcement Learning. arXiv:2406.00518[^1]

Hafner et al. (2023). Mastering Diverse Domains through World Models (DreamerV3). Nature, arXiv:2301.04104[^6]

Lee et al. (2023). DreamSmooth: Improving Model-Based RL via Reward Smoothing. GitHub danijar/dreamsmooth[^4]

Emergent Mind. Potential-Based Reward Shaping in RL.[^10]

Badnava et al. (2019). A new Potential-Based Reward Shaping for Reinforcement Learning Agent. arXiv:1902.06239[^11]

Wang et al. (2023). Efficient Potential-based Exploration in Reinforcement Learning. NeurIPS.[^12]

Andrychowicz et al. (2017). Hindsight Experience Replay. NIPS.[^18]

Chane-Sane et al. (2025). Offline RL under Sparse Rewards via LLM-Guided Sub-goal Temporal Order. arXiv:2601.08107[^19]

Hu et al. (2025). DreamerV3-XP: Optimizing exploration through uncertainty. arXiv:2510.21418[^15]

Hafner et al. (2019). Dream to Control: Learning Behaviors by Latent Imagination. arXiv:1912.01603[^2]

Lee et al. (2023). DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing. arXiv:2311.01450[^5]

Lee, V. (2023). DreamSmooth: Improving MBRL via Reward Smoothing. [findingtheta.com](https://vint-1.github.io/dreamsmooth/)[^14]

Wen et al. (2021). Characterizing the Gap Between Actor-Critic and Policy Gradient. ICML.[^13]

Pathak et al. (2017). Curiosity-driven Exploration by Self-supervised Prediction (ICM). ICML.[^8]

Khanzada et al. (2025). InDRiVE: Intrinsic Disagreement-based Reinforcement for Vehicle Exploration. arXiv:2503.05573[^16]

Shihab et al. (2025). What Can You Do When You Have Zero Rewards During Training? arXiv:2410.03847[^20]

Khanzada et al. (2025). InDRiVE GitHub/Technical Report.[^17]

Finding Theta. The Evolution of Imagination: A Deep Dive into DreamerV3 and Its Conquest of Minecraft. Blog.[^3]

Researchers. (2025). Coupled Variational Reinforcement Learning for Language Models. arXiv:2512.12576[^9]
`<span style="display:none">`[^22][^24][^26][^28][^30][^32][^34][^36][^38][^40][^42][^44][^46][^48][^50][^52][^54][^56][^58][^60][^62][^64][^66][^68][^70][^72][^74][^76][^78][^80][^82][^84][^86][^88][^90][^92][^94][^95]

<div align="center">⁂</div>

[^1]: https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/AirHockeyChallenge_SpaceR.pdf
    
[^2]: https://arxiv.org/abs/2301.04104
    
[^3]: https://findingtheta.com/blog/the-evolution-of-imagination-a-deep-dive-into-dreamerv3-and-its-conquest-of-minecraft
    
[^4]: https://github.com/vint-1/dreamsmooth
    
[^5]: https://arxiv.org/abs/2311.01450
    
[^6]: https://www.nature.com/articles/s41586-025-08744-2
    
[^7]: https://www.alphanome.ai/post/the-sparse-reward-problem-a-major-challenge-in-training-ai-agents
    
[^8]: https://www.emergentmind.com/topics/random-network-distillation-rnd
    
[^9]: https://arxiv.org/html/2512.12576v1
    
[^10]: https://www.emergentmind.com/topics/potential-based-reward-shaping
    
[^11]: https://arxiv.org/abs/1902.06239
    
[^12]: https://proceedings.neurips.cc/paper_files/paper/2023/file/79f7f00cbe3003cea4d0c2326b4c0b42-Paper-Conference.pdf
    
[^13]: https://proceedings.mlr.press/v139/wen21b/wen21b.pdf
    
[^14]: https://vint-1.github.io/dreamsmooth/
    
[^15]: https://arxiv.org/html/2510.21418v1
    
[^16]: https://arxiv.org/html/2503.05573v1
    
[^17]: https://arxiv.org/abs/2503.05573
    
[^18]: https://www.youtube.com/watch?v=0EDiC__RcCA
    
[^19]: https://arxiv.org/html/2601.08107v1
    
[^20]: https://arxiv.org/html/2510.03971v1
    
[^21]: https://arxiv.org/abs/2406.00518
    
[^22]: https://elib.uni-stuttgart.de/items/f0120d92-2c97-437b-b274-88fb24720cb1
    
[^23]: https://www.hrl.uni-bonn.de/publications/2023/handling-sparse-rewards-in-reinforcment-learning-using-model-predictive-control
    
[^24]: https://www.youtube.com/watch?v=Au6R92JHH5Q
    
[^25]: https://openreview.net/forum?id=RN7RzMxwjC
    
[^26]: https://research.google/blog/deep-hierarchical-planning-from-pixels/
    
[^27]: https://www.youtube.com/watch?v=0Ey02HT_1Ho
    
[^28]: https://www.reddit.com/r/reinforcementlearning/comments/1defbq8/d_how_does_dreamerv3_do_so_well_on_sparsereward/
    
[^29]: https://www2.informatik.uni-hamburg.de/wtm/publications/2021/WWW21/IJCNN-0880-Wulur_PIP_copyright_.pdf
    
[^30]: https://autolab.berkeley.edu/assets/publications/media/2022-12-DayDreamer-CoRL.pdf
    
[^31]: https://proceedings.neurips.cc/paper_files/paper/2020/file/6101903146e4bbf4999c449d78441606-Paper.pdf
    
[^32]: https://milvus.io/ai-quick-reference/what-is-intrinsic-motivation-in-reinforcement-learning
    
[^33]: https://pathak22.github.io/noreward-rl/resources/icml17.pdf
    
[^34]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11525000/
    
[^35]: http://bair.berkeley.edu/blog/2020/10/06/plan2explore/
    
[^36]: https://openreview.net/forum?id=OjCWG58ZyY
    
[^37]: https://openreview.net/forum?id=5E5sd3TWGD
    
[^38]: https://ramanans1.github.io/docs/ramanan_approved_masters_thesis.pdf
    
[^39]: https://www.emergentmind.com/topics/hindsight-experience-replay-her
    
[^40]: https://arxiv.org/html/2508.20294v2
    
[^41]: https://arxiv.org/html/2510.22940v2
    
[^42]: https://github.com/danijar/dreamerv3
    
[^43]: https://proceedings.neurips.cc/paper/2021/file/4ebccfb3e317c7789f04f7a558df4537-Paper.pdf
    
[^44]: https://proceedings.mlr.press/v232/rafiee23a/rafiee23a.pdf
    
[^45]: https://openreview.net/forum?id=S1lOTC4tDS
    
[^46]: https://arxiv.org/html/2508.20294v3
    
[^47]: https://arxiv.org/html/2501.19128v1
    
[^48]: https://www.merl.com/publications/docs/TR2022-139.pdf
    
[^49]: https://projecteuclid.org/journals/bayesian-analysis/volume-8/issue-3/On-the-Prior-and-Posterior-Distributions-Used-in-Graphical-Modelling/10.1214/13-BA819.pdf
    
[^50]: https://www.emergentmind.com/topics/dense-reward-functions
    
[^51]: https://proceedings.mlr.press/v238/wang24g/wang24g.pdf
    
[^52]: https://proceedings.neurips.cc/paper_files/paper/2023/file/65496a4902252d301cdf219339bfbf9e-Paper-Conference.pdf
    
[^53]: https://arxiv.org/pdf/2212.05331.pdf
    
[^54]: https://www.uber.com/en-DE/blog/go-explore/
    
[^55]: https://www.reddit.com/r/reinforcementlearning/comments/ph8r0y/exploration_in_actorcritic/
    
[^56]: https://www.ijcai.org/proceedings/2020/0290.pdf
    
[^57]: https://stackoverflow.com/questions/51751163/actor-critic-policy-loss-going-to-zero-with-no-improvement
    
[^58]: https://www.reddit.com/r/MachineLearning/comments/8w5grj/r_learning_montezumas_revenge_from_a_single/
    
[^59]: https://arxiv.org/html/2407.00324v2
    
[^60]: https://arxiv.org/html/2511.00423v1
    
[^61]: https://arxiv.org/pdf/2509.03790.pdf
    
[^62]: https://par.nsf.gov/servlets/purl/10538655
    
[^63]: https://aclanthology.org/2025.findings-acl.1302.pdf
    
[^64]: https://arxiv.org/html/2105.05716v6
    
[^65]: https://aclanthology.org/2024.emnlp-main.515.pdf
    
[^66]: https://arxiv.org/html/2512.00005v1
    
[^67]: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
    
[^68]: https://arxiv.org/html/2512.04302v1
    
[^69]: https://groups.uni-paderborn.de/lea/share/lehre/reinforcementlearning/lecture_slides/built/Lecture12.pdf
    
[^70]: https://arxiv.org/html/2510.01857v1
    
[^71]: https://apxml.com/courses/advanced-reinforcement-learning/chapter-4-advanced-exploration-strategies/random-network-distillation
    
[^72]: https://openreview.net/pdf?id=cYksYKbf6K
    
[^73]: https://www.talkrl.com/episodes/danijar-hafner-on-dreamer-v4/transcript
    
[^74]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11239510/
    
[^75]: https://www.mathworks.com/help/reinforcement-learning/ug/custom-ppo-training-loop-with-random-network-distillation.html
    
[^76]: https://www.reddit.com/r/reinforcementlearning/comments/1p39ikh/strategies_for_rl_with_selfplay_for_games_where/
    
[^77]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6705923/
    
[^78]: https://arxiv.org/pdf/2503.02552.pdf
    
[^79]: https://proceedings.mlr.press/v162/graesser22a/graesser22a.pdf
    
[^80]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0263373
    
[^81]: https://www.semanticscholar.org/paper/InDRiVE:-Intrinsic-Disagreement-based-Reinforcement-Khanzada-Kwon/0b32d9781745580d9247b5418ee56c2ec5674aa9
    
[^82]: https://arxiv.org/html/2410.11359v1
    
[^83]: https://arxiv.org/html/2510.25801v1
    
[^84]: https://www.talkrl.com/episodes/max-schwarzer/transcript
    
[^85]: https://www.reddit.com/r/reinforcementlearning/comments/1mv4jn1/dreamer_v3_with_storm_4_months_to_build/
    
[^86]: https://arxiv.org/html/2510.07242v1
    
[^87]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12695655/
    
[^88]: https://arxiv.org/pdf/2301.04104.pdf
    
[^89]: https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html
    
[^90]: https://research.google/blog/efficiently-initializing-reinforcement-learning-with-prior-policies/
    
[^91]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8621907/
    
[^92]: https://arxiv.org/html/2410.03847v1
    
[^93]: https://arxiv.org/html/2412.17344v1
    
[^94]: http://papers.neurips.cc/paper/9057-energy-inspired-models-learning-with-sampler-induced-distributions.pdf
    
[^95]: https://ieeexplore.ieee.org/iel7/34/10490207/10328687.pdf
