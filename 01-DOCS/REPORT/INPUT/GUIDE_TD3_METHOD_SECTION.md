# Carl's Method Section: TD3 Implementation Guide

## Overview
This section covers the Twin Delayed DDPG (TD3) implementation and all domain-specific modifications developed for the hockey environment. This is approximately **1.5-2 pages** of the report (of your ~3.5 pages total).

---

## 1. TD3 Background and Core Principles

### What to Include:
- **Brief history:** Position TD3 in the context of actor-critic methods (DQN → DDPG → TD3)
- **Three key TD3 innovations:**
  1. Twin Critic Networks (addressing overestimation bias)
  2. Delayed Policy Updates (reduce policy lag behind value function)
  3. Target Smoothing with Clipped Noise (robustness)

### Math to Present:
- Bellman target: $Q(s,a) = E[r + \gamma \min_{i=1,2} Q_i'(s', \pi'(s' + \epsilon))]$
- Actor loss: $\mathcal{L}_{\pi} = -E[Q_1(s, \pi(s))]$
- Target network update: $\theta_i' \leftarrow \tau \theta_i + (1-\tau) \theta_i'$

### Key Insight:
Explain WHY each innovation matters:
- Twin critics: Empirical evidence shows Q-learning overestimation leads to poor policies
- Delayed updates: Prevents policy chasing a moving value function
- Target smoothing: Adds robustness without reintroducing overestimation

---

## 2. Network Architecture and Configuration

### Actor Network Description
```
Input: 18-D state vector (positions, velocities, puck state)
  ↓
Hidden Layer 1: 256 units (ReLU)
  ↓
Hidden Layer 2: 256 units (ReLU)
  ↓
Output: 4-D action vector (scaled to [-1,1])
```

### Critic Network Description (×2 networks)
```
Input: 22-D concatenation [state (18-D) + action (4-D)]
  ↓
Hidden Layer 1: 256 units (ReLU)
  ↓
Hidden Layer 2: 256 units (ReLU)
  ↓
Hidden Layer 3: 128 units (ReLU)
  ↓
Output: 1-D Q-value scalar
```

### Hyperparameters Table

Create a professional table with these parameters:

| Category | Parameter | Value | Justification |
|----------|-----------|-------|---|
| **Optimization** | Actor LR | 3×10⁻⁴ | Standard for continuous control |
| | Critic LR | 3×10⁻⁴ | Matched to actor for stability |
| | Optimizer | Adam | Industry standard, good convergence |
| **Replay Buffer** | Size | 500,000 | Sufficient diversity, memory efficient |
| | Batch Size | 256 | Standard; large enough for stable estimates |
| **TD3-Specific** | Policy Freq. | 2 | Update policy every 2 critic updates |
| | Discount γ | 0.99 | Standard for continuous control |
| | Soft Update τ | 0.005 | Conservative; slow target tracking |
| | Target Noise σ | 0.2 | Exploration smoothing |
| | Noise Clip | 0.5 | Prevents extreme smoothing |
| **Exploration** | OU Noise σ₀ | 1.0 | Start fully exploratory |
| | OU Noise σ_min | 0.05 | Maintain baseline noise |
| | Decay Rate | varies (0.9997-0.99985) | Gradual → deterministic policy |

**What to say:** Discuss why these choices are reasonable for hockey, reference TD3 paper for standard values, and note any domain-specific tweaks.

---

## 3. Domain-Specific Modifications and Innovations

This is the **most important part** where you differentiate from vanilla TD3. For each modification:
- Explain the PROBLEM it solves
- Describe the SOLUTION in detail (with equations)
- Justify WHY it works

### Modification 1: Q-Value Clipping

**Problem:**
- Preliminary experiments showed Q-values growing unbounded (→1000+)
- Caused gradient explosions in critic loss
- Training became unstable after ~15k episodes

**Solution:**
```
Q_clipped(s,a) = clamp(Q(s,a), -25, 25)
```

**Analysis to Include:**
- Show plot: Q-value progression WITH vs WITHOUT clipping
- Demonstrate that clipping prevents instability
- Argue that [-25, 25] range is sufficient for the hockey reward scale (max/min rewards are bounded)
- Note: This is a HARD clamp, more aggressive than typical approaches

**Equation:**
```
Q(s,a) ← clip(Q(s,a), -Q_max, Q_max)
where Q_max = 25.0
```

---

### Modification 2: Value Function Regularization (Anti-Lazy Learning)

**Problem:**
- Empirical observation: Without explicit incentive for action, agent learns inaction is often optimal
- When far from puck (60% of state space), "do nothing" receives high Q-value
- Agent becomes passive, rarely takes offensive actions
- Results in failures against active opponents

**Solution:**

Introduce regularization penalty when agent learns passive action is better:

```
L_vf_reg = λ_vf · max(0, Q(s, a_passive) - Q(s, a_active))
```

Where:
- `a_passive` = [0, 0, 0, 0] (do nothing)
- `a_active` = π(s) (learned action)
- Applied only when: distance_to_puck > threshold (e.g., 3.0)
- λ_vf = 0.1 (regularization strength)

**Physical Interpretation:**
This regularization says: "When puck is far, learned action should be better than inaction." It encourages the agent to be proactive while still allowing defensive play when needed.

**Analysis to Include:**
- Compare learning curves WITH vs WITHOUT this regularization
- Show qualitative behaviors: animated policy comparisons if possible
- Measure: average action magnitudes, offensive action frequency
- Discuss why this is important for game play (offense requires initiative)

---

### Modification 3: Dual Replay Buffer with Dynamic Anchor Mixing

**Problem:**
- Standard single replay buffer in self-play → catastrophic forgetting
- Agent overfits to current self-play opponents, forgets how to beat weak baseline
- Performance against weak opponent degrades: 92% → 75% as training continues

**Solution:**

Maintain TWO replay buffers with intelligent mixing:

```
Buffer_anchor: Experiences vs weak opponent (1/3 capacity = 167k)
Buffer_pool:   Experiences vs self-play (2/3 capacity = 333k)

Training batch:
  - 1/3 sampled from Buffer_anchor
  - 2/3 sampled from Buffer_pool
```

**Dynamic Adjustment:**

Monitor validation performance and adaptively change the ratio:

```
if performance_drop_from_best > 10%:
    anchor_ratio ← 0.70  # Increase weak opponent training
else:
    anchor_ratio ← 0.50  # Baseline mix

Performance tracked every 100 episodes against weak opponent (deterministic eval)
```

**Intuition:**
- Anchor buffer acts as "memory" of how to beat weak opponent
- Self-play pool buffer captures latest strategy developments
- When performance drops, we rebalance to prevent forgetting
- This is proven curriculum learning + anti-forgetting mechanism

**Analysis to Include:**
- Plot showing win-rate vs weak opponent over 97.5k episodes
- Show anchor_ratio adaptation over time (when does it jump to 0.70?)
- Ablation: performance of constant 0.50 ratio vs dynamic adaptation
- Demonstrate this prevents catastrophic forgetting

---

### Modification 4: Gradient Clipping

**Problem:**
- High-dimensional action spaces + large networks → potential gradient explosions
- Critic loss can produce very large gradients

**Solution:**
```
Clip gradient norm to 1.0 during backprop
```

**Implementation Detail:**
Applied to both actor and critic networks during `backward()` pass.

**Analysis:**
Brief mention that this is standard practice; show gradient statistics before/after clipping if you tracked them.

---

## 4. Exploration and Exploitation Strategy

### Ornstein-Uhlenbeck Noise

**Why OU Noise?**
- More temporally correlated than Gaussian noise
- Encourages exploration in smooth directions (natural for physics)
- Better than pure ε-greedy for continuous control

**Formulation:**
```
dε_t = -θ(ε_t - μ)dt + σ dW_t

In practice:
ε_t ~ N(0, σ)
a_t = π(s_t) + ε_t  (clipped to [-1, 1])
```

**Decay Schedule:**
```
σ(t) = σ_0 · (σ_min / σ_0)^(t / T_decay)

Example with 500k episodes:
σ(0) = 1.0
σ(500k) ≈ 0.05

Linear interpolation recommended
```

**What to Show:**
- Plot: noise magnitude decay over training
- Discuss: exploration-exploitation trade-off
- Justify: why this decay schedule works for hockey

### Evaluation (Deterministic Policy)

**Important distinction:**
- Training: Uses noise for exploration
- Evaluation: No noise (ε=0.0)

---

## 5. Reward Shaping Integration

This is crucial for explaining WHAT domain knowledge you're adding.

### 5.1 Potential-Based Reward Shaping (PBRS)

**Theory:**
Standard PBRS theorem: Adding $\Phi(s)$ bonus doesn't change optimal policy if properly formulated.

**Your Conservative Approach:**
```
r_shaped = r + α · Φ(s)
where α = 0.005 (very conservative!)
```

**Why α=0.005?**
- Prevents reward hacking (agent optimizing bonus instead of true reward)
- Avoids Q-value explosion
- Still provides learning signal for difficult parts of task

**The Potential Function Φ(s):**

Your Φ combines 4 hand-crafted components. For EACH component, explain:

#### Component 1: Offensive Progress (Φ_off)
```
Φ_off(s) = 1.5 × (1.0 - min(dist_to_opponent_goal / 10.0, 1.0))
```
- **Range:** [0, 1.5]
- **Meaning:** Rewards puck being close to opponent goal
- **Max bonus:** 0.5 × 1.5 × 1.5 = 1.125 per step when puck at goal
- **Intuition:** Goals are good; this provides signal toward scoring

#### Component 2: Puck Proximity (Φ_prox)
```
Φ_prox(s) = -1.5 × tanh(dist_to_puck / 1.5)
```
- **Range:** [-1.5, 0]
- **Meaning:** Rewards agent being close to puck
- **Tanh reason:** Soft saturation; far distances clipped, close distances emphasized
- **Why negative base:** Uses negative potential for proximity (reward for decreasing distance)
- **Intuition:** Can't score if you're not near the puck

#### Component 3: Defensive Lane (Φ_defense)
```
Φ_defense(s) = 1.0 × exp(-(dist_to_lane)² / 0.5)
```
- **Range:** [0, 1.0]
- **Gaussian peak:** At center of defensive lane
- **Variance:** 0.5 (controls width of reward region)
- **Intuition:** Good defensive positioning between puck and own goal

#### Component 4: Cushion Potential (Φ_cushion)
```
Φ_cushion(s) = -2.0 × tanh(ReLU(p_x + 2.0))
```
- **Purpose:** Penalize being TOO far forward (risky)
- **ReLU(·):** Only active when p_x > -2.0 (already forward)
- **Tanh scaling:** Soft penalty, increases with forward position
- **Intuition:** Aggressive attacking = risky; encourage balanced positioning

**Total Potential:**
```
Φ(s) = Φ_off + Φ_prox + Φ_defense + Φ_cushion
```

**PBRS Annealing:**
During self-play (after 5k episodes), linearly decay α from 0.005 → 0:
- Allows agent to learn intrinsic value function
- Reduces dependence on hand-crafted rewards
- Prevents overfitting to reward shaping

**What to Show:**
- Plot: Component-wise contribution over training
- Plot: Total PBRS bonus magnitude over time
- Ablation: training curves WITHOUT reward shaping
- Analysis: Does PBRS accelerate convergence? By how much?

---

### 5.2 Strategic Reward Bonuses

Beyond PBRS, add small bonuses for tactical behaviors:

| Behavior | Bonus | Rationale |
|----------|-------|-----------|
| Puck touch | +0.06 | Incentivize contact |
| Proximity bonus | +0.01 | Fine-grained distance signal |
| Direction toward goal | +0.12 | Encourage forward movement |
| Goal proximity | +0.015 | Final stage signal |
| Clear shot (unblocked) | +0.15 | Valuable strategic action |
| Blocked shot | -0.20 | Penalize blocked attempts |
| Attack diversity | +0.5 per side (max +1.5) | Encourage varied tactics |
| Opponent forcing | +0.1× dist_opponent_moved | Reward defensive effectiveness |

**How these work together:**
- Sparse rewards (goal ±1) provide ground truth
- PBRS provides dense guidance signal
- Strategic bonuses add tactical encouragement
- Conservative scaling prevents reward hacking

---

## 6. Summary of TD3 Implementation

Create a concise summary table:

| Aspect | Value | Source |
|--------|-------|--------|
| Base Algorithm | TD3 (Fujimoto et al. 2018) | [Citation] |
| Core Innovations | Twin critics, Delayed updates, Target smoothing | Original TD3 paper |
| Domain Modifications | Q-clipping, VF regularization, Dual buffers, PBRS | This work |
| Implementation Language | Python 3.9+ | [Framework info] |
| Network Framework | PyTorch | [Details] |
| Final Architecture Size | Actor: 256×256, Critic: 256×256×128 | [Hyperparameter details] |

---

## Approximate Word Count Target
- **For this section: 800-1200 words**
- This covers Method section thoroughly for TD3
- Leave 1.5 pages more for Experiments section

## Figures You Need to Create

1. **Architecture Diagram**
   - Simple schematic of actor/critic networks
   - Show input/output dimensions
   - Highlight where modifications apply

2. **Reward Shaping Visualization**
   - 4 subplots showing each Φ component
   - Heatmaps over state space (puck position + agent position)
   - Show which regions reward which behaviors

3. **PBRS Annealing Schedule**
   - α coefficient over 100k episodes
   - Linear decay from 0.005 → 0 over 5k episode window

## Key References to Cite

1. **TD3 Paper:** Fujimoto, van Hoof, Meger (2018) - "Addressing function approximation error in actor-critic methods"
2. **DDPG (for context):** Lillicrop et al. (2015) - "Continuous control with deep reinforcement learning"
3. **PBRS Theory:** Ng, Harada, Russell (1999) - "Policy Invariance Under Reward Transformations"
4. **Actor-Critic Foundations:** Konda & Tsitsiklis (2000) or Sutton & Barto (2018)

---

## Critical Points to Make Clear

1. **Why TD3 for hockey?** Continuous action space, sample efficiency, stability
2. **Why these modifications?** Specific problems encountered, empirical evidence solutions work
3. **Why this reward shaping?** Domain knowledge (hockey is about goal-proximity and puck-proximity)
4. **Why conservative α=0.005?** Prevent Q-value explosion, but still guide learning
5. **Ablation validation:** Multiple ablations show each modification helps

---

## Writing Tips

- **Use numbered equations** with descriptions of each term
- **Provide intuition** before mathematics
- **Be specific** about values, thresholds, decay rates
- **Justify choices** based on preliminary experiments or literature
- **Stay focused** on YOUR modifications, not vanilla TD3 theory
- **Use concrete examples** (e.g., "when agent is 5m from puck...")
- **Cross-reference** to tables/figures

