# TD3 Development Journey: Lessons Learned

**Authors:** Serhat Alpay, Carl Kueschall
**Date:** January 2025
**Status:** Concluded - Transitioning to DreamerV3

---

## Executive Summary

This document chronicles our attempt to train a TD3 (Twin Delayed DDPG) agent to play air hockey using the laser-hockey-gym environment. After extensive experimentation with reward shaping, hyperparameter tuning, and various training strategies, we concluded that **TD3 is fundamentally unsuited for this sparse-reward, long-horizon competitive task**.

**Key Finding:** TD3's 1-step temporal difference learning cannot propagate sparse goal rewards (~250 timesteps) backward effectively. All reward shaping attempts led to exploitable local optima.

---

## 1. The Problem Domain

### Environment: laser-hockey-gym
- **Observation space:** 18-dimensional continuous
- **Action space:** 4-dimensional continuous
- **Episode length:** ~250 timesteps (NORMAL mode)
- **Reward structure:** Sparse (+10 win, -10 loss, 0 tie)
- **Challenge:** Goals are rare events; credit assignment must span hundreds of timesteps

### Why This Is Hard
1. **Sparse rewards:** A successful goal-scoring trajectory might occur once every 100+ episodes
2. **Long horizon:** 250 timesteps between action and reward
3. **Competitive dynamics:** Opponent actively prevents scoring
4. **Continuous control:** Precise paddle movements required

---

## 2. TD3 Implementation

### Core Algorithm (Fujimoto et al., 2018)
Our implementation followed the TD3 paper faithfully:
- Twin critic networks (Q1, Q2) to reduce overestimation bias
- Delayed policy updates (every 2 critic updates)
- Target policy smoothing with clipped Gaussian noise (σ=0.2, clip=0.5)
- Gaussian exploration noise N(0, 0.1)
- Soft target updates (τ=0.005)

### Architecture
```
Actor:  18 → 256 → 256 → 4 (tanh output)
Critic: 18+4 → 256 → 256 → 128 → 1 (twin networks)
```

### Key Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-3 | TD3 paper default |
| Batch size | 100 | TD3 paper default |
| Buffer size | 1M | Large for long training |
| Gamma | 0.99 | Standard for sparse rewards |
| Warmup episodes | 2000 | ~2% of 100k training |
| Gradient clipping | 1.0 | Prevents explosion |
| Q-value clipping | 25.0 (soft) | Prevents Q-explosion |

---

## 3. Reward Shaping Evolution

### The Fundamental Dilemma

| Approach | Result |
|----------|--------|
| **No shaping** | Goals too rare, rewards don't propagate, agent stuck |
| **With PBRS** | Agent exploits shaping, farms rewards without scoring |

### PBRS V1: Basic Distance Potentials
- φ_chase: -distance(agent, puck)
- φ_attack: -distance(puck, opponent_goal)
- **Result:** Agent learned to hover near puck without engaging

### PBRS V2: Added Velocity Components
- Added puck velocity toward goal
- Added agent velocity toward puck
- **Result:** Agent learned to oscillate, creating velocity rewards

### PBRS V3.x: Two-Component Dense Path
Final design philosophy:
```python
φ(s) = W_CHASE × φ_chase + W_ATTACK × φ_attack
W_CHASE = 0.5 × K
W_ATTACK = 0.5 × (K + 1)
```

**Key insight:** W_ATTACK > W_CHASE ensures forward shooting is net positive.

**Ablations performed:**
- K=1: W_CHASE=0.5, W_ATTACK=1.0
- K=2: W_CHASE=1.0, W_ATTACK=1.5
- K=3: W_CHASE=1.5, W_ATTACK=2.0

**Result:** ALL K values produced identical failure mode.

### PBRS Scaling Experiments
Tested scales: 0.003, 0.001, 0.0005, 0.0001

**Result:** Lower scales delayed but didn't prevent exploitation.

---

## 4. The Failure Mode: Passive Positioning Exploit

### What We Observed (From Training GIFs)

Across ALL runs, the agent learned the SAME exploit:

1. **Passive Midfield Positioning:** Agent stands at halfway line, between puck and opponent
2. **Consistent Angling:** Always angled toward top-left of arena
3. **Zero Engagement:** Does not actively pursue or shoot the puck
4. **Farming Rewards:** Collects PBRS rewards from opponent's actions

### Why This Happens

The PBRS gradient creates a LOCAL OPTIMUM:
- Being "close to puck" = consistent positive shaping
- Shooting = risky (might miss, lose position)
- The agent found that passive existence maximizes expected PBRS

### Metrics That Revealed the Problem

| Metric | Healthy Value | Exploit Value |
|--------|---------------|---------------|
| `shoot_action_when_possess` | +0.5 to +1.0 | -0.6 to -0.9 |
| Action magnitude | Varies | Saturated ~1.39 |
| Q-values | -1 to +1 | -3.5 to -4.5 |
| Win rate trajectory | Monotonic increase | Rise then crash |

---

## 5. Additional Attempted Solutions

### VF Regularization (Anti-Lazy)
Added penalty when Q(passive) > Q(active):
```python
vf_reg_loss = λ × max(0, Q(s, a_passive) - Q(s, a_active))
```
**Result:** Insufficient. Addresses symptom, not cause.

### Epsilon Reset on Self-Play
Reset exploration noise when self-play activates:
```python
if episode == self_play_start:
    agent.eps = 0.5  # Re-explore
```
**Result:** Temporary improvement, then collapse to same exploit.

### PBRS Annealing
Gradually reduce PBRS weight:
```python
pbrs_weight = max(min_weight, 1.0 - (episode - anneal_start) / anneal_duration)
```
**Result:** Agent exploits PBRS before annealing takes effect.

### Self-Play with PFSP
Prioritized Fictitious Self-Play for opponent diversity:
- Variance mode: Prioritize ~50% win rate opponents
- Hard mode: Prioritize hardest opponents

**Result:** Improved generalization but didn't fix core exploitation.

---

## 6. Root Cause Analysis

### Why TD3 Fails on This Task

1. **1-Step TD Learning:**
   - TD3 updates Q(s,a) based on r + γQ(s',a')
   - Reward signal propagates ONE step per training iteration
   - 250 timesteps → 250+ episodes to propagate goal reward to initial actions

2. **Insufficient Exploration:**
   - Gaussian noise N(0, 0.1) rarely generates goal-scoring trajectories
   - Agent converges to local optimum before experiencing enough goals

3. **Function Approximation Error:**
   - Neural networks interpolate poorly in sparse-reward regions
   - Q-values become unreliable far from experienced trajectories

### The Mathematical Reality

For a goal at timestep T=250:
- Reward r_T = +1 (scaled)
- Q(s_0, a_0) should increase
- Required: 250 consecutive Bellman backups
- With batch sampling: ~250 × (buffer_size / batch_size) iterations minimum
- In practice: Never converges before exploitation sets in

---

## 7. What Would Have Worked (Research Findings)

### DreamerV3 (Model-Based RL)
- World model imagines 50+ timesteps into future
- Credit assignment through imagination, not slow TD backups
- Achieved 2nd place Robot Air Hockey Challenge 2023 with SPARSE REWARDS ONLY
- No PBRS needed

### Curriculum Learning
- Pre-train on TRAIN_SHOOTING mode (easier, shorter episodes)
- Transfer to NORMAL mode
- Self-play for generalization

### Intrinsic Curiosity Module (ICM)
- Reward novel states, penalize "boring" passive positions
- Breaks local optima by making exploitation unrewarding over time

---

## 8. Files and Artifacts

### Core Implementation (02-SRC/TD3/)
| File | Purpose |
|------|---------|
| `train_hockey.py` | Main training loop |
| `agents/td3_agent.py` | TD3 algorithm implementation |
| `rewards/pbrs.py` | PBRS V3.3 implementation |
| `opponents/self_play.py` | Self-play with PFSP |
| `config/parser.py` | CLI argument parser |

### Documentation (01-DOCS/)
| File | Purpose |
|------|---------|
| `methodology/reward-design/PBRS_*.md` | PBRS version history |
| `debugging/ROOT_CAUSE_ANALYSIS.md` | Bug investigations |
| `DreamerV3/RESEARCH_RESULTS.md` | Comprehensive research on alternatives |

### Key W&B Runs Analyzed
- `wandb_run_2TD3-Hockey-NORMAL-weak-*.txt` (K=2 ablation)
- `wandb_run_3TD3-Hockey-NORMAL-weak-*.txt` (K=3 ablation)
- `wandb_run_best-performer-*.txt` (Best TD3 attempt)

---

## 9. Lessons for the Report

### What to Highlight

1. **TD3 is designed for dense rewards:** The algorithm assumes frequent reward signal. Hockey violates this assumption.

2. **PBRS theoretical guarantees don't hold in practice:** Policy invariance requires sufficient exploration, which never happens.

3. **Reward hacking is predictable:** State-based potentials create exploitable landscapes. This is well-documented in literature.

4. **Model-based RL solves the credit assignment problem:** Imagination-based planning propagates rewards instantly.

### Experimental Evidence to Include

- PBRS ablation results (K=1, 2, 3 all fail identically)
- Training curves showing rise-then-crash pattern
- GIF evidence of passive positioning exploit
- Comparison with DreamerV3 Robot Air Hockey Challenge results

### Honest Assessment

TD3 was a reasonable starting point but fundamentally mismatched to the problem. The 3+ weeks spent on PBRS tuning taught valuable lessons about reward shaping limitations, but the core algorithm choice was wrong from the start for this sparse-reward domain.

---

## 10. Transition to DreamerV3

### Why DreamerV3
- Proven on identical task (Robot Air Hockey Challenge 2023)
- Works with sparse rewards (no shaping exploitation)
- Self-play integration is straightforward
- Imagination-based credit assignment solves TD3's core weakness

### What We Keep
- Self-play infrastructure (PFSP opponent selection)
- W&B logging and metrics tracking
- Hockey environment wrapper
- Evaluation framework

### What We Archive
- TD3 agent implementation
- PBRS reward shaping
- All TD3-specific hyperparameter tuning code

---

## References

1. Fujimoto et al. (2018) - "Addressing Function Approximation Error in Actor-Critic Methods" (TD3 paper)
2. Ng et al. (1999) - "Policy Invariance Under Reward Transformations" (PBRS theory)
3. Orsula et al. (2024) - "Learning to Play Air Hockey with Model-Based Deep RL" (DreamerV3 hockey)
4. Perplexity Research Results - `/01-DOCS/DreamerV3/RESEARCH_RESULTS.md`

---

*This document serves as a reference for the final project report and documents the complete TD3 development journey from January 2025.*
