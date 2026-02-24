# DreamerV3 Hockey Agent: Final 72-Hour Tournament Optimization Plan

## A) Most Likely Root Causes (Ranked by Probability)

### 1. Opponent Selection Bug Causing Skewed Training Distribution (Probability: ~70%)

The reported symptom—self-play manager anchor counters looking balanced, but actual logged opponent counts being strongly skewed—is the most likely primary cause of late regression. If the self-play opponent selection overrides or interacts incorrectly with the mixed-opponent logic, the agent may train predominantly against self-play opponents in later phases, causing it to "forget" how to exploit fixed-bot weaknesses. This directly explains why fixed-bot win rates peaked and then regressed (combined_win_rate from ~0.99 to ~0.88, strong_win_rate from ~1.00 to ~0.84) while world model metrics remained stable. The world model is not collapsing—the policy is shifting its strategy distribution. In the AlphaStar league, a similar failure mode was observed: when opponent sampling becomes unbalanced, the main agent's robustness deteriorates significantly, with worst-case win rates dropping to 1% in the original AlphaStar setup.[^1][^2]

### 2. Self-Play Cycling / Non-Transitive Forgetting (Probability: ~60%)

Self-play in competitive games with continuous action spaces inherently produces cycling dynamics. Research consistently shows that "various widely used self-play algorithms exhibit cyclic policy evolutions". The late regression in pfsp_min_winrate (~0.62-0.67 → ~0.47) is a classic cycling signature: the agent learns to beat recent pool members but forgets how to handle older strategies. PFSP mitigates this but does not eliminate it, especially when the pool is small (20 agents) and the game has non-transitive elements. The DreamerV3 air hockey paper at Robot Air Hockey Challenge 2023 explicitly noted that "agents are prone to overfitting when trained solely against a single playstyle", and even with self-play, the learning curve stagnated in reward terms while behavioral quality improved—suggesting cycling beneath aggregate metrics.[^3][^4][^1]

### 3. Insufficient Fixed-Bot Anchor Ratio During Self-Play (Probability: ~50%)

With 50% anchor ratio, if combined with the selection bug above, the effective anchor fraction could be much lower. AlphaStar-style training uses careful PFSP weighting where the main agent plays against the *entire* league including frozen historical agents. ROA-Star improved on this by ensuring "there always exist evenly matched opponents for MA" through goal-conditioned exploiters. A 50% anchor may be too low for maintaining fixed-bot performance during aggressive self-play phases, particularly when the self-play pool itself has limited diversity.[^2]

### 4. No Explicit Anti-Forgetting Mechanism on the Policy (Probability: ~45%)

Pure gradient-descent policy optimization with no regularization toward historical performance will naturally drift. Policy Consolidation research demonstrates that in competitive self-play settings, regularizing the current policy by its own history at multiple timescales significantly improves continual learning. EWC-style approaches have shown reduced forgetting in alternating-task RL, and EMA of weights produces solutions that "not only generalize better but also exhibit improved robustness". The current setup has no such mechanism.[^5][^6][^7][^8][^9][^10]

### 5. Checkpoint Selection Bias (Probability: ~30%)

Selecting best_selfplay_336k based on combined_win_rate against fixed bots may not maximize robustness against arbitrary opponents. The DreamerV3 air hockey paper used multi-strategy evaluation, and ROA-Star used worst-case win rate and Relative Population Performance (RPP) as robustness metrics rather than average performance. A checkpoint that scores 93.5% combined against fixed bots may actually be less robust than a nearby checkpoint with 90% combined but better self-play pool coverage.[^1][^2]

***

## B) Top Interventions (Ranked)

### Intervention 1: Fix Opponent Selection Bug

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | Correct the code path where self-play activation overrides or skews the intended opponent distribution. Ensure actual opponent sampling matches configured ratios (50% anchor / 50% self-play pool). |
| **Evidence quality** | Direct from project diagnostics. Distribution bugs are the #1 silent failure in self-play systems[^2][^3]. |
| **Expected gain** | **Large** (+3-8% combined win rate if this is the primary cause) |
| **Time to implement** | 1-3 hours (debugging + patching) |
| **Risk of backfire** | Near zero — this is a bug fix, not a design change |
| **Compatibility** | Direct fix to existing codebase |

### Intervention 2: Increase Anchor Ratio to 65-70% Fixed Bots

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | Reduce self-play fraction from 50% to 30-35%, ensuring the agent continuously trains against fixed bots and retains performance. The remaining 30-35% self-play maintains diversity pressure. |
| **Evidence quality** | Strong from AlphaStar league design (main agent trains against entire league including frozen historical)[^2] and Gigaflow's advantage filtering[^11]. |
| **Expected gain** | **Medium** (+2-5% fixed-bot win rate stability) |
| **Time to implement** | 0.5 hours (config change) |
| **Risk of backfire** | Low-medium. Too little self-play may reduce diversity. 65% is a conservative adjustment. |
| **Compatibility** | Direct config parameter |

### Intervention 3: EMA Shadow Weights for Actor Network

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | Maintain an exponential moving average of the actor's weights during training (decay α=0.995-0.999). Use EMA weights for evaluation/tournament. EMA reduces noise and provides implicit regularization, producing solutions that "differ from last-iterate solutions" with improved robustness and consistency[^8][^9]. |
| **Evidence quality** | Strong systematic study showing EMA improves generalization, robustness to noise, prediction consistency, and calibration across many settings[^8]. Polyak averaging is standard for critic target networks (τ=0.005)[^12][^13], extending to the actor is natural. |
| **Expected gain** | **Medium** (+1-3% win rate, significantly smoother performance) |
| **Time to implement** | 1-2 hours (~20 lines of code) |
| **Risk of backfire** | Very low — EMA is a pure read operation during training, only used at evaluation |
| **Compatibility** | Fully compatible, adds zero training overhead |

### Intervention 4: Lower Learning Rate Continuation from Best Checkpoint

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | Resume from best_selfplay_336k with 0.3-0.5× the original actor LR. Lower LR reduces the amplitude of cycling and prevents large policy swings. Combined with bug fix and higher anchor ratio, this fine-tunes the policy without destabilizing it. |
| **Evidence quality** | Standard practice. Reduced LR near convergence is well-established[^8][^14]. DreamerV3 with fixed hyperparameters may benefit from manual LR reduction at the fine-tuning stage[^15]. |
| **Expected gain** | **Medium** (+1-4% robustness, reduced cycling) |
| **Time to implement** | 0.5 hours (config change on resume) |
| **Risk of backfire** | Low. Worst case: slower learning, marginal improvement. |
| **Compatibility** | Direct config change |

### Intervention 5: Expanded Checkpoint Evaluation with Worst-Case Selection

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | Instead of selecting the single best checkpoint by average combined win rate, evaluate top-10 checkpoints against multiple opponent types (weak, strong, self-play pool members, random checkpoints) and select the one with the best worst-case win rate. ROA-Star demonstrated this is far more predictive of tournament robustness than average metrics (44% worst-case vs AlphaStar's 1%)[^2]. |
| **Evidence quality** | Strong. ROA-Star and AlphaStar both use worst-case and RPP metrics for model selection[^2]. |
| **Expected gain** | **Medium** (+2-5% tournament performance through better selection, no training needed) |
| **Time to implement** | 2-4 hours (evaluation script) |
| **Risk of backfire** | Zero — pure evaluation, no model changes |
| **Compatibility** | Fully compatible |

### Intervention 6: Replay Buffer Opponent Tagging and Balanced Sampling

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | Tag replay buffer sequences by opponent type (weak/strong/self-play-agent-ID). During training, sample uniformly across opponent categories rather than relying on temporal ordering. This ensures the world model and policy continue to train on diverse experience. WMAR (World Models with Augmented Replay) showed that distribution-matching replay buffers improve continual learning with DreamerV3[^16]. |
| **Evidence quality** | Moderate. WMAR demonstrates the principle for DreamerV3[^16]. Prioritized replay has known benefits for sample efficiency[^15]. |
| **Expected gain** | **Small-Medium** (+1-3%) |
| **Time to implement** | 3-5 hours (buffer modification + sampling logic) |
| **Risk of backfire** | Low. Main risk: slower convergence if balance is wrong. |
| **Compatibility** | Requires modification to replay buffer code |

### Intervention 7: PFSP Temperature Reduction

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | Reduce PFSP variance/temperature so opponent sampling is more uniform across the pool rather than heavily prioritizing specific opponents. High PFSP variance causes over-specialization against a few pool members while neglecting others, producing the observed pfsp_min_winrate drops[^3][^2]. |
| **Evidence quality** | Moderate. AlphaStar used PFSP proportional to win rate[^2]; reducing temperature makes sampling more uniform, preventing worst-case neglect. |
| **Expected gain** | **Small-Medium** (+1-2% on min win rate metrics) |
| **Time to implement** | 0.5-1 hour (config change) |
| **Risk of backfire** | Low |
| **Compatibility** | Direct parameter adjustment |

### Intervention 8: KL Penalty Toward Frozen Best-Checkpoint Policy

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | Add a small KL divergence penalty between the current actor distribution and the frozen best_selfplay_336k actor distribution, preventing the policy from drifting too far during fine-tuning. This is a lightweight version of Policy Consolidation[^7][^10] and the piKL framework used in Diplomacy AI[^17][^18]. |
| **Evidence quality** | Strong theoretical and empirical support. piKL-Hedge in Diplomacy showed KL-regularized search maintains human-like play while improving strength[^17]. Policy Consolidation improved self-play continual learning[^7]. |
| **Expected gain** | **Medium** (+2-4% robustness) |
| **Time to implement** | 3-5 hours (modify actor loss to include KL term) |
| **Risk of backfire** | Medium. Too strong KL coefficient prevents learning; too weak has no effect. Requires tuning λ. |
| **Compatibility** | Requires actor loss modification in DreamerV3 |

### Intervention 9: Multi-Strategy Ensemble at Deployment

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | If time permits, train a second agent with a defensive reward bias (heavier penalty for conceding), then ensemble the two at deployment by selecting the more conservative action or score-based switching. The DreamerV3 air hockey paper achieved 2nd place with this approach[^1][^19]. |
| **Evidence quality** | Direct precedent—same algorithm, similar environment[^1]. |
| **Expected gain** | **Medium-Large** if well-calibrated (+3-7%), but requires training time |
| **Time to implement** | 15-24 hours (full training run + ensemble logic) |
| **Risk of backfire** | Medium. Ensemble switching logic can introduce inconsistency. |
| **Compatibility** | Requires second training run and deployment wrapper |

### Intervention 10: Increase Imagination Horizon from 15 to 25-30

| Attribute | Assessment |
|-----------|-----------|
| **Mechanism** | The air hockey paper found longer imagination horizons (50) produced more stable learning and better performance than shorter ones (10, 25)[^1][^19]. Current horizon of 15 may limit the agent's ability to plan multi-step strategies. Increasing to 25-30 trades VRAM for potentially better credit assignment with sparse rewards. |
| **Evidence quality** | Direct experimental evidence from DreamerV3 air hockey[^1]. DreamerV3 can handle high replay ratios and horizons[^15][^20]. |
| **Expected gain** | **Small-Medium** (+1-3%, but uncertain given 3-day timeline) |
| **Time to implement** | 0.5 hours (config change) + requires full re-training to benefit |
| **Risk of backfire** | Medium. Requires more VRAM, slower training per step. Late deadline risk. |
| **Compatibility** | Config change only |

***

## C) Final Attempt Plan (72-Hour Execution)

### Hour 0-3: Bug Fix and Diagnostics (CRITICAL PATH)

**Step 1 (Hour 0-1): Opponent Selection Audit**
- Add explicit logging: for every episode, log `actual_opponent_type` (weak/strong/self_play_id) alongside `intended_opponent_type`
- Run 100 episodes with current config, verify distribution matches intent
- Check the code path: does `self_play_manager.select_opponent()` override or interact with `mixed_opponent_ratio`?
- Specifically check: when self-play is enabled and anchor is triggered, is the anchor actually loading a fixed bot or falling through to a self-play agent?

**Step 2 (Hour 1-2): Apply Bug Fix**
- Fix any distribution skew found
- Verify with logging that post-fix distribution is correct

**Step 3 (Hour 2-3): Baseline Evaluation**
- Run 100-episode deterministic eval of best_selfplay_336k against weak, strong, and 3-5 recent self-play checkpoints
- Record: win rates, avg reward, and per-opponent performance
- This is the baseline for all subsequent experiments

### Hour 3-6: Quick Config Changes + Launch Primary Run

**Apply simultaneously:**
- Anchor ratio: 65% fixed bots (33% weak, 32% strong), 35% self-play pool
- Actor LR: 0.5× original value (e.g., if 3e-5, reduce to 1.5e-5)
- PFSP temperature: reduce by 30-50% (e.g., if variance=0.5, try 0.3)
- EMA shadow weights: implement for actor (α=0.998), begin tracking immediately
- Resume from best_selfplay_336k

**Monitoring schedule (every 10k steps):**
- Eval against weak bot (20 episodes)
- Eval against strong bot (20 episodes)
- Eval EMA weights separately against same bots
- Log: pfsp_min_winrate, winrate_vs_pool_overall, winrate_newest_third

**Early stop criteria:**
- If combined_win_rate drops below 0.85 for 3 consecutive evals → stop, revert
- If combined_win_rate exceeds 0.95 AND pfsp_min_winrate > 0.55 → checkpoint as candidate

**Run duration: ~18-24 hours**

### Hour 6-8: Implement KL Regularization (Parallel Task)

While primary run is executing:
- Implement KL penalty toward frozen best_selfplay_336k actor
- Use `λ_KL = 0.01` as starting point
- This is the backup intervention if primary run shows cycling again

### Hour 24-30: Evaluate Primary Run Results

**Checkpoint selection protocol:**
1. Identify top-5 checkpoints by combined_win_rate
2. For each, run 50-episode eval against: weak, strong, 3 self-play pool members
3. Compute worst-case win rate across all opponents
4. Also evaluate EMA shadow weights at each checkpoint
5. Select the checkpoint (regular or EMA) with the best worst-case win rate

**Decision point:**
- If best checkpoint achieves combined_win_rate ≥ 0.95 AND worst-case ≥ 0.70 → **ACCEPT**, proceed to polish
- If combined_win_rate ≥ 0.90 but worst-case < 0.60 → Launch **Fallback Run** with KL regularization
- If combined_win_rate < 0.90 → Revert to original best_selfplay_336k, focus on evaluation protocol

### Hour 30-48: Fallback Run (If Needed) or Polish Run

**Fallback (if primary disappointing):**
- Resume from best_selfplay_336k with KL regularization (λ=0.01)
- Anchor ratio 70% fixed, 30% self-play
- Actor LR: 0.3× original
- Same monitoring protocol
- Run for 12-18 hours

**Polish run (if primary successful):**
- Take best checkpoint from primary
- Continue with 80% fixed bots, 20% self-play, 0.2× LR
- Run for 6-12 hours as fine-tuning
- Very conservative—just refining without risking regression

### Hour 48-60: Final Evaluation and Model Selection

- Run comprehensive eval: 100 episodes against weak, strong, self-play pool members
- Evaluate both regular and EMA weights for top-3 candidates
- Run tournament-style round-robin between all candidates
- Select final model based on worst-case win rate

### Hour 60-72: Tournament Preparation

- Lock final model
- Test competition client submission
- Verify inference latency (<20ms per step if applicable)
- Run smoke test against all available opponents
- Prepare backup model (original best_selfplay_336k) in case primary fails at submission

### Exact Hyperparameter Summary

| Parameter | Primary Run | Fallback Run | Polish Run |
|-----------|------------|-------------|------------|
| Resume from | best_selfplay_336k | best_selfplay_336k | Primary best |
| Actor LR multiplier | 0.5× | 0.3× | 0.2× |
| Anchor ratio | 65% fixed | 70% fixed | 80% fixed |
| Self-play pool ratio | 35% | 30% | 20% |
| PFSP temperature | 0.7× original | 0.7× original | 0.5× original |
| EMA decay (actor) | 0.998 | 0.998 | 0.998 |
| KL regularization λ | 0 | 0.01 | 0.005 |
| DreamSmooth | ON (keep) | ON (keep) | ON (keep) |
| Two-Hot | ON (keep) | ON (keep) | ON (keep) |
| Imagination horizon | 15 (keep) | 15 (keep) | 15 (keep) |
| Replay ratio | 32 (keep) | 32 (keep) | 32 (keep) |

***

## D) Minimal Patch Set

### Patch 1: Opponent Selection Bug Fix (HIGHEST PRIORITY)

**Diagnostic check (pseudo-code):**
```
# In training loop, add after opponent selection:
log(f"Episode {ep}: intended={intended_opponent}, actual={actual_opponent.__class__.__name__}, "
    f"is_selfplay={is_selfplay_opponent}, opponent_id={opponent_id}")

# After 100 episodes, verify:
# count(actual==weak) / total ≈ configured_weak_ratio
# count(actual==strong) / total ≈ configured_strong_ratio  
# count(actual==selfplay) / total ≈ configured_selfplay_ratio
```

**Likely bug location:** Check whether `self_play_manager.select_opponent()` is called *before* or *after* the anchor/fixed-bot decision. If the self-play manager always returns an opponent and the calling code doesn't check the anchor flag, fixed-bot selection gets silently overridden.

**Fix pattern:**
```
# BEFORE (buggy):
opponent = self_play_manager.select_opponent()  # Always returns SP agent

# AFTER (fixed):
if random.random() < anchor_ratio:
    opponent = select_fixed_bot(weak_ratio)  # Guaranteed fixed bot
else:
    opponent = self_play_manager.select_opponent()  # SP pool
```

### Patch 2: EMA Shadow Weights

```
# At initialization:
ema_actor_params = {k: v.clone() for k, v in actor.state_dict().items()}
ema_decay = 0.998

# After each actor gradient step:
with torch.no_grad():
    for k, v in actor.state_dict().items():
        ema_actor_params[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

# For evaluation:
def load_ema_weights(actor, ema_params):
    original = {k: v.clone() for k, v in actor.state_dict().items()}
    actor.load_state_dict(ema_params)
    return original  # save for restoring
```

### Patch 3: Enhanced Logging

```
# Every eval_interval steps, log:
metrics = {
    'combined_win_rate': (weak_wins + strong_wins) / (weak_total + strong_total),
    'worst_case_win_rate': min(weak_wr, strong_wr, *selfplay_wrs),
    'ema_combined_win_rate': ema_eval_combined,
    'pfsp_min_winrate': min_wr_against_any_pool_member,
    'opponent_distribution': {type: count for type, count in opponent_counts.items()},
}
```

### Patch 4: KL Regularization (for fallback)

```
# In actor loss computation (imagination training):
# frozen_actor = copy of actor at best_selfplay_336k, requires_grad=False

with torch.no_grad():
    frozen_dist = frozen_actor(imagined_states)  # Get frozen policy distribution

current_dist = actor(imagined_states)
kl_penalty = torch.distributions.kl_divergence(current_dist, frozen_dist).mean()

actor_loss = original_actor_loss + lambda_kl * kl_penalty
```

***

## E) Evaluation Protocol for Arbitrary Opponent Robustness

### Tier 1: Fixed-Bot Evaluation (5 minutes, run always)

- 50 episodes vs weak bot (deterministic)
- 50 episodes vs strong bot (deterministic)
- Metrics: win rate, avg reward, goal differential
- **Threshold: combined ≥ 0.90**

### Tier 2: Self-Play Pool Cross-Evaluation (15 minutes)

- 20 episodes vs each of the 5 most recent self-play checkpoints
- 20 episodes vs each of the 5 oldest surviving pool members
- Metrics: per-opponent win rate, worst-case win rate, mean win rate
- **Threshold: worst-case ≥ 0.55, mean ≥ 0.70**

### Tier 3: Exploitability Probe (30 minutes)

- Take 3 diverse checkpoints from training history (early, mid, late)
- Run 30 episodes each
- If any checkpoint beats the candidate at >40% win rate, flag as exploitable
- **Threshold: no single opponent beats candidate at >40%**

### Tier 4: Behavioral Diversity Check (15 minutes)

- Run 50 episodes vs strong bot, log action distributions per game phase
- Compare action variance to baseline—if significantly lower, the agent may have collapsed to a narrow strategy
- **Threshold: action entropy within 0.7× of baseline**

### Decision Matrix

| Tier 1 | Tier 2 | Tier 3 | Decision |
|--------|--------|--------|----------|
| ≥0.90 | worst≥0.55 | No exploit | **ACCEPT** — submit this model |
| ≥0.90 | worst<0.55 | Any | **CONDITIONAL** — accept only if better than current best |
| <0.90 | Any | Any | **REJECT** — revert to previous best |
| ≥0.95 | worst≥0.70 | No exploit | **STRONG ACCEPT** — high confidence submission |

### Comparing Two Candidates

When choosing between two passing candidates, rank by:
1. Worst-case win rate (most important for tournament)
2. Combined fixed-bot win rate
3. Self-play pool mean win rate
4. Action entropy (prefer higher diversity as tiebreaker)

***

## F) If I Can Only Do 1 Thing

**Fix the opponent selection bug and verify the training distribution.**

This is the single highest expected-value action because:

1. **It costs almost nothing** (1-3 hours of debugging and a config fix).
2. **It addresses the most likely root cause** (~70% probability this is the primary issue).
3. **It is prerequisite to all other interventions**—no amount of fancy regularization or curriculum tuning will help if the agent is training against the wrong opponent distribution.
4. **The evidence is strong**: the symptom (anchor counters balanced, actual counts skewed) is a classic logging-vs-reality mismatch. AlphaStar's own post-mortem revealed that opponent sampling imbalances were a primary cause of fragility, leading to the main agent being "easily defeated by Grandmaster players using some uncommon strategies".[^2]
5. **Even if the bug isn't the full story**, understanding the true opponent distribution is essential diagnostic information for all subsequent decisions.

After fixing the bug, the single best action is to **resume training with 65% fixed-bot anchoring, 0.5× actor LR, and EMA shadow weights**—a 30-minute config change that combines three low-risk interventions.

***

## Executive Recommendation (One Page)

The late-stage regression pattern (high peak → decline in both fixed-bot and self-play metrics) with stable world model losses is characteristic of a **policy-level distribution shift**, not a learning failure. The most likely cause is an opponent selection bug that skews the actual training distribution away from the configured ratio, compounded by natural cycling in self-play without anti-forgetting mechanisms.

**The 72-hour plan has three phases:**

**Phase 1 (Hours 0-6): Fix and Launch.** Audit and fix the opponent selection code. Apply conservative config changes (65% anchor, 0.5× LR, EMA). Launch primary training run from best checkpoint.

**Phase 2 (Hours 6-48): Train and Monitor.** Monitor every 10k steps. The primary run should improve within 50-100k steps if the distribution fix was correct. Have KL-regularized fallback ready. Evaluate both regular and EMA weights.

**Phase 3 (Hours 48-72): Select and Ship.** Use worst-case win rate (not average) to select the final model. Run the 4-tier evaluation protocol. Submit the most robust candidate, keeping the original best_selfplay_336k as backup.

**Key insight from the literature**: The DreamerV3 air hockey paper demonstrated that self-play is essential but that agents overfit to specific playstyles without diversity. ROA-Star showed that worst-case metrics are far more predictive than average metrics for tournament robustness. The Gigaflow self-play paper confirmed that scale matters, but also that advantage filtering (focusing on high-impact transitions) improves efficiency dramatically.[^11][^21][^1][^2]

**Do not**: Rewrite the architecture. Change DreamSmooth or Two-Hot. Increase imagination horizon (not enough time to benefit). Train from scratch.

**Do**: Fix the bug. Lower the LR. Add EMA. Anchor harder on fixed bots. Select by worst-case.

***

## Run Sheet: Prioritized Experiments

### Experiment 1: Bug Fix + Conservative Fine-Tuning (PRIMARY)
- **Start:** Hour 3 (after bug fix confirmed)
- **Duration:** 18-24 hours
- **Config:** Resume best_selfplay_336k, 65% anchor, 0.5× actor LR, EMA α=0.998, PFSP temp 0.7×
- **Monitor:** Combined win rate, worst-case win rate, pfsp_min_winrate every 10k steps
- **Stop if:** Combined WR < 0.85 for 3 consecutive evals
- **Success criterion:** Combined ≥ 0.95 AND worst-case ≥ 0.70

### Experiment 2: KL-Regularized Fine-Tuning (FALLBACK)
- **Start:** Hour 30 (only if Experiment 1 disappoints)
- **Duration:** 12-18 hours
- **Config:** Resume best_selfplay_336k, 70% anchor, 0.3× actor LR, KL λ=0.01, EMA α=0.998
- **Monitor:** Same metrics
- **Success criterion:** Combined ≥ 0.92 AND worst-case ≥ 0.60

### Experiment 3: Polish Run (IF TIME PERMITS)
- **Start:** After best candidate selected from Exp 1 or 2
- **Duration:** 6-12 hours
- **Config:** 80% anchor, 0.2× LR, KL λ=0.005
- **Purpose:** Fine-grain stabilization of already-good checkpoint
- **Success criterion:** No regression from input checkpoint

---

## References

1. [Learning to Play Air Hockey with Model-Based Deep Reinforcement ...](https://arxiv.org/html/2406.00518v1) - Our approach revolves around leveraging DreamerV3 for training reinforcement learning agents to play...

2. [[PDF] A Robust and Opponent-Aware League Training Method for ...](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf) - Based on these improvements, we train a better and superhuman AI with orders of magnitude less resou...

3. [A Survey on Self-play Methods in Reinforcement Learning](https://arxiv.org/html/2408.01072v1) - This paper first clarifies the preliminaries of self-play, including the multi-agent reinforcement l...

4. [[PDF] A Comparison of Self-Play Algorithms Under a Generalized ...](https://eprints.whiterose.ac.uk/id/eprint/171068/1/Self_play_framework_paper_3_.pdf) - Our results indicate that, throughout training, various widely used self-play algorithms exhibit cyc...

5. [On the Design of Safe Continual RL Methods for Control of ...](https://www.arxiv.org/pdf/2502.15922.pdf) - We show that agents trained with CPO, a safe RL algorithm, experience higher catastrophic forgetting...

6. [Multi-task Learning and Catastrophic Forgetting in ...](https://arxiv.org/abs/1909.10008) - by J Ribeiro · 2019 · Cited by 20 — Abstract:In this paper we investigate two hypothesis regarding t...

7. [[PDF] Policy Consolidation for Continual Reinforcement Learning](http://proceedings.mlr.press/v97/kaplanis19a/kaplanis19a.pdf) - We propose a method for tackling catastrophic forgetting in deep reinforcement learning that is agno...

8. [Exponential Moving Average of Weights in Deep Learning - arXiv](https://arxiv.org/html/2411.18704v1) - In this work, we present a systematic study of the Exponential Moving Average (EMA) of weights. We f...

9. [Exponential Moving Average of Weights in Deep Learning](https://openreview.net/forum?id=2M9CUnYnBA) - In this work, we present a systematic study of the Exponential Moving Average (EMA) of weights. We f...

10. [Policy Consolidation for Continual Reinforcement Learning](https://proceedings.mlr.press/v97/kaplanis19a.html) - by C Kaplanis · 2019 · Cited by 87 — We propose a method for tackling catastrophic forgetting in dee...

11. [Robust Autonomy Emerges from Self-Play - arXiv](https://arxiv.org/html/2502.03349v1) - We show that robust and naturalistic driving emerges entirely from self-play in simulation at unprec...

12. [Scaling Off-Policy Reinforcement Learning with Batch and Weight ...](https://arxiv.org/html/2502.07523v1) - This work eliminates the need for drastic interventions, such as network resets, and offers a simple...

13. [[PDF] Scaling Off-Policy Reinforcement Learning with Batch and Weight ...](https://openreview.net/pdf?id=Pr2fNUGU06) - As is common, we use Polyak-averaging with a τ = 0.005 from the critic network to the target network...

14. [DreamerV3 for Traffic Signal Control: Hyperparameter Tuning and ...](https://arxiv.org/abs/2503.02279) - Using the SUMO simulation platform, the two hyperparameters (training ratio and model size) of the D...

15. [Mastering Diverse Domains through World Models - ar5iv - arXiv](https://ar5iv.labs.arxiv.org/html/2301.04104) - We present DreamerV3, a general algorithm that learns to master diverse domains while using fixed hy...

16. [Augmenting Replay in World Models for Continual Reinforcement ...](https://arxiv.org/html/2401.16650v2) - We present WMAR, World Models with Augmented Replay, a model-based RL algorithm with a world model a...

17. [Modeling Strong and Human-Like Gameplay with KL- ...](https://web.mit.edu/~gfarina/www/2022/human_like_pikl_icml22/human_like_pikl.icml22.pdf) - by AP Jacob · Cited by 82 — Imi- tation learning is effective at predicting human actions but may no...

18. [Modeling Strong and Human-Like Gameplay with KL ...](https://proceedings.mlr.press/v162/jacob22a/jacob22a.pdf) - by AP Jacob · 2022 · Cited by 82 — Abstract. We consider the task of accurately modeling strong huma...

19. [[PDF] Learning to Play Air Hockey with Model-Based Deep Reinforcement ...](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/AirHockeyChallenge_SpaceR.pdf) - We use the concept of multiple strategies during the self-play training process and within the multi...

20. [Mastering diverse control tasks through world models | Nature](https://www.nature.com/articles/s41586-025-08744-2) - The replay ratio affects the number of gradient updates performed by the agent. Figure 6 shows robus...

21. [Robust Autonomy Emerges from Self-Play](https://proceedings.mlr.press/v267/cusumano-towner25a.html) - We show that robust and naturalistic driving emerges entirely from self-play in simulation at unprec...

