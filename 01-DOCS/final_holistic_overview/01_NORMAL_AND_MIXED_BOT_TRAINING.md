# Normal and Mixed-Bot Training (DreamerV3)

## Scope
This document describes the training modes where the learner trains only against fixed built-in opponents (`weak`, `strong`) or a controlled weak/strong mixture, without self-play pool sampling.

Primary implementation paths:
- `02-SRC/DreamerV3/train_hockey.py`
- `02-SRC/DreamerV3/dreamer.py`
- `02-SRC/DreamerV3/networks.py`
- `02-SRC/DreamerV3/buffer.py`
- `02-SRC/DreamerV3/opponents/fixed.py`

## Why this mode exists
1. It is the most stable bootstrapping regime.
2. It fills the replay buffer with grounded, low-noise transitions before adversarial non-stationarity from self-play.
3. It saturates fixed-opponent competence quickly and creates strong continuation checkpoints for later league/self-play phases.

## Opponent mechanics in this mode
### Single fixed opponent
- `--opponent weak` or `--opponent strong`
- `create_opponent()` instantiates `FixedOpponent` wrapping `BasicOpponent(weak=...)`.

### Mixed fixed opponents
- `--mixed_opponents --mixed_weak_prob <p>`
- Per episode, before self-play activation, opponent is sampled as:
  - weak with probability `p`
  - strong with probability `1-p`
- This reduces overfitting to one bot behavior policy.

## End-to-end training loop in code
## 1) Initialization
- Parse CLI and apply overrides into `config`.
- Build hockey env (`Mode.NORMAL` unless changed).
- Derive dimensions:
  - observation size = 18
  - action size = 4 (agent action, half of env 8D joint action)
- Create `Dreamer` agent.

## 2) Optional resume
- `agent.loadCheckpoint(path, load_optimizers=True)` by default.
- If optimizer parameter groups mismatch, code retries with `load_optimizers=False` to restore model weights only.

## 3) Warmup / buffer priming
- With `--warmup_episodes N`, collect N real episodes.
- Each transition `(obs, action, reward, next_obs, done)` is pushed to replay buffer.
- If warmup is small, code still enforces minimum transitions:
  - at least `batch_size * batch_length` before gradient updates.

## 4) Main iteration (repeated until `gradientSteps`)
For each interaction cycle:
1. Run `replayRatio` gradient updates.
2. For each update:
   - `buffer.sample(batch_size, batch_length)`
   - `agent.worldModelTraining(batch)`
   - `agent.behaviorTraining(full_states_from_world_model)`
3. Run `interaction_episodes` real episodes against selected fixed/mixed opponent.
4. Add transitions to buffer.
5. Update aggregate counters (`totalEpisodes`, `totalEnvSteps`, `totalGradientSteps`).

## World model training internals (`worldModelTraining`)
### State-space construction
- Observation is transformed with `symlog` before encoding.
- RSSM roll-through over sequence:
  - recurrent GRU state `h`
  - categorical latent `z` (16x16 by default)
  - prior from `h`, posterior from `h + encoded_obs`

### Loss terms
1. Reconstruction loss
- Decoder predicts symlog-observation distribution.

2. Reward prediction loss
- Default: Two-Hot Symlog categorical loss.
- Ablation mode: Gaussian NLL when `--use_gaussian_heads`.

3. KL regularization
- Prior/posterior KL with asymmetric weights (`betaPrior`, `betaPosterior`) and `freeNats` threshold.

4. Optional continuation loss
- Bernoulli continue predictor if enabled.

### Sparse-reward handling in this phase
- Inverse-frequency weighting for sparse reward events (`|reward| > 1`), capped.
- Optional DreamSmooth via replay buffer sampling path (`dreamsmooth_ema`).
- Rich sparse diagnostics are logged.

## Behavior training internals (`behaviorTraining`)
### Imagined rollouts
- Start from posterior states from world-model batch.
- Roll imagination horizon steps in latent space using actor + RSSM prior.

### Targets and returns
- Predict imagined rewards and values.
- Compute TD(lambda) returns via `computeLambdaValues`.
- Normalize advantages with `Moments` percentile scaling.

### Actor objective
- Policy gradient on normalized advantages + entropy bonus.
- Mean regularization term (`mean_reg_scale`) to avoid tanh-saturation drift.

### Critic objective
- Default: Two-Hot value cross-entropy.
- Ablation: Gaussian NLL.
- Slow critic EMA updated every critic step for stability monitoring.

## Logging/evaluation/checkpointing in this mode
1. Frequent train logs
- Win rate over recent window, reward, losses, entropy, timings.

2. Periodic fixed-opponent eval
- Evaluates against weak and strong.
- Logs combined win rate.

3. Optional GIF logging
- Records weak and strong eval rollouts.

4. Checkpointing
- Periodic and final checkpoints include model + optimizer + counters.

## What this regime is best for
1. Fast stabilization from scratch.
2. Learning core dynamics and control priors.
3. Producing robust base checkpoints for later self-play/league stages.

## Known bottlenecks/limits
1. Fixed bots induce limited strategic diversity.
2. Agent can overfit to anchor bot tendencies.
3. Tournament transfer can lag without later pool-based adversarial diversity.
