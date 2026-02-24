# Holistic Overview: DreamerV3 Hockey Training Strategy

## Purpose of this overview
This file connects all training regimes and support systems into one coherent map so the full strategy can be explained clearly in a professor-facing presentation.

Supporting detailed reports in this folder:
1. `01_NORMAL_AND_MIXED_BOT_TRAINING.md`
2. `02_SELF_PLAY_TRAINING.md`
3. `03_LEAGUE_CURATED_POOL_TRAINING.md`
4. `04_EVALUATION_AND_MODEL_SELECTION_PIPELINE.md`

## System boundaries
Active training system is DreamerV3 under:
- `02-SRC/DreamerV3/`

Key components:
1. `train_hockey.py`
- orchestration loop, opponent control, logging/eval/checkpoints.

2. `dreamer.py`
- world model + imagination behavior training.

3. `networks.py`
- RSSM parts and model heads.

4. `buffer.py`
- replay sequence sampling and optional DreamSmooth smoothing.

5. `opponents/`
- fixed bot wrappers, PFSP, self-play manager.

6. `scripts/`
- matrix and pool-selection evaluation tooling.

## End-to-end dataflow
## 1) Real environment interaction
- Agent observes 18D state, emits 4D action.
- Opponent emits 4D action.
- Environment stepped with concatenated 8D action.
- Transition stored in replay buffer.

## 2) World model training on replay sequences
- Sample contiguous sequences.
- Encode symlog observation.
- Roll RSSM (recurrent + categorical latent).
- Optimize reconstruction + reward + KL (+ optional continuation).

## 3) Imagination policy learning
- Start from posterior latent states.
- Roll imagined trajectories in latent dynamics.
- Predict imagined rewards/values.
- Compute lambda returns and normalized advantages.
- Update actor and critic.

## 4) Opponent curriculum layer
- Depending on phase: fixed bots, self-play pool, or curated league bootstrap.

## 5) Evaluation and selection layer
- fixed eval, probe eval, robust score.
- offline matrix/pool selection scripts to choose checkpoints and league composition.

## 6) Checkpoint lifecycle
- save periodic checkpoints.
- select best robust candidates.
- continue training from selected checkpoint.

## Distinct training regimes and their role
## Regime A: Normal single-opponent fixed-bot
- Simplest stable baseline.
- Useful for early validation and debugging.

## Regime B: Mixed fixed-bot
- Pre-self-play stabilization against weak+strong distribution.
- Builds stronger general prior than single-bot training.

## Regime C: Self-play (pool + PFSP)
- Introduces dynamic adversarial curriculum.
- Improves robustness beyond fixed-bot overfitting.

## Regime D: League-curated continuation
- Starts from strong checkpoint.
- Seeds pool with selected diverse high-value checkpoints.
- Uses weighted priors + PFSP to accelerate robust gains.

## What each regime contributes
1. Regime A/B contributes stability and competence floor.
2. Regime C contributes adaptive robustness and strategic breadth.
3. Regime D contributes large diversity and better sample efficiency under time constraints.

## Why this strategy is compute-efficient
1. Most expensive exploration is front-loaded in mixed/fixed phases.
2. Continuations reuse strong checkpoints rather than restarting.
3. Curated league avoids wasting self-play on low-value or redundant opponents.
4. Evaluation scripts downselect large checkpoint sets into high-quality pools.

## Core algorithmic choices and their rationale
## RSSM with categorical latents
- Better uncertainty handling and representation expressivity for nontrivial dynamics.

## Two-Hot Symlog heads (default)
- Better suited to sparse, multi-modal reward/value targets than plain Gaussian.

## Sparse-event reward weighting
- Prevents rare goal events from being gradient-diluted.

## Percentile value normalization (`Moments`)
- Stabilizes actor updates despite return-scale drift.

## Slow critic EMA
- Reduces rapid critic drift and bootstrapping instability.

## PFSP opponent sampling
- Focuses training on informative opponents, not only easy or impossible ones.

## Runtime signals that indicate healthy training
1. Buffer grows and stays near capacity after warm phase.
2. Entropy remains positive and not collapsed.
3. Pool-overall and stratified self-play winrates trend upward without extreme forgetting.
4. Probe min/mean and robust score improve over checkpoints.
5. Fixed weak/strong remain high while cp robustness improves.

## Runtime signals that indicate problems
1. Bootstrap failures from missing files or wrong absolute paths.
2. Pool size stuck near 1 after activation.
3. High fixed-bot score but poor cp-worst or probe-min.
4. Compatibility failures from old checkpoint architectures.
5. Opponent distribution mismatch with intended anchor/self-play ratio.

## How all pieces connect operationally
1. Train with selected regime and save frequent checkpoints.
2. Run evaluation scripts on checkpoint sets.
3. Build curated league directory from recommended pool.
4. Resume from best robust checkpoint with league-bootstrap self-play.
5. Monitor online robust metrics and repeat selection loop.

This creates a closed optimization loop:
- training -> checkpoint generation -> robust evaluation -> league curation -> stronger training.

## Practical narrative for presentation
A defensible high-level narrative is:
1. We started with stable world-model learning against fixed anchors.
2. We then introduced self-play to escape fixed-opponent overfitting.
3. We finally added a curated checkpoint league and weighted PFSP to maximize diversity and robustness under hard time constraints.
4. Selection was robustness-first (worst-case and quantile aware), not single-metric cherry-picking.

## What this overview enables next
This folder now provides enough structure to derive visualization assets that are faithful to implementation:
1. pipeline/dataflow diagram,
2. phase-timeline diagram,
3. opponent-sampling schematic,
4. checkpoint-selection flowchart,
5. metric-dashboard storyboard for training-health interpretation.
