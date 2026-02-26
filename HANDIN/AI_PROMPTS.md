# AI Prompts Log

This file documents code files in this hand-in that explicitly state they were written entirely by AI.
Replace each placeholder with the original full prompt(s) used.

## 1) `DreamerV3/scripts/download_wandb_run.py`

- Declaration found in file header: code written entirely by Claude Code.
- Prompt(s) used:

```text
Create a script that will download every single wandb metric that we are currently using with the option to pull only a fraction of the data as a cli arg. ensure that this dynamically scales for future metrics that we add, meaning that we won't have to update this file 1billion times.
```

## 2) `DreamerV3/scripts/eval_summit_matrix.py`

- Declaration found in file header: code written entirely by Claude Code.
- Prompt(s) used:

```text
Given the following plan, please implement a script to run a massive checkpoint comparison matrix (candidate vs fixed weak/strong and candidate vs checkpoint opponents), with an efficient parallel worker systemand resumable outputs, and produce a ranked summary CSV with the target columns: checkpoint, fixed_weak_win_rate, fixed_strong_win_rate, fixed_combined_win_rate, cp_mean_win_rate, cp_q20_win_rate, cp_worst_win_rate, robust_score.
```

## 4) `DreamerV3/visualization/gif_recorder.py`

- Declaration found in file header: code written entirely by Claude Code.
- Prompt(s) used:

```text
implement this now, we need a script that uses gif-capture to record n episodes and upload them to wandb for inspection. 
```
