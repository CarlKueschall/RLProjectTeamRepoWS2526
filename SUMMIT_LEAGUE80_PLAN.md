# Summit Final 24h Plan (League 80/10/10)

## Goal
Resume from the current best checkpoint and train with:
- 80% checkpoint league opponents
- 10% weak anchor
- 10% strong anchor

## What Was Implemented
- Added self-play bootstrap support in:
  - `02-SRC/DreamerV3/opponents/self_play.py`
  - `02-SRC/DreamerV3/train_hockey.py`
- New CLI flags:
  - `--self_play_bootstrap_dir`
  - `--self_play_bootstrap_glob`
  - `--self_play_bootstrap_max`
  - `--self_play_bootstrap_strategy`
- New sbatch file:
  - `02-SRC/DreamerV3/sbatch/summit_final24h_league80.sbatch`

## Execution Commands

1. Sync code to cluster:
```bash
rsync -av \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='.git' \
  --exclude='wandb' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  /Users/carlkueschall/workspace/RLProjectHockey/02-SRC \
  stud432@login3.tcml.uni-tuebingen.de:~/
```

2. Upload resume checkpoint into DreamerV3 root (same dir as `train_hockey.py`):
```bash
scp /Users/carlkueschall/workspace/RLProjectHockey/02-SRC/DreamerV3/checkpoints/training/results_checkpoints__weak_seed42_20260223_031428_474k.pth \
  stud432@login3.tcml.uni-tuebingen.de:~/02-SRC/DreamerV3/
```

3. Upload league checkpoints (Dreamer + optional TD3):
```bash
ssh stud432@login3.tcml.uni-tuebingen.de "mkdir -p ~/02-SRC/DreamerV3/checkpoints/summit-evaluation/opponents"

scp /Users/carlkueschall/workspace/RLProjectHockey/02-SRC/DreamerV3/checkpoints/summit-evaluation/opponents/*.pth \
  stud432@login3.tcml.uni-tuebingen.de:~/02-SRC/DreamerV3/checkpoints/summit-evaluation/opponents/

scp /Users/carlkueschall/workspace/RLProjectHockey/99-ARCHIVE/TD3/results_checkpoints_TD3_Hockey_NORMAL_weak_27500_seed102.pth \
  stud432@login3.tcml.uni-tuebingen.de:~/02-SRC/DreamerV3/checkpoints/summit-evaluation/opponents/
```

4. Verify remote files:
```bash
ssh stud432@login3.tcml.uni-tuebingen.de "ls -lh ~/02-SRC/DreamerV3/checkpoints/training | rg 'results_checkpoints__weak_seed42_20260223_031428_474k.pth' && ls -lh ~/02-SRC/DreamerV3/checkpoints/summit-evaluation/opponents | wc -l"
```

5. Submit training job:
```bash
ssh stud432@login3.tcml.uni-tuebingen.de "cd ~/02-SRC/DreamerV3 && sbatch sbatch/summit_final24h_league80.sbatch"
```

6. Track logs:
```bash
ssh stud432@login3.tcml.uni-tuebingen.de "cd ~/02-SRC/DreamerV3 && tail -f job.*.out"
```

## Expected Training Behavior
- Self-play activates at episode 1.
- Pool is immediately bootstrapped from `checkpoints/summit-evaluation/opponents`.
- Opponent split converges to ~80% self-play and ~20% anchors (balanced weak/strong).
- Robustness metrics logged during eval:
  - `eval/probe_min_win_rate`
  - `eval/probe_mean_win_rate`
  - `eval/robust_score`
