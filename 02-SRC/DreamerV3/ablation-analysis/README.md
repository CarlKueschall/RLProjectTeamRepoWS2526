# Ablation Analysis

Folder for analyzing ablation runs, comparing metrics, hyperparameters, and generating compact reports for the RL Hockey project.

## Structure

```
ablation-analysis/
├── input/          # Place your ablation run metrics here
├── output/         # Generated reports, figures, summaries
└── README.md
```

## Input: Naming Convention

Place metrics files in `input/` with descriptive names. Suggested format:

```
<ablation_type>_<variant>_<optional_suffix>.<ext>
```

Examples:
- `dreamsmooth_ON_weak_seed42.json` / `.csv` / `.txt`
- `dreamsmooth_OFF_weak_seed42.json`
- `twohot_ON_weak_seed42.json`
- `twohot_MSE_weak_seed42.json`
- `auxiliary_ON_weak_seed42.json`
- `auxiliary_OFF_weak_seed42.json`

Supported formats: JSON (W&B export), CSV, or structured text. Include hyperparameters and metric series (e.g. `stats/win_rate`, `eval/combined_win_rate`, `world/loss`, etc.).

## Output

After running `python analyze_ablations.py`:

| File | Description |
|------|-------------|
| `runs_summary.json` | Extracted metrics for all runs |
| `metric_series.json` | Time series (gradient_steps, win_rate, mean_reward) for plotting |
| `comparison_table.csv` | Compact CSV for report tables |
| `comparison_dreamsmooth.md` | DreamSmooth ON vs OFF |
| `comparison_twohot.md` | Two-Hot vs Gaussian |
| `comparison_auxiliary.md` | Auxiliary tasks ON vs OFF |
| `overall_observations.md` | Cross-ablation summary |
| `learnings.md` | Key takeaways |
| `synthesis_report.md` | Why and how—deeper analysis |

## Usage

1. Place W&B run exports in `input/` (e.g. via `download_wandb_run.py`).
2. Run `python analyze_ablations.py` to parse and extract metrics.
3. Read the markdown reports in `output/` for analysis and report text.
