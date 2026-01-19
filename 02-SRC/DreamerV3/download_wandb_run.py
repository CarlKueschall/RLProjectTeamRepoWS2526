"""
AI Usage Declaration:
This file was developed with assistance from Claude Code.

W&B Run Data Downloader for DreamerV3 Hockey

Downloads complete run data from W&B for analysis.
Outputs a text file that can be shared for debugging.

Usage:
    python download_wandb_run.py --run_name "DreamerV3-small-weak-seed43"
    python download_wandb_run.py --run_id "abc123xyz"
    python download_wandb_run.py --run_name "..." --max_chars 50000
"""

import argparse
import numpy as np
import wandb
from pathlib import Path
from collections import defaultdict


# =============================================================================
# METRIC CATEGORIES - Comprehensive DreamerV3 Metrics
# =============================================================================
#
# These are all the metrics logged by our DreamerV3 implementation.
# Organized by category for easy analysis.
#
# -----------------------------------------------------------------------------
# TRAINING PROGRESS (what you check first)
# -----------------------------------------------------------------------------
# stats/
#   gradient_steps       - Total gradient updates
#   env_steps            - Total environment steps
#   episodes             - Total episodes completed
#   win_rate             - Rolling win rate (last 100)
#   mean_reward          - Rolling mean reward (last 100)
#   wins, losses, draws  - Cumulative counts
#   buffer_size          - Replay buffer size
#
# episode/
#   length_mean/std/min/max    - Episode duration stats
#   reward_mean/std            - Sparse reward stats
#   shaped_reward_mean         - PBRS contribution
#   win_rate_100               - Win rate last 100 episodes
#   draw_rate_100              - Timeout rate
#
# time/
#   elapsed_hours              - Training time
#   steps_per_second           - Env throughput
#   episodes_per_hour          - Episode throughput
#   gradient_steps_per_second  - Training speed
#
# eval/
#   win_rate, mean_reward      - Evaluation performance
#   wins, losses, draws        - Eval outcome counts
#
# -----------------------------------------------------------------------------
# WORLD MODEL (is the model learning the environment?)
# -----------------------------------------------------------------------------
# world/
#   loss                 - Total world model loss
#   recon_loss           - Observation reconstruction loss
#   reward_loss          - Reward prediction loss
#   kl_loss              - KL divergence loss
#   continue_loss        - Episode termination prediction loss
#
#   recon_error_mean/std/max   - Reconstruction quality distribution
#   reward_pred_error_mean/std - Reward prediction accuracy
#   reward_pred_mean           - Predicted reward mean
#   reward_actual_mean         - Actual reward mean
#
#   latent_entropy             - Diversity of latent space (higher=better)
#   prior_posterior_kl         - World model uncertainty
#
#   continue_pred_mean         - Predicted continue probability
#   continue_actual_mean       - Actual continue rate
#
# -----------------------------------------------------------------------------
# BEHAVIOR / ACTOR-CRITIC (is the policy learning?)
# -----------------------------------------------------------------------------
# behavior/
#   actor_loss                 - Policy loss
#   critic_loss                - Value function loss
#
#   entropy_mean/std/min/max   - Policy entropy distribution
#   logprobs_mean/std          - Action log probabilities
#
#   advantages_mean/std/min/max      - Policy gradient signal
#   advantages_abs_mean              - Signal magnitude
#
# values/
#   mean/std/min/max           - Critic value predictions
#   lambda_returns_mean/std/min/max  - TD(λ) targets
#   norm_low/high/scale        - Value normalization stats
#
# -----------------------------------------------------------------------------
# IMAGINATION (what does the agent "see" when imagining?)
# -----------------------------------------------------------------------------
# imagination/
#   reward_mean/std/min/max    - Predicted rewards in imagination
#   reward_abs_mean            - Reward magnitude
#   reward_nonzero_frac        - Fraction with any reward signal
#   reward_significant_frac    - Fraction with sparse-level rewards
#   continue_mean/min          - Episode continuation in imagination
#
# -----------------------------------------------------------------------------
# ACTIONS (what is the policy outputting?)
# -----------------------------------------------------------------------------
# actions/
#   mean/std/min/max           - Action distribution stats
#   abs_mean                   - Action magnitude
#   dim0/1/2/3_mean            - Per-dimension action means
#
# -----------------------------------------------------------------------------
# GRADIENTS (is training stable?)
# -----------------------------------------------------------------------------
# gradients/
#   world_model_norm           - World model gradient norm
#   actor_norm                 - Actor gradient norm
#   critic_norm                - Critic gradient norm
#
# -----------------------------------------------------------------------------
# SPARSE REWARD SIGNAL (is the agent learning from sparse rewards?)
# -----------------------------------------------------------------------------
# sparse_signal/
#   event_rate_in_batch        - How often sparse rewards appear in training
#   num_sparse_events          - Count of sparse events per batch
#   reward_variance            - Reward variance (should be >0)
#   reward_min/max/range       - Reward range in batch
#
#   sparse_pred_error          - Prediction error ON sparse rewards
#   sparse_pred_mean           - What model predicts for sparse rewards
#   sparse_actual_mean         - Actual sparse reward values
#   sparse_sign_accuracy       - Does model predict correct sign? (critical!)
#   nonsparse_pred_error       - Prediction error on dense rewards
#   sparse_vs_nonsparse_error_ratio - Is sparse harder to predict?
#
#   lambda_return_abs_mean     - Lambda return magnitude (should be >0)
#   lambda_return_nonzero_frac - Fraction with signal
#   lambda_return_significant_frac - Fraction with strong signal
#
#   value_lambda_gap_mean      - How much imagination adds to values
#   value_lambda_gap_abs_mean  - Gap magnitude
#
#   advantage_nonzero_frac     - Fraction of meaningful advantages
#   advantage_significant_frac - Fraction of strong advantages
#
# -----------------------------------------------------------------------------
# PBRS COMPONENTS (reward shaping breakdown)
# -----------------------------------------------------------------------------
# pbrs/
#   episode_total              - Total PBRS reward per episode
#   episode_mean               - Mean PBRS per step
#   chase_total                - Chase component contribution
#   attack_total               - Attack component contribution
#   chase_ratio                - Fraction from chase
#   attack_ratio               - Fraction from attack
#
# -----------------------------------------------------------------------------
# REWARD COMPOSITION (balance between sparse and shaped)
# -----------------------------------------------------------------------------
# reward_composition/
#   sparse_fraction            - % of total from sparse (should be >0.5)
#   pbrs_fraction              - % of total from PBRS (should be <0.5)
#   sparse_mean                - Avg sparse reward
#   pbrs_mean                  - Avg PBRS reward
#   total_mean                 - Avg total reward
#   pbrs_to_sparse_ratio       - |PBRS|/|sparse| (should be <1)
#
# -----------------------------------------------------------------------------
# REWARD HACKING DETECTION (is the agent gaming PBRS?)
# -----------------------------------------------------------------------------
# reward_hacking/
#   pbrs_std                   - PBRS variance
#   sparse_std                 - Sparse reward variance
#   pbrs_winrate_correlation   - PBRS-win correlation (should be >0)
#   pbrs_when_win              - Avg PBRS in winning episodes
#   pbrs_when_loss             - Avg PBRS in losing episodes
#   loss_pbrs_minus_win_pbrs   - Should be <0 (more PBRS when winning)
#
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description='Download W&B run data for analysis')

    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name (e.g., "DreamerV3-small-weak-seed43")')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID (alternative to run_name)')
    parser.add_argument('--project', type=str, default='rl-hockey',
                        help='W&B project name (default: rl-hockey)')
    parser.add_argument('--entity', type=str, default='carlkueschalledu',
                        help='W&B entity/username (default: carlkueschalledu)')
    parser.add_argument('--max_chars', type=int, default=100000,
                        help='Maximum characters in output (default: 100000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: ./wandb_run_<name>.txt)')
    parser.add_argument('--include_metrics', type=str, nargs='+', default=None,
                        help='Specific metrics to include (default: ALL metrics)')

    return parser.parse_args()


def get_run(entity, project, run_name=None, run_id=None):
    """Get W&B run by name or ID"""
    api = wandb.Api()

    if run_id:
        run = api.run(f"{entity}/{project}/{run_id}")
    elif run_name:
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        runs_list = list(runs)

        if len(runs_list) == 0:
            raise ValueError(f"No run found with name: {run_name}")
        elif len(runs_list) > 1:
            print(f"Warning: Found {len(runs_list)} runs with name '{run_name}'")
            print("Using most recent run")
            runs_list.sort(key=lambda r: r.created_at, reverse=True)

        run = runs_list[0]
    else:
        raise ValueError("Must provide either run_name or run_id")

    return run


def discretize_data(data, max_points=200):
    """Intelligently discretize data to reduce size while preserving trends."""
    if len(data) <= max_points:
        return data

    indices = []

    # Keep first 50 points (early training)
    n_start = min(50, len(data) // 4)
    indices.extend(range(n_start))

    # Keep last 50 points (final performance)
    n_end = min(50, len(data) // 4)
    indices.extend(range(len(data) - n_end, len(data)))

    # Sample middle uniformly
    n_middle = max_points - n_start - n_end
    if n_middle > 0:
        middle_start = n_start
        middle_end = len(data) - n_end
        middle_indices = np.linspace(middle_start, middle_end - 1, n_middle, dtype=int)
        indices.extend(middle_indices)

    indices = sorted(set(indices))
    return [data[i] for i in indices]


def format_run_data(run, include_metrics=None, max_chars=100000):
    """Format run data for analysis"""

    config = run.config
    summary = run.summary

    # Fetch history
    print(f"Fetching run history for: {run.name}")
    try:
        history = run.history()
        print(f"  Fetched {len(history)} rows with {len(history.columns)} columns")
    except Exception as e:
        print(f"  Failed to fetch history: {e}")
        history = None

    # Build output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"W&B RUN DATA: {run.name}")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Metadata
    output_lines.append("## RUN METADATA")
    output_lines.append(f"Run ID: {run.id}")
    output_lines.append(f"Created: {run.created_at}")
    output_lines.append(f"State: {run.state}")
    output_lines.append(f"Duration: {summary.get('_runtime', 'N/A')} seconds")
    output_lines.append("")

    # Configuration - show key DreamerV3 settings
    output_lines.append("## CONFIGURATION")
    config_groups = {
        'Environment': ['opponent', 'mode'],
        'Training': ['batch_size', 'batch_length', 'replay_ratio', 'gradient_steps',
                     'imagination_horizon', 'discount', 'lambda_'],
        'Learning Rates': ['lr_world', 'lr_actor', 'lr_critic'],
        'Entropy': ['entropy_scale'],
        'PBRS': ['use_pbrs', 'pbrs_scale', 'pbrs_w_chase', 'pbrs_w_attack'],
        'Architecture': ['recurrent_size', 'latent_length', 'latent_classes'],
        'Meta': ['seed', 'algorithm'],
    }

    for section, keys in config_groups.items():
        section_values = [(k, config.get(k)) for k in keys if config.get(k) is not None]
        if section_values:
            output_lines.append(f"  [{section}]")
            for key, value in section_values:
                output_lines.append(f"    {key}: {value}")
    output_lines.append("")

    # Summary
    output_lines.append("## FINAL SUMMARY")
    runtime = summary.get('_runtime', 0)
    if runtime:
        output_lines.append(f"  runtime: {runtime/3600:.2f} hours ({runtime:.0f} seconds)")
    output_lines.append("")

    # Process metrics
    if history is not None and not history.empty:
        metrics_data = defaultdict(list)

        # Get all W&B metrics (columns with "/" in name)
        all_metrics = [col for col in history.columns if '/' in col]

        if include_metrics:
            metrics_to_process = [m for m in include_metrics if m in history.columns]
        else:
            metrics_to_process = all_metrics

        print(f"  Processing {len(metrics_to_process)} metrics...")

        for metric in metrics_to_process:
            values = history[metric].dropna()
            if len(values) > 0:
                metrics_data[metric] = values.tolist()

        print(f"  Extracted data for {len(metrics_data)} metrics")
    else:
        metrics_data = {}

    # Estimate size and discretize
    current_size = sum(len(line) for line in output_lines)
    estimated_size = current_size + sum(len(d) * 20 for d in metrics_data.values())

    if estimated_size > max_chars:
        factor = int(np.ceil(estimated_size / max_chars))
        max_points = max(50, 200 // factor)
        output_lines.append(f"## NOTE: Data discretized to ~{max_points} points per metric")
        output_lines.append("")
    else:
        max_points = 10000

    # Format metrics by category
    output_lines.append("## METRICS DATA")
    output_lines.append("")

    if not metrics_data:
        output_lines.append("No metrics found")
    else:
        # Group metrics by prefix
        metric_groups = defaultdict(list)
        for metric in sorted(metrics_data.keys()):
            prefix = metric.split('/')[0]
            metric_groups[prefix].append(metric)

        # Define display order
        group_order = [
            'stats', 'episode', 'time', 'eval',  # Progress
            'world',  # World model
            'behavior', 'values',  # Actor-critic
            'imagination',  # Imagination
            'actions',  # Actions
            'gradients',  # Gradients
            'sparse_signal',  # Sparse rewards
            'pbrs', 'reward_composition', 'reward_hacking',  # Reward analysis
        ]

        # Add any groups not in order
        for group in metric_groups:
            if group not in group_order:
                group_order.append(group)

        for group in group_order:
            if group not in metric_groups:
                continue

            output_lines.append(f"### {group.upper()}")

            for metric in sorted(metric_groups[group]):
                data = metrics_data[metric]
                if not data:
                    continue

                # Discretize
                if len(data) > max_points:
                    data = discretize_data(data, max_points)

                metric_name = metric.split('/')[-1]
                output_lines.append(f"#### {metric_name}")
                output_lines.append(f"  Points: {len(data)}")

                try:
                    numeric = [float(v) for v in data]
                    output_lines.append(f"  Min: {min(numeric):.4f}, Max: {max(numeric):.4f}, Mean: {np.mean(numeric):.4f}")
                    output_lines.append("  Data: [")
                    for i in range(0, len(numeric), 10):
                        chunk = numeric[i:i+10]
                        output_lines.append(f"    {', '.join(f'{v:.4f}' for v in chunk)},")
                    output_lines.append("  ]")
                except (ValueError, TypeError) as e:
                    output_lines.append(f"  Error: {e}")

                output_lines.append("")

            output_lines.append("")

    output_text = "\n".join(output_lines)

    if len(output_text) > max_chars:
        print(f"Warning: Output ({len(output_text)} chars) exceeds max ({max_chars})")

    return output_text


def main():
    args = parse_args()

    print(f"Connecting to W&B...")
    run = get_run(args.entity, args.project, args.run_name, args.run_id)
    print(f"Found run: {run.name} (ID: {run.id})")

    output = format_run_data(run, args.include_metrics, args.max_chars)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"./wandb_run_{run.name.replace('/', '_')}.txt")

    output_path.write_text(output)
    print(f"\nSaved to: {output_path}")
    print(f"Size: {len(output):,} characters, {len(output.splitlines()):,} lines")

    # Preview
    print("\n" + "=" * 80)
    print("PREVIEW (first 40 lines):")
    print("=" * 80)
    for line in output.splitlines()[:40]:
        print(line)
    print("...")


if __name__ == "__main__":
    main()
