"""
W&B Async Training Run Data Downloader

Downloads complete run data from W&B for async training runs and formats it for analysis.
Automatically discretizes data to fit within character limits for chat analysis.

Usage:
    python download_wandb_async.py --run_name "TD3-ASYNC-NORMAL-weak-seed42"
    python download_wandb_async.py --run_id "abc123xyz"
    python download_wandb_async.py --run_name "..." --max_chars 50000
"""

import argparse
import json
import wandb
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Download W&B async run data for analysis')

    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name (e.g., "TD3-ASYNC-NORMAL-weak-seed42")')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID (alternative to run_name)')
    parser.add_argument('--project', type=str, default='rl-hockey',
                        help='W&B project name (default: rl-hockey)')
    parser.add_argument('--entity', type=str, default='carlkueschalledu',
                        help='W&B entity/username (default: carlkueschalledu)')
    parser.add_argument('--max_chars', type=int, default=100000,
                        help='Maximum characters in output (default: 100000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: ./wandb_async_run_data.txt)')

    return parser.parse_args()


def get_run(entity, project, run_name=None, run_id=None):
    """Get W&B run by name or ID"""
    api = wandb.Api()

    if run_id:
        run = api.run(f"{entity}/{project}/{run_id}")
    elif run_name:
        # Search for run by name
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


def get_async_metrics():
    """Define metrics logged by async training (from async_orchestrator.py)"""
    return [
        # Performance metrics (from AsyncMetricsTracker.get_metrics_dict)
        'performance/win_rate',
        'performance/rolling_win_rate',
        'performance/wins',
        'performance/losses',
        'performance/ties',
        'performance/total_episodes',

        # Reward metrics
        'rewards/episode_reward',
        'rewards/sparse_reward',

        # Training dynamics
        'training/episode_length',
        'training/eps_per_sec',
        'training/elapsed_time',
        'training/epsilon',
        'training/buffer_size',
        'training/train_step',

        # Loss metrics
        'losses/critic_loss',
        'losses/actor_loss',

        # Q-value metrics
        'values/Q_mean',
        'values/Q_std',

        # Gradient metrics
        'gradients/critic_grad_norm',
        'gradients/actor_grad_norm',

        # Behavior metrics
        'behavior/action_magnitude',
        'behavior/puck_distance',
        'behavior/puck_touches',

        # Evaluation metrics
        'eval/win_rate_vs_weak',
        'eval/win_rate_vs_strong',
        'eval/avg_reward_weak',
        'eval/avg_reward_strong',

        # Scoring metrics
        'scoring/goals_scored',
        'scoring/goals_conceded',

        # PER metrics
        'per/beta',
        'per/max_priority',

        # Async-specific metrics
        'async/num_workers',
        'async/collector_episodes',
        'async/train_steps',
        'async/avg_train_time',
        'async/buffer_queue_size',
    ]


def discretize_data(data, max_points=200):
    """
    Intelligently discretize data to reduce size while preserving trends.

    Strategy:
    - Keep first 50 points (early training critical)
    - Keep last 50 points (final performance)
    - Adaptively sample middle based on variance
    """
    if len(data) <= max_points:
        return data

    indices = []

    # Always keep first 50 points (early training)
    n_start = min(50, len(data) // 4)
    indices.extend(range(n_start))

    # Always keep last 50 points (final performance)
    n_end = min(50, len(data) // 4)
    indices.extend(range(len(data) - n_end, len(data)))

    # Sample middle based on available budget
    n_middle = max_points - n_start - n_end
    if n_middle > 0:
        middle_start = n_start
        middle_end = len(data) - n_end

        # Uniform sampling of middle
        middle_indices = np.linspace(middle_start, middle_end - 1, n_middle, dtype=int)
        indices.extend(middle_indices)

    # Sort and remove duplicates
    indices = sorted(set(indices))

    return [data[i] for i in indices]


def format_run_data(run, max_chars=100000):
    """Format run data for chat analysis"""

    # Get run metadata
    config = run.config
    summary = run.summary

    # Metrics to fetch
    metrics_to_fetch = get_async_metrics()

    # Fetch history data
    print(f"Fetching run history for: {run.name}")

    try:
        print(f"  Attempting to fetch full history...")
        history = run.history()
        print(f"  Fetched {len(history)} rows with {len(history.columns)} columns")
    except Exception as e:
        print(f"  Failed: {e}")
        history = None

    if history is None or history.empty:
        print(f"  WARNING: No history data fetched!")
        metrics_data = {}
    else:
        # Debug: Print available columns
        all_columns = list(history.columns)
        available_metrics = [col for col in all_columns if '/' in col]

        print(f"  Available W&B metrics: {len(available_metrics)}")

        # Show which requested metrics are actually available
        available_requested = [m for m in metrics_to_fetch if m in history.columns]
        missing_requested = [m for m in metrics_to_fetch if m not in history.columns]
        if available_requested:
            print(f"  Found {len(available_requested)} requested metrics")
        if missing_requested:
            print(f"  Not found: {missing_requested[:10]}")

    # Organize data by metric
    if history is not None and not history.empty:
        metrics_data = defaultdict(list)

        metrics_to_process = [m for m in metrics_to_fetch if m in history.columns]
        if not metrics_to_process:
            print(f"  No requested metrics found. Using all available...")
            metrics_to_process = [col for col in history.columns if '/' in col]

        print(f"  Processing {len(metrics_to_process)} metrics...")

        for metric in metrics_to_process:
            if metric in history.columns:
                values = history[metric].dropna()
                if len(values) > 0:
                    metrics_data[metric] = values.tolist()

        print(f"  Extracted data for {len(metrics_data)} metrics")
    else:
        metrics_data = {}

    # Build output text
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"W&B ASYNC RUN DATA: {run.name}")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Run metadata
    output_lines.append("## RUN METADATA")
    output_lines.append(f"Run ID: {run.id}")
    output_lines.append(f"Created: {run.created_at}")
    output_lines.append(f"State: {run.state}")
    output_lines.append(f"Duration: {summary.get('_runtime', 'N/A')} seconds")
    output_lines.append("")

    # Config (async-specific)
    output_lines.append("## CONFIGURATION")
    key_config = [
        'mode', 'opponent', 'num_workers', 'max_episodes',
        'lr_actor', 'lr_critic', 'batch_size', 'buffer_size',
        'eps', 'eps_min', 'eps_decay', 'gamma', 'tau',
        'policy_freq', 'target_update_freq', 'weight_sync_interval',
        'use_per', 'per_alpha', 'per_beta_start',
        'reward_shaping', 'pbrs_scale',
        'q_clip', 'q_clip_mode', 'grad_clip',
        'hidden_actor', 'hidden_critic',
        'eval_interval', 'eval_episodes', 'seed'
    ]
    for key in key_config:
        if key in config:
            output_lines.append(f"  {key}: {config[key]}")
    output_lines.append("")

    # Summary metrics
    output_lines.append("## FINAL SUMMARY")
    key_summary = [
        'final_win_rate', 'best_win_rate', 'total_episodes',
        'total_train_steps', 'training_time_seconds', 'eps_per_sec'
    ]
    for key in key_summary:
        if key in summary:
            val = summary[key]
            if isinstance(val, float):
                output_lines.append(f"  {key}: {val:.4f}")
            else:
                output_lines.append(f"  {key}: {val}")
    output_lines.append("")

    # Estimate size and discretize if needed
    current_size = sum(len(line) for line in output_lines)
    estimated_metrics_size = sum(len(data) * 20 for data in metrics_data.values())
    total_estimated = current_size + estimated_metrics_size

    if total_estimated > max_chars:
        discretization_factor = int(np.ceil(total_estimated / max_chars))
        max_points_per_metric = max(50, 200 // discretization_factor)
        output_lines.append(f"## NOTE: Data discretized to fit {max_chars} char limit")
        output_lines.append(f"  Showing ~{max_points_per_metric} points per metric")
        output_lines.append("")
    else:
        max_points_per_metric = 10000

    # Format metrics data
    output_lines.append("## METRICS DATA")
    output_lines.append("")

    if not metrics_data:
        output_lines.append("NO METRICS FOUND")
        output_lines.append("")
    else:
        # Group metrics by category for readability
        categories = {
            'Performance': ['performance/'],
            'Rewards': ['rewards/'],
            'Training': ['training/'],
            'Losses': ['losses/'],
            'Values': ['values/'],
            'Gradients': ['gradients/'],
            'Behavior': ['behavior/'],
            'Evaluation': ['eval/'],
            'Scoring': ['scoring/'],
            'PER': ['per/'],
            'Async': ['async/'],
        }

        for category, prefixes in categories.items():
            category_metrics = [m for m in sorted(metrics_data.keys())
                              if any(m.startswith(p) for p in prefixes)]
            if not category_metrics:
                continue

            output_lines.append(f"### {category}")
            output_lines.append("")

            for metric in category_metrics:
                data = metrics_data[metric]
                if not data:
                    continue

                # Discretize if needed
                if len(data) > max_points_per_metric:
                    data = discretize_data(data, max_points_per_metric)

                # Short metric name
                short_name = metric.split('/')[-1]

                try:
                    numeric_data = [float(v) for v in data]
                    output_lines.append(f"**{short_name}** (n={len(data)})")
                    output_lines.append(f"  min={min(numeric_data):.4f}, max={max(numeric_data):.4f}, mean={np.mean(numeric_data):.4f}")

                    # Compact data format
                    formatted = ", ".join(f"{v:.4f}" for v in numeric_data)
                    output_lines.append(f"  [{formatted}]")
                    output_lines.append("")
                except (ValueError, TypeError):
                    output_lines.append(f"**{short_name}**: [non-numeric data]")
                    output_lines.append("")

    # Join and check size
    output_text = "\n".join(output_lines)

    if len(output_text) > max_chars:
        print(f"Warning: Output ({len(output_text)} chars) exceeds max ({max_chars})")

    return output_text


def main():
    args = parse_args()

    print(f"Connecting to W&B...")
    run = get_run(args.entity, args.project, args.run_name, args.run_id)
    print(f"Found run: {run.name} (ID: {run.id})")

    output = format_run_data(run, args.max_chars)

    if args.output:
        output_path = Path(args.output)
    else:
        safe_name = run.name.replace('/', '_').replace(' ', '_')
        output_path = Path(f"./wandb_async_{safe_name}.txt")

    output_path.write_text(output)
    print(f"\nData saved to: {output_path}")
    print(f"Size: {len(output):,} characters")
    print(f"Lines: {len(output.splitlines()):,}")
    print("\nYou can now copy this file and paste it in chat for analysis!")

    # Preview
    print("\n" + "=" * 80)
    print("PREVIEW (first 60 lines):")
    print("=" * 80)
    for line in output.splitlines()[:60]:
        print(line)
    print("...")
    print("=" * 80)


if __name__ == "__main__":
    main()
