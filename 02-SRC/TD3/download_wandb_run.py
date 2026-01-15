"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.

W&B Run Data Downloader and Discretizer

Downloads complete run data from W&B and formats it for analysis.
Automatically discretizes data to fit within character limits for chat analysis.

Usage:
    python download_wandb_run.py --run_name "TD3-Hockey-NORMAL-weak-lr0.0003-seed42"
    python download_wandb_run.py --run_id "abc123xyz"
    python download_wandb_run.py --run_name "..." --max_chars 50000
"""

import argparse
import json
import wandb
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Download W&B run data for analysis')

    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name (e.g., "TD3-Hockey-NORMAL-weak-lr0.0003-seed42")')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID (alternative to run_name)')
    parser.add_argument('--project', type=str, default='rl-hockey',
                        help='W&B project name (default: rl-hockey)')
    parser.add_argument('--entity', type=str, default='carlkueschalledu',
                        help='W&B entity/username (default: carlkueschalledu)')
    parser.add_argument('--max_chars', type=int, default=100000,
                        help='Maximum characters in output (default: 100000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: ./wandb_run_data.txt)')
    parser.add_argument('--include_metrics', type=str, nargs='+', default=None,
                        help='Specific metrics to include (default: all critical metrics)')

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


def get_critical_metrics():
    """Define critical metrics to always include (matches train_hockey.py logging)"""
    return [
        # Performance metrics
        'performance/cumulative_win_rate',
        'performance/wins',
        'performance/losses',
        'performance/ties',

        # Reward metrics (shaped vs sparse)
        'rewards/p1',
        'rewards/p2',
        'rewards/sparse_only',
        'rewards/sparse_ratio',

        # Scoring metrics
        'scoring/goals_scored',
        'scoring/goals_conceded',

        # Training dynamics
        'training/epsilon',
        'training/eps_per_sec',
        'training/episode',
        'training/pbrs_enabled',
        'training/calls_per_episode',
        'training/avg_episode_length',
        'training/tie_penalty',
        'training/lr_actor',
        'training/lr_critic',
        'training/episode_block_size',
        'training/episodes_in_current_block',

        # Loss metrics (CRITICAL for diagnosing training stability)
        'losses/critic_loss',
        'losses/actor_loss',

        # Gradient norm metrics (CRITICAL for monitoring learning health)
        # NEW FIX: Added gradient norm logging to detect vanishing/exploding gradients
        'gradients/critic_grad_norm',
        'gradients/actor_grad_norm',

        # PBRS (Potential-Based Reward Shaping)
        'pbrs/avg_per_episode',
        'pbrs/annealing_weight',

        # Self-play configuration and status
        'selfplay/active',
        'selfplay/pool_size',
        'selfplay/weak_ratio_target',
        'selfplay/anchor_ratio_current',
        'selfplay/episode_opponent_type_weak',
        'selfplay/episode_opponent_type_strong',
        'selfplay/episode_opponent_type_selfplay',

        # Self-play anchor buffer balance (weak vs strong episodes)
        'selfplay/anchor_weak_episodes',
        'selfplay/anchor_strong_episodes',
        'selfplay/anchor_weak_ratio',
        'selfplay/anchor_strong_ratio',
        'selfplay/anchor_balance_score',

        # Self-play opponent selection and PFSP metrics
        'selfplay/opponent_pool_index',
        'selfplay/opponent_checkpoint_episode',
        'selfplay/opponent_age_episodes',
        'selfplay/pfsp_num_opponents_tracked',
        'selfplay/pfsp_avg_winrate',
        'selfplay/pfsp_std_winrate',
        'selfplay/pfsp_min_winrate',
        'selfplay/pfsp_max_winrate',
        'selfplay/pfsp_median_winrate',
        'selfplay/pfsp_diversity_metric',

        # Self-play regression tracking and rollback
        'selfplay/best_eval_vs_weak',
        'selfplay/consecutive_eval_drops',
        'selfplay/rollback_enabled',

        # Evaluation metrics (comprehensive three-way evaluation)
        # Evaluation vs WEAK opponent (baseline)
        'eval/weak/win_rate',
        'eval/weak/win_rate_decisive',
        'eval/weak/tie_rate',
        'eval/weak/loss_rate',
        'eval/weak/avg_reward',
        'eval/weak/wins',
        'eval/weak/losses',
        'eval/weak/ties',

        # Evaluation vs STRONG opponent (training opponent)
        'eval/strong/win_rate',
        'eval/strong/win_rate_decisive',
        'eval/strong/tie_rate',
        'eval/strong/loss_rate',
        'eval/strong/avg_reward',
        'eval/strong/wins',
        'eval/strong/losses',
        'eval/strong/ties',

        # Evaluation vs SELF-PLAY opponent (if active)
        'eval/selfplay/win_rate',
        'eval/selfplay/win_rate_decisive',
        'eval/selfplay/tie_rate',
        'eval/selfplay/loss_rate',
        'eval/selfplay/avg_reward',
        'eval/selfplay/wins',
        'eval/selfplay/losses',
        'eval/selfplay/ties',
        'eval/selfplay/opponent_age',

        # Behavioral metrics (CRITICAL for lazy learning detection)
        'behavior/action_magnitude_avg',
        'behavior/action_magnitude_max',
        'behavior/lazy_action_ratio',
        'behavior/dist_to_puck_avg',
        'behavior/dist_to_puck_min',
        'behavior/puck_touches',
        'behavior/time_near_puck',
        'behavior/distance_traveled',
        'behavior/velocity_avg',
        'behavior/velocity_max',

        # Value function health
        'values/Q_avg',
        'values/Q_std',
        'values/Q_min',
        'values/Q_max',

        # Prioritized Experience Replay (PER) metrics
        # Buffer state
        'per/beta',
        'per/max_priority',
        'per/total_priority',
        'per/n_entries',
        # IS weight distribution (shows correction strength)
        'per/is_weight_mean',
        'per/is_weight_std',
        'per/is_weight_min',
        'per/is_weight_max',
        # TD error distribution (shows priority spread)
        'per/td_error_mean',
        'per/td_error_std',
        'per/td_error_min',
        'per/td_error_max',

        # Shoot/Keep behavior metrics (possession strategy)
        'behavior/shoot_action_avg',
        'behavior/shoot_action_when_possess',
        'behavior/possession_ratio',
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


def format_run_data(run, include_metrics=None, max_chars=100000):
    """Format run data for chat analysis"""

    # Get run metadata
    config = run.config
    summary = run.summary

    # Determine which metrics to include
    if include_metrics:
        metrics_to_fetch = include_metrics
    else:
        metrics_to_fetch = get_critical_metrics()

    # Fetch history data
    print(f"Fetching run history for: {run.name}")

    # Fetch all available data (don't filter by keys to avoid losing data)
    try:
        print(f"  Attempting to fetch full history (no key filtering)...")
        history = run.history()  # Fetch everything, no filtering
        print(f"  ✓ Fetched {len(history)} rows with {len(history.columns)} columns")
    except Exception as e:
        print(f"  ✗ Failed with default fetch: {e}")
        print(f"  Trying alternative fetch method...")
        try:
            # Fallback: try with explicit samples parameter
            history = run.history(samples=None)  # Fetch all samples
            print(f"  ✓ Fetched {len(history)} rows with {len(history.columns)} columns")
        except Exception as e2:
            print(f"  ✗ Also failed with samples=None: {e2}")
            print(f"  This run may have no logged metrics or API access issues")
            history = None

    if history is None or history.empty:
        print(f"  ⚠ WARNING: No history data fetched! The run may not have metrics logged.")
        print(f"  Returning config/summary only...")
        metrics_data = {}
    else:
        # Debug: Print available columns
        all_columns = list(history.columns)
        available_metrics = [col for col in all_columns if '/' in col]  # Only W&B metrics
        non_metric_cols = [col for col in all_columns if '/' not in col]

        print(f"  ✓ Available W&B metrics: {len(available_metrics)}")
        print(f"    Non-metric columns: {non_metric_cols}")
        if available_metrics:
            print(f"    Sample W&B metrics: {available_metrics[:15]}")

        # Show which requested metrics are actually available
        available_requested = [m for m in metrics_to_fetch if m in history.columns]
        missing_requested = [m for m in metrics_to_fetch if m not in history.columns]
        if available_requested:
            print(f"  ✓ Found {len(available_requested)} requested metrics")
        if missing_requested:
            print(f"    Not found (not logged): {missing_requested[:10]}")

    # Organize data by metric (only if history was successfully fetched)
    if history is not None and not history.empty:
        metrics_data = defaultdict(list)

        # Use available metrics if they exist, otherwise use all
        metrics_to_process = [m for m in metrics_to_fetch if m in history.columns]
        if not metrics_to_process:
            print(f"  No requested metrics found. Using all available W&B metrics...")
            metrics_to_process = [col for col in history.columns if '/' in col]

        print(f"  Processing {len(metrics_to_process)} metrics from {len(history)} history rows...")

        # Extract metric values from history
        collected_count = 0
        for metric in metrics_to_process:
            if metric in history.columns:
                values = history[metric].dropna()  # Remove NaN values
                if len(values) > 0:
                    metrics_data[metric] = values.tolist()
                    collected_count += 1

        print(f"  ✓ Successfully extracted data for {collected_count} metrics")
        if collected_count == 0:
            print(f"    ⚠ WARNING: No metrics with data found!")
    else:
        metrics_data = {}

    # Build output text
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"W&B RUN DATA: {run.name}")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Run metadata
    output_lines.append("## RUN METADATA")
    output_lines.append(f"Run ID: {run.id}")
    output_lines.append(f"Created: {run.created_at}")
    output_lines.append(f"State: {run.state}")
    output_lines.append(f"Duration: {run.summary.get('_runtime', 'N/A')} seconds")
    output_lines.append("")

    # Config
    output_lines.append("## CONFIGURATION")
    key_config = ['algorithm', 'mode', 'opponent', 'learning_rate_actor', 'learning_rate_critic',
                  'batch_size', 'buffer_size', 'max_episodes', 'random_seed',
                  'reward_shaping', 'self_play_start', 'self_play_pool_size',
                  'eps', 'eps_min', 'eps_decay', 'gamma', 'tau']
    for key in key_config:
        if key in config:
            output_lines.append(f"  {key}: {config[key]}")
    output_lines.append("")

    # Summary metrics
    output_lines.append("## FINAL SUMMARY")
    key_summary = ['final_win_rate', 'wins', 'losses', 'ties', 'total_episodes']
    for key in key_summary:
        if key in summary:
            output_lines.append(f"  {key}: {summary[key]}")
    output_lines.append("")

    # Estimate size and discretize if needed
    current_size = sum(len(line) for line in output_lines)
    estimated_metrics_size = sum(len(data) * 20 for data in metrics_data.values())  # ~20 chars per datapoint
    total_estimated = current_size + estimated_metrics_size

    # Calculate discretization factor
    if total_estimated > max_chars:
        discretization_factor = int(np.ceil(total_estimated / max_chars))
        max_points_per_metric = max(50, 200 // discretization_factor)
        output_lines.append(f"## NOTE: Data discretized to fit {max_chars} char limit")
        output_lines.append(f"  Showing ~{max_points_per_metric} points per metric (of {len(next(iter(metrics_data.values())) if metrics_data else [])} total)")
        output_lines.append("")
    else:
        max_points_per_metric = 10000  # No discretization needed

    # Format metrics data
    output_lines.append("## METRICS DATA")
    output_lines.append("")

    if not metrics_data:
        output_lines.append("⚠ NO METRICS FOUND")
        output_lines.append("")
        output_lines.append("This run has no logged metric history. Possible causes:")
        output_lines.append("  1. Run crashed before first metric was logged")
        output_lines.append("  2. Metrics not being logged in training script (check wandb.log calls)")
        output_lines.append("  3. W&B API access issue")
        output_lines.append("  4. Run is still in progress (wait for it to complete)")
        output_lines.append("")
    else:
        for metric in sorted(metrics_data.keys()):
            data = metrics_data[metric]

            if not data:
                continue

            # Discretize if needed
            if len(data) > max_points_per_metric:
                data = discretize_data(data, max_points_per_metric)

            output_lines.append(f"### {metric}")
            output_lines.append(f"  Total points: {len(data)}")

            # Handle cases where data might not be numeric
            try:
                numeric_data = [float(v) for v in data]
                output_lines.append(f"  Min: {min(numeric_data):.4f}, Max: {max(numeric_data):.4f}, Mean: {np.mean(numeric_data):.4f}")

                # Format data compactly
                output_lines.append("  Data: [")

                # Group into lines of 10 values for readability
                for i in range(0, len(numeric_data), 10):
                    chunk = numeric_data[i:i+10]
                    formatted_chunk = ", ".join(f"{v:.4f}" for v in chunk)
                    output_lines.append(f"    {formatted_chunk},")

                output_lines.append("  ]")
            except (ValueError, TypeError) as e:
                output_lines.append(f"  ✗ Could not format data: {e}")

            output_lines.append("")

    # Join and check size
    output_text = "\n".join(output_lines)

    if len(output_text) > max_chars:
        print(f"Warning: Output size ({len(output_text)} chars) exceeds max_chars ({max_chars})")
        print("Consider reducing max_chars or selecting fewer metrics")

    return output_text


def main():
    args = parse_args()

    # Get run
    print(f"Connecting to W&B...")
    run = get_run(args.entity, args.project, args.run_name, args.run_id)
    print(f"Found run: {run.name} (ID: {run.id})")

    # Format data
    output = format_run_data(run, args.include_metrics, args.max_chars)

    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"./wandb_run_{run.name.replace('/', '_')}.txt")

    output_path.write_text(output)
    print(f"\n✅ Data saved to: {output_path}")
    print(f"   Size: {len(output):,} characters")
    print(f"   Lines: {len(output.splitlines()):,}")
    print("\nYou can now copy this file and paste it in chat for analysis!")

    # Also print first few lines as preview
    print("\n" + "=" * 80)
    print("PREVIEW (first 50 lines):")
    print("=" * 80)
    for i, line in enumerate(output.splitlines()[:50]):
        print(line)
    print("...")
    print("=" * 80)


if __name__ == "__main__":
    main()
