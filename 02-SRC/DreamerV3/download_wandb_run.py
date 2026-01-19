"""
AI Usage Declaration:
This file was developed with assistance from Claude Code.

W&B Run Data Downloader and Visualizer

Downloads complete run data from W&B and creates visualization plots.
Supports both TD3 and DreamerV3 runs with automatic metric detection.

Usage:
    python download_wandb_run.py --run_name "DreamerV3-NORMAL-weak-seed42"
    python download_wandb_run.py --run_id "abc123xyz"
    python download_wandb_run.py --run_name "..." --max_chars 50000
"""

import argparse
import json
import wandb
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
                        help='Specific metrics to include (default: ALL metrics - no filtering)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--plot_output', type=str, default=None,
                        help='Output path for plot (default: ./wandb_run_<name>.png)')

    return parser.parse_args()


def plot_dreamer_metrics(history, run_name, output_path):
    """Generate comprehensive DreamerV3 training visualization."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
        return

    # Set up the figure with subplots (5 rows for more metrics)
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'DreamerV3 Training: {run_name}', fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.25)

    # Helper to plot with smoothing
    def plot_metric(ax, data, label, color, smooth_window=10):
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        x = np.arange(len(data))
        ax.plot(x, data, alpha=0.3, color=color)
        if len(data) > smooth_window:
            smoothed = np.convolve(data, np.ones(smooth_window)/smooth_window, mode='valid')
            ax.plot(np.arange(len(smoothed)) + smooth_window//2, smoothed, color=color, label=label)
        else:
            ax.plot(x, data, color=color, label=label)

    def get_metric(name):
        if name in history.columns:
            return history[name].dropna().values
        return None

    # 1. Episode Reward
    ax1 = fig.add_subplot(gs[0, 0])
    reward_data = get_metric('episode/reward')
    plot_metric(ax1, reward_data, 'Reward', 'blue')
    ax1.set_title('Episode Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.legend()

    # 2. Win Rate
    ax2 = fig.add_subplot(gs[0, 1])
    win_rate = get_metric('stats/win_rate')
    if win_rate is not None:
        ax2.plot(win_rate, color='green', label='Win Rate')
        ax2.fill_between(range(len(win_rate)), win_rate, alpha=0.3, color='green')
    ax2.set_title('Win Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.legend()

    # 3. Episode Length
    ax3 = fig.add_subplot(gs[0, 2])
    length_data = get_metric('episode/length')
    plot_metric(ax3, length_data, 'Length', 'purple')
    ax3.set_title('Episode Length')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.legend()

    # 4. World Model Loss
    ax4 = fig.add_subplot(gs[1, 0])
    world_loss = get_metric('world/loss')
    plot_metric(ax4, world_loss, 'Total', 'red')
    ax4.set_title('World Model Loss')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.set_yscale('log')
    ax4.legend()

    # 5. World Model Loss Components
    ax5 = fig.add_subplot(gs[1, 1])
    recon_loss = get_metric('world/recon_loss')
    reward_loss = get_metric('world/reward_loss')
    kl_loss = get_metric('world/kl_loss')
    continue_loss = get_metric('world/continue_loss')
    if recon_loss is not None:
        plot_metric(ax5, recon_loss, 'Recon', 'blue')
    if reward_loss is not None:
        plot_metric(ax5, reward_loss, 'Reward', 'green')
    if kl_loss is not None:
        plot_metric(ax5, kl_loss, 'KL', 'orange')
    if continue_loss is not None:
        plot_metric(ax5, continue_loss, 'Continue', 'purple')
    ax5.set_title('World Model Components')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Loss')
    ax5.legend()

    # 6. KL Divergence
    ax6 = fig.add_subplot(gs[1, 2])
    kl_value = get_metric('world/kl_value')
    plot_metric(ax6, kl_value, 'KL Value', 'orange')
    ax6.set_title('KL Divergence (Prior vs Posterior)')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('KL')
    ax6.legend()

    # 7. Actor Loss
    ax7 = fig.add_subplot(gs[2, 0])
    actor_loss = get_metric('behavior/actor_loss')
    plot_metric(ax7, actor_loss, 'Actor Loss', 'blue')
    ax7.set_title('Actor Loss')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Loss')
    ax7.legend()

    # 8. Critic Loss
    ax8 = fig.add_subplot(gs[2, 1])
    critic_loss = get_metric('behavior/critic_loss')
    plot_metric(ax8, critic_loss, 'Critic Loss', 'red')
    ax8.set_title('Critic Loss')
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Loss')
    ax8.legend()

    # 9. Policy Entropy
    ax9 = fig.add_subplot(gs[2, 2])
    entropy = get_metric('behavior/entropy')
    plot_metric(ax9, entropy, 'Entropy', 'purple')
    ax9.set_title('Policy Entropy')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Entropy')
    ax9.legend()

    # 10. Gradient Norms
    ax10 = fig.add_subplot(gs[3, 0])
    grad_world = get_metric('grad/world')
    grad_actor = get_metric('grad/actor')
    grad_critic = get_metric('grad/critic')
    if grad_world is not None:
        plot_metric(ax10, grad_world, 'World', 'red')
    if grad_actor is not None:
        plot_metric(ax10, grad_actor, 'Actor', 'blue')
    if grad_critic is not None:
        plot_metric(ax10, grad_critic, 'Critic', 'green')
    ax10.set_title('Gradient Norms')
    ax10.set_xlabel('Episode')
    ax10.set_ylabel('Norm')
    ax10.legend()

    # 11. Value Estimates
    ax11 = fig.add_subplot(gs[3, 1])
    value_mean = get_metric('behavior/value_mean')
    target_mean = get_metric('behavior/target_mean')
    if value_mean is not None:
        plot_metric(ax11, value_mean, 'Value', 'blue')
    if target_mean is not None:
        plot_metric(ax11, target_mean, 'Target', 'green')
    ax11.set_title('Value Estimates')
    ax11.set_xlabel('Episode')
    ax11.set_ylabel('Value')
    ax11.legend()

    # 12. Imagined Rewards
    ax12 = fig.add_subplot(gs[3, 2])
    imagine_reward = get_metric('imagine/reward_mean')
    imagine_continue = get_metric('imagine/continue_mean')
    if imagine_reward is not None:
        ax12_twin = ax12.twinx()
        plot_metric(ax12, imagine_reward, 'Reward Mean', 'blue')
        if imagine_continue is not None:
            plot_metric(ax12_twin, imagine_continue, 'Continue Prob', 'orange')
            ax12_twin.set_ylabel('Continue Prob', color='orange')
            ax12_twin.tick_params(axis='y', labelcolor='orange')
    ax12.set_title('Imagination Stats')
    ax12.set_xlabel('Episode')
    ax12.set_ylabel('Reward', color='blue')
    ax12.tick_params(axis='y', labelcolor='blue')

    # 13. Terminal Reward Prediction (CRITICAL for sparse reward learning)
    ax13 = fig.add_subplot(gs[4, 0])
    terminal_pred = get_metric('world/terminal_pred_mean')
    terminal_target = get_metric('world/terminal_target_mean')
    if terminal_pred is not None:
        plot_metric(ax13, terminal_pred, 'Predicted', 'blue')
    if terminal_target is not None:
        plot_metric(ax13, terminal_target, 'Target', 'green')
    ax13.set_title('Terminal Reward Prediction')
    ax13.set_xlabel('Episode')
    ax13.set_ylabel('Reward')
    ax13.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax13.legend()

    # 14. Terminal Count per Batch
    ax14 = fig.add_subplot(gs[4, 1])
    terminal_count = get_metric('world/terminal_count')
    if terminal_count is not None:
        ax14.bar(range(len(terminal_count)), terminal_count, alpha=0.7, color='purple')
        ax14.axhline(y=np.mean(terminal_count) if terminal_count is not None and len(terminal_count) > 0 else 0,
                     color='red', linestyle='--', label=f'Mean: {np.mean(terminal_count):.1f}')
    ax14.set_title('Terminal Samples per Batch')
    ax14.set_xlabel('Episode')
    ax14.set_ylabel('Count')
    ax14.legend()

    # 15. Advantage Distribution
    ax15 = fig.add_subplot(gs[4, 2])
    advantage = get_metric('behavior/advantage_mean')
    policy_loss = get_metric('behavior/policy_loss')
    if advantage is not None:
        plot_metric(ax15, advantage, 'Advantage', 'blue')
    if policy_loss is not None:
        ax15_twin = ax15.twinx()
        plot_metric(ax15_twin, policy_loss, 'Policy Loss', 'red')
        ax15_twin.set_ylabel('Policy Loss', color='red')
        ax15_twin.tick_params(axis='y', labelcolor='red')
    ax15.set_title('Advantage & Policy Loss')
    ax15.set_xlabel('Episode')
    ax15.set_ylabel('Advantage', color='blue')
    ax15.tick_params(axis='y', labelcolor='blue')
    ax15.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved to: {output_path}")


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


def get_all_metrics():
    """
    Return None to indicate we want ALL metrics from the run.

    The script will automatically fetch all W&B metrics (those with "/" in name).
    This ensures we never miss any logged metrics.

    DreamerV3 metric categories:
    - episode/: reward, length, outcome, time, train_time
    - stats/: win_rate, total_steps, buffer_size, buffer_episodes
    - world/: loss, recon_loss, reward_loss, continue_loss, kl_loss, kl_value,
              terminal_count, terminal_pred_mean, terminal_target_mean
    - behavior/: actor_loss, critic_loss, policy_loss, entropy, value_mean, target_mean, advantage_mean
    - grad/: world, actor, critic
    - imagine/: reward_mean, reward_std, continue_mean
    - eval_weak/: win_rate, loss_rate, draw_rate, mean_reward, mean_length, goals_scored, goals_conceded
    - eval_strong/: win_rate, loss_rate, draw_rate, mean_reward, mean_length, goals_scored, goals_conceded

    TD3 metric categories (for reference):
    - performance/: cumulative_win_rate, wins, losses, ties
    - rewards/: p1, p2, sparse_only, sparse_ratio
    - scoring/: goals_scored, goals_conceded
    - training/: epsilon, eps_per_sec, episode
    - losses/: critic_loss, actor_loss
    - gradients/: critic_grad_norm, actor_grad_norm
    - eval/weak/: win_rate, loss_rate, avg_reward
    - eval/strong/: win_rate, loss_rate, avg_reward
    """
    return None  # None means fetch ALL metrics


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
    # If include_metrics is None, we'll fetch ALL metrics from the run
    metrics_to_fetch = include_metrics  # None means all

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

        # Show metrics availability
        if metrics_to_fetch is not None:
            available_requested = [m for m in metrics_to_fetch if m in history.columns]
            missing_requested = [m for m in metrics_to_fetch if m not in history.columns]
            if available_requested:
                print(f"  ✓ Found {len(available_requested)} requested metrics")
            if missing_requested:
                print(f"    Not found (not logged): {missing_requested[:10]}")
        else:
            print(f"  ✓ Fetching ALL {len(available_metrics)} W&B metrics (no filtering)")

    # Organize data by metric (only if history was successfully fetched)
    if history is not None and not history.empty:
        metrics_data = defaultdict(list)

        # Determine which metrics to process
        if metrics_to_fetch is not None:
            # User specified specific metrics
            metrics_to_process = [m for m in metrics_to_fetch if m in history.columns]
            if not metrics_to_process:
                print(f"  No requested metrics found. Using all available W&B metrics...")
                metrics_to_process = [col for col in history.columns if '/' in col]
        else:
            # Fetch ALL W&B metrics (columns with "/" in name)
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

    # Config - detect algorithm type and show relevant keys
    output_lines.append("## CONFIGURATION")

    # Check if DreamerV3 or TD3 based on config keys
    is_dreamer = any(k.startswith('arch/') or k.startswith('imagination/') or k.startswith('dreamer/') for k in config.keys())

    if is_dreamer:
        # DreamerV3 config keys
        config_sections = {
            'Environment': ['env/mode', 'env/opponent', 'env/obs_dim', 'env/action_dim'],
            'Architecture': ['arch/hidden_size', 'arch/latent_size', 'arch/recurrent_size'],
            'Imagination': ['imagination/horizon', 'imagination/batch_size'],
            'Training': ['train/batch_size', 'train/batch_length', 'train/lr_world',
                        'train/lr_actor', 'train/lr_critic', 'train/gamma', 'train/lambda_gae'],
            'DreamerV3': ['dreamer/kl_free', 'dreamer/entropy_scale'],
            'Meta': ['seed', 'max_steps', 'device'],
        }
    else:
        # TD3 config keys
        config_sections = {
            'Environment': ['mode', 'opponent'],
            'Training': ['learning_rate_actor', 'learning_rate_critic', 'batch_size',
                        'buffer_size', 'max_episodes', 'gamma', 'tau'],
            'Exploration': ['eps', 'eps_min', 'eps_decay'],
            'Self-Play': ['self_play_start', 'self_play_pool_size'],
            'Meta': ['random_seed', 'algorithm'],
        }

    for section, keys in config_sections.items():
        section_values = [(k, config.get(k)) for k in keys if k in config]
        if section_values:
            output_lines.append(f"  [{section}]")
            for key, value in section_values:
                output_lines.append(f"    {key}: {value}")
    output_lines.append("")

    # Summary metrics
    output_lines.append("## FINAL SUMMARY")
    # Try common summary keys
    summary_keys = ['final_win_rate', 'wins', 'losses', 'ties', 'draws',
                    'total_episodes', 'total_steps', '_runtime']
    for key in summary_keys:
        if key in summary:
            value = summary[key]
            if key == '_runtime':
                # Format runtime nicely
                hours = value / 3600
                output_lines.append(f"  runtime: {hours:.2f} hours ({value:.0f} seconds)")
            else:
                output_lines.append(f"  {key}: {value}")
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

    # Generate plot if requested
    if args.plot:
        print("\nGenerating visualization plots...")
        try:
            history = run.history()
            if history is not None and not history.empty:
                # Determine plot output path
                if args.plot_output:
                    plot_path = Path(args.plot_output)
                else:
                    plot_path = Path(f"./wandb_plot_{run.name.replace('/', '_')}.png")

                plot_dreamer_metrics(history, run.name, plot_path)
            else:
                print("  ⚠ No history data available for plotting")
        except Exception as e:
            print(f"  ✗ Failed to generate plot: {e}")

    print("\nYou can now copy the text file and paste it in chat for analysis!")

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
