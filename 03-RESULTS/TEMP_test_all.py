#!/usr/bin/env python3
"""
Comprehensive Checkpoint Testing Script for DreamerV3 Hockey Agents.

AI Usage Declaration:
This file was developed with assistance from Claude Code.

This script:
1. Discovers all .pth checkpoints in the DreamerV3 directory
2. Tests each checkpoint against both weak and strong opponents (100 episodes each)
3. Records video samples of gameplay
4. Produces comprehensive text reports
5. Generates beautiful visualization graphs

Usage:
    conda activate py310
    python TEMP_test_all.py                    # Test all checkpoints
    python TEMP_test_all.py --episodes 50      # Fewer episodes per test
    python TEMP_test_all.py --checkpoints best_weak.pth best_strong.pth  # Specific checkpoints
    python TEMP_test_all.py --no-record        # Skip video recording
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

from dreamer import Dreamer
from utils import loadConfig, seedEverything
from opponents import FixedOpponent


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EpisodeResult:
    """Result of a single episode."""
    outcome: str  # 'win', 'loss', 'draw'
    reward: float
    steps: int


@dataclass
class TestResult:
    """Result of testing a checkpoint against one opponent."""
    checkpoint_name: str
    checkpoint_path: str
    opponent: str
    episodes: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    loss_rate: float
    draw_rate: float
    mean_reward: float
    std_reward: float
    mean_steps: float
    std_steps: float
    gradient_steps: int
    training_episodes: int
    test_duration_seconds: float


@dataclass
class CheckpointSummary:
    """Summary of a checkpoint's performance across all opponents."""
    checkpoint_name: str
    checkpoint_path: str
    gradient_steps: int
    training_episodes: int
    weak_win_rate: float
    strong_win_rate: float
    combined_win_rate: float
    weak_mean_reward: float
    strong_mean_reward: float


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get best available device."""
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_episode(env, agent, opponent, render: bool = False) -> Tuple[EpisodeResult, List]:
    """Run a single evaluation episode."""
    obs, _ = env.reset()
    h, z = None, None
    total_reward = 0.0
    steps = 0
    frames = []
    done = False

    while not done:
        if render:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)

        action, h, z = agent.act(obs, h, z)
        obs_opponent = env.obs_agent_two()
        action_opponent = opponent.act(obs_opponent)
        next_obs, reward, done, truncated, info = env.step(np.hstack([action, action_opponent]))
        done = done or truncated
        total_reward += reward
        obs = next_obs
        steps += 1

    # Final frame
    if render:
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)

    if info.get('winner', 0) == 1:
        outcome = 'win'
    elif info.get('winner', 0) == -1:
        outcome = 'loss'
    else:
        outcome = 'draw'

    return EpisodeResult(outcome=outcome, reward=total_reward, steps=steps), frames


def test_checkpoint(
    agent: Dreamer,
    checkpoint_name: str,
    checkpoint_path: str,
    opponent_type: str,
    episodes: int,
    env,
    record_episodes: int = 5,
    verbose: bool = False
) -> Tuple[TestResult, List[List]]:
    """Test a checkpoint against a specific opponent."""

    opponent = FixedOpponent(weak=(opponent_type == "weak"))

    start_time = time.time()

    results = []
    all_frames = []

    for ep in range(episodes):
        opponent.reset()
        render = ep < record_episodes
        result, frames = run_episode(env, agent, opponent, render=render)
        results.append(result)

        if frames:
            all_frames.append(frames)

        if verbose and (ep + 1) % 20 == 0:
            wins = sum(1 for r in results if r.outcome == 'win')
            print(f"    Progress: {ep+1}/{episodes} | Win rate: {wins/(ep+1):.1%}")

    duration = time.time() - start_time

    # Aggregate results
    wins = sum(1 for r in results if r.outcome == 'win')
    losses = sum(1 for r in results if r.outcome == 'loss')
    draws = sum(1 for r in results if r.outcome == 'draw')
    rewards = [r.reward for r in results]
    steps = [r.steps for r in results]

    return TestResult(
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
        opponent=opponent_type,
        episodes=episodes,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / episodes,
        loss_rate=losses / episodes,
        draw_rate=draws / episodes,
        mean_reward=np.mean(rewards),
        std_reward=np.std(rewards),
        mean_steps=np.mean(steps),
        std_steps=np.std(steps),
        gradient_steps=agent.totalGradientSteps,
        training_episodes=agent.totalEpisodes,
        test_duration_seconds=duration
    ), all_frames


def save_video(frames_list: List[List], output_path: str, fps: int = 30) -> bool:
    """Save recorded frames as MP4 video."""
    try:
        import imageio
    except ImportError:
        print("  Warning: imageio not available, skipping video save")
        return False

    if not frames_list:
        return False

    video_frames = []
    for ep_idx, frames in enumerate(frames_list):
        if not frames:
            continue
        video_frames.extend(frames)
        if ep_idx < len(frames_list) - 1:
            separator = np.zeros_like(frames[0])
            separator[:] = 30
            for _ in range(15):
                video_frames.append(separator)

    if not video_frames:
        return False

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                    macro_block_size=1, pixelformat='yuv420p')
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()
        return True
    except Exception:
        try:
            imageio.mimsave(output_path, video_frames, fps=fps)
            return True
        except Exception:
            return False


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(summaries: List[CheckpointSummary], results: List[TestResult], output_dir: str):
    """Create comprehensive visualization graphs."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Warning: matplotlib not available, skipping visualizations")
        return

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    os.makedirs(output_dir, exist_ok=True)

    # Sort summaries by gradient steps
    summaries = sorted(summaries, key=lambda x: x.gradient_steps)

    # Color palette
    colors = {
        'weak': '#3498db',      # Blue
        'strong': '#e74c3c',    # Red
        'combined': '#2ecc71',  # Green
        'draw': '#95a5a6',      # Gray
        'background': '#f8f9fa'
    }

    # =========================================================================
    # Figure 1: Main Performance Overview
    # =========================================================================
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # --- Plot 1: Win Rates Bar Chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(summaries))
    width = 0.25

    bars1 = ax1.bar(x - width, [s.weak_win_rate * 100 for s in summaries],
                    width, label='vs Weak', color=colors['weak'], alpha=0.8)
    bars2 = ax1.bar(x, [s.strong_win_rate * 100 for s in summaries],
                    width, label='vs Strong', color=colors['strong'], alpha=0.8)
    bars3 = ax1.bar(x + width, [s.combined_win_rate * 100 for s in summaries],
                    width, label='Combined', color=colors['combined'], alpha=0.8)

    ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
    ax1.set_title('Win Rates by Checkpoint', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.checkpoint_name[:20] + '...' if len(s.checkpoint_name) > 20
                         else s.checkpoint_name for s in summaries],
                        rotation=45, ha='right', fontsize=8)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                ax1.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    # --- Plot 2: Win Rate vs Gradient Steps (Line Plot) ---
    ax2 = fig.add_subplot(gs[0, 1])

    grad_steps = [s.gradient_steps / 1000 for s in summaries]  # Convert to thousands

    ax2.plot(grad_steps, [s.weak_win_rate * 100 for s in summaries],
             'o-', color=colors['weak'], linewidth=2, markersize=8, label='vs Weak')
    ax2.plot(grad_steps, [s.strong_win_rate * 100 for s in summaries],
             's-', color=colors['strong'], linewidth=2, markersize=8, label='vs Strong')
    ax2.plot(grad_steps, [s.combined_win_rate * 100 for s in summaries],
             '^-', color=colors['combined'], linewidth=2, markersize=8, label='Combined')

    ax2.fill_between(grad_steps, [s.weak_win_rate * 100 for s in summaries],
                     alpha=0.1, color=colors['weak'])
    ax2.fill_between(grad_steps, [s.strong_win_rate * 100 for s in summaries],
                     alpha=0.1, color=colors['strong'])

    ax2.set_xlabel('Gradient Steps (thousands)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance vs Training Progress', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Mean Rewards ---
    ax3 = fig.add_subplot(gs[1, 0])

    x = np.arange(len(summaries))
    width = 0.35

    bars1 = ax3.bar(x - width/2, [s.weak_mean_reward for s in summaries],
                    width, label='vs Weak', color=colors['weak'], alpha=0.8)
    bars2 = ax3.bar(x + width/2, [s.strong_mean_reward for s in summaries],
                    width, label='vs Strong', color=colors['strong'], alpha=0.8)

    ax3.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Episode Rewards', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.checkpoint_name[:20] + '...' if len(s.checkpoint_name) > 20
                         else s.checkpoint_name for s in summaries],
                        rotation=45, ha='right', fontsize=8)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Plot 4: Outcome Distribution (Stacked Bar) ---
    ax4 = fig.add_subplot(gs[1, 1])

    # Get all results for stacked bar
    weak_results = {s.checkpoint_name: None for s in summaries}
    strong_results = {s.checkpoint_name: None for s in summaries}

    for r in results:
        if r.opponent == 'weak':
            weak_results[r.checkpoint_name] = r
        else:
            strong_results[r.checkpoint_name] = r

    x = np.arange(len(summaries))
    width = 0.4

    # Weak opponent stack
    weak_wins = [weak_results[s.checkpoint_name].win_rate * 100 if weak_results[s.checkpoint_name] else 0 for s in summaries]
    weak_draws = [weak_results[s.checkpoint_name].draw_rate * 100 if weak_results[s.checkpoint_name] else 0 for s in summaries]
    weak_losses = [weak_results[s.checkpoint_name].loss_rate * 100 if weak_results[s.checkpoint_name] else 0 for s in summaries]

    ax4.bar(x - width/2, weak_wins, width, label='Win', color=colors['combined'], alpha=0.8)
    ax4.bar(x - width/2, weak_draws, width, bottom=weak_wins, label='Draw', color=colors['draw'], alpha=0.8)
    ax4.bar(x - width/2, weak_losses, width, bottom=[w+d for w,d in zip(weak_wins, weak_draws)],
            label='Loss', color=colors['strong'], alpha=0.8)

    # Strong opponent stack
    strong_wins = [strong_results[s.checkpoint_name].win_rate * 100 if strong_results[s.checkpoint_name] else 0 for s in summaries]
    strong_draws = [strong_results[s.checkpoint_name].draw_rate * 100 if strong_results[s.checkpoint_name] else 0 for s in summaries]
    strong_losses = [strong_results[s.checkpoint_name].loss_rate * 100 if strong_results[s.checkpoint_name] else 0 for s in summaries]

    ax4.bar(x + width/2, strong_wins, width, color=colors['combined'], alpha=0.8)
    ax4.bar(x + width/2, strong_draws, width, bottom=strong_wins, color=colors['draw'], alpha=0.8)
    ax4.bar(x + width/2, strong_losses, width, bottom=[w+d for w,d in zip(strong_wins, strong_draws)],
            color=colors['strong'], alpha=0.8)

    ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Checkpoint (Left=Weak, Right=Strong)', fontsize=12, fontweight='bold')
    ax4.set_title('Outcome Distribution', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.checkpoint_name[:15] + '...' if len(s.checkpoint_name) > 15
                         else s.checkpoint_name for s in summaries],
                        rotation=45, ha='right', fontsize=8)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_ylim(0, 105)

    plt.suptitle('DreamerV3 Checkpoint Evaluation Results', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(output_dir, 'evaluation_overview.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    # =========================================================================
    # Figure 2: Detailed Performance Heatmap
    # =========================================================================
    fig, ax = plt.figure(figsize=(14, max(6, len(summaries) * 0.5))), plt.gca()

    # Create data matrix
    metrics = ['Weak Win%', 'Strong Win%', 'Combined%', 'Weak Reward', 'Strong Reward']
    data = []
    for s in summaries:
        row = [
            s.weak_win_rate * 100,
            s.strong_win_rate * 100,
            s.combined_win_rate * 100,
            s.weak_mean_reward,
            s.strong_mean_reward
        ]
        data.append(row)

    data = np.array(data)

    # Normalize for color mapping (per column)
    data_normalized = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        if col.max() != col.min():
            data_normalized[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            data_normalized[:, j] = 0.5

    # Create heatmap
    im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(summaries)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_yticklabels([s.checkpoint_name[:30] for s in summaries], fontsize=9)

    # Add text annotations
    for i in range(len(summaries)):
        for j in range(len(metrics)):
            val = data[i, j]
            text_color = 'white' if data_normalized[i, j] < 0.3 or data_normalized[i, j] > 0.7 else 'black'
            if j < 3:  # Percentages
                text = f'{val:.1f}%'
            else:  # Rewards
                text = f'{val:.2f}'
            ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')

    ax.set_title('Performance Heatmap (Green = Better)', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Relative Performance')

    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    # =========================================================================
    # Figure 3: Best Checkpoints Comparison
    # =========================================================================
    if len(summaries) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Sort by different criteria
        best_weak = sorted(summaries, key=lambda x: x.weak_win_rate, reverse=True)[:5]
        best_strong = sorted(summaries, key=lambda x: x.strong_win_rate, reverse=True)[:5]
        best_combined = sorted(summaries, key=lambda x: x.combined_win_rate, reverse=True)[:5]

        for ax, data, title, color in [
            (axes[0], best_weak, 'Best vs Weak', colors['weak']),
            (axes[1], best_strong, 'Best vs Strong', colors['strong']),
            (axes[2], best_combined, 'Best Combined', colors['combined'])
        ]:
            names = [s.checkpoint_name[:20] for s in data]
            if title == 'Best vs Weak':
                values = [s.weak_win_rate * 100 for s in data]
            elif title == 'Best vs Strong':
                values = [s.strong_win_rate * 100 for s in data]
            else:
                values = [s.combined_win_rate * 100 for s in data]

            bars = ax.barh(range(len(names)), values, color=color, alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('Win Rate (%)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlim(0, 105)
            ax.invert_yaxis()

            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9)

        plt.suptitle('Top 5 Checkpoints by Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_checkpoints.png'), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    print(f"  Visualizations saved to {output_dir}/")


# ============================================================================
# REPORTING
# ============================================================================

def generate_text_report(summaries: List[CheckpointSummary], results: List[TestResult],
                         output_path: str, total_duration: float):
    """Generate comprehensive text report."""

    lines = []
    lines.append("=" * 80)
    lines.append("DREAMERV3 CHECKPOINT EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Test Duration: {total_duration/60:.1f} minutes")
    lines.append(f"Checkpoints Tested: {len(summaries)}")
    lines.append("")

    # Summary table
    lines.append("-" * 80)
    lines.append("SUMMARY TABLE")
    lines.append("-" * 80)
    lines.append(f"{'Checkpoint':<40} {'Weak%':>8} {'Strong%':>8} {'Combined':>10} {'Steps':>12}")
    lines.append("-" * 80)

    for s in sorted(summaries, key=lambda x: x.combined_win_rate, reverse=True):
        name = s.checkpoint_name[:38] if len(s.checkpoint_name) > 38 else s.checkpoint_name
        lines.append(f"{name:<40} {s.weak_win_rate*100:>7.1f}% {s.strong_win_rate*100:>7.1f}% "
                    f"{s.combined_win_rate*100:>9.1f}% {s.gradient_steps:>12,}")

    lines.append("")

    # Best performers
    lines.append("-" * 80)
    lines.append("BEST PERFORMERS")
    lines.append("-" * 80)

    best_weak = max(summaries, key=lambda x: x.weak_win_rate)
    best_strong = max(summaries, key=lambda x: x.strong_win_rate)
    best_combined = max(summaries, key=lambda x: x.combined_win_rate)

    lines.append(f"Best vs Weak:     {best_weak.checkpoint_name}")
    lines.append(f"                  Win Rate: {best_weak.weak_win_rate*100:.1f}%")
    lines.append("")
    lines.append(f"Best vs Strong:   {best_strong.checkpoint_name}")
    lines.append(f"                  Win Rate: {best_strong.strong_win_rate*100:.1f}%")
    lines.append("")
    lines.append(f"Best Combined:    {best_combined.checkpoint_name}")
    lines.append(f"                  Combined: {best_combined.combined_win_rate*100:.1f}% "
                f"(Weak: {best_combined.weak_win_rate*100:.1f}%, Strong: {best_combined.strong_win_rate*100:.1f}%)")
    lines.append("")

    # Detailed results
    lines.append("-" * 80)
    lines.append("DETAILED RESULTS")
    lines.append("-" * 80)

    for s in sorted(summaries, key=lambda x: x.gradient_steps):
        lines.append("")
        lines.append(f"Checkpoint: {s.checkpoint_name}")
        lines.append(f"  Path: {s.checkpoint_path}")
        lines.append(f"  Training: {s.gradient_steps:,} gradient steps, {s.training_episodes:,} episodes")
        lines.append("")

        # Find corresponding detailed results
        weak_result = next((r for r in results if r.checkpoint_name == s.checkpoint_name and r.opponent == 'weak'), None)
        strong_result = next((r for r in results if r.checkpoint_name == s.checkpoint_name and r.opponent == 'strong'), None)

        if weak_result:
            lines.append(f"  vs Weak ({weak_result.episodes} episodes):")
            lines.append(f"    Win Rate:  {weak_result.win_rate*100:5.1f}% ({weak_result.wins} wins)")
            lines.append(f"    Loss Rate: {weak_result.loss_rate*100:5.1f}% ({weak_result.losses} losses)")
            lines.append(f"    Draw Rate: {weak_result.draw_rate*100:5.1f}% ({weak_result.draws} draws)")
            lines.append(f"    Reward:    {weak_result.mean_reward:+.2f} ± {weak_result.std_reward:.2f}")
            lines.append(f"    Avg Steps: {weak_result.mean_steps:.1f} ± {weak_result.std_steps:.1f}")

        if strong_result:
            lines.append(f"  vs Strong ({strong_result.episodes} episodes):")
            lines.append(f"    Win Rate:  {strong_result.win_rate*100:5.1f}% ({strong_result.wins} wins)")
            lines.append(f"    Loss Rate: {strong_result.loss_rate*100:5.1f}% ({strong_result.losses} losses)")
            lines.append(f"    Draw Rate: {strong_result.draw_rate*100:5.1f}% ({strong_result.draws} draws)")
            lines.append(f"    Reward:    {strong_result.mean_reward:+.2f} ± {strong_result.std_reward:.2f}")
            lines.append(f"    Avg Steps: {strong_result.mean_steps:.1f} ± {strong_result.std_steps:.1f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text


def save_json_results(summaries: List[CheckpointSummary], results: List[TestResult], output_path: str):
    """Save results as JSON for programmatic access."""
    data = {
        'generated': datetime.now().isoformat(),
        'summaries': [asdict(s) for s in summaries],
        'detailed_results': [asdict(r) for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def find_checkpoints(base_dir: str, specific_checkpoints: Optional[List[str]] = None) -> List[str]:
    """Find all checkpoint files in the directory."""
    if specific_checkpoints:
        # Use specific checkpoints
        checkpoints = []
        for cp in specific_checkpoints:
            path = os.path.join(base_dir, cp) if not os.path.isabs(cp) else cp
            if os.path.exists(path):
                checkpoints.append(path)
            else:
                print(f"Warning: Checkpoint not found: {cp}")
        return checkpoints

    # Find all .pth files
    checkpoints = []

    # Check main directory
    for f in os.listdir(base_dir):
        if f.endswith('.pth'):
            checkpoints.append(os.path.join(base_dir, f))

    # Check results/checkpoints subdirectory
    results_dir = os.path.join(base_dir, 'results', 'checkpoints')
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.endswith('.pth'):
                checkpoints.append(os.path.join(results_dir, f))

    return checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive DreamerV3 Checkpoint Testing")

    parser.add_argument("--checkpoints", nargs='+', default=None,
                        help="Specific checkpoint files to test (default: all)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per opponent per checkpoint (default: 100)")
    parser.add_argument("--record-episodes", type=int, default=5,
                        help="Episodes to record per test (default: 5)")
    parser.add_argument("--no-record", action="store_true",
                        help="Skip video recording")
    parser.add_argument("--output-dir", type=str, default="test_results",
                        help="Output directory for results (default: test_results)")
    parser.add_argument("--config", type=str, default="hockey.yml",
                        help="Config file (default: hockey.yml)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/mps/cpu, default: auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    config = loadConfig(args.config)
    device = get_device(args.device)
    seedEverything(args.seed)

    print("=" * 70)
    print("DREAMERV3 COMPREHENSIVE CHECKPOINT TESTING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Episodes per test: {args.episodes}")
    print(f"Record episodes: {0 if args.no_record else args.record_episodes}")
    print(f"Output directory: {output_dir}")
    print()

    # Find checkpoints
    checkpoints = find_checkpoints(base_dir, args.checkpoints)

    if not checkpoints:
        print("ERROR: No checkpoints found!")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {os.path.basename(cp)}")
    print()

    # Create environment
    env = h_env.HockeyEnv(mode=Mode.NORMAL, keep_mode=True)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0] // 2
    action_low = env.action_space.low[:action_size].tolist()
    action_high = env.action_space.high[:action_size].tolist()

    # Run tests
    all_results: List[TestResult] = []
    summaries: List[CheckpointSummary] = []

    total_start = time.time()

    for cp_idx, checkpoint_path in enumerate(checkpoints):
        checkpoint_name = os.path.basename(checkpoint_path)

        print(f"\n[{cp_idx+1}/{len(checkpoints)}] Testing: {checkpoint_name}")
        print("-" * 60)

        # Load agent
        try:
            agent = Dreamer(observation_size, action_size, action_low, action_high, device, config.dreamer)
            agent.loadCheckpoint(checkpoint_path)
            print(f"  Loaded: {agent.totalGradientSteps:,} gradient steps, {agent.totalEpisodes:,} episodes")
        except Exception as e:
            print(f"  ERROR loading checkpoint: {e}")
            continue

        weak_result = None
        strong_result = None

        # Test vs weak
        print(f"\n  Testing vs WEAK ({args.episodes} episodes)...")
        weak_result, weak_frames = test_checkpoint(
            agent, checkpoint_name, checkpoint_path, "weak",
            args.episodes, env,
            record_episodes=0 if args.no_record else args.record_episodes,
            verbose=args.verbose
        )
        all_results.append(weak_result)
        print(f"  → Win rate: {weak_result.win_rate*100:.1f}% | "
              f"Reward: {weak_result.mean_reward:+.2f} | "
              f"Time: {weak_result.test_duration_seconds:.1f}s")

        # Save weak video
        if weak_frames and not args.no_record:
            video_path = os.path.join(output_dir, 'videos', f'{checkpoint_name}_vs_weak.mp4')
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            if save_video(weak_frames, video_path):
                print(f"  → Video saved: {video_path}")

        # Test vs strong
        print(f"\n  Testing vs STRONG ({args.episodes} episodes)...")
        strong_result, strong_frames = test_checkpoint(
            agent, checkpoint_name, checkpoint_path, "strong",
            args.episodes, env,
            record_episodes=0 if args.no_record else args.record_episodes,
            verbose=args.verbose
        )
        all_results.append(strong_result)
        print(f"  → Win rate: {strong_result.win_rate*100:.1f}% | "
              f"Reward: {strong_result.mean_reward:+.2f} | "
              f"Time: {strong_result.test_duration_seconds:.1f}s")

        # Save strong video
        if strong_frames and not args.no_record:
            video_path = os.path.join(output_dir, 'videos', f'{checkpoint_name}_vs_strong.mp4')
            if save_video(strong_frames, video_path):
                print(f"  → Video saved: {video_path}")

        # Create summary
        combined_win_rate = (weak_result.win_rate + strong_result.win_rate) / 2
        summary = CheckpointSummary(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            gradient_steps=agent.totalGradientSteps,
            training_episodes=agent.totalEpisodes,
            weak_win_rate=weak_result.win_rate,
            strong_win_rate=strong_result.win_rate,
            combined_win_rate=combined_win_rate,
            weak_mean_reward=weak_result.mean_reward,
            strong_mean_reward=strong_result.mean_reward
        )
        summaries.append(summary)

        print(f"\n  COMBINED: {combined_win_rate*100:.1f}%")

    env.close()

    total_duration = time.time() - total_start

    # Generate outputs
    print("\n" + "=" * 70)
    print("GENERATING REPORTS AND VISUALIZATIONS")
    print("=" * 70)

    # Text report
    report_path = os.path.join(output_dir, f'evaluation_report_{timestamp}.txt')
    report_text = generate_text_report(summaries, all_results, report_path, total_duration)
    print(f"\n  Text report saved: {report_path}")

    # JSON results
    json_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
    save_json_results(summaries, all_results, json_path)
    print(f"  JSON results saved: {json_path}")

    # Visualizations
    print("\n  Creating visualizations...")
    create_visualizations(summaries, all_results, os.path.join(output_dir, 'plots'))

    # Print summary to console
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nTotal Duration: {total_duration/60:.1f} minutes")
    print(f"Checkpoints Tested: {len(summaries)}")
    print()

    # Sort by combined and print top 5
    top_summaries = sorted(summaries, key=lambda x: x.combined_win_rate, reverse=True)[:5]
    print("TOP 5 CHECKPOINTS (by combined win rate):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Checkpoint':<35} {'Weak%':>8} {'Strong%':>9} {'Combined':>10}")
    print("-" * 70)

    for i, s in enumerate(top_summaries, 1):
        name = s.checkpoint_name[:33] if len(s.checkpoint_name) > 33 else s.checkpoint_name
        print(f"{i:<6} {name:<35} {s.weak_win_rate*100:>7.1f}% {s.strong_win_rate*100:>8.1f}% "
              f"{s.combined_win_rate*100:>9.1f}%")

    print()
    print("=" * 70)
    print(f"Full report: {report_path}")
    print(f"Visualizations: {os.path.join(output_dir, 'plots')}/")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
