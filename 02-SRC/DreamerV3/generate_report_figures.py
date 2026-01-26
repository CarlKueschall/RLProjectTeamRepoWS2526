#!/usr/bin/env python3
"""
Generate figures for the DreamerV3 Hockey Report.

This script creates publication-quality figures from W&B run data.
Figures are saved to 03-RESULTS/REPORT/INPUT/figures/

Usage:
    python generate_report_figures.py --wandb_entity YOUR_ENTITY --wandb_project rl-hockey
    python generate_report_figures.py --placeholder  # Generate placeholder figures
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Style settings for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '03-RESULTS', 'REPORT', 'INPUT', 'figures')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def smooth(data, window=10):
    """Simple moving average smoothing that preserves array length."""
    if len(data) < window:
        return data
    # Use 'same' mode and handle edges
    smoothed = np.convolve(data, np.ones(window)/window, mode='same')
    return smoothed


def generate_placeholder_training_curve():
    """Generate placeholder 3-phase training curve."""
    ensure_dir(FIGURES_DIR)

    # Simulated data for 3 phases
    np.random.seed(42)

    # Phase 1: Rapid improvement with noise (0-268k steps)
    x1 = np.linspace(0, 268, 200)
    y1 = 72 * (1 - np.exp(-x1/80)) + np.random.randn(200) * 5

    # Phase 2: Stabilization (192k-270k steps, continuing from ~72%)
    x2 = np.linspace(268, 350, 60)
    y2 = 72 + (85-72) * (1 - np.exp(-(x2-268)/30)) + np.random.randn(60) * 3

    # Phase 3: Fine-tuning (260k-340k steps)
    x3 = np.linspace(350, 450, 80)
    y3 = 85 + (88.5-85) * (1 - np.exp(-(x3-350)/40)) + np.random.randn(80) * 2

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each phase with different colors
    ax.plot(x1, smooth(np.clip(y1, 0, 100), 5), 'b-', linewidth=2, label='Phase 1: Mixed + Self-play')
    ax.plot(x2, smooth(np.clip(y2, 0, 100), 3), 'g-', linewidth=2, label='Phase 2: Mixed only')
    ax.plot(x3, smooth(np.clip(y3, 0, 100), 3), 'r-', linewidth=2, label='Phase 3: Fine-tuning')

    # Phase transition lines
    ax.axvline(x=268, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=350, color='gray', linestyle='--', alpha=0.5)

    # Annotations
    ax.annotate('Self-play\nremoved', xy=(268, 75), xytext=(220, 50),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    ax.annotate('LR reduced', xy=(350, 86), xytext=(400, 70),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    ax.set_xlabel('Gradient Steps (thousands)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('3-Phase Training Curriculum')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 460)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add final performance annotation
    ax.annotate(f'Final: 88.5%', xy=(450, 88.5), xytext=(420, 95),
                fontsize=10, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.savefig(os.path.join(FIGURES_DIR, 'training_curve_placeholder.png'))
    plt.close()
    print(f"Saved: training_curve_placeholder.png")


def generate_placeholder_ablation_dreamsmooth():
    """Generate placeholder DreamSmooth ablation figures."""
    ensure_dir(FIGURES_DIR)
    np.random.seed(43)

    steps = np.linspace(0, 150, 100)

    # WITH DreamSmooth - faster learning, better final
    with_ds = 85 * (1 - np.exp(-steps/40)) + np.random.randn(100) * 4

    # WITHOUT DreamSmooth - slower, worse final
    without_ds = 60 * (1 - np.exp(-steps/60)) + np.random.randn(100) * 5

    # Win rate plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, smooth(np.clip(with_ds, 0, 100), 5), 'b-', linewidth=2, label='With DreamSmooth')
    ax.plot(steps, smooth(np.clip(without_ds, 0, 100), 5), 'r--', linewidth=2, label='Without DreamSmooth')
    ax.set_xlabel('Gradient Steps (thousands)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('DreamSmooth Ablation: Win Rate')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_dreamsmooth_winrate.png'))
    plt.close()
    print(f"Saved: ablation_dreamsmooth_winrate.png")

    # Reward plot
    with_ds_reward = 7 * (1 - np.exp(-steps/50)) - 3 + np.random.randn(100) * 0.5
    without_ds_reward = 4 * (1 - np.exp(-steps/70)) - 3 + np.random.randn(100) * 0.6

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, smooth(with_ds_reward, 5), 'b-', linewidth=2, label='With DreamSmooth')
    ax.plot(steps, smooth(without_ds_reward, 5), 'r--', linewidth=2, label='Without DreamSmooth')
    ax.set_xlabel('Gradient Steps (thousands)')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('DreamSmooth Ablation: Reward')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_dreamsmooth_reward.png'))
    plt.close()
    print(f"Saved: ablation_dreamsmooth_reward.png")


def generate_placeholder_ablation_auxiliary():
    """Generate placeholder Auxiliary Tasks ablation figures."""
    ensure_dir(FIGURES_DIR)
    np.random.seed(44)

    steps = np.linspace(0, 150, 100)

    # WITH Auxiliary Tasks
    with_aux = 85 * (1 - np.exp(-steps/45)) + np.random.randn(100) * 4

    # WITHOUT Auxiliary Tasks - slightly worse
    without_aux = 75 * (1 - np.exp(-steps/50)) + np.random.randn(100) * 5

    # Win rate plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, smooth(np.clip(with_aux, 0, 100), 5), 'b-', linewidth=2, label='With Auxiliary Tasks')
    ax.plot(steps, smooth(np.clip(without_aux, 0, 100), 5), 'r--', linewidth=2, label='Without Auxiliary Tasks')
    ax.set_xlabel('Gradient Steps (thousands)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Auxiliary Tasks Ablation: Win Rate')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_auxiliary_winrate.png'))
    plt.close()
    print(f"Saved: ablation_auxiliary_winrate.png")

    # World model loss plot
    with_aux_loss = 3 * np.exp(-steps/30) + 0.5 + np.random.randn(100) * 0.1
    without_aux_loss = 3 * np.exp(-steps/40) + 0.7 + np.random.randn(100) * 0.12

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, smooth(with_aux_loss, 5), 'b-', linewidth=2, label='With Auxiliary Tasks')
    ax.plot(steps, smooth(without_aux_loss, 5), 'r--', linewidth=2, label='Without Auxiliary Tasks')
    ax.set_xlabel('Gradient Steps (thousands)')
    ax.set_ylabel('World Model Loss')
    ax.set_title('Auxiliary Tasks Ablation: World Model Quality')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_auxiliary_worldloss.png'))
    plt.close()
    print(f"Saved: ablation_auxiliary_worldloss.png")


def generate_all_placeholders():
    """Generate all placeholder figures."""
    print("Generating placeholder figures for report...")
    print(f"Output directory: {FIGURES_DIR}")
    print()

    generate_placeholder_training_curve()
    generate_placeholder_ablation_dreamsmooth()
    generate_placeholder_ablation_auxiliary()

    print()
    print("All placeholder figures generated!")
    print("Replace with real data after running ablations.")


def main():
    parser = argparse.ArgumentParser(description='Generate report figures')
    parser.add_argument('--placeholder', action='store_true',
                        help='Generate placeholder figures with simulated data')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity for fetching real data')
    parser.add_argument('--wandb_project', type=str, default='rl-hockey',
                        help='W&B project name')

    args = parser.parse_args()

    if args.placeholder:
        generate_all_placeholders()
    elif args.wandb_entity:
        print("W&B data fetching not yet implemented.")
        print("Use --placeholder for now, then update with real data.")
    else:
        print("Usage:")
        print("  python generate_report_figures.py --placeholder")
        print("  python generate_report_figures.py --wandb_entity YOUR_ENTITY")


if __name__ == '__main__':
    main()
