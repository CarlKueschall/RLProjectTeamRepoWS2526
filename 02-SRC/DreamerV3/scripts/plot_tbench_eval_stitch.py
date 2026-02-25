#!/usr/bin/env python3
"""
Create a stitched eval/combined_win_rate plot across the four tbench runs.

This script pulls live data from W&B and exports:
  1) A stitched benchmark plot (PNG)
  2) The stitched raw data (CSV)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


@dataclass
class RunSpec:
    label: str
    run_id: str
    color: str


RUN_ORDER: List[RunSpec] = [
    RunSpec("tbench-from-scratch", "hxi50hzk", "#1f77b4"),
    RunSpec("tbench-selfplay-only", "y3kb2dat", "#2ca02c"),
    RunSpec("tbench-selfplay-league-round-1", "65uupd8c", "#ff7f0e"),
    RunSpec("tbench-selfplay-league-round-2", "nzo74r5t", "#d62728"),
]


def get_default_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "03-RESULTS" / "REPORT" / "figures" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir / "final_benchmark_eval_progression_stitch.png",
        out_dir / "final_benchmark_eval_progression_stitch_data.csv",
    )


def fetch_eval_series(api: wandb.Api, path: str) -> Tuple[np.ndarray, np.ndarray]:
    run = api.run(path)
    # run.history() is sampled but sufficient for eval curves and much faster
    # than scanning full unsampled history for long runs.
    hist = run.history(samples=20000)
    if hist.empty or "eval/combined_win_rate" not in hist.columns:
        raise RuntimeError(f"No eval/combined_win_rate points found for {path}")

    hist = hist.sort_values("_step").reset_index(drop=True)

    if "stats/gradient_steps" in hist.columns:
        grad = hist["stats/gradient_steps"].ffill()
    elif "gradient_steps" in hist.columns:
        grad = hist["gradient_steps"].ffill()
    else:
        grad = pd.Series([np.nan] * len(hist))

    grad = grad.fillna(hist["_step"])
    eval_rows = hist[hist["eval/combined_win_rate"].notna()].copy()
    eval_rows["__grad"] = grad[eval_rows.index].astype(float)

    arr = eval_rows[["__grad", "eval/combined_win_rate"]].to_numpy(dtype=float)
    if len(arr) == 0:
        raise RuntimeError(f"No eval/combined_win_rate points found for {path}")

    # Keep last value per x if duplicates exist.
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    uniq_x: List[float] = []
    uniq_y: List[float] = []
    for x, y in arr:
        if uniq_x and x == uniq_x[-1]:
            uniq_y[-1] = y
        else:
            uniq_x.append(float(x))
            uniq_y.append(float(y))

    return np.array(uniq_x), np.array(uniq_y)


def build_continuous_segments(
    run_data: List[Tuple[RunSpec, np.ndarray, np.ndarray]]
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Merge runs into one continuous x-axis in real gradient steps.
    Overlap handling: for each subsequent run, drop points with
    gradient_steps <= last kept gradient step.
    """
    merged: List[Dict] = []
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    last_x = -np.inf

    for spec, x, y in run_data:
        keep_mask = x > last_x
        kept_x = x[keep_mask]
        kept_y = y[keep_mask]
        dropped = int((~keep_mask).sum())

        if len(kept_x) == 0:
            merged.append(
                {
                    "spec": spec,
                    "x_raw": x,
                    "y_raw": y,
                    "x_kept": kept_x,
                    "y_kept": kept_y,
                    "start": None,
                    "end": None,
                    "dropped": dropped,
                }
            )
            continue

        merged.append(
            {
                "spec": spec,
                "x_raw": x,
                "y_raw": y,
                "x_kept": kept_x,
                "y_kept": kept_y,
                "start": float(kept_x[0]),
                "end": float(kept_x[-1]),
                "dropped": dropped,
            }
        )
        all_x.append(kept_x)
        all_y.append(kept_y)
        last_x = float(kept_x[-1])

    if not all_x:
        raise RuntimeError("No non-overlapping points left after merge.")

    return merged, np.concatenate(all_x), np.concatenate(all_y)


def export_csv(segments: List[Dict], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase",
                "run_id",
                "gradient_steps",
                "x_ksteps",
                "eval_combined_win_rate",
                "dropped_overlap_prefix_points",
            ]
        )
        for seg in segments:
            spec: RunSpec = seg["spec"]
            for gx, y in zip(seg["x_kept"], seg["y_kept"]):
                writer.writerow(
                    [
                        spec.label,
                        spec.run_id,
                        int(gx),
                        float(gx / 1000.0),
                        float(y),
                        seg["dropped"],
                    ]
                )


def make_plot(segments: List[Dict], all_x: np.ndarray, all_y: np.ndarray, out_png: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    # One global timeline line (all kept datapoints merged)
    ax.plot(
        all_x / 1000.0,
        all_y,
        color="#333333",
        linewidth=1.8,
        alpha=0.7,
        label="Merged timeline",
        zorder=1,
    )

    for i, seg in enumerate(segments):
        spec: RunSpec = seg["spec"]
        if seg["start"] is None:
            continue
        xs = seg["x_kept"] / 1000.0
        ys = seg["y_kept"]
        ax.axvspan(seg["start"] / 1000.0, seg["end"] / 1000.0, color=spec.color, alpha=0.05, lw=0)
        label = spec.label
        # Dense segments become unreadable with markers everywhere.
        # Keep raw accuracy in a light trace and add a smooth overlay.
        if len(xs) > 140:
            ax.plot(xs, ys, color=spec.color, linewidth=1.2, alpha=0.28)
            win = max(7, (len(xs) // 30) | 1)  # odd window
            kernel = np.ones(win) / win
            ys_smooth = np.convolve(ys, kernel, mode="same")
            ax.plot(
                xs,
                ys_smooth,
                color=spec.color,
                linewidth=3.0,
                alpha=0.95,
                label=label,
            )
        else:
            ax.plot(
                xs,
                ys,
                color=spec.color,
                linewidth=2.8,
                marker="o",
                markersize=4.0,
                alpha=0.95,
                label=label,
            )
        # Boundary lines (except first segment start).
        if i > 0:
            ax.axvline(
                seg["start"] / 1000.0, color="#666666", linestyle="--", linewidth=1.1, alpha=0.7
            )

        # Phase label above each segment center.
        cx = (seg["start"] + seg["end"]) / 2 / 1000.0
        ax.text(
            cx,
            1.015,
            f"Phase {i+1}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=spec.color,
        )

    ax.set_ylim(0.0, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("eval/combined_win_rate", fontsize=12)
    ax.set_xlabel("Gradient steps (thousands)", fontsize=12)
    ax.set_title(
        "Benchmark Progression (single timeline, overlap-clipped across runs)",
        fontsize=16,
        fontweight="bold",
        pad=34,
    )
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.15)

    leg = ax.legend(loc="lower right", frameon=True, fontsize=10, ncol=1)
    leg.get_frame().set_alpha(0.92)
    leg.get_frame().set_facecolor("#f7f7f7")

    # Reserve extra top space so title and phase labels never collide.
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch tbench eval curves from W&B")
    parser.add_argument("--entity", type=str, default="carlkueschalledu")
    parser.add_argument("--project", type=str, default="rl-hockey")
    parser.add_argument("--output_png", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    default_png, default_csv = get_default_paths()
    out_png = Path(args.output_png) if args.output_png else default_png
    out_csv = Path(args.output_csv) if args.output_csv else default_csv
    out_png.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    fetched: List[Tuple[RunSpec, np.ndarray, np.ndarray]] = []
    print(f"Fetching eval/combined_win_rate from {args.entity}/{args.project} ...")
    for spec in RUN_ORDER:
        path = f"{args.entity}/{args.project}/{spec.run_id}"
        x, y = fetch_eval_series(api, path)
        fetched.append((spec, x, y))
        print(
            f"  {spec.label}: {len(y)} points, gradient steps {int(x[0])} -> {int(x[-1])},"
            f" last={y[-1]:.4f}"
        )

    segments, all_x, all_y = build_continuous_segments(fetched)
    make_plot(segments, all_x, all_y, out_png)
    export_csv(segments, out_csv)

    print(f"\nSaved stitched plot: {out_png}")
    print(f"Saved stitched data: {out_csv}")


if __name__ == "__main__":
    main()
