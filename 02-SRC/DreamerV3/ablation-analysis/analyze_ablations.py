#!/usr/bin/env python3
"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

# Parse W&B ablation run files, extract metrics. Outputs JSON and CSV.

import re
import json
from pathlib import Path
from collections import defaultdict

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"


def parse_wandb_file(path: Path) -> dict:
    """Parse a wandb_run_*.txt file and extract metadata + metrics."""
    text = path.read_text()
    lines = text.splitlines()

    result = {
        "file": path.name,
        "run_name": None,
        "metadata": {},
        "config": {},
        "metrics": {},
        "metric_series": {},
    }

    # Run name from header
    for line in lines[:5]:
        if line.startswith("W&B RUN DATA:"):
            result["run_name"] = line.replace("W&B RUN DATA:", "").strip()
            break

    # Parse sections
    current_section = None
    current_metric = None
    current_data = []

    for line in lines:
        if line.startswith("## RUN METADATA"):
            current_section = "metadata"
            continue
        elif line.startswith("## CONFIGURATION"):
            current_section = "config"
            continue
        elif line.startswith("## METRICS DATA"):
            current_section = "metrics"
            continue

        if current_section == "metadata":
            if ":" in line and not line.strip().startswith("["):
                k, v = line.split(":", 1)
                result["metadata"][k.strip()] = v.strip()

        elif current_section == "metrics":
            if line.startswith("#### "):
                if current_metric:
                    result["metric_series"][current_metric] = current_data
                current_metric = line.replace("####", "").strip()
                current_data = []
            elif line.strip().startswith("Data: ["):
                pass
            elif re.match(r"^\s+[\d\-.,\s]+[,]?\s*$", line):
                # Parse data line: "    0.0000, 0.2700, 0.5800, ..."
                nums = re.findall(r"-?\d+\.?\d*", line)
                current_data.extend([float(n) for n in nums])
            elif "Min:" in line and "Max:" in line:
                m = re.search(r"Min:\s*([\d.-]+).*Max:\s*([\d.-]+).*Mean:\s*([\d.-]+)", line)
                if m and current_metric:
                    result["metrics"][current_metric] = {
                        "min": float(m.group(1)),
                        "max": float(m.group(2)),
                        "mean": float(m.group(3)),
                    }

    if current_metric:
        result["metric_series"][current_metric] = current_data

    return result


def get_metric_at_steps(run: dict, metric_key: str, steps_key: str = "gradient_steps") -> tuple:
    """Get (steps, values) for a metric, aligned by index."""
    steps_data = run["metric_series"].get(steps_key, [])
    metric_data = run["metric_series"].get(metric_key, [])
    n = min(len(steps_data), len(metric_data))
    return (steps_data[:n], metric_data[:n])


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    runs = {}
    for f in sorted(INPUT_DIR.glob("wandb_run_*.txt")):
        run = parse_wandb_file(f)
        name = run["run_name"]
        runs[name] = run

    # Summary table
    summary = []
    for name, run in runs.items():
        wr = run["metrics"].get("win_rate", {})
        mr = run["metrics"].get("mean_reward", {})
        gs = run["metric_series"].get("gradient_steps", [])
        max_steps = max(gs) if gs else 0
        summary.append({
            "run": name,
            "max_gradient_steps": max_steps,
            "win_rate_final": wr.get("max", 0),
            "mean_reward_final": mr.get("max", 0),
            "win_rate_mean": wr.get("mean", 0),
            "mean_reward_mean": mr.get("mean", 0),
        })

    (OUTPUT_DIR / "runs_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Parsed {len(runs)} runs, wrote runs_summary.json")

    # Extract series for plotting (gradient_steps vs win_rate, mean_reward)
    series_export = {}
    for name, run in runs.items():
        steps, win_rate = get_metric_at_steps(run, "win_rate")
        _, mean_reward = get_metric_at_steps(run, "mean_reward")
        series_export[name] = {
            "gradient_steps": steps,
            "win_rate": win_rate,
            "mean_reward": mean_reward,
        }
    (OUTPUT_DIR / "metric_series.json").write_text(json.dumps(series_export, indent=2))
    print("Wrote metric_series.json")


if __name__ == "__main__":
    main()
