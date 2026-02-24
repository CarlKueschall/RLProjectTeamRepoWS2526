#!/usr/bin/env python3
"""
Fast checkpoint league selection for self-play pool curation.

Goal:
- Evaluate many checkpoints quickly.
- Produce a concrete shortlist to include in self-play pool.

Method (2-stage):
1) Stage 1 (fast screening):
   - Fixed-bot eval (weak/strong) for each checkpoint.
   - Sparse checkpoint-vs-checkpoint directed matches.
   - Rank by robust screening score.
2) Stage 2 (focused refinement on top-K):
   - Denser checkpoint-vs-checkpoint matches (optionally bidirectional).
   - Build final robustness summary and pairwise score matrix.
3) Pool recommendation:
   - Greedy selection maximizing quality + diversity.

Outputs:
- stage1_matches.csv
- stage1_summary.csv
- stage2_matches.csv (unless --skip-stage2)
- final_summary.csv
- recommended_pool.txt
- recommended_pool.md
"""

import argparse
import csv
import datetime as dt
import os
import random
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Fast checkpoint pool selection for self-play.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints/summit-evaluation/opponents",
        help="Directory with .pth checkpoints",
    )
    parser.add_argument(
        "--checkpoints-file",
        type=str,
        default=None,
        help="Optional newline-separated checkpoint list (overrides checkpoints-dir)",
    )
    parser.add_argument("--config", type=str, default="hockey.yml", help="Config for test_hockey.py")
    parser.add_argument("--device", type=str, default=None, help="Optional device override")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--sample-seed", type=int, default=123, help="Seed for stage1 matchup sampling")
    parser.add_argument("--max-workers", type=int, default=6, help="Parallel workers")
    parser.set_defaults(skip_incompatible=True)
    parser.add_argument(
        "--skip-incompatible",
        dest="skip_incompatible",
        action="store_true",
        help="Pre-check and skip checkpoints incompatible with current architecture (default: enabled)",
    )
    parser.add_argument(
        "--no-skip-incompatible",
        dest="skip_incompatible",
        action="store_false",
        help="Disable compatibility pre-check (not recommended for mixed legacy folders)",
    )

    # Stage 1 (screening)
    parser.add_argument("--episodes-fixed-stage1", type=int, default=40, help="Episodes vs weak/strong in stage1")
    parser.add_argument("--episodes-cp-stage1", type=int, default=14, help="Episodes per cp-vs-cp match in stage1")
    parser.add_argument(
        "--stage1-opponents-per-candidate",
        type=int,
        default=10,
        help="How many cp opponents to sample per candidate in stage1",
    )
    parser.add_argument(
        "--stage1-neighbor-span",
        type=int,
        default=2,
        help="Always include +/- this many temporal neighbors in stage1",
    )

    # Stage 2 (refinement)
    parser.add_argument("--skip-stage2", action="store_true", help="Skip stage2 and recommend from stage1 only")
    parser.add_argument("--stage2-top-k", type=int, default=24, help="Top-K from stage1 to refine in stage2")
    parser.add_argument("--episodes-fixed-stage2", type=int, default=80, help="Episodes vs weak/strong in stage2")
    parser.add_argument("--episodes-cp-stage2", type=int, default=40, help="Episodes per cp-vs-cp in stage2")
    parser.add_argument(
        "--stage2-bidirectional",
        action="store_true",
        help="Evaluate both A->B and B->A in stage2 (recommended)",
    )

    # Final pool selection
    parser.add_argument("--pool-size", type=int, default=24, help="Final recommended pool size")
    parser.add_argument("--quality-weight", type=float, default=0.75, help="Weight for quality in greedy selection")
    parser.add_argument("--diversity-weight", type=float, default=0.25, help="Weight for diversity in greedy selection")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/summit-evaluation/results",
        help="Base output dir",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print job counts")
    return parser.parse_args()


def step_key(path: str):
    name = Path(path).name.lower()
    m = re.search(r"(\d+)k", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{5,7})", name)
    if m:
        return int(m.group(1))
    return 10**9


def discover_checkpoints_from_dir(path: Path):
    return sorted([str(p.resolve()) for p in path.glob("*.pth")], key=step_key)


def read_checkpoint_list(path: Path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        out.append(str(p))
    return out


def parse_eval_output(stdout: str):
    win_m = re.search(r"Win rate:\s+([0-9.]+)%\s+\((\d+)\s+wins\)", stdout)
    loss_m = re.search(r"Loss rate:\s+([0-9.]+)%\s+\((\d+)\s+losses\)", stdout)
    draw_m = re.search(r"Draw rate:\s+([0-9.]+)%\s+\((\d+)\s+draws\)", stdout)
    reward_m = re.search(r"Reward:\s+([+-]?[0-9.]+)\s+\+/-\s+([0-9.]+)", stdout)
    steps_m = re.search(r"Avg steps:\s+([0-9.]+)", stdout)
    if not (win_m and loss_m and draw_m and reward_m and steps_m):
        raise RuntimeError("Could not parse test_hockey output:\n" + stdout[-1500:])
    return {
        "win_rate": float(win_m.group(1)) / 100.0,
        "loss_rate": float(loss_m.group(1)) / 100.0,
        "draw_rate": float(draw_m.group(1)) / 100.0,
        "wins": int(win_m.group(2)),
        "losses": int(loss_m.group(2)),
        "draws": int(draw_m.group(2)),
        "avg_reward": float(reward_m.group(1)),
        "reward_std": float(reward_m.group(2)),
        "avg_steps": float(steps_m.group(1)),
    }


def run_eval(test_script: Path, config: str, checkpoint: str, episodes: int, seed: int, device: str, opponent: str = None, opponent_checkpoint: str = None):
    cmd = [
        sys.executable,
        str(test_script),
        "--checkpoint",
        checkpoint,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--config",
        config,
    ]
    if device:
        cmd.extend(["--device", device])
    if opponent is not None:
        cmd.extend(["--opponent", opponent])
    if opponent_checkpoint is not None:
        cmd.extend(["--opponent_checkpoint", opponent_checkpoint])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Evaluation command failed:\n"
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + proc.stdout[-2000:]
            + "\nSTDERR:\n"
            + proc.stderr[-2000:]
        )
    parsed = parse_eval_output(proc.stdout)
    return parsed, cmd


def run_one_job(test_script: Path, config: str, seed: int, device: str, job: dict):
    parsed, cmd = run_eval(
        test_script=test_script,
        config=config,
        checkpoint=job["candidate_path"],
        episodes=job["episodes"],
        seed=seed,
        device=device,
        opponent=job.get("fixed_opponent"),
        opponent_checkpoint=job.get("opponent_path") if job["opponent_type"] == "checkpoint" else None,
    )
    return {
        "stage": job["stage"],
        "candidate": job["candidate"],
        "candidate_path": job["candidate_path"],
        "opponent_type": job["opponent_type"],
        "opponent": job["opponent"],
        "opponent_path": job["opponent_path"],
        "episodes": job["episodes"],
        "seed": seed,
        "command": " ".join(cmd),
        **parsed,
    }


def run_jobs(jobs, test_script: Path, config: str, seed: int, device: str, workers: int):
    rows = []
    failures = []
    total = len(jobs)
    if workers <= 1:
        for i, job in enumerate(jobs, start=1):
            label = f"{job['candidate']} vs {job['opponent_type']}:{job['opponent']}"
            print(f"[{i}/{total}] {label}")
            try:
                rows.append(run_one_job(test_script, config, seed, device, job))
            except Exception as e:
                failures.append({"job": job, "error": str(e)})
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            future_map = {
                ex.submit(run_one_job, test_script, config, seed, device, job): job for job in jobs
            }
            completed = 0
            for fut in as_completed(future_map):
                completed += 1
                job = future_map[fut]
                label = f"{job['candidate']} vs {job['opponent_type']}:{job['opponent']}"
                try:
                    rows.append(fut.result())
                    print(f"[{completed}/{total}] OK  {label}")
                except Exception as e:
                    failures.append({"job": job, "error": str(e)})
                    print(f"[{completed}/{total}] FAIL {label}\n{e}\n")
    return rows, failures


def prefilter_compatible_checkpoints(checkpoints, config_path: str):
    """
    Pre-check checkpoint compatibility by trying model-only load once per checkpoint.
    This avoids flooding stage evaluation with repeated load failures.
    """
    from utils import loadConfig
    from dreamer import Dreamer
    import hockey.hockey_env as h_env
    from hockey.hockey_env import Mode
    import torch

    cfg = loadConfig(config_path)
    device = torch.device("cpu")

    env = h_env.HockeyEnv(mode=Mode.NORMAL, keep_mode=True)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0] // 2
    action_low = env.action_space.low[:action_size].tolist()
    action_high = env.action_space.high[:action_size].tolist()

    agent = Dreamer(observation_size, action_size, action_low, action_high, device, cfg.dreamer)
    compatible = []
    skipped = []

    for cp in checkpoints:
        try:
            agent.loadCheckpoint(cp, load_optimizers=False)
            compatible.append(cp)
        except Exception as e:
            msg = str(e).strip().splitlines()[0] if str(e).strip() else "unknown load error"
            skipped.append({"checkpoint": cp, "error": msg})

    env.close()
    return compatible, skipped


def write_rows_csv(path: Path, rows):
    fieldnames = [
        "stage",
        "candidate",
        "candidate_path",
        "opponent_type",
        "opponent",
        "opponent_path",
        "episodes",
        "seed",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "wins",
        "losses",
        "draws",
        "avg_reward",
        "reward_std",
        "avg_steps",
        "command",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def quantile(vals, q):
    if not vals:
        return float("nan")
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q))


def build_stage1_jobs(checkpoints, args):
    rng = random.Random(args.sample_seed)
    n = len(checkpoints)
    jobs = []
    idx_map = {p: i for i, p in enumerate(checkpoints)}

    # Fixed bot jobs for all checkpoints.
    for cp in checkpoints:
        name = Path(cp).name
        for fixed_opp in ("weak", "strong"):
            jobs.append(
                {
                    "stage": "stage1",
                    "candidate": name,
                    "candidate_path": cp,
                    "opponent_type": "fixed",
                    "opponent": fixed_opp,
                    "opponent_path": fixed_opp,
                    "episodes": args.episodes_fixed_stage1,
                    "fixed_opponent": fixed_opp,
                }
            )

    # Sparse directed cp-vs-cp jobs.
    for cp in checkpoints:
        i = idx_map[cp]
        others = [x for x in checkpoints if x != cp]
        forced = set()

        span = max(0, args.stage1_neighbor_span)
        for d in range(1, span + 1):
            j1 = i + d
            j2 = i - d
            if j1 < n:
                forced.add(checkpoints[j1])
            if j2 >= 0:
                forced.add(checkpoints[j2])

        remain = [x for x in others if x not in forced]
        k_total = min(len(others), max(1, args.stage1_opponents_per_candidate))
        k_extra = max(0, k_total - len(forced))
        sampled = rng.sample(remain, k=min(k_extra, len(remain)))
        targets = sorted(list(forced) + sampled, key=step_key)

        for opp in targets:
            jobs.append(
                {
                    "stage": "stage1",
                    "candidate": Path(cp).name,
                    "candidate_path": cp,
                    "opponent_type": "checkpoint",
                    "opponent": Path(opp).name,
                    "opponent_path": opp,
                    "episodes": args.episodes_cp_stage1,
                }
            )
    return jobs


def summarize_stage_rows(rows):
    by_candidate = {}
    for r in rows:
        by_candidate.setdefault(r["candidate"], []).append(r)

    summaries = []
    for cand_name, cand_rows in by_candidate.items():
        fixed_rows = [r for r in cand_rows if r["opponent_type"] == "fixed"]
        cp_rows = [r for r in cand_rows if r["opponent_type"] == "checkpoint"]

        weak_wr = next((r["win_rate"] for r in fixed_rows if r["opponent"] == "weak"), float("nan"))
        strong_wr = next((r["win_rate"] for r in fixed_rows if r["opponent"] == "strong"), float("nan"))
        fixed_combined = statistics.mean([x for x in (weak_wr, strong_wr) if x == x]) if fixed_rows else float("nan")

        cp_winrates = [r["win_rate"] for r in cp_rows]
        cp_mean = statistics.mean(cp_winrates) if cp_winrates else float("nan")
        cp_worst = min(cp_winrates) if cp_winrates else float("nan")
        cp_q20 = quantile(cp_winrates, 0.20)
        cp_std = statistics.pstdev(cp_winrates) if len(cp_winrates) > 1 else 0.0

        stage_quality = (
            (0.55 * cp_q20 if cp_q20 == cp_q20 else 0.0)
            + (0.25 * cp_mean if cp_mean == cp_mean else 0.0)
            + (0.20 * fixed_combined if fixed_combined == fixed_combined else 0.0)
        )

        summaries.append(
            {
                "candidate": cand_name,
                "candidate_path": next(r["candidate_path"] for r in cand_rows),
                "fixed_weak_wr": weak_wr,
                "fixed_strong_wr": strong_wr,
                "fixed_combined_wr": fixed_combined,
                "cp_mean_wr": cp_mean,
                "cp_worst_wr": cp_worst,
                "cp_q20_wr": cp_q20,
                "cp_std_wr": cp_std,
                "cp_matchups": len(cp_rows),
                "stage_quality": stage_quality,
            }
        )
    summaries.sort(key=lambda x: x["stage_quality"], reverse=True)
    return summaries


def build_stage2_jobs(top_paths, args):
    jobs = []
    # Fixed jobs on refined set.
    for cp in top_paths:
        name = Path(cp).name
        for fixed_opp in ("weak", "strong"):
            jobs.append(
                {
                    "stage": "stage2",
                    "candidate": name,
                    "candidate_path": cp,
                    "opponent_type": "fixed",
                    "opponent": fixed_opp,
                    "opponent_path": fixed_opp,
                    "episodes": args.episodes_fixed_stage2,
                    "fixed_opponent": fixed_opp,
                }
            )

    if args.stage2_bidirectional:
        for a in top_paths:
            for b in top_paths:
                if a == b:
                    continue
                jobs.append(
                    {
                        "stage": "stage2",
                        "candidate": Path(a).name,
                        "candidate_path": a,
                        "opponent_type": "checkpoint",
                        "opponent": Path(b).name,
                        "opponent_path": b,
                        "episodes": args.episodes_cp_stage2,
                    }
                )
    else:
        for i, a in enumerate(top_paths):
            for j in range(i + 1, len(top_paths)):
                b = top_paths[j]
                jobs.append(
                    {
                        "stage": "stage2",
                        "candidate": Path(a).name,
                        "candidate_path": a,
                        "opponent_type": "checkpoint",
                        "opponent": Path(b).name,
                        "opponent_path": b,
                        "episodes": args.episodes_cp_stage2,
                    }
                )
    return jobs


def summarize_stage2_with_pairing(rows, top_names):
    by_candidate = {}
    pair_map = {}
    for r in rows:
        by_candidate.setdefault(r["candidate"], []).append(r)
        if r["opponent_type"] == "checkpoint":
            pair_map[(r["candidate"], r["opponent"])] = r

    summaries = []
    pair_scores = {c: {} for c in top_names}

    for cand in top_names:
        cand_rows = by_candidate.get(cand, [])
        fixed_rows = [r for r in cand_rows if r["opponent_type"] == "fixed"]

        weak_wr = next((r["win_rate"] for r in fixed_rows if r["opponent"] == "weak"), float("nan"))
        strong_wr = next((r["win_rate"] for r in fixed_rows if r["opponent"] == "strong"), float("nan"))
        fixed_combined = statistics.mean([x for x in (weak_wr, strong_wr) if x == x]) if fixed_rows else float("nan")

        cp_scores = []
        for opp in top_names:
            if opp == cand:
                continue
            ab = pair_map.get((cand, opp))
            ba = pair_map.get((opp, cand))
            if ab and ba:
                # Side-bias aware estimate from both directions:
                # candidate win rate as player-1 + candidate win rate as player-2 (opp losses).
                s = 0.5 * (ab["win_rate"] + ba["loss_rate"])
            elif ab:
                s = ab["win_rate"]
            elif ba:
                s = ba["loss_rate"]
            else:
                continue
            pair_scores[cand][opp] = s
            cp_scores.append(s)

        cp_mean = statistics.mean(cp_scores) if cp_scores else float("nan")
        cp_worst = min(cp_scores) if cp_scores else float("nan")
        cp_q20 = quantile(cp_scores, 0.20)
        cp_std = statistics.pstdev(cp_scores) if len(cp_scores) > 1 else 0.0

        final_quality = (
            (0.60 * cp_q20 if cp_q20 == cp_q20 else 0.0)
            + (0.25 * cp_mean if cp_mean == cp_mean else 0.0)
            + (0.15 * fixed_combined if fixed_combined == fixed_combined else 0.0)
        )

        summaries.append(
            {
                "candidate": cand,
                "candidate_path": next((r["candidate_path"] for r in cand_rows), ""),
                "fixed_weak_wr": weak_wr,
                "fixed_strong_wr": strong_wr,
                "fixed_combined_wr": fixed_combined,
                "cp_mean_wr": cp_mean,
                "cp_worst_wr": cp_worst,
                "cp_q20_wr": cp_q20,
                "cp_std_wr": cp_std,
                "cp_matchups": len(cp_scores),
                "final_quality": final_quality,
            }
        )
    summaries.sort(key=lambda x: x["final_quality"], reverse=True)
    return summaries, pair_scores


def minmax_norm(values_by_key):
    vals = list(values_by_key.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in values_by_key}
    return {k: (v - lo) / (hi - lo) for k, v in values_by_key.items()}


def cosine_distance(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def recommend_pool(summaries, pair_scores, pool_size, quality_weight, diversity_weight):
    names = [s["candidate"] for s in summaries]
    name_to_summary = {s["candidate"]: s for s in summaries}
    quality_raw = {s["candidate"]: s["final_quality"] for s in summaries}
    quality_norm = minmax_norm(quality_raw)

    # Build dense pair-score vectors for diversity. Missing entries -> candidate mean score.
    vecs = {}
    all_opps = names[:]
    for c in names:
        vals = []
        known = list(pair_scores.get(c, {}).values())
        fill = float(np.mean(known)) if known else 0.5
        for o in all_opps:
            if o == c:
                vals.append(0.5)
            else:
                vals.append(pair_scores.get(c, {}).get(o, fill))
        vecs[c] = np.asarray(vals, dtype=np.float64)

    selected = []
    records = []
    target = min(pool_size, len(names))
    remaining = set(names)

    while len(selected) < target:
        best = None
        best_score = -1e9
        best_div = 0.0
        for c in sorted(remaining):
            q = quality_norm[c]
            if not selected:
                d = 1.0
            else:
                d = min(cosine_distance(vecs[c], vecs[s]) for s in selected)
            score = quality_weight * q + diversity_weight * d
            if score > best_score:
                best_score = score
                best = c
                best_div = d
        selected.append(best)
        remaining.remove(best)
        records.append(
            {
                "rank": len(selected),
                "candidate": best,
                "candidate_path": name_to_summary[best]["candidate_path"],
                "final_quality": quality_raw[best],
                "quality_norm": quality_norm[best],
                "min_diversity_to_selected": best_div,
                "blend_score": best_score,
                "cp_q20_wr": name_to_summary[best]["cp_q20_wr"],
                "cp_worst_wr": name_to_summary[best]["cp_worst_wr"],
                "cp_mean_wr": name_to_summary[best]["cp_mean_wr"],
                "fixed_combined_wr": name_to_summary[best]["fixed_combined_wr"],
            }
        )
    return records


def write_summary_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent  # DreamerV3/
    test_script = root / "test_hockey.py"

    if args.checkpoints_file:
        checkpoints = read_checkpoint_list(Path(args.checkpoints_file).expanduser().resolve())
    else:
        checkpoints = discover_checkpoints_from_dir((root / args.checkpoints_dir).resolve())

    if len(checkpoints) < 3:
        raise RuntimeError(f"Need at least 3 checkpoints, found {len(checkpoints)}")
    for p in checkpoints:
        if not Path(p).exists():
            raise RuntimeError(f"Checkpoint not found: {p}")

    original_count = len(checkpoints)
    skipped_incompatible = []
    if args.skip_incompatible:
        try:
            checkpoints, skipped_incompatible = prefilter_compatible_checkpoints(checkpoints, args.config)
            print(
                f"Compatibility pre-check: kept {len(checkpoints)}/{original_count}, "
                f"skipped {len(skipped_incompatible)} incompatible"
            )
        except Exception as e:
            print(f"Warning: compatibility pre-check failed, continuing without filter: {e}")
            checkpoints = checkpoints
            skipped_incompatible = []

    if len(checkpoints) < 3:
        raise RuntimeError(
            f"Need at least 3 compatible checkpoints after filtering, found {len(checkpoints)} "
            f"(from {original_count} total)"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (root / args.output_dir).resolve() / f"pool_select_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = out_dir / "plan.txt"
    with plan.open("w") as f:
        f.write(f"Checkpoints compatible ({len(checkpoints)}/{original_count}):\n")
        for p in checkpoints:
            f.write(f"- {p}\n")
        if skipped_incompatible:
            f.write(f"\nSkipped incompatible ({len(skipped_incompatible)}):\n")
            for item in skipped_incompatible:
                f.write(f"- {item['checkpoint']} :: {item['error']}\n")
        f.write("\nArgs:\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"- {k}: {v}\n")

    print(f"Output directory: {out_dir}")
    print(f"Checkpoints discovered: {original_count}")
    print(f"Checkpoints usable: {len(checkpoints)}")

    if skipped_incompatible:
        skipped_csv = out_dir / "skipped_incompatible.csv"
        with skipped_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["checkpoint", "error"])
            writer.writeheader()
            for item in skipped_incompatible:
                writer.writerow(item)
        print(f"Skipped incompatible list: {skipped_csv}")

    # Stage 1
    stage1_jobs = build_stage1_jobs(checkpoints, args)
    print(f"Stage1 jobs: {len(stage1_jobs)}")
    if args.dry_run:
        print("Dry run complete.")
        return

    stage1_rows, stage1_fail = run_jobs(
        stage1_jobs,
        test_script=test_script,
        config=args.config,
        seed=args.seed,
        device=args.device,
        workers=args.max_workers,
    )
    write_rows_csv(out_dir / "stage1_matches.csv", stage1_rows)

    if stage1_fail:
        with (out_dir / "stage1_failures.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["candidate", "candidate_path", "opponent_type", "opponent", "opponent_path", "episodes", "error"],
            )
            writer.writeheader()
            for item in stage1_fail:
                j = item["job"]
                writer.writerow(
                    {
                        "candidate": j["candidate"],
                        "candidate_path": j["candidate_path"],
                        "opponent_type": j["opponent_type"],
                        "opponent": j["opponent"],
                        "opponent_path": j["opponent_path"],
                        "episodes": j["episodes"],
                        "error": item["error"],
                    }
                )

    stage1_summary = summarize_stage_rows(stage1_rows)
    write_summary_csv(
        out_dir / "stage1_summary.csv",
        stage1_summary,
        [
            "candidate",
            "candidate_path",
            "fixed_weak_wr",
            "fixed_strong_wr",
            "fixed_combined_wr",
            "cp_mean_wr",
            "cp_worst_wr",
            "cp_q20_wr",
            "cp_std_wr",
            "cp_matchups",
            "stage_quality",
        ],
    )

    top_k = min(args.stage2_top_k, len(stage1_summary))
    top_rows = stage1_summary[:top_k]
    top_paths = [r["candidate_path"] for r in top_rows]
    top_names = [Path(p).name for p in top_paths]
    with (out_dir / "stage2_candidates.txt").open("w") as f:
        for p in top_paths:
            f.write(f"{p}\n")

    final_summary = None
    pair_scores = None

    if args.skip_stage2:
        # Reuse stage1 scores for recommendation if requested.
        final_summary = []
        for r in top_rows:
            final_summary.append(
                {
                    **r,
                    "final_quality": r["stage_quality"],
                }
            )
        # Pair score proxy from stage1 directed rows.
        pair_scores = {n: {} for n in top_names}
        for row in stage1_rows:
            if row["opponent_type"] != "checkpoint":
                continue
            c = row["candidate"]
            o = row["opponent"]
            if c in pair_scores and o in pair_scores:
                pair_scores[c][o] = row["win_rate"]
    else:
        stage2_jobs = build_stage2_jobs(top_paths, args)
        print(f"Stage2 candidates: {len(top_paths)} | Stage2 jobs: {len(stage2_jobs)}")
        stage2_rows, stage2_fail = run_jobs(
            stage2_jobs,
            test_script=test_script,
            config=args.config,
            seed=args.seed,
            device=args.device,
            workers=args.max_workers,
        )
        write_rows_csv(out_dir / "stage2_matches.csv", stage2_rows)
        if stage2_fail:
            with (out_dir / "stage2_failures.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["candidate", "candidate_path", "opponent_type", "opponent", "opponent_path", "episodes", "error"],
                )
                writer.writeheader()
                for item in stage2_fail:
                    j = item["job"]
                    writer.writerow(
                        {
                            "candidate": j["candidate"],
                            "candidate_path": j["candidate_path"],
                            "opponent_type": j["opponent_type"],
                            "opponent": j["opponent"],
                            "opponent_path": j["opponent_path"],
                            "episodes": j["episodes"],
                            "error": item["error"],
                        }
                    )

        final_summary, pair_scores = summarize_stage2_with_pairing(stage2_rows, top_names)
        write_summary_csv(
            out_dir / "final_summary.csv",
            final_summary,
            [
                "candidate",
                "candidate_path",
                "fixed_weak_wr",
                "fixed_strong_wr",
                "fixed_combined_wr",
                "cp_mean_wr",
                "cp_worst_wr",
                "cp_q20_wr",
                "cp_std_wr",
                "cp_matchups",
                "final_quality",
            ],
        )

    selected = recommend_pool(
        summaries=final_summary,
        pair_scores=pair_scores,
        pool_size=args.pool_size,
        quality_weight=args.quality_weight,
        diversity_weight=args.diversity_weight,
    )

    write_summary_csv(
        out_dir / "recommended_pool.csv",
        selected,
        [
            "rank",
            "candidate",
            "candidate_path",
            "final_quality",
            "quality_norm",
            "min_diversity_to_selected",
            "blend_score",
            "cp_q20_wr",
            "cp_worst_wr",
            "cp_mean_wr",
            "fixed_combined_wr",
        ],
    )

    with (out_dir / "recommended_pool.txt").open("w") as f:
        for r in selected:
            f.write(f"{r['candidate_path']}\n")

    md = out_dir / "recommended_pool.md"
    with md.open("w") as f:
        f.write("# Recommended Self-Play Pool\n\n")
        f.write(f"Selected {len(selected)} checkpoints.\n\n")
        f.write("| Rank | Checkpoint | Blend | Quality | Diversity | CP Q20 | CP Worst | CP Mean | Fixed Combined |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in selected:
            f.write(
                f"| {r['rank']} | `{r['candidate']}` | {r['blend_score']:.4f} | {r['final_quality']:.4f} | "
                f"{r['min_diversity_to_selected']:.4f} | {r['cp_q20_wr']:.4f} | {r['cp_worst_wr']:.4f} | "
                f"{r['cp_mean_wr']:.4f} | {r['fixed_combined_wr']:.4f} |\n"
            )
        f.write("\n## Usage\n\n")
        f.write("Use `recommended_pool.txt` as the league/probe file for training.\n")

    print("Done.")
    print(f"- Stage1 matches: {out_dir / 'stage1_matches.csv'}")
    print(f"- Stage1 summary: {out_dir / 'stage1_summary.csv'}")
    if not args.skip_stage2:
        print(f"- Stage2 matches: {out_dir / 'stage2_matches.csv'}")
        print(f"- Final summary: {out_dir / 'final_summary.csv'}")
    print(f"- Recommended pool paths: {out_dir / 'recommended_pool.txt'}")
    print(f"- Recommended pool report: {out_dir / 'recommended_pool.md'}")


if __name__ == "__main__":
    main()
