#!/usr/bin/env python3
"""
Summit evaluation matrix runner.

Runs:
1) Fixed-bot benchmark per candidate (weak + strong)
2) Cross-play matrix: candidates vs opponent checkpoints

Outputs:
- all_matches.csv
- candidate_summary.csv
- candidate_ranking.md
"""

import argparse
import csv
import datetime as dt
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run checkpoint evaluation matrix for summit push.")
    parser.add_argument(
        "--candidates-dir",
        type=str,
        default="checkpoints/summit-evaluation/candidates",
        help="Directory containing candidate checkpoints",
    )
    parser.add_argument(
        "--opponents-dir",
        type=str,
        default="checkpoints/summit-evaluation/opponents",
        help="Directory containing opponent checkpoints",
    )
    parser.add_argument(
        "--candidates-file",
        type=str,
        default=None,
        help="Optional newline-separated candidate checkpoint list (overrides candidates-dir)",
    )
    parser.add_argument(
        "--opponents-file",
        type=str,
        default=None,
        help="Optional newline-separated opponent checkpoint list (overrides opponents-dir)",
    )
    parser.add_argument("--episodes-fixed", type=int, default=100, help="Episodes for weak/strong fixed eval")
    parser.add_argument("--episodes-checkpoint", type=int, default=30, help="Episodes per checkpoint-vs-checkpoint eval")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (cuda/mps/cpu)")
    parser.add_argument("--config", type=str, default="hockey.yml", help="Config file passed to test_hockey.py")
    parser.add_argument(
        "--include-self-match",
        action="store_true",
        help="Include candidate vs same-named opponent checkpoint matchups",
    )
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional cap for candidates")
    parser.add_argument("--max-opponents", type=int, default=None, help="Optional cap for opponents")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/summit-evaluation/results",
        help="Directory for CSV/Markdown outputs",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent evaluation workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining jobs if one evaluation fails; write failures to failed_jobs.csv",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only")
    return parser.parse_args()


def read_checkpoint_list_from_file(path: Path):
    checkpoints = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        checkpoints.append(str(Path(line).expanduser()))
    return checkpoints


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


def parse_eval_output(stdout: str):
    win_m = re.search(r"Win rate:\s+([0-9.]+)%\s+\((\d+)\s+wins\)", stdout)
    loss_m = re.search(r"Loss rate:\s+([0-9.]+)%\s+\((\d+)\s+losses\)", stdout)
    draw_m = re.search(r"Draw rate:\s+([0-9.]+)%\s+\((\d+)\s+draws\)", stdout)
    reward_m = re.search(r"Reward:\s+([+-]?[0-9.]+)\s+\+/-\s+([0-9.]+)", stdout)
    steps_m = re.search(r"Avg steps:\s+([0-9.]+)", stdout)
    if not (win_m and loss_m and draw_m and reward_m and steps_m):
        raise RuntimeError("Could not parse test_hockey.py output:\n" + stdout[-1500:])
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


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    test_script = root / "test_hockey.py"
    output_root = (root / args.output_dir).resolve()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"matrix_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.candidates_file:
        candidates = read_checkpoint_list_from_file(Path(args.candidates_file).expanduser().resolve())
    else:
        candidates = discover_checkpoints_from_dir((root / args.candidates_dir).resolve())

    if args.opponents_file:
        opponents = read_checkpoint_list_from_file(Path(args.opponents_file).expanduser().resolve())
    else:
        opponents = discover_checkpoints_from_dir((root / args.opponents_dir).resolve())

    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if args.max_opponents is not None:
        opponents = opponents[: args.max_opponents]

    if not candidates:
        raise RuntimeError("No candidate checkpoints found.")
    if not opponents:
        raise RuntimeError("No opponent checkpoints found.")

    for p in candidates + opponents:
        if not Path(p).exists():
            raise RuntimeError(f"Checkpoint does not exist: {p}")

    plan_txt = out_dir / "plan.txt"
    with plan_txt.open("w") as f:
        f.write("Candidates:\n")
        for c in candidates:
            f.write(f"- {c}\n")
        f.write("\nOpponents:\n")
        for o in opponents:
            f.write(f"- {o}\n")
        f.write(
            f"\nEpisodes fixed: {args.episodes_fixed}\nEpisodes checkpoint: {args.episodes_checkpoint}\nSeed: {args.seed}\n"
        )

    jobs = []

    print(f"Output directory: {out_dir}")
    
    for cand in candidates:
        cand_name = Path(cand).name

        for fixed_opp in ("weak", "strong"):
            jobs.append(
                {
                    "candidate": cand_name,
                    "candidate_path": cand,
                    "opponent_type": "fixed",
                    "opponent": fixed_opp,
                    "opponent_path": fixed_opp,
                    "episodes": args.episodes_fixed,
                    "fixed_opponent": fixed_opp,
                }
            )

        for opp in opponents:
            opp_name = Path(opp).name
            if (not args.include_self_match) and cand_name == opp_name:
                continue
            jobs.append(
                {
                    "candidate": cand_name,
                    "candidate_path": cand,
                    "opponent_type": "checkpoint",
                    "opponent": opp_name,
                    "opponent_path": opp,
                    "episodes": args.episodes_checkpoint,
                }
            )

    total_jobs = len(jobs)
    print(
        f"Candidates: {len(candidates)} | Opponents: {len(opponents)} | Jobs: {total_jobs} | "
        f"Workers: {max(1, args.max_workers)}"
    )

    if args.dry_run:
        for i, j in enumerate(jobs, start=1):
            if j["opponent_type"] == "fixed":
                print(f"[{i}/{total_jobs}] {j['candidate']} vs fixed:{j['opponent']}")
            else:
                print(f"[{i}/{total_jobs}] {j['candidate']} vs cp:{j['opponent']}")
        print("Dry run complete.")
        return

    rows = []
    failures = []

    if args.max_workers <= 1:
        for i, job in enumerate(jobs, start=1):
            label = f"{job['candidate']} vs " + (
                f"fixed:{job['opponent']}" if job["opponent_type"] == "fixed" else f"cp:{job['opponent']}"
            )
            print(f"[{i}/{total_jobs}] {label}")
            try:
                row = run_one_job(
                    test_script=test_script,
                    config=args.config,
                    seed=args.seed,
                    device=args.device,
                    job=job,
                )
                rows.append(row)
            except Exception as e:
                failures.append({"job": job, "error": str(e)})
                print(f"FAILED: {label}\n{e}\n")
                if not args.continue_on_error:
                    raise
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            future_map = {
                ex.submit(
                    run_one_job,
                    test_script,
                    args.config,
                    args.seed,
                    args.device,
                    job,
                ): job
                for job in jobs
            }
            completed = 0
            for fut in as_completed(future_map):
                completed += 1
                job = future_map[fut]
                label = f"{job['candidate']} vs " + (
                    f"fixed:{job['opponent']}" if job["opponent_type"] == "fixed" else f"cp:{job['opponent']}"
                )
                try:
                    row = fut.result()
                    rows.append(row)
                    print(f"[{completed}/{total_jobs}] OK  {label}")
                except Exception as e:
                    failures.append({"job": job, "error": str(e)})
                    print(f"[{completed}/{total_jobs}] FAIL {label}\n{e}\n")
                    if not args.continue_on_error:
                        raise

    if failures:
        failed_csv = out_dir / "failed_jobs.csv"
        with failed_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "candidate",
                    "candidate_path",
                    "opponent_type",
                    "opponent",
                    "opponent_path",
                    "episodes",
                    "error",
                ],
            )
            writer.writeheader()
            for item in failures:
                writer.writerow(
                    {
                        "candidate": item["job"]["candidate"],
                        "candidate_path": item["job"]["candidate_path"],
                        "opponent_type": item["job"]["opponent_type"],
                        "opponent": item["job"]["opponent"],
                        "opponent_path": item["job"]["opponent_path"],
                        "episodes": item["job"]["episodes"],
                        "error": item["error"],
                    }
                )
        print(f"Warnings: {len(failures)} failed jobs written to {failed_csv}")

    # Defensive re-create in case output directory was removed during long runs.
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = out_dir / "all_matches.csv"
    fieldnames = [
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
    with all_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

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
        cp_mean = statistics.mean([r["win_rate"] for r in cp_rows]) if cp_rows else float("nan")
        cp_worst = min([r["win_rate"] for r in cp_rows]) if cp_rows else float("nan")
        cp_std = statistics.pstdev([r["win_rate"] for r in cp_rows]) if len(cp_rows) > 1 else 0.0
        robust_score = (
            (0.60 * cp_worst if cp_worst == cp_worst else 0.0)
            + (0.25 * fixed_combined if fixed_combined == fixed_combined else 0.0)
            + (0.15 * cp_mean if cp_mean == cp_mean else 0.0)
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
                "cp_std_wr": cp_std,
                "cp_matchups": len(cp_rows),
                "robust_score": robust_score,
            }
        )

    summaries.sort(key=lambda x: x["robust_score"], reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "candidate_summary.csv"
    summary_fields = [
        "candidate",
        "candidate_path",
        "fixed_weak_wr",
        "fixed_strong_wr",
        "fixed_combined_wr",
        "cp_mean_wr",
        "cp_worst_wr",
        "cp_std_wr",
        "cp_matchups",
        "robust_score",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_md = out_dir / "candidate_ranking.md"
    with ranking_md.open("w") as f:
        f.write("# Candidate Ranking\n\n")
        f.write("| Rank | Candidate | Robust Score | Fixed Combined | CP Mean | CP Worst | CP Std |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for idx, s in enumerate(summaries, start=1):
            f.write(
                f"| {idx} | `{s['candidate']}` | {s['robust_score']:.4f} | {s['fixed_combined_wr']:.4f} | "
                f"{s['cp_mean_wr']:.4f} | {s['cp_worst_wr']:.4f} | {s['cp_std_wr']:.4f} |\n"
            )

    print("Done.")
    print(f"- Raw matches: {all_csv}")
    print(f"- Candidate summary: {summary_csv}")
    print(f"- Ranking markdown: {ranking_md}")


if __name__ == "__main__":
    main()
