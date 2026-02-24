#!/usr/bin/env python3
"""Prepare a clean league directory from recommended_pool.csv.

Creates symlinks/copies for selected checkpoints and writes helper files:
- league_checkpoints.txt
- league_weights.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare league directory from recommended pool CSV")
    p.add_argument("--recommended-csv", type=Path, required=True,
                   help="Path to recommended_pool.csv")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for curated league pool")
    p.add_argument("--top-k", type=int, default=0,
                   help="Keep only top-k rows (0 = all)")
    p.add_argument("--source-dir", type=Path, default=None,
                   help="Optional fallback directory to resolve checkpoints by basename")
    p.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink",
                   help="How to materialize files in output dir")
    return p.parse_args()


def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    rec_csv = args.recommended_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir.expanduser().resolve() if args.source_dir else None

    rows = []
    with rec_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if args.top_k and args.top_k > 0:
        rows = rows[: args.top_k]

    if not rows:
        raise RuntimeError(f"No rows found in {rec_csv}")

    league_rows = []

    for row in rows:
        src = Path(row["candidate_path"]).expanduser().resolve()
        if not src.exists() and source_dir is not None:
            alt = source_dir / src.name
            if alt.exists():
                src = alt.resolve()
        if not src.exists():
            raise FileNotFoundError(
                f"Missing checkpoint from CSV: {src}"
                + (f" (and not found in source-dir {source_dir})" if source_dir else "")
            )

        rank = int(row["rank"])
        name = src.name
        dst = out_dir / name
        materialize(src, dst, args.link_mode)

        league_rows.append(
            {
                "rank": rank,
                "checkpoint": name,
                "checkpoint_path": str(dst),
                "source_path": str(src),
                "blend_score": float(row.get("blend_score", 1.0)),
                "quality_norm": float(row.get("quality_norm", 1.0)),
                "final_quality": float(row.get("final_quality", 1.0)),
            }
        )

    # Preserve rank order
    league_rows.sort(key=lambda r: r["rank"])

    checkpoints_txt = out_dir / "league_checkpoints.txt"
    with checkpoints_txt.open("w") as f:
        for r in league_rows:
            f.write(f"{r['checkpoint_path']}\n")

    weights_csv = out_dir / "league_weights.csv"
    with weights_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "checkpoint",
                "checkpoint_path",
                "source_path",
                "blend_score",
                "quality_norm",
                "final_quality",
            ],
        )
        writer.writeheader()
        for r in league_rows:
            writer.writerow(r)

    print(f"Prepared league pool: {out_dir}")
    print(f"Selected checkpoints: {len(league_rows)}")
    print(f"Checkpoints list: {checkpoints_txt}")
    print(f"Weights CSV: {weights_csv}")


if __name__ == "__main__":
    main()
