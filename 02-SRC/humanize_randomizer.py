#!/usr/bin/env python3
"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

# Produces a random "rulebook" for humanizing code. Run before editing a file;
# the output config guides spacing, comment style, loop style, etc.
# Makes code look like a student or human wrote it: inconsistent, varied, not polished.
#
# Usage:
#   python humanize_randomizer.py -f path/to/file.py -o rulebook.json
#   python humanize_randomizer.py -f path/to/file.py --rulebook -o RULEBOOK.md
#   python humanize_randomizer.py --seed 42   # fixed seed for reproducibility

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Generate random humanization rulebook")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--output", "-o", type=Path, default=None, help="Write config to file (default: stdout)")
    p.add_argument("--file", "-f", type=Path, default=None, help="Target file (used for file-specific seed if no --seed)")
    p.add_argument("--rulebook", action="store_true", help="Output human-readable rulebook instead of JSON")
    return p.parse_args()


def make_spacing_weights(rng: random.Random) -> list[tuple[int, float]]:
    """Weights for 0, 1, 2, 3 blank lines between blocks. Most humans use 1 or 2, sometimes 0, rarely 3."""
    # base: (num_lines, weight) - weights are relative, we normalize
    options = [
        (0, rng.uniform(0.05, 0.25)),   # no blank - cramped, happens sometimes
        (1, rng.uniform(0.35, 0.65)),   # one blank - most common
        (2, rng.uniform(0.15, 0.45)),   # two blanks - also common
        (3, rng.uniform(0.02, 0.12)),   # three - rare, "breather" sections
    ]
    total = sum(w for _, w in options)
    return [(n, w / total) for n, w in options]


def make_spacing_sequence(rng: random.Random, length: int, weights: list[tuple[int, float]]) -> list[int]:
    """Generate a sequence of blank-line counts for `length` gaps between blocks."""
    nums, probs = zip(*weights)
    return rng.choices(nums, weights=probs, k=length)


def main():
    args = parse_args()

    if args.seed is not None:
        seed = args.seed
    elif args.file is not None:
        # deterministic per file: hash of path + size
        seed = hash((str(args.file.resolve()), args.file.stat().st_size)) % (2**32)
    else:
        seed = random.randrange(2**32)

    rng = random.Random(seed)

    # -------------------------------------------------------------------------
    # Comment frequency: 1-10. Low = sparse comments, high = chatty.
    # Students vary a lot here.
    comment_frequency = rng.randint(1, 10)

    # Comment length: 1-10. Low = terse ("# init"), high = longer explanations.
    comment_length = rng.randint(1, 10)

    # Prefer explicit for-loops over comprehensions. 0-1. Higher = more explicit.
    # Students often write explicit loops; experienced devs use comprehensions.
    prefer_explicit_loops = rng.uniform(0.4, 0.95)

    # Use intermediate variables vs inline. 0-1. Higher = more intermediate vars.
    # Students sometimes break things out more.
    prefer_intermediate_vars = rng.uniform(0.3, 0.85)

    # Spacing weights
    spacing_weights = make_spacing_weights(rng)

    # Pre-generate a spacing sequence for up to 100 block boundaries.
    # Editor consumes from this when inserting blanks.
    spacing_sequence = make_spacing_sequence(rng, 100, spacing_weights)

    # Occasional "oops" extra blank: small prob of inserting 1 extra somewhere.
    # Mimics accidental double-enter.
    occasional_extra_blank_prob = rng.uniform(0.0, 0.08)

    # Comment style: 1-10. Low = very terse, high = more explanatory.
    # Correlated with comment_length but can differ.
    comment_verbosity = rng.randint(1, 10)

    # Section separator style. 0 = none, 1 = # ---, 2 = # ===, 3 = ### lines
    section_separator = rng.choice([0, 0, 1, 1, 2, 3])  # 0 and 1 most common

    # Trailing commas: 0 = no, 1 = sometimes, 2 = yes when multi-line
    trailing_comma_style = rng.choice([0, 1, 1, 2])

    # Bracket/paren style: "same line" vs "next line" for long lines.
    # 0 = prefer same line, 1 = mixed, 2 = prefer next line
    bracket_style = rng.choice([0, 1, 1, 2])

    # Tiny prob of a casual/typo in a comment (student notes feel).
    # 0 = never, 0.02-0.08 = rare
    casual_comment_prob = rng.uniform(0.0, 0.06)

    config = {
        "seed": seed,
        "comment_frequency": comment_frequency,
        "comment_length": comment_length,
        "comment_verbosity": comment_verbosity,
        "prefer_explicit_loops": round(prefer_explicit_loops, 3),
        "prefer_intermediate_vars": round(prefer_intermediate_vars, 3),
        "spacing_weights": [(n, round(w, 4)) for n, w in spacing_weights],
        "spacing_sequence": spacing_sequence,
        "occasional_extra_blank_prob": round(occasional_extra_blank_prob, 3),
        "casual_comment_prob": round(casual_comment_prob, 3),
        "section_separator": section_separator,
        "trailing_comma_style": trailing_comma_style,
        "bracket_style": bracket_style,
    }

    if args.rulebook:
        out = format_rulebook(config)
    else:
        out = json.dumps(config, indent=2)

    if args.output:
        args.output.write_text(out)
        print(f"Wrote rulebook to {args.output} (seed={seed})", file=sys.stderr)
    else:
        print(out)


def format_rulebook(config: dict) -> str:
    """Turn config into a markdown rulebook for the editor to follow."""
    lines = [
        "# Humanization Rulebook",
        "",
        f"Seed: {config['seed']}",
        "",
        "## Spacing",
        "- Between logical blocks, use the spacing_sequence in order (0, 1, 2, or 3 blank lines).",
        f"- Weights: {config['spacing_weights']}",
        f"- Occasionally add 1 extra blank line with prob {config['occasional_extra_blank_prob']} (accidental double-enter feel).",
        "",
        "## Comments",
        f"- Casual/typo in comment with prob {config.get('casual_comment_prob', 0)} (optional, student-notes feel).",
        f"- Frequency (1-10): {config['comment_frequency']} — {'sparse' if config['comment_frequency'] <= 3 else 'moderate' if config['comment_frequency'] <= 6 else 'chatty'}",
        f"- Length (1-10): {config['comment_length']} — {'terse' if config['comment_length'] <= 3 else 'medium' if config['comment_length'] <= 6 else 'explanatory'}",
        f"- Verbosity (1-10): {config['comment_verbosity']}",
        "",
        "## Code Style",
        f"- Prefer explicit for-loops over comprehensions: {config['prefer_explicit_loops']*100:.0f}% of the time",
        f"- Prefer intermediate variables over long inline expressions: {config['prefer_intermediate_vars']*100:.0f}% of the time",
        f"- Section separators: {['none', '# ---', '# ===', '### lines'][config['section_separator']]}",
        f"- Trailing commas: {['no', 'sometimes', 'yes when multi-line'][config['trailing_comma_style']]}",
        f"- Bracket style: {['same line', 'mixed', 'next line'][config['bracket_style']]}",
        "",
        "## Spacing Sequence (use in order)",
        f"First 20: {config['spacing_sequence'][:20]}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
