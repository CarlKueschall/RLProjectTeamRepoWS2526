# Slide 5 — Self-Play and League Curation Pipeline

## On-Slide Text
- Build large candidate set from multiple runs/seeds
- Evaluate via summit matrix:
  - candidate vs fixed weak/strong
  - candidate vs checkpoint opponents
- Select pool by blended objective:
  - quality (robust cross-play)
  - diversity (non-redundant behaviors)
- Continue training from top checkpoint with weighted league bootstrap

## Visual
Use a compact process flow:
`Candidate checkpoints -> Matrix eval -> Quality + Diversity ranking -> Weighted league pool -> Final continuation run`

## Speaker Notes (25s)
"We moved from ad-hoc self-play to an explicit league-curation pipeline. Instead of picking by average win rate only, we rank checkpoints with robustness-aware metrics and a diversity term, then bootstrap continuation training with weighted sampling from that curated pool."

## Slide Build
1. Process flow line.
2. Reveal quality and diversity criterion boxes.
