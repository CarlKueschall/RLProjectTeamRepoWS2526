0. Visual Identity Specification

All generated images must conform to the following design system:

Typography:
- Primary typeface: Inter (fallback: Helvetica Neue, SF Pro Display)
- Headings: Inter SemiBold or Inter Bold, tracked at -0.02em
- Body/labels: Inter Regular, 14–16pt equivalent
- Code/math: JetBrains Mono or SF Mono
- No serif fonts. No handwriting fonts. No decorative type.

Color Palette (Anthropic-derived):
┌────────────────────────┬─────────────────┬─────────┬───────────────────────────────────────────┐
│          Role          │      Color      │   Hex   │                   Usage                   │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Background (dark)      │ Warm Charcoal   │ #1A1A1A │ Primary canvas                            │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Background (alt)       │ Deep Clay       │ #2D2520 │ Panel backgrounds, cards                  │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Surface                │ Warm Ash        │ #3A3530 │ Elevated containers, boxes                │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Primary Accent         │ Anthropic Coral │ #E07A5F │ Headers, key highlights, primary callouts │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Secondary Accent       │ Warm Amber      │ #D4A574 │ Secondary labels, connecting lines        │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Tertiary Accent        │ Soft Sand       │ #E8D5B7 │ Tertiary elements, subtle borders         │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Text Primary           │ Warm White      │ #F5F0EB │ Body text, equations                      │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Text Secondary         │ Muted Cream     │ #B8AFA6 │ Captions, annotations                     │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Semantic: Constraint   │ Dusty Rose      │ #C97C7C │ Constraints, boundaries, limits           │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Semantic: Objective    │ Slate Blue      │ #7C9CB8 │ Objectives, targets, goals                │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Semantic: Solution     │ Warm Gold       │ #D4A843 │ Optimal points, solutions, results        │
├────────────────────────┼─────────────────┼─────────┼───────────────────────────────────────────┤
│ Semantic: Danger/Error │ Burnt Sienna    │ #C4553A │ Misconceptions, errors, warnings          │
└────────────────────────┴─────────────────┴─────────┴───────────────────────────────────────────┘
Visual Grammar:
- Corners: 8px radius on all containers. No sharp rectangles. No fully circular panels.
- Borders: 1px #3A3530 on cards. Use Anthropic Coral borders only for primary emphasis.
- Shadows: Subtle warm-toned drop shadows (rgba(224, 122, 95, 0.08)), never cool/blue shadows.
- Spacing: 16px base grid. Breathe. Whitespace is a feature, not wasted space.
- Lines/arrows: 2px stroke, rounded caps. Use Warm Amber for connectors, Anthropic Coral for emphasis arrows.
- No gradients except subtle vignettes on the background. Flat color fills only.
- No stock imagery, no photographic textures, no decorative shapes without semantic purpose.

---

Generate one 16:9 slide image (1920x1080), clean and high-density, explaining the league curation pipeline.

Visual identity:
- Dark background #0F172A
- Inter type
- Text #E5E7EB
- Process arrows in cyan #22D3EE
- Decision/selection highlights in orange #FB923C
- Success outputs in emerald #34D399

Layout:
- Top: title.
- Middle: left-to-right pipeline with 5 process nodes.
- Bottom: two compact metric cards for selection criteria.

Exact text:
Title:
"Self-Play and League Curation Pipeline"

Pipeline nodes (left to right):
1. "Collect candidate checkpoints\n(multi-run, multi-seed)"
2. "Summit matrix evaluation\n(vs fixed + checkpoint opponents)"
3. "Score robustness\n(CP_worst, quantiles, robust score)"
4. "Greedy quality + diversity selection"
5. "Weighted league bootstrap for continuation"

Bottom card A title:
"Quality"
Bottom card A body:
"High robust cross-play against diverse opponents"

Bottom card B title:
"Diversity"
Bottom card B body:
"Avoid redundant policies; maximize behavioral coverage"

Right-side stamp:
"Output: curated league pool for final training"

Visual instructions:
- Use rounded rectangular nodes with clear directional arrows.
- Keep line spacing tight but readable.
- Make node 4 visually emphasized (orange border/glow).

Validation:
- Pipeline must read clearly in under 3 seconds.
- No ambiguous loops.
- Selection criteria must be visibly tied to node 4.
