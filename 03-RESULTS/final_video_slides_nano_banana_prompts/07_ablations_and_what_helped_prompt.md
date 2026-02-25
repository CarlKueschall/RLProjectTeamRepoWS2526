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

Generate one 16:9 ablation summary slide (1920x1080) with a crisp scientific style.

References:
- If provided, place these as small multiples:
  - DreamSmooth combined win rate
  - DreamSmooth sparse prediction error
  - Two-Hot combined win rate
- If references are missing, recreate equivalent schematic trend plots with correct qualitative message.

Style:
- Same deck identity (dark #0F172A, Inter, high contrast).
- Use color coding:
  - DreamSmooth findings in emerald/cyan
  - Two-Hot findings in amber/orange
  - Caveat text in muted gray-blue

Layout:
- Top: title.
- Middle: 3 mini-chart panels in a row.
- Right or bottom: conclusions card with 3 bullet verdicts.

Exact text:
Title:
"Ablations: What Helped, What Was Neutral"

Conclusion card heading:
"Interpretation"

Conclusion bullets:
- "DreamerV3 core already handles sparse hockey strongly"
- "DreamSmooth improved late-stage stability and sparse prediction quality"
- "Two-Hot vs Gaussian was mixed to neutral in this setup"

Caveat line (small text):
"Single-seed evidence is directional, not full statistical proof."

Visual constraints:
- Make DreamSmooth benefit visually obvious (green check icon).
- Mark Two-Hot as neutral/mixed (neutral icon, not red failure icon).
- Keep chart labels readable but compact.

Validation:
- Slide must communicate nuanced interpretation, not binary win/lose.
- No contradiction between charts and verdict text.
- Keep total text concise and legible.
