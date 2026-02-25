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

Create one 16:9 results slide (1920x1080), same deck style, focused on robust checkpoint selection.

Reference handling:
- If benchmark progression plot is attached, embed it prominently.
- If not attached, render a clean synthetic line-chart style placeholder with the same narrative shape (late-stage strong checkpoints highlighted).

Style:
- Dark background #0F172A, Inter typography.
- Primary text #E5E7EB.
- Emphasis colors:
  - Highlight checkpoints in emerald #34D399
  - Robustness metrics in cyan #22D3EE
  - Warnings/caveats in orange #FB923C

Layout:
- Top: title and one-line thesis.
- Middle: chart area (about 65% width) + right insight panel (35% width).
- Bottom: shortlist bar.

Exact text:
Title:
"Results: Robust Selection Beats Single-Metric Picking"

Thesis line:
"Fixed-bot performance saturated; robust cross-play metrics became the key discriminator."

Right panel heading:
"Selection signals"

Right panel bullets:
- "probe_mean_win_rate"
- "probe_min_win_rate"
- "robust score / CP_worst"

Bottom shortlist text:
"Late robust shortlist: 556k, 612k, 634k"

Visual requirements:
- Clearly mark 556k, 612k, 634k on chart or in adjacent callouts.
- Include a small caution tag: "Non-transitive matchups require robustness metrics".
- Keep numbers concise; prioritize interpretability.

Validation:
- Chart + right panel must feel like one argument, not two unrelated blocks.
- Shortlist must be visually unmissable.
- No tiny unreadable axis labels.
