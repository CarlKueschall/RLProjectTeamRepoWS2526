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

Create one 16:9 technical presentation slide (1920x1080) that centers the training curriculum insight.

Reference handling:
- If an image is attached (training phases visual), use it as the central hero visual.
- If no image is attached, recreate an equivalent 3-phase curriculum graphic from the text below.

Style:
- Same deck style: dark slate background #0F172A, Inter font.
- Main text #E5E7EB, accents cyan #22D3EE, orange #FB923C, emerald #34D399.

Layout:
- Top 15%: title.
- Middle 65%: hero visual (image or recreated diagram).
- Bottom 20%: concise quantitative takeaway band.

Exact text:
Title:
"Training Curriculum: Why Performance Improved Late"

Overlay labels on/near visual:
- "Phase A: Anchor competence (~20%)"
- "Phase B: Bootstrap self-play (~20%)"
- "Phase C: Curated-league self-play (~60%)"

Bottom takeaway text:
"Fixed-bot training is necessary but saturates early; most late gains come from curated league self-play."

If recreating without reference image:
- Build a clean 3-stage horizontal progression with increasing complexity and opponent diversity.
- Include small subtitles:
  - A: "Weak/strong anchors"
  - B: "PFSP self-play introduction"
  - C: "Weighted curated opponent pool"

Validation:
- The 20/20/60 split must be immediately visible.
- The visual should communicate progression, not three isolated blocks.
- Avoid text crowding; prioritize readability.
