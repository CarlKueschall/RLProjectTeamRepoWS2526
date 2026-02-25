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

Create a single 16:9 technical presentation slide (1920x1080) with a polished, modern academic look.

Design system:
- Font family: Inter (fallback Helvetica).
- Background: very dark slate (#0F172A).
- Primary text: off-white (#E5E7EB).
- Accent 1: cyan (#22D3EE).
- Accent 2: orange (#FB923C).
- Accent 3: emerald (#34D399).
- Use subtle shadows, no gradients except very soft background vignette.
- Strictly no logos, no watermark, no slide numbers.

Layout:
- Top 25%: title and subtitle block.
- Middle 45%: concise core-message bullets (left) + pipeline strip (right).
- Bottom 30%: one short “thesis sentence” in highlighted callout bar.

Exact on-slide text:
Title:
"DreamerV3 Hockey: From Sparse Rewards to Tournament-Ready Policy"

Subtitle:
"World-model RL with recursive self-play and curated league training"

Bullets (left):
- "Challenge: sparse rewards (+10 / -10 / 0) in adversarial continuous control"
- "Method: DreamerV3 + self-play PFSP + curated opponent league"
- "Key insight: late gains come from league-dominant self-play, not fixed-bot overfitting"

Pipeline strip (right, connected rounded boxes with arrows):
"Scratch Training" -> "Self-Play Expansion" -> "League Curation" -> "Final Checkpoint"

Bottom callout text:
"Core message: Robustness under opponent diversity is the primary objective."

Visual tone:
- Professional, sharp, non-generic.
- High readability from distance.
- Keep text concise and balanced; avoid crowding.

Validation:
- Ensure all text is spelled exactly as above.
- Ensure strong visual hierarchy: title > bullets > callout.
- Ensure pipeline is clear at first glance.
