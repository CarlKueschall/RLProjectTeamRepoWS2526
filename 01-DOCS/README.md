# 01-DOCS Documentation Index

**Last Updated:** January 2025
**Project:** RL Hockey Agent with TD3 and Self-Play

---

## 📚 Quick Navigation

### For Active Development
- **Self-Play Debugging**: [`implementation/self-play/`](#self-play-implementation)
- **Monitoring & Metrics**: [`implementation/monitoring/`](#monitoring--metrics)
- **Bug Tracking**: [`debugging/`](#debugging--analysis)

### For Report Writing
- **LaTeX Files**: [`report/INPUT/`](#report-materials)
- **Figures**: [`report/INPUT/figures/`](#report-materials)
- **Writing Guides**: [`report/INPUT/guides/`](#report-materials)

### For Understanding Methodology
- **Reward Design**: [`methodology/reward-design/`](#reward-design)
- **Research Background**: [`methodology/research/`](#research-materials)
- **Algorithm Theory**: [`methodology/TD3-paper.md`](#algorithm-documentation)

---

## 📂 Detailed Structure

### Methodology

#### Reward Design
**Location:** `methodology/reward-design/`

Evolutionary development of Potential-Based Reward Shaping (PBRS) for the hockey environment.

- **`PBRS_REWARD_DESIGN_V2.md`** - Initial reward function design with basic shaping
- **`PBRS_REWARD_DESIGN_V3.md`** - Refined version with improved potential function
- **`PBRS_REWARD_DESIGN_V3.1.md`** ⭐ **CURRENT** - Latest iteration with optimized hyperparameters

💡 *Tip: Review all versions to understand the design evolution and rationale behind current choices.*

#### Research Materials
**Location:** `methodology/research/`

- **`pbrs_research_prompt.md`** - Comprehensive research questions and prompts for optimal PBRS design
- **`comprehensive_perplexity_research_results.md`** - External research findings and literature review results

#### Algorithm Documentation
**Location:** `methodology/`

- **`TD3-paper.md`** - Core Twin Delayed Deep Deterministic Policy Gradient algorithm documentation and implementation notes

---

### Implementation

#### Self-Play Implementation
**Location:** `implementation/self-play/`

Documentation for self-play training system implementation and debugging.

- **`SELFPLAY_STATUS_LOGGING.md`** - Logging system for tracking self-play training status
- **`SELFPLAY_ACTIVATION_BUG_FIX.md`** - Critical bug fix for self-play activation logic
- **`SELFPLAY_METRICS_DEBUG.md`** - Debugging metrics collection during self-play
- **`SELF-PLAY_FIXES_SUMMARY.md`** - Comprehensive summary of all self-play fixes applied

🔧 *Use case: Start here when debugging self-play behavior or understanding the training loop.*

#### Monitoring & Metrics
**Location:** `implementation/monitoring/`

- **`WANDB_METRICS_GUIDE.md`** - Guide to Weights & Biases metrics dashboard and key metrics to track
- **`images/`** - Screenshots and diagrams supporting monitoring documentation

#### Tournament & Evaluation
**Location:** `implementation/`

- **`TOURNAMENT_KEEP_MODE.md`** - Documentation for tournament keep mode feature and opponent selection

---

### Debugging & Analysis

**Location:** `debugging/`

Critical analyses and investigations into system behavior.

- **`ROOT_CAUSE_ANALYSIS.md`** - Deep dive root cause analysis of major issues
- **`COMPREHENSIVE_CODE_AUDIT.md`** - Full codebase audit findings and recommendations
- **`TA_INQUIRY_WEAK_BOT_DISCREPANCY.md`** - Investigation into discrepancies with weak bot performance

🐛 *Start here when investigating unexpected behavior or performance issues.*

---

### Report Materials

**Location:** `report/`

#### LaTeX Documents
**Location:** `report/INPUT/`

- **`main.tex`** ⭐ **PRIMARY** - Main report document
- **`main_full.tex`** - Extended version with additional sections
- **`main_AI.tex`** - AI-focused version of the report
- **`main.bib`** - Bibliography and references
- **`latex_template.tex`** - Template for formatting

#### Figures
**Location:** `report/INPUT/figures/`

Organized experimental results and visualizations:
- `normal_vs_Self-Play/` - Self-play vs normal training comparisons
- `Hidden_Size_Comparisons/` - Network architecture experiments
- `strong-vs-weak-opponent/` - Opponent difficulty analysis
- `PSFP_weighting_curve/` - Prioritized Fictitious Self-Play weighting
- `RS_vs_no_RS/` - Reward shaping impact analysis

#### Writing Guides
**Location:** `report/INPUT/guides/`

Structured guides for report writing:
- **`INDEX.md`** - Overview of all guides
- **`README_GUIDES.md`** - How to use the guide system
- **`QUICK_REFERENCE.md`** - Quick lookup for common patterns
- **`REVISED_TOC.md`** - Report table of contents and structure
- **`GUIDE_OVERALL_STRATEGY.md`** - High-level writing strategy
- **`GUIDE_COMPACT_APPROACH.md`** - Approach for concise writing
- **`GUIDE_TD3_METHOD_SECTION.md`** - How to write the TD3 methodology section
- **`GUIDE_TD3_EXPERIMENTS_SECTION.md`** - How to write TD3 experiments section
- **`GUIDE_SELF_PLAY_SECTION.md`** - How to write self-play section

📝 *Use these guides to maintain consistency and quality when writing the final report.*

---

### Data

**Location:** `data/`

#### Game Replays
**Location:** `data/game-replays/tournament/`

Experimental data from tournament evaluations:
- `tournament_game_data_against_weak.pkl` - Games vs weak opponent
- `tournament_game_data_against_strong.pkl` - Games vs strong opponent

---

## 🔍 Common Tasks

### "I need to update the reward function"
1. Review version history: `methodology/reward-design/`
2. Edit current version: `PBRS_REWARD_DESIGN_V3.1.md`
3. Document changes in a new version file if significant

### "Self-play isn't working correctly"
1. Check known issues: `implementation/self-play/SELF-PLAY_FIXES_SUMMARY.md`
2. Review metrics: `implementation/monitoring/WANDB_METRICS_GUIDE.md`
3. If new issue, document in `debugging/`

### "I'm writing the methods section"
1. Start with: `report/INPUT/guides/GUIDE_TD3_METHOD_SECTION.md`
2. Reference: `methodology/TD3-paper.md`
3. Edit: `report/INPUT/main.tex`

### "I need to understand what went wrong"
1. Recent fixes: `implementation/self-play/SELF-PLAY_FIXES_SUMMARY.md`
2. Deep investigations: `debugging/ROOT_CAUSE_ANALYSIS.md`
3. Code quality: `debugging/COMPREHENSIVE_CODE_AUDIT.md`

---

## 📌 Document Naming Conventions

- **UPPERCASE.md** - Primary documentation and guides
- **lowercase.md** - Supporting materials and auxiliary docs
- **V2, V3, V3.1** - Version numbers for iterative documents (keep all versions)
- **_GUIDE** suffix - Writing/usage guides
- **_FIX** suffix - Bug fix documentation

---

## 🎯 Version Control Note

All files in this directory are git version-controlled. Feel free to edit and experiment - you can always revert changes through git history.

---

**Questions or suggestions for improving this documentation structure?**
Open an issue or update this README directly.
