# 01-DOCS/REPORT/INPUT Directory Index

## Files Overview

### LaTeX Report Files (The Actual Report)

| File | Size | Purpose |
|---|---|---|
| **report_main.tex** | 8.5 KB | **MAIN REPORT FILE** - Edit this, fill in [TODO] sections |
| **main.bib** | 1.9 KB | Bibliography database (auto-populated with 6 key papers) |
| **latex_template.tex** | 1.0 KB | Original template (reference only, don't edit) |

---

### Guide Documents (How to Write Your Report)

| File | Size | Purpose | When to Read |
|---|---|---|---|
| **README_GUIDES.md** | 5.4 KB | Navigation guide explaining all guides | First (2 min read) |
| **QUICK_REFERENCE.md** | 6.4 KB | One-page checklist and quick lookup | During writing (print it!) |
| **GUIDE_COMPACT_APPROACH.md** | 10 KB | **PRIMARY GUIDE** - How to fill [TODO] sections | While filling TODOs |
| **GUIDE_OVERALL_STRATEGY.md** | 18 KB | Overall report structure and strategy | For context (quick skim) |
| **GUIDE_TD3_METHOD_SECTION.md** | 14 KB | Detailed TD3 algorithm explanation | For understanding (reference) |
| **GUIDE_TD3_EXPERIMENTS_SECTION.md** | 19 KB | Detailed experiment design | For understanding (reference) |
| **GUIDE_SELF_PLAY_SECTION.md** | 20 KB | Detailed self-play system explanation | For understanding (reference) |

---

## How to Use This Directory

### Step 1: Understand Your Task
1. Read `README_GUIDES.md` (2 minutes)
2. Skim `GUIDE_OVERALL_STRATEGY.md` (5 minutes)
3. Print or bookmark `QUICK_REFERENCE.md` for while you work

### Step 2: Prepare Your Data
1. Extract all data from WandB, code, evaluations
2. Follow the **Data Extraction Checklist** in `GUIDE_COMPACT_APPROACH.md`

### Step 3: Create Figures & Tables
1. Open `GUIDE_COMPACT_APPROACH.md`
2. Find your [TODO] section
3. See what figure/table to create
4. Create and save as PDF/image

### Step 4: Write Prose
1. Keep to word limits (180 words Methods, 200 Experiments, 150 Self-Play, 250 Discussion = 780 total)
2. Use detailed guides (`GUIDE_TD3_METHOD_SECTION.md`, etc.) for inspiration on what to discuss
3. But DON'T write lengthy sections - keep it brief

### Step 5: Integrate Into LaTeX
1. Open `report_main.tex`
2. Find each [TODO] section
3. Replace with your figure/table
4. Compile with `pdflatex` + `bibtex`

---

## The Three [TODO] Sections You Must Fill

### 1. In Experiments: "Include training curves and quantitative results"
**Create:** 2-3 figure panels (architecture/LR/exploration ablations) + 1 results table
**See:** `GUIDE_COMPACT_APPROACH.md` → Section 3a

### 2. In Experiments: "Comparative metrics table"
**Create:** TD3 vs DDPG comparison table
**See:** `GUIDE_COMPACT_APPROACH.md` → Section 3b

### 3. In Self-Play: "Pool evolution visualization, PFSP weight analysis"
**Create:** Pool evolution figure + PFSP weighting curve
**See:** `GUIDE_COMPACT_APPROACH.md` → Section 4

---

## File Purposes Explained

### report_main.tex
**What it is:** Your actual report file that will be submitted.

**Structure:**
- Title + Table of Contents (auto-generated)
- Introduction: 0.25 pages (mostly done)
- Methods: 0.5 pages (fully done)
- Experiments: 0.75 pages (partial - has [TODOs])
- Self-Play: 0.5 pages (partial - has [TODO])
- Discussion: 0.5 pages (mostly done)
- References: Bibliography (auto-generated from main.bib)

**What to do:** Replace [TODO] sections with figures/tables

**Compilation:**
```bash
cd /Users/carlkueschall/workspace/RLProjectHockey/01-DOCS/REPORT/INPUT
pdflatex report_main.tex
bibtex report_main
pdflatex report_main.tex
pdflatex report_main.tex
# Output: report_main.pdf
```

---

### main.bib
**What it is:** Bibliography database (BibTeX format)

**Contains:** 6 papers
- Fujimoto et al. (TD3)
- Lillicrop et al. (DDPG)
- Haarnoja et al. (SAC)
- Hessel et al. (Rainbow)
- Wang et al. (Dueling Networks)
- Hafner et al. (Dreamer)

**What to do:** Don't edit unless adding new references. It's auto-populated and works with `report_main.tex`.

---

### README_GUIDES.md
**What it is:** Navigation guide explaining all 7 guide documents.

**Contains:**
- Which guide to use for what
- Key changes from original to compact approach
- Working approach (phases 1-4)
- FAQ

**When to read:** First thing (2-3 minutes)

---

### QUICK_REFERENCE.md
**What it is:** One-page quick lookup card for common tasks.

**Contains:**
- The three [TODO] sections and what to do for each
- Data extraction checklist
- Writing checklist
- Timeline
- Pro tips
- Figure/table creation code examples
- Common mistakes

**When to read:** Keep it printed/open while you work

**Best for:** Quick answers ("What should my table look like?" "How much should I write for Methods?")

---

### GUIDE_COMPACT_APPROACH.md (⭐ PRIMARY GUIDE)
**What it is:** Step-by-step instructions for filling [TODO] sections with figures/tables.

**Contains:**
- Detailed breakdown of each [TODO] section
- Exact figure specifications (axes, labels, what data to plot)
- Exact table structure (columns, rows, what data to include)
- Data extraction checklist (what WandB metrics you need)
- LaTeX code examples for inserting figures/tables
- Figure and table checklists
- Word count management
- Timeline

**When to read:** While actively filling [TODO] sections

**Best for:** "What figure should I create for [TODO #1]?"

---

### GUIDE_OVERALL_STRATEGY.md
**What it is:** High-level overview of report structure and strategy.

**Contains:**
- Compact vs original structure comparison
- Page budget breakdown
- Section-by-section overview
- Writing timeline
- Data organization tips
- Quality checklist

**When to read:** First 5 minutes (skim for context)

**Best for:** Understanding overall approach and page constraints

---

### GUIDE_TD3_METHOD_SECTION.md
**What it is:** In-depth explanation of what to write in Methods section.

**Contains:**
- TD3 background and principles
- Network architecture details
- Each of 4 domain modifications explained
- Exploration and reward shaping
- What figures to create
- Approximate word count targets

**Status:** Written for 1.5-page section, but you only need 0.5 pages. Use for LEARNING MATERIAL, not word count guide.

**When to read:** If you want to deeply understand TD3, or if writing your prose

**Best for:** "Why does Q-clipping matter?" "What is PBRS and why use α=0.005?"

---

### GUIDE_TD3_EXPERIMENTS_SECTION.md
**What it is:** In-depth explanation of what ablations to run and how to present results.

**Contains:**
- Ablation study specifications (architecture, LR, exploration, reward shaping)
- TD3 vs DDPG comparison details
- Evaluation methodology
- 6-8 figure specifications
- Results tables with exact columns/rows
- Data extraction guidance

**Status:** Written for 1.5-page section, but you only need 0.75 pages. Use for IDEAS, not word count guide.

**When to read:** If you want detailed ablation design, or need ideas for what to measure

**Best for:** "What should I measure for architecture ablation?" "What figures best show these results?"

---

### GUIDE_SELF_PLAY_SECTION.md
**What it is:** In-depth explanation of self-play system and how to present it.

**Contains:**
- Pool management details
- PFSP curriculum explanation
- Anti-forgetting mechanisms
- Performance gates and regression rollback
- What metrics to track
- Figure specifications

**Status:** Written for 0.75-page section, which MATCHES your current template. This is most aligned.

**When to read:** If you want deep understanding of your self-play system

**Best for:** "How do I visualize pool evolution?" "What should I say about PFSP?"

---

## Reading Order Recommendation

### If you have 5 minutes:
1. `README_GUIDES.md` (2 min)
2. `QUICK_REFERENCE.md` (3 min)
3. Print `QUICK_REFERENCE.md` for reference

### If you have 30 minutes:
1. `README_GUIDES.md` (5 min)
2. `GUIDE_OVERALL_STRATEGY.md` - skim structure (10 min)
3. `GUIDE_COMPACT_APPROACH.md` - read full (15 min)

### If you have 2 hours (comprehensive):
1. All of above
2. `GUIDE_TD3_METHOD_SECTION.md` - skim for TD3 understanding (30 min)
3. `GUIDE_TD3_EXPERIMENTS_SECTION.md` - skim for ablation ideas (30 min)
4. `GUIDE_SELF_PLAY_SECTION.md` - read for self-play details (20 min)

---

## Typical Workflow

```
START HERE (5 min)
    ↓
README_GUIDES.md
    ↓
Print QUICK_REFERENCE.md
    ↓
Extract data from WandB (2-3 hours)
    ↓
Follow GUIDE_COMPACT_APPROACH.md to fill [TODOs] (4-5 hours)
    ├─ Create figures
    ├─ Create tables
    └─ Insert into LaTeX
    ↓
Compile PDF and check layout (1 hour)
    ↓
Done! (Submit report_main.pdf)
```

---

## Common Questions

**Q: Which file should I edit?**
A: `report_main.tex` - this is your actual report.

**Q: Where do I find [TODO] sections?**
A: In `report_main.tex`, search for "[TODO:" to find all 3.

**Q: Should I read all 7 guides?**
A: No. Primary guide is `GUIDE_COMPACT_APPROACH.md`. Others are reference.

**Q: Can I delete the other guides after I'm done?**
A: Yes, but keep them until submission in case you need to revise.

**Q: How do I know if I'm doing it right?**
A: Check against `QUICK_REFERENCE.md` and `GUIDE_COMPACT_APPROACH.md` specs.

**Q: What if my data doesn't match the guide expectations?**
A: Use your actual data. The guides show typical specs; yours may differ slightly.

**Q: Should I modify the Methods section beyond what's written?**
A: Minimal. It's complete. Only add equations if you want to clarify for your reader.

---

## File Sizes Summary

```
Total guide size:      ~81 KB (for reference/learning)
Total LaTeX size:      ~12 KB (the actual report)
Total needed:          ~5 KB compiled PDF + figures/tables

Guidelines compliance:
✓ Maximum 8 pages (you'll use ~5)
✓ Maximum 2 people content
✓ References excluded from page count
✓ Compact and efficient
```

---

## Success Indicators

✓ You've read `README_GUIDES.md`
✓ You've printed or bookmarked `QUICK_REFERENCE.md`
✓ You understand the 3 [TODO] sections
✓ You know how to extract your data
✓ You can follow the figures/tables specifications
✓ You can compile `report_main.tex` to PDF
✓ You're ready to fill in the blanks!

---

**Ready to start? → Open `GUIDE_COMPACT_APPROACH.md` and follow it step-by-step.**

