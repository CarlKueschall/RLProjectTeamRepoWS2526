#!/bin/bash
#
# TD3 to DreamerV3 Migration Cleanup Script
# This script archives all TD3-specific files to 99-ARCHIVE/TD3/
#
# Run with: bash cleanup_for_dreamer.sh
# Or make executable: chmod +x cleanup_for_dreamer.sh && ./cleanup_for_dreamer.sh
#

set -e  # Exit on error

PROJECT_ROOT="/Users/carlkueschall/workspace/RLProjectHockey"
ARCHIVE_DIR="${PROJECT_ROOT}/99-ARCHIVE/TD3"
TD3_DIR="${PROJECT_ROOT}/02-SRC/TD3"

echo "============================================"
echo "TD3 to DreamerV3 Migration Cleanup"
echo "============================================"
echo ""
echo "This script will archive TD3-specific files to:"
echo "  ${ARCHIVE_DIR}"
echo ""
echo "Files that are reusable (self-play, evaluation, metrics, visualization)"
echo "will remain in place."
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

# Create archive directory structure
echo ""
echo "Creating archive directory structure..."
mkdir -p "${ARCHIVE_DIR}/agents"
mkdir -p "${ARCHIVE_DIR}/rewards"
mkdir -p "${ARCHIVE_DIR}/config"
mkdir -p "${ARCHIVE_DIR}/sbatch"
mkdir -p "${ARCHIVE_DIR}/analysis"
mkdir -p "${ARCHIVE_DIR}/guides"
mkdir -p "${ARCHIVE_DIR}/plans"
mkdir -p "${ARCHIVE_DIR}/pbrs_docs"
mkdir -p "${ARCHIVE_DIR}/reports"
mkdir -p "${ARCHIVE_DIR}/docs"
mkdir -p "${ARCHIVE_DIR}/wandb_exports"
mkdir -p "${ARCHIVE_DIR}/logs"
mkdir -p "${ARCHIVE_DIR}/checkpoints"
mkdir -p "${ARCHIVE_DIR}/results"
mkdir -p "${ARCHIVE_DIR}/videos"
mkdir -p "${ARCHIVE_DIR}/tests"

echo "Done."

# ============================================
# ARCHIVE TD3 AGENT IMPLEMENTATION
# ============================================
echo ""
echo "Archiving TD3 agent implementation..."

# Move agent files (these are TD3-specific)
if [ -f "${TD3_DIR}/agents/td3_agent.py" ]; then
    mv "${TD3_DIR}/agents/td3_agent.py" "${ARCHIVE_DIR}/agents/"
    echo "  Moved: agents/td3_agent.py"
fi

if [ -f "${TD3_DIR}/agents/model.py" ]; then
    mv "${TD3_DIR}/agents/model.py" "${ARCHIVE_DIR}/agents/"
    echo "  Moved: agents/model.py"
fi

if [ -f "${TD3_DIR}/agents/memory.py" ]; then
    mv "${TD3_DIR}/agents/memory.py" "${ARCHIVE_DIR}/agents/"
    echo "  Moved: agents/memory.py"
fi

if [ -f "${TD3_DIR}/agents/noise.py" ]; then
    mv "${TD3_DIR}/agents/noise.py" "${ARCHIVE_DIR}/agents/"
    echo "  Moved: agents/noise.py"
fi

if [ -f "${TD3_DIR}/agents/device.py" ]; then
    mv "${TD3_DIR}/agents/device.py" "${ARCHIVE_DIR}/agents/"
    echo "  Moved: agents/device.py"
fi

if [ -f "${TD3_DIR}/agents/__init__.py" ]; then
    cp "${TD3_DIR}/agents/__init__.py" "${ARCHIVE_DIR}/agents/"
    echo "  Copied: agents/__init__.py (kept original for reference)"
fi

# ============================================
# ARCHIVE PBRS REWARD SHAPING
# ============================================
echo ""
echo "Archiving PBRS reward shaping..."

if [ -f "${TD3_DIR}/rewards/pbrs.py" ]; then
    mv "${TD3_DIR}/rewards/pbrs.py" "${ARCHIVE_DIR}/rewards/"
    echo "  Moved: rewards/pbrs.py"
fi

if [ -f "${TD3_DIR}/rewards/base.py" ]; then
    mv "${TD3_DIR}/rewards/base.py" "${ARCHIVE_DIR}/rewards/"
    echo "  Moved: rewards/base.py"
fi

if [ -f "${TD3_DIR}/rewards/__init__.py" ]; then
    cp "${TD3_DIR}/rewards/__init__.py" "${ARCHIVE_DIR}/rewards/"
    echo "  Copied: rewards/__init__.py"
fi

# ============================================
# ARCHIVE TRAINING SCRIPTS
# ============================================
echo ""
echo "Archiving training scripts..."

if [ -f "${TD3_DIR}/train_hockey.py" ]; then
    mv "${TD3_DIR}/train_hockey.py" "${ARCHIVE_DIR}/"
    echo "  Moved: train_hockey.py"
fi

# Archive config (TD3-specific parser)
if [ -f "${TD3_DIR}/config/parser.py" ]; then
    mv "${TD3_DIR}/config/parser.py" "${ARCHIVE_DIR}/config/"
    echo "  Moved: config/parser.py"
fi

if [ -f "${TD3_DIR}/config/__init__.py" ]; then
    cp "${TD3_DIR}/config/__init__.py" "${ARCHIVE_DIR}/config/"
    echo "  Copied: config/__init__.py"
fi

# ============================================
# ARCHIVE SBATCH FILES
# ============================================
echo ""
echo "Archiving sbatch files..."

# From TD3 directory
for f in "${TD3_DIR}"/train_hockey*.sbatch; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/sbatch/"
        echo "  Moved: $(basename $f)"
    fi
done

# From project root
for f in "${PROJECT_ROOT}"/train_hockey*.sbatch; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/sbatch/"
        echo "  Moved: $(basename $f) (from root)"
    fi
done

if [ -f "${PROJECT_ROOT}/tournament_client.sbatch" ]; then
    mv "${PROJECT_ROOT}/tournament_client.sbatch" "${ARCHIVE_DIR}/sbatch/"
    echo "  Moved: tournament_client.sbatch"
fi

# ============================================
# ARCHIVE MARKDOWN DOCUMENTATION
# ============================================
echo ""
echo "Archiving markdown documentation..."

# Analysis files
for f in "${TD3_DIR}"/ANALYSIS_*.md; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/analysis/"
        echo "  Moved: $(basename $f)"
    fi
done

# Guide files
for f in "${TD3_DIR}"/*_GUIDE.md; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/guides/"
        echo "  Moved: $(basename $f)"
    fi
done

# Plan files
for f in "${TD3_DIR}"/*_PLAN.md; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/plans/"
        echo "  Moved: $(basename $f)"
    fi
done

# PBRS documentation
for f in "${TD3_DIR}"/PBRS_*.md; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/pbrs_docs/"
        echo "  Moved: $(basename $f)"
    fi
done

# Performance reports
for f in "${TD3_DIR}"/PERFORMANCE_REPORT_*.md; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/reports/"
        echo "  Moved: $(basename $f)"
    fi
done

# Other docs
if [ -f "${TD3_DIR}/REWARD_SHAPING_RESEARCH_PROMPT.md" ]; then
    mv "${TD3_DIR}/REWARD_SHAPING_RESEARCH_PROMPT.md" "${ARCHIVE_DIR}/docs/"
    echo "  Moved: REWARD_SHAPING_RESEARCH_PROMPT.md"
fi

# ============================================
# ARCHIVE WANDB EXPORTS AND LOGS
# ============================================
echo ""
echo "Archiving W&B exports and logs..."

# W&B text exports
for f in "${TD3_DIR}"/wandb_run_*.txt; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/wandb_exports/"
        echo "  Moved: $(basename $f)"
    fi
done

# Training logs
if [ -f "${TD3_DIR}/training.log" ]; then
    mv "${TD3_DIR}/training.log" "${ARCHIVE_DIR}/logs/"
    echo "  Moved: training.log"
fi

for f in "${TD3_DIR}"/worker_*.log; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/logs/"
        echo "  Moved: $(basename $f)"
    fi
done

# W&B directory (if exists and not empty)
if [ -d "${TD3_DIR}/wandb" ]; then
    mv "${TD3_DIR}/wandb" "${ARCHIVE_DIR}/"
    echo "  Moved: wandb/"
fi

# W&B analysis directory
if [ -d "${TD3_DIR}/wandb_analysis" ]; then
    mv "${TD3_DIR}/wandb_analysis" "${ARCHIVE_DIR}/"
    echo "  Moved: wandb_analysis/"
fi

# ============================================
# ARCHIVE CHECKPOINTS
# ============================================
echo ""
echo "Archiving checkpoints..."

for f in "${TD3_DIR}"/results_checkpoints_*.pth; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/checkpoints/"
        echo "  Moved: $(basename $f)"
    fi
done

# Results directory
if [ -d "${TD3_DIR}/results" ]; then
    mv "${TD3_DIR}/results" "${ARCHIVE_DIR}/"
    echo "  Moved: results/"
fi

# ============================================
# ARCHIVE TEST VIDEOS
# ============================================
echo ""
echo "Archiving test videos..."

for f in "${TD3_DIR}"/*.mp4; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE_DIR}/videos/"
        echo "  Moved: $(basename $f)"
    fi
done

# ============================================
# ARCHIVE TESTS
# ============================================
echo ""
echo "Archiving TD3-specific tests..."

if [ -f "${TD3_DIR}/tests/test_pbrs_potential.py" ]; then
    mv "${TD3_DIR}/tests/test_pbrs_potential.py" "${ARCHIVE_DIR}/tests/"
    echo "  Moved: tests/test_pbrs_potential.py"
fi

# ============================================
# CLEANUP EMPTY DIRECTORIES
# ============================================
echo ""
echo "Cleaning up empty directories..."

# Remove agents dir if empty (we'll recreate for DreamerV3)
if [ -d "${TD3_DIR}/agents" ] && [ -z "$(ls -A ${TD3_DIR}/agents 2>/dev/null)" ]; then
    rmdir "${TD3_DIR}/agents"
    echo "  Removed empty: agents/"
fi

# Remove rewards dir if empty
if [ -d "${TD3_DIR}/rewards" ] && [ -z "$(ls -A ${TD3_DIR}/rewards 2>/dev/null)" ]; then
    rmdir "${TD3_DIR}/rewards"
    echo "  Removed empty: rewards/"
fi

# Remove config dir if empty
if [ -d "${TD3_DIR}/config" ] && [ -z "$(ls -A ${TD3_DIR}/config 2>/dev/null)" ]; then
    rmdir "${TD3_DIR}/config"
    echo "  Removed empty: config/"
fi

# Remove tests dir if empty
if [ -d "${TD3_DIR}/tests" ] && [ -z "$(ls -A ${TD3_DIR}/tests 2>/dev/null)" ]; then
    rmdir "${TD3_DIR}/tests"
    echo "  Removed empty: tests/"
fi

# ============================================
# RENAME TD3 DIRECTORY
# ============================================
echo ""
echo "Renaming TD3 directory to DreamerV3..."

if [ -d "${TD3_DIR}" ]; then
    mv "${TD3_DIR}" "${PROJECT_ROOT}/02-SRC/DreamerV3"
    echo "  Renamed: 02-SRC/TD3 -> 02-SRC/DreamerV3"
fi

# ============================================
# SUMMARY
# ============================================
echo ""
echo "============================================"
echo "CLEANUP COMPLETE"
echo "============================================"
echo ""
echo "Archived to: ${ARCHIVE_DIR}"
echo ""
echo "Files KEPT in 02-SRC/DreamerV3/ (reusable):"
echo "  - opponents/ (self-play, PFSP)"
echo "  - evaluation/ (evaluator)"
echo "  - metrics/ (metrics_tracker)"
echo "  - visualization/ (gif_recorder, frame_capture)"
echo "  - test_hockey.py (adapt for DreamerV3)"
echo "  - download_wandb_run.py"
echo "  - requirements.txt"
echo ""
echo "Next steps:"
echo "  1. Review the archived files in 99-ARCHIVE/TD3/"
echo "  2. Update 02-SRC/DreamerV3/__init__.py"
echo "  3. Create new agents/ directory for DreamerV3"
echo "  4. Follow IMPLEMENTATION_PLAN.md for Phase 1"
echo ""
echo "Documentation:"
echo "  - 01-DOCS/TD3_JOURNEY.md (TD3 lessons learned)"
echo "  - 01-DOCS/DreamerV3/IMPLEMENTATION_PLAN.md (full plan)"
echo "  - 01-DOCS/DreamerV3/RESEARCH_RESULTS.md (research)"
echo ""
