# RL Hockey Report — Overleaf Ready

Upload this entire folder to Overleaf. Set **main.tex** as the main document.

## Structure

```
REPORT/
├── main.tex      # Main document
├── main.bib      # Bibliography
├── figures/      # All figures (flat)
│   ├── final_benchmark_eval_progression_stitch.png
│   ├── dreamsmooth-*.png
│   ├── twohot-*.png
│   ├── training_phases.jpg
│   ├── ai-use.png
│   └── recommended_pool_tradeoff.png
└── README.md
```

## Compilation

- **Overleaf:** Upload folder, set main document to `main.tex`, compile.
- **Local:** `pdflatex main && bibtex main && pdflatex main && pdflatex main`
